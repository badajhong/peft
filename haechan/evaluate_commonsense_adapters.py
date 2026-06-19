#!/usr/bin/env python
"""Evaluate saved adapters on commonsense reasoning tasks."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B"
DEFAULT_EXPERIMENT_NAME = "commonsense"
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_NUM_BEAMS = 4
DEFAULT_DTYPE = "float32"
DEFAULT_SEED = 42
SUPPORTED_TASKS = ("boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa")
ANSWER_PATTERNS = {
    "boolq": r"true|false",
    "piqa": r"solution1|solution2",
    "social_i_qa": r"answer1|answer2|answer3|answer4|answer5",
    "ARC-Challenge": r"answer1|answer2|answer3|answer4|answer5",
    "ARC-Easy": r"answer1|answer2|answer3|answer4|answer5",
    "openbookqa": r"answer1|answer2|answer3|answer4|answer5",
    "hellaswag": r"ending1|ending2|ending3|ending4",
    "winogrande": r"option1|option2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PEFT adapters or merged checkpoints on commonsense reasoning test.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--adapter-root", default=None, help="Directory containing adapter/model subdirectories.")
    parser.add_argument(
        "--adapter-paths",
        nargs="*",
        default=[],
        help="Specific PEFT adapter dirs or merged model dirs to evaluate.",
    )
    parser.add_argument("--include-base", action="store_true", help="Also evaluate the base model.")
    parser.add_argument("--data-root", default=None, help="Root containing <task>/test.json.")
    parser.add_argument("--tasks", nargs="*", choices=SUPPORTED_TASKS, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-id", default=None, help="Override the model id inferred from adapter_config.json.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate every example.")
    parser.add_argument("--max-input-length", type=int, default=None, help="Maximum prompt length.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-beams", type=int, default=None, help="Beam count for deterministic generation.")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16", "fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print resolved config, data files, and targets.")
    return parser.parse_args()


def normalize_dtype(dtype_name: str) -> str:
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    return aliases.get(dtype_name, dtype_name)


def torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = normalize_dtype(dtype_name)
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolved_settings(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    dtype_name = normalize_dtype(args.dtype or DEFAULT_DTYPE)
    tasks = args.tasks or list(SUPPORTED_TASKS)
    unknown_tasks = [task for task in tasks if task not in SUPPORTED_TASKS]
    if unknown_tasks:
        raise ValueError(f"Unsupported commonsense tasks: {', '.join(unknown_tasks)}")

    return {
        "model_id": model_id,
        "tasks": list(tasks),
        "batch_size": args.batch_size or DEFAULT_EVAL_BATCH_SIZE,
        "num_beams": args.num_beams or DEFAULT_NUM_BEAMS,
        "dtype": dtype_name,
        "max_input_length": args.max_input_length or DEFAULT_MAX_INPUT_LENGTH,
        "seed": args.seed if args.seed is not None else DEFAULT_SEED,
    }


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(__file__).resolve().parent / "results" / "commonsense_eval" / f"{DEFAULT_EXPERIMENT_NAME}-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def candidate_data_roots(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.data_root:
        candidates.append(Path(args.data_root).expanduser())
    env_root = os.environ.get("COMMONSENSE_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates.extend(
        [
            Path.cwd() / "dataset",
            script_dir / "dataset",
            repo_root / "dataset",
            repo_root / "LLM-Adapters" / "dataset",
            repo_root.parent / "LLM-Adapters" / "dataset",
            repo_root / "LoRA-XS" / "data" / "commonsense",
            Path.home() / "haechan" / "LLM-Adapters" / "dataset",
            Path.home() / "haechan" / "cara_commonsense" / "dataset",
        ]
    )
    return [path.resolve() for path in candidates]


def find_data_root(args: argparse.Namespace, tasks: list[str]) -> Path:
    searched = []
    for root in candidate_data_roots(args):
        searched.append(root)
        if all((root / task / "test.json").exists() for task in tasks):
            return root
    searched_text = "\n".join(f"  - {path}" for path in searched)
    raise FileNotFoundError(
        "Could not find commonsense eval files. Expected <data-root>/<task>/test.json for all tasks.\n"
        f"Searched:\n{searched_text}\n"
        "Pass --data-root /path/to/dataset if your data is elsewhere."
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list or JSONL records in {path}")
    records = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at {path}:{index}")
        if "instruction" not in item or "answer" not in item:
            raise ValueError(f"Record {index} in {path} must contain instruction and answer")
        records.append(item)
    return records


def load_task_data(data_root: Path, task: str, max_samples: int) -> list[dict[str, Any]]:
    records = load_json_records(data_root / task / "test.json")
    if max_samples > 0:
        records = records[:max_samples]
    return records


def generate_prompt(instruction: str, input_text: str | None = None) -> str:
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def extract_answer(task: str, text: str) -> str:
    pattern = ANSWER_PATTERNS[task]
    matches = re.findall(pattern, text.strip().lower())
    return matches[0] if matches else ""


def normalize_label(label: Any) -> str:
    return str(label).strip().lower()


def batch_records(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=True) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "model",
        "path",
        "path_type",
        "task",
        "num_samples",
        "correct",
        "accuracy",
        "eval_seconds",
        "samples_per_second",
        "status",
        "error",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def is_adapter_path(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def is_merged_model_path(path: Path) -> bool:
    return (path / "config.json").exists() and not is_adapter_path(path)


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def adapter_base_model(path: Path) -> str | None:
    if not is_adapter_path(path):
        return None
    config = read_json_file(path / "adapter_config.json")
    base_model = config.get("base_model_name_or_path")
    return str(base_model) if base_model else None


def merged_model_id(path: Path) -> str | None:
    if not is_merged_model_path(path):
        return None
    config = read_json_file(path / "config.json")
    for key in ("_name_or_path", "name_or_path"):
        value = config.get(key)
        if value:
            return str(value)
    return str(path)


def infer_model_id(args: argparse.Namespace, target_paths: list[Path]) -> tuple[str, str]:
    if args.model_id:
        return args.model_id, "--model-id"

    inferred = []
    for path in target_paths:
        if is_adapter_path(path):
            base_model = adapter_base_model(path)
            if base_model:
                inferred.append((path, base_model))
        elif is_merged_model_path(path):
            model_id = merged_model_id(path)
            if model_id:
                inferred.append((path, model_id))

    unique_model_ids = sorted({model_id for _, model_id in inferred})
    if len(unique_model_ids) == 1:
        return unique_model_ids[0], "adapter_config"
    if len(unique_model_ids) > 1:
        details = "\n".join(f"  - {path}: {model_id}" for path, model_id in inferred)
        raise ValueError(
            "Evaluation targets were trained from different base models. "
            "Run them separately or pass --model-id explicitly.\n"
            f"{details}"
        )

    return DEFAULT_MODEL_ID, "default"


def path_label(path: Path | None) -> str:
    if path is None:
        return "base"
    if path.name.startswith("checkpoint-") and path.parent.name:
        return f"{path.parent.name}-{path.name}"
    return path.name


def discover_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item).expanduser().resolve() for item in args.adapter_paths]
    if args.adapter_root:
        root = Path(args.adapter_root).expanduser().resolve()
        if is_adapter_path(root) or is_merged_model_path(root):
            paths.append(root)
        elif root.exists():
            for config_path in sorted(root.rglob("adapter_config.json")):
                paths.append(config_path.parent.resolve())
            for config_path in sorted(root.rglob("config.json")):
                candidate = config_path.parent.resolve()
                if not is_adapter_path(candidate):
                    paths.append(candidate)
        else:
            raise FileNotFoundError(f"Adapter root does not exist: {root}")

    unique_paths = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        if not is_adapter_path(path) and not is_merged_model_path(path):
            raise FileNotFoundError(f"Expected adapter_config.json or config.json in: {path}")
        unique_paths.append(path)
        seen.add(path)
    return unique_paths


def load_tokenizer(model_id: str, max_input_length: int, trust_remote_code: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tokenizer.model_max_length = max_input_length
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_model(
    *,
    model_id: str,
    target_path: Path | None,
    dtype_name: str,
    attn_implementation: str | None,
    trust_remote_code: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, str | None]:
    path_type = None
    model_source = str(target_path) if target_path is not None and is_merged_model_path(target_path) else model_id
    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_source,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if normalize_dtype(dtype_name) != "float32":
        kwargs["torch_dtype"] = torch_dtype(dtype_name)

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    if target_path is not None and is_adapter_path(target_path):
        path_type = "adapter"
        model = PeftModel.from_pretrained(model, target_path)
    elif target_path is not None:
        path_type = "merged"

    model.to(device)
    model.eval()
    return model, path_type


def cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=max_input_length,
        padding=True,
        return_tensors="pt",
    ).to(device)
    generation_config = GenerationConfig(num_beams=num_beams, do_sample=False)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def evaluate_task(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    label: str,
    task: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    path: Path | None,
    path_type: str | None,
    args: argparse.Namespace,
    settings: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    correct = 0
    prediction_rows = []

    batches = batch_records(records, settings["batch_size"])
    for batch_index, batch in enumerate(tqdm(batches, desc=f"{label}/{task}", leave=False)):
        prompts = [generate_prompt(str(item["instruction"]), item.get("input")) for item in batch]
        outputs = generate_batch(
            model,
            tokenizer,
            prompts,
            max_input_length=settings["max_input_length"],
            max_new_tokens=args.max_new_tokens,
            num_beams=settings["num_beams"],
        )
        for offset, (item, output) in enumerate(zip(batch, outputs)):
            prediction = extract_answer(task, output)
            gold = normalize_label(item["answer"])
            is_correct = prediction == gold
            correct += int(is_correct)
            prediction_rows.append(
                {
                    "model": label,
                    "path": str(path) if path is not None else None,
                    "path_type": path_type or "base",
                    "task": task,
                    "id": item.get("id", batch_index * settings["batch_size"] + offset),
                    "instruction": item["instruction"],
                    "prompt": prompts[offset],
                    "generation": output,
                    "prediction": prediction,
                    "answer": gold,
                    "correct": is_correct,
                }
            )

    seconds = time.perf_counter() - start
    total = len(records)
    row = {
        "model": label,
        "path": str(path) if path is not None else None,
        "path_type": path_type or "base",
        "task": task,
        "num_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "eval_seconds": seconds,
        "samples_per_second": total / seconds if seconds > 0 else 0.0,
        "status": "success",
    }

    if args.save_predictions:
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(pred_dir / f"{label}-{task}.jsonl", prediction_rows)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row["model"]
        if row.get("status") != "success" or row.get("task") == "mean":
            continue
        model_summary = by_model.setdefault(model, {"tasks": {}, "mean_accuracy": 0.0})
        model_summary["tasks"][row["task"]] = row["accuracy"]

    for model, model_summary in by_model.items():
        values = list(model_summary["tasks"].values())
        model_summary["mean_accuracy"] = sum(values) / len(values) if values else 0.0

    return {"models": by_model}


def main() -> None:
    args = parse_args()
    target_paths = discover_paths(args)
    model_id, model_id_source = infer_model_id(args, target_paths)
    settings = resolved_settings(args, model_id)
    if not settings["model_id"]:
        raise ValueError("Could not resolve model id. Pass --model-id or evaluate an adapter with adapter_config.json.")

    set_seed(settings["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = make_output_dir(args)
    data_root = find_data_root(args, settings["tasks"])
    targets: list[tuple[str, Path | None]] = []
    if args.include_base:
        targets.append(("base", None))
    targets.extend((path_label(path), path) for path in target_paths)
    if not targets:
        raise ValueError("No evaluation targets. Pass --adapter-root, --adapter-paths, or --include-base.")

    task_data = {task: load_task_data(data_root, task, args.max_samples) for task in settings["tasks"]}

    with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(vars(args)), handle, indent=2, ensure_ascii=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            jsonable(
                {
                    "settings": settings,
                    "defaults": {
                        "model_id": DEFAULT_MODEL_ID,
                        "batch_size": DEFAULT_EVAL_BATCH_SIZE,
                        "max_input_length": DEFAULT_MAX_INPUT_LENGTH,
                        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                        "num_beams": DEFAULT_NUM_BEAMS,
                        "dtype": DEFAULT_DTYPE,
                        "seed": DEFAULT_SEED,
                        "tasks": list(SUPPORTED_TASKS),
                    },
                    "model_id_source": model_id_source,
                    "data_root": data_root,
                    "targets": [{"label": label, "path": path} for label, path in targets],
                }
            ),
            handle,
            indent=2,
            ensure_ascii=True,
        )

    print(f"python={Path(sys.executable).resolve()} version={sys.version.split()[0]}")
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'} output_dir={output_dir}")
    print(f"model={settings['model_id']} source={model_id_source} dtype={settings['dtype']}")
    print(f"data_root={data_root}")
    print(f"tasks={', '.join(f'{task}({len(records)})' for task, records in task_data.items())}")
    print(f"eval: batch_size={settings['batch_size']} num_beams={settings['num_beams']} max_new_tokens={args.max_new_tokens}")
    print(f"targets={', '.join(label for label, _ in targets)}")

    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(settings["model_id"], settings["max_input_length"], args.trust_remote_code)
    csv_path = output_dir / "results.csv"
    rows: list[dict[str, Any]] = []

    for label, target_path in targets:
        model = None
        path_type = None
        try:
            print(f"\n=== {label} ===")
            model, path_type = load_model(
                model_id=settings["model_id"],
                target_path=target_path,
                dtype_name=settings["dtype"],
                attn_implementation=args.attn_implementation,
                trust_remote_code=args.trust_remote_code,
                device=device,
            )
            for task, records in task_data.items():
                row = evaluate_task(
                    model=model,
                    tokenizer=tokenizer,
                    label=label,
                    task=task,
                    records=records,
                    output_dir=output_dir,
                    path=target_path,
                    path_type=path_type,
                    args=args,
                    settings=settings,
                )
                append_csv(csv_path, row)
                rows.append(row)
                print(f"[{label}/{task}] acc={row['accuracy'] * 100:.2f} correct={row['correct']}/{row['num_samples']}")
        except Exception as exc:
            row = {
                "model": label,
                "path": str(target_path) if target_path is not None else None,
                "path_type": path_type or ("base" if target_path is None else "unknown"),
                "status": "failed",
                "error": str(exc),
            }
            append_csv(csv_path, row)
            rows.append(row)
            print(f"[{label}] failed: {exc}")
            if args.stop_on_error:
                raise
        finally:
            if model is not None:
                del model
            cleanup_device(device)

        summary = {"rows": rows, **summarize(rows)}
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(jsonable(summary), handle, indent=2, ensure_ascii=True)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    if args.save_predictions:
        print(f"Saved predictions: {output_dir / 'predictions'}")


if __name__ == "__main__":
    main()
