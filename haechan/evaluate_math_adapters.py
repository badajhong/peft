#!/usr/bin/env python
"""Evaluate saved PEFT adapters on GSM8K and MATH."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from decimal import Decimal, DivisionByZero, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from peft import PeftConfig, PeftModel, get_peft_model


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B"
DEFAULT_PROMPT_TEMPLATE = "Question: {query} Think step by step.\nAnswer: "
DEFAULT_LORAXS_ROOT = "LoRA-XS"
DEFAULT_LORAXS_RECONSTRUCT_CONFIG = "LoRA-XS/config/reconstruct_config.yaml"
DEFAULT_MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
ANSWER_MARKERS = (
    "####",
    "The final answer is:",
    "The final answer is",
    "The answer is:",
    "The answer is",
    "final answer is:",
    "final answer is",
    "answer is:",
    "answer is",
)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PEFT adapters on GSM8K and MATH with exact final-answer matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-root", default=None, help="Directory containing method subdirs with adapter_config.json.")
    parser.add_argument("--adapter-paths", nargs="*", default=[], help="Specific adapter directories to evaluate.")
    parser.add_argument("--include-base", action="store_true", help="Also evaluate the base model without an adapter.")
    parser.add_argument("--datasets", nargs="+", choices=("gsm8k", "math"), default=["gsm8k", "math"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate the full split.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--loraxs-root", default=DEFAULT_LORAXS_ROOT, help="Official LoRA-XS repository folder.")
    parser.add_argument(
        "--loraxs-reconstruct-config",
        default=DEFAULT_LORAXS_RECONSTRUCT_CONFIG,
        help="Official LoRA-XS reconstruction YAML.",
    )
    parser.add_argument("--loraxs-reconstruction-type", default="svd")

    parser.add_argument("--gsm8k-dataset", default="openai/gsm8k")
    parser.add_argument("--gsm8k-config", default="main")
    parser.add_argument("--gsm8k-split", default="test")
    parser.add_argument("--math-dataset", default="EleutherAI/hendrycks_math")
    parser.add_argument("--math-configs", nargs="*", default=list(DEFAULT_MATH_CONFIGS))
    parser.add_argument("--math-split", default="test")
    return parser.parse_args()


def torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(__file__).resolve().parent / "results" / "math_eval" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_latest_adapter_root() -> Path | None:
    roots = []
    base = Path(__file__).resolve().parent / "results" / "math_time_compare"
    for root in base.glob("*/adapters"):
        if any(child.is_dir() and (child / "adapter_config.json").exists() for child in root.iterdir()):
            roots.append(root)
    if not roots:
        return None
    return max(roots, key=lambda path: path.stat().st_mtime)


def discover_adapter_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in args.adapter_paths]

    adapter_root = Path(args.adapter_root).expanduser().resolve() if args.adapter_root else None
    if adapter_root is None and not paths and not args.include_base:
        adapter_root = find_latest_adapter_root()
        if adapter_root is not None:
            print(f"Using latest adapter root: {adapter_root}")

    if adapter_root is not None:
        if (adapter_root / "adapter_config.json").exists():
            paths.append(adapter_root)
        else:
            paths.extend(
                sorted(
                    child.resolve()
                    for child in adapter_root.iterdir()
                    if child.is_dir() and (child / "adapter_config.json").exists()
                )
            )

    unique_paths: list[Path] = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter_config.json in adapter path: {path}")
        unique_paths.append(path)
        seen.add(path)
    return unique_paths


def model_label(adapter_path: Path | None) -> str:
    if adapter_path is None:
        return "base"
    return adapter_path.name


def load_tokenizer(args: argparse.Namespace) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer.model_max_length = args.max_input_length
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_adapter_state_dict(adapter_path: Path) -> dict[str, torch.Tensor]:
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        return load_file(safetensors_path)

    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(f"Could not find adapter_model.safetensors or adapter_model.bin in {adapter_path}")


def is_loraxs_adapter(adapter_path: Path) -> bool:
    state_dict = load_adapter_state_dict(adapter_path)
    return any("default_lora_latent_mapping" in key for key in state_dict)


def map_loraxs_state_key(key: str) -> str:
    key = key.replace(".lora_A.weight", ".lora_A.default.weight")
    key = key.replace(".lora_B.weight", ".lora_B.default.weight")
    return key


def load_loraxs_adapter(
    model: torch.nn.Module,
    adapter_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    math_train_module = import_math_train_module()
    peft_config = PeftConfig.from_pretrained(adapter_path)
    model = get_peft_model(model, peft_config)
    math_train_module.apply_loraxs_initialization(model, args)

    state_dict = load_adapter_state_dict(adapter_path)
    mapped_state_dict = {map_loraxs_state_key(key): value for key, value in state_dict.items()}
    model.load_state_dict(mapped_state_dict, strict=False)
    model.to(device)
    return model


def import_math_train_module() -> Any:
    module_name = "_haechan_math_peft_time_compare"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().with_name("math_peft_time_compare.py")
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not import math_peft_time_compare.py from {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def load_model(args: argparse.Namespace, adapter_path: Path | None, device: torch.device) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": args.model_id,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.dtype != "float32":
        kwargs["torch_dtype"] = torch_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    if adapter_path is not None:
        if is_loraxs_adapter(adapter_path):
            model = load_loraxs_adapter(model, adapter_path, args, device)
        else:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.to(device)
    model.eval()
    return model


def extract_braced(text: str, open_brace: int) -> str | None:
    if open_brace >= len(text) or text[open_brace] != "{":
        return None
    depth = 0
    for idx in range(open_brace, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : idx]
    return None


def extract_boxed(text: str) -> str | None:
    answer = None
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        open_brace = text.find("{", match.start())
        answer = extract_braced(text, open_brace)
    return answer


def extract_after_marker(text: str) -> str | None:
    lowered = text.lower()
    best_pos = -1
    best_marker = None
    for marker in ANSWER_MARKERS:
        pos = lowered.rfind(marker.lower())
        if pos > best_pos:
            best_pos = pos
            best_marker = marker
    if best_marker is None:
        return None
    answer = text[best_pos + len(best_marker) :].strip()
    return answer.split("\n", 1)[0].strip()


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def extract_final_answer(text: str, dataset_name: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed:
        return boxed

    marked = extract_after_marker(text)
    if marked:
        if dataset_name == "gsm8k":
            number = extract_last_number(marked)
            return number or marked
        return marked

    if dataset_name == "gsm8k":
        return extract_last_number(text)
    return extract_last_number(text) or text.strip()


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    answer = answer.strip(".$ ")
    answer = answer.rstrip(".!,;:")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\!", "").replace("\\,", "")
    answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    answer = answer.replace("$", "").replace("%", "")
    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)
    answer = re.sub(r"\s+", "", answer)
    return answer


def decimal_from_answer(answer: str | None) -> Decimal | None:
    answer = normalize_answer(answer)
    if answer is None:
        return None
    answer = answer.strip()
    try:
        frac_match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", answer)
        if frac_match:
            numerator = Decimal(frac_match.group(1))
            denominator = Decimal(frac_match.group(2))
            if denominator == 0:
                return None
            return numerator / denominator
        if "/" in answer and answer.count("/") == 1:
            numerator, denominator = answer.split("/")
            denominator_decimal = Decimal(denominator)
            if denominator_decimal == 0:
                return None
            return Decimal(numerator) / denominator_decimal
        return Decimal(answer)
    except (DivisionByZero, InvalidOperation, ValueError):
        return None


def answers_match(prediction: str | None, gold: str | None) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    pred_decimal = decimal_from_answer(pred_norm)
    gold_decimal = decimal_from_answer(gold_norm)
    return (pred_decimal is not None) and (gold_decimal is not None) and (pred_decimal == gold_decimal)


def load_gsm8k(args: argparse.Namespace) -> Dataset:
    ds = load_dataset(args.gsm8k_dataset, args.gsm8k_config, split=args.gsm8k_split)
    records = []
    for idx, example in enumerate(ds):
        records.append(
            {
                "id": f"gsm8k-{idx}",
                "dataset": "gsm8k",
                "query": str(example["question"]),
                "solution": str(example["answer"]),
                "gold_answer": extract_final_answer(str(example["answer"]), "gsm8k"),
            }
        )
    return Dataset.from_list(records)


def load_math(args: argparse.Namespace) -> Dataset:
    datasets = []
    for config in args.math_configs:
        ds = load_dataset(args.math_dataset, config, split=args.math_split)
        records = []
        for idx, example in enumerate(ds):
            records.append(
                {
                    "id": f"math-{config}-{idx}",
                    "dataset": "math",
                    "math_config": config,
                    "level": str(example.get("level", "")),
                    "type": str(example.get("type", "")),
                    "query": str(example["problem"]),
                    "solution": str(example["solution"]),
                    "gold_answer": extract_final_answer(str(example["solution"]), "math"),
                }
            )
        datasets.append(Dataset.from_list(records))
    return concatenate_datasets(datasets)


def sample_dataset(ds: Dataset, args: argparse.Namespace) -> Dataset:
    if args.max_samples <= 0:
        return ds
    sample_count = min(args.max_samples, len(ds))
    return ds.shuffle(seed=args.seed).select(range(sample_count))


def load_eval_datasets(args: argparse.Namespace) -> dict[str, Dataset]:
    loaded = {}
    if "gsm8k" in args.datasets:
        loaded["gsm8k"] = sample_dataset(load_gsm8k(args), args)
    if "math" in args.datasets:
        loaded["math"] = sample_dataset(load_math(args), args)
    return loaded


def generate_batch(model: torch.nn.Module, tokenizer: Any, prompts: list[str], args: argparse.Namespace) -> list[str]:
    device = next(model.parameters()).device
    batch = tokenizer(
        prompts,
        truncation=True,
        max_length=args.max_input_length,
        padding=True,
        return_tensors="pt",
    ).to(device)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p
    outputs = model.generate(**batch, **generate_kwargs)
    generated = outputs[:, batch["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=True) + "\n")


def evaluate_one_dataset(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    ds: Dataset,
    dataset_name: str,
    label: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.perf_counter()
    prediction_rows = []
    correct = 0

    for start_idx in tqdm(range(0, len(ds), args.batch_size), desc=f"{label}/{dataset_name}", leave=False):
        sliced = ds[start_idx : start_idx + args.batch_size]
        prompts = [args.prompt_template.format(query=query) for query in sliced["query"]]
        generations = generate_batch(model, tokenizer, prompts, args)
        for offset, generation in enumerate(generations):
            gold_answer = sliced["gold_answer"][offset]
            pred_answer = extract_final_answer(generation, dataset_name)
            is_correct = answers_match(pred_answer, gold_answer)
            correct += int(is_correct)
            prediction_rows.append(
                {
                    "model": label,
                    "dataset": dataset_name,
                    "id": sliced["id"][offset],
                    "math_config": sliced.get("math_config", [""] * len(generations))[offset],
                    "prompt": prompts[offset],
                    "generation": generation,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "correct": is_correct,
                    "solution": sliced["solution"][offset],
                }
            )

    seconds = time.perf_counter() - start
    total = len(ds)
    row = {
        "model": label,
        "dataset": dataset_name,
        "num_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "eval_seconds": seconds,
        "samples_per_second": total / seconds if seconds > 0 else 0.0,
    }
    if args.save_predictions:
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(pred_dir / f"{label}-{dataset_name}.jsonl", prediction_rows)
    return row


def append_csv(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "model",
        "adapter_path",
        "dataset",
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = make_output_dir(args)
    with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(vars(args)), handle, indent=2, ensure_ascii=True)

    adapter_paths = discover_adapter_paths(args)
    targets: list[tuple[str, Path | None]] = []
    if args.include_base:
        targets.append(("base", None))
    targets.extend((model_label(path), path) for path in adapter_paths)
    if not targets:
        raise ValueError("No model target to evaluate. Pass --adapter-root, --adapter-paths, or --include-base.")

    tokenizer = load_tokenizer(args)
    eval_datasets = load_eval_datasets(args)
    device = get_device()
    csv_path = output_dir / "results.csv"
    summary_rows = []

    print(f"device={device} output_dir={output_dir}")
    print(f"datasets={', '.join(eval_datasets)}")
    print(f"models={', '.join(label for label, _ in targets)}")

    for label, adapter_path in targets:
        try:
            print(f"\n=== {label} ===")
            model = load_model(args, adapter_path, device)
            for dataset_name, ds in eval_datasets.items():
                row = evaluate_one_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    ds=ds,
                    dataset_name=dataset_name,
                    label=label,
                    output_dir=output_dir,
                    args=args,
                )
                row["adapter_path"] = str(adapter_path) if adapter_path is not None else None
                row["status"] = "success"
                append_csv(csv_path, row)
                summary_rows.append(row)
                print(
                    f"[{label}/{dataset_name}] accuracy={row['accuracy']:.4f} "
                    f"correct={row['correct']}/{row['num_samples']} "
                    f"time={row['eval_seconds']:.1f}s"
                )
        except Exception as exc:
            row = {
                "model": label,
                "adapter_path": str(adapter_path) if adapter_path is not None else None,
                "status": "failed",
                "error": str(exc),
            }
            append_csv(csv_path, row)
            summary_rows.append(row)
            print(f"[{label}] failed: {exc}")
            if args.stop_on_error:
                raise
        finally:
            try:
                del model
            except UnboundLocalError:
                pass
            cleanup_device(device)

        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(jsonable(summary_rows), handle, indent=2, ensure_ascii=True)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    if args.save_predictions:
        print(f"Saved predictions: {output_dir / 'predictions'}")


if __name__ == "__main__":
    main()
