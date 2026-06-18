#!/usr/bin/env python
"""Training-time comparison for Llama 3.2 3B math fine-tuning methods.

This script focuses on timing PEFT adapter training under the same model,
dataset subset, sequence length, batch size, and optimizer settings.
It records model loading time, adapter setup time, pure training time, memory,
loss, trainable parameter counts, and tokens/sec.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from peft import BOFTConfig, CaraConfig, LoraConfig, OFTConfig, PsoftConfig, TaskType, get_peft_model


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B"
DEFAULT_DATASET = "meta-math/MetaMathQA-40K"
DEFAULT_QUERY_TEMPLATE = "Question: {query} Think step by step.\nAnswer: "
SUPPORTED_METHODS = ("lora", "cara", "oft", "boft", "psoft", "pissa", "dora", "lora-xs")
DEFAULT_LORAXS_ROOT = "LoRA-XS"
DEFAULT_LORAXS_RECONSTRUCT_CONFIG = "LoRA-XS/config/reconstruct_config.yaml"
FIXED_DATASET_NAME = "meta-math/MetaMathQA-40K"
FIXED_DATASET_CONFIG = None
FIXED_DATASET_SPLIT = "train"
FIXED_QUERY_FIELD = "query"
FIXED_RESPONSE_FIELD = "response"
FIXED_TRAIN_SAMPLES = 0
FIXED_MAX_SEQ_LENGTH = 512
FIXED_EFFECTIVE_BATCH_SIZE = 64
FIXED_EPOCHS = 2
FIXED_OPTIMIZER = "adamw"
FIXED_WARMUP_RATIO = 0.1
FIXED_LR_SCHEDULER = "cosine"


class UnsupportedMethodError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PEFT training time on a math CausalLM task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--query-template", default=DEFAULT_QUERY_TEMPLATE)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPPORTED_METHODS),
        help="Methods to run. Comma-separated values are also accepted.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=FIXED_TRAIN_SAMPLES,
        help="Deprecated: training always uses all MetaMathQA-40K rows.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Deprecated: training uses fixed 2 epochs.")
    parser.add_argument("--max-seq-length", type=int, default=FIXED_MAX_SEQ_LENGTH, help="Fixed to 512.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Micro-batch size. Effective training batch size is fixed to 64 via gradient accumulation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Ignored: computed automatically from fixed effective batch size 64.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--mask-prompt-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-steps-to-ignore", type=int, default=2)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print adapter configs; do not load/train model.")
    parser.add_argument("--save-adapters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-dir-name", default="adapters")
    parser.add_argument("--wandb-project", default=None, help="If set, log each method as a separate W&B run.")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--wandb-run-prefix", default="math-peft")
    parser.add_argument("--wandb-tags", nargs="*", default=[])

    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="auto",
        help="'auto' uses PEFT model defaults, 'all-linear' targets all linear layers, or comma-separated modules.",
    )

    parser.add_argument("--cara-noise-alpha", type=float, default=0.01)
    parser.add_argument("--cara-noise-step-interval", type=int, default=5)
    parser.add_argument("--pissa-init", default="pissa_niter_4")
    parser.add_argument("--boft-block-size", type=int, default=4)
    parser.add_argument("--boft-n-butterfly-factor", type=int, default=1)
    parser.add_argument("--psoft-rank", type=int, default=256)
    parser.add_argument("--psoft-alpha", type=int, default=256)
    parser.add_argument("--psoft-svd", choices=("full", "lowrank"), default="lowrank")
    parser.add_argument("--psoft-svd-lowrank-niter", type=int, default=10)
    parser.add_argument("--psoft-use-cayley-neumann", action="store_true")
    parser.add_argument("--loraxs-root", default=DEFAULT_LORAXS_ROOT, help="Official LoRA-XS repository folder.")
    parser.add_argument(
        "--loraxs-reconstruct-config",
        default=DEFAULT_LORAXS_RECONSTRUCT_CONFIG,
        help="Official LoRA-XS reconstruction YAML.",
    )
    parser.add_argument("--loraxs-reconstruction-type", default="svd")
    return parser.parse_args()


def normalize_methods(raw_methods: list[str]) -> list[str]:
    normalized: list[str] = []
    aliases = {
        "loraxs": "lora-xs",
        "lora_xs": "lora-xs",
        "lora-xs": "lora-xs",
        "loarxs": "lora-xs",
        "loar_xs": "lora-xs",
        "loar-xs": "lora-xs",
        "oftv2": "oft",
        "oft-v2": "oft",
    }
    for item in raw_methods:
        for token in item.split(","):
            method = token.strip().lower()
            if not method:
                continue
            normalized.append(aliases.get(method, method))
    return normalized


def apply_fixed_training_hyperparams(args: argparse.Namespace) -> argparse.Namespace:
    args.micro_batch_size = args.batch_size
    args.dataset_name = FIXED_DATASET_NAME
    args.dataset_config = FIXED_DATASET_CONFIG
    args.dataset_split = FIXED_DATASET_SPLIT
    args.query_field = FIXED_QUERY_FIELD
    args.response_field = FIXED_RESPONSE_FIELD
    args.train_samples = FIXED_TRAIN_SAMPLES
    args.max_steps = None
    args.max_seq_length = FIXED_MAX_SEQ_LENGTH
    args.epochs = FIXED_EPOCHS
    args.batch_size = FIXED_EFFECTIVE_BATCH_SIZE
    args.effective_batch_size = FIXED_EFFECTIVE_BATCH_SIZE
    args.optimizer = FIXED_OPTIMIZER
    args.warmup_ratio = FIXED_WARMUP_RATIO
    args.lr_scheduler = FIXED_LR_SCHEDULER

    if args.micro_batch_size <= 0:
        raise ValueError("--batch-size is the micro-batch size and must be positive.")
    if args.micro_batch_size > FIXED_EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            f"--batch-size is a micro-batch size and cannot exceed fixed effective batch size "
            f"{FIXED_EFFECTIVE_BATCH_SIZE}."
        )
    if FIXED_EFFECTIVE_BATCH_SIZE % args.micro_batch_size != 0:
        raise ValueError(
            f"Fixed effective batch size {FIXED_EFFECTIVE_BATCH_SIZE} must be divisible by micro-batch size "
            f"{args.micro_batch_size}. Try --batch-size 1, 2, 4, 8, 16, 32, or 64."
        )
    args.gradient_accumulation_steps = FIXED_EFFECTIVE_BATCH_SIZE // args.micro_batch_size
    return args


def parse_target_modules(value: str) -> list[str] | str | None:
    value = value.strip()
    if value.lower() == "auto":
        return None
    if value == "all-linear":
        return "all-linear"
    return [part.strip() for part in value.split(",") if part.strip()]


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


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
    }


def current_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated() / 2**20


def current_reserved_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_reserved() / 2**20


def cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return jsonable(config.to_dict())
    return jsonable(config)


def compact_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "method",
        "status",
        "started_at",
        "finished_at",
        "model_id",
        "dataset_name",
        "dataset_config",
        "dataset_split",
        "train_samples",
        "num_train_examples",
        "epochs",
        "batch_size",
        "micro_batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "optimizer",
        "lr_scheduler",
        "warmup_ratio",
        "loraxs_initialization_seconds",
        "loraxs_root",
        "loraxs_reconstruct_config",
        "loraxs_reconstruction_type",
        "loraxs_rank",
        "loraxs_r_squared",
        "loraxs_latent_modules",
        "warmup_steps",
        "optimizer_steps",
        "micro_steps",
        "train_seconds",
        "avg_step_seconds_after_warmup",
        "tokens_per_second",
        "samples_per_second",
        "loss_first",
        "loss_last",
        "grad_norm_first",
        "grad_norm_last",
        "grad_norm_avg",
        "learning_rate_last",
        "trainable_params",
        "total_params",
        "trainable_percent",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "model_load_seconds",
        "adapter_setup_seconds",
        "adapter_save_path",
        "adapter_save_seconds",
        "adapter_file_size_mb",
        "end_to_end_seconds",
        "error",
    ]
    summary = {key: row.get(key) for key in keys if key in row}
    if "peft_config" in row:
        summary["peft_config"] = row["peft_config"]
    return jsonable(summary)


def write_method_summary(output_dir: Path, row: dict[str, Any]) -> None:
    summary = compact_summary(row)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "current_method_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    method = str(row.get("method", "unknown"))
    with (summaries_dir / f"{method}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)


def write_overall_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_rows = [compact_summary(row) for row in rows]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, ensure_ascii=True)


def init_wandb_run(method: str, args: argparse.Namespace, output_dir: Path, peft_config: Any) -> Any | None:
    if not args.wandb_project or args.dry_run:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Install it with `pip install wandb`.") from exc

    group = args.wandb_group or output_dir.name
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=f"{args.wandb_run_prefix}-{method}",
        tags=[*args.wandb_tags, method],
        mode=args.wandb_mode,
        config={
            "method": method,
            "model_id": args.model_id,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "query_field": args.query_field,
            "response_field": args.response_field,
            "train_samples": args.train_samples,
            "epochs": args.epochs,
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "effective_batch_size": args.effective_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "dtype": args.dtype,
            "gradient_checkpointing": args.gradient_checkpointing,
            "target_modules": args.target_modules,
            "loraxs_root": args.loraxs_root,
            "loraxs_reconstruct_config": args.loraxs_reconstruct_config,
            "loraxs_reconstruction_type": args.loraxs_reconstruction_type,
            "peft_config": config_to_dict(peft_config),
            "local_output_dir": str(output_dir),
        },
        reinit=True,
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("summary/*")
    print(f"[{method}] wandb run: {run.url}")
    return run


def build_peft_config(method: str, args: argparse.Namespace) -> Any:
    target_modules = parse_target_modules(args.target_modules)
    method = method.lower()

    if method == "lora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
    if method == "cara":
        return CaraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            noise_alpha=args.cara_noise_alpha,
            noise_step_interval=args.cara_noise_step_interval,
            target_modules=target_modules,
        )
    if method == "oft":
        return OFTConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            oft_block_size=0,
            target_modules=target_modules,
            module_dropout=0.0,
            bias="none",
            use_cayley_neumann=True,
        )
    if method == "boft":
        return BOFTConfig(
            task_type=TaskType.CAUSAL_LM,
            boft_block_size=args.boft_block_size,
            boft_block_num=0,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=target_modules,
            boft_dropout=0.0,
            bias="none",
        )
    if method == "psoft":
        return PsoftConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.psoft_rank,
            psoft_alpha=args.psoft_alpha,
            psoft_dropout=0.0,
            target_modules=target_modules,
            ab_svd_init="psoft_init",
            psoft_svd=args.psoft_svd,
            psoft_svd_lowrank_niter=args.psoft_svd_lowrank_niter,
            psoft_orth=True,
            psoft_mag_a=True,
            psoft_mag_b=True,
            use_cayley_neumann=args.psoft_use_cayley_neumann,
        )
    if method == "pissa":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            init_lora_weights=args.pissa_init,
        )
    if method == "dora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_dora=True,
        )
    if method == "lora-xs":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
    raise UnsupportedMethodError(f"Unknown method: {method}")


def resolve_existing_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(Path(__file__).resolve().parents[1] / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Path not found: {path_value}")


def import_loraxs_initialization_module(loraxs_root: Path) -> Any:
    utils_dir = loraxs_root / "utils"
    init_file = utils_dir / "__init__.py"
    module_file = utils_dir / "initialization_utils.py"
    if not module_file.exists():
        raise FileNotFoundError(f"Missing official LoRA-XS initialization file: {module_file}")

    package_name = "_official_loraxs_utils"
    if package_name not in sys.modules:
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            init_file,
            submodule_search_locations=[str(utils_dir)],
        )
        if package_spec is None or package_spec.loader is None:
            raise ImportError(f"Could not import LoRA-XS utils package from {utils_dir}")
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package
        package_spec.loader.exec_module(package)

    module_name = f"{package_name}.initialization_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_spec = importlib.util.spec_from_file_location(module_name, module_file)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not import LoRA-XS initialization module from {module_file}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def patch_loraxs_runtime_dtype_support(initialization_module: Any) -> None:
    if getattr(initialization_module, "_haechan_dtype_patch_applied", False):
        return

    def get_replacement_module(weight: torch.Tensor, module_name: str, type: str, writer: Any, reconstruct_config: Any):
        cfg = reconstruct_config[type]
        if type != "svd":
            raise NotImplementedError(f"{type} is currently not supported.")
        _, enc, dec = initialization_module.get_linear_rec_svd(
            weight.detach().float().cpu().numpy(),
            cfg["rank"],
            cfg["n_iter"],
            cfg["random_state"],
        )
        final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
        return final_enc, final_dec

    initialization_module.get_replacement_module = get_replacement_module
    initialization_module._haechan_dtype_patch_applied = True


def align_loraxs_latent_dtype(model: torch.nn.Module, adapter_name: str = "default") -> int:
    aligned = 0
    for module in model.modules():
        latent = getattr(module, "default_lora_latent_mapping", None)
        lora_a = getattr(module, "lora_A", None)
        if latent is None or lora_a is None or adapter_name not in lora_a:
            continue
        reference_weight = lora_a[adapter_name].weight
        latent.to(device=reference_weight.device, dtype=reference_weight.dtype)
        aligned += 1
    return aligned


def apply_loraxs_initialization(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("LoRA-XS needs PyYAML. Install it with `pip install pyyaml`.") from exc

    loraxs_root = resolve_existing_path(args.loraxs_root)
    config_path = resolve_existing_path(args.loraxs_reconstruct_config)
    initialization_module = import_loraxs_initialization_module(loraxs_root)
    patch_loraxs_runtime_dtype_support(initialization_module)

    with config_path.open("r", encoding="utf-8") as stream:
        reconstruct_config = yaml.load(stream, Loader=yaml.FullLoader)
    if not isinstance(reconstruct_config, dict):
        raise ValueError(f"Invalid LoRA-XS reconstruction config: {config_path}")

    adapter_name = "default"
    if not hasattr(model, "peft_config") or adapter_name not in model.peft_config:
        raise ValueError("LoRA-XS initialization expects a PEFT model with the default adapter.")

    lora_config = model.peft_config[adapter_name]
    if isinstance(lora_config.target_modules, str):
        raise UnsupportedMethodError(
            "Official LoRA-XS initialization expects target_modules to be a resolved list/set. "
            "Use --target-modules auto or a comma-separated module list, not all-linear."
        )

    reconstruction_type = args.loraxs_reconstruction_type
    if reconstruction_type not in reconstruct_config:
        raise ValueError(f"LoRA-XS reconstruction type {reconstruction_type!r} is missing in {config_path}.")

    reconstruct_config[reconstruction_type]["rank"] = lora_config.r
    initialization_module.find_and_initialize(
        model,
        {adapter_name: lora_config},
        adapter_name=adapter_name,
        reconstr_type=reconstruction_type,
        writer=None,
        reconstruct_config=reconstruct_config,
    )
    latent_modules = align_loraxs_latent_dtype(model, adapter_name)
    return {
        "loraxs_root": str(loraxs_root),
        "loraxs_reconstruct_config": str(config_path),
        "loraxs_reconstruction_type": reconstruction_type,
        "loraxs_rank": lora_config.r,
        "loraxs_r_squared": reconstruct_config.get("r_squared"),
        "loraxs_latent_modules": latent_modules,
    }


def get_text_fields(example: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    if args.query_field not in example:
        raise KeyError(f"Missing query field {args.query_field!r}. Available fields: {sorted(example)}")
    if args.response_field not in example:
        raise KeyError(f"Missing response field {args.response_field!r}. Available fields: {sorted(example)}")
    return str(example[args.query_field]), str(example[args.response_field])


def build_dataset(args: argparse.Namespace, tokenizer: Any) -> Dataset:
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    else:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.train_samples > 0:
        sample_count = min(args.train_samples, len(dataset))
        dataset = dataset.shuffle(seed=args.seed).select(range(sample_count))

    eos = tokenizer.eos_token or ""

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        query, response = get_text_fields(example, args)
        prompt = args.query_template.format(query=query)
        text = prompt + response + eos
        encoded = tokenizer(text, truncation=True, max_length=args.max_seq_length)
        labels = list(encoded["input_ids"])

        if args.mask_prompt_loss:
            prompt_ids = tokenizer(prompt, truncation=True, max_length=args.max_seq_length)["input_ids"]
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            if all(label == -100 for label in labels) and labels:
                labels[-1] = encoded["input_ids"][-1]

        encoded["labels"] = labels
        return encoded

    return dataset.map(tokenize, remove_columns=dataset.column_names)


class CausalLMCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features
        ]
        batch = self.tokenizer.pad(model_features, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        labels = []
        for item in features:
            item_labels = list(item["labels"])
            pad_len = max_len - len(item_labels)
            labels.append(item_labels + [-100] * pad_len)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def make_dataloader(dataset: Dataset, tokenizer: Any, args: argparse.Namespace) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=CausalLMCollator(tokenizer),
        generator=generator,
    )


def next_batch(iterator: Any, dataloader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def clip_or_get_grad_norm(model: torch.nn.Module, max_grad_norm: float) -> float:
    parameters = [param for param in model.parameters() if param.grad is not None]
    if not parameters:
        return 0.0

    if max_grad_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        return float(grad_norm.detach().cpu().item())

    norms = [torch.linalg.vector_norm(param.grad.detach(), ord=2) for param in parameters]
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    return float(total_norm.detach().cpu().item())


def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    wandb_run: Any | None,
) -> dict[str, Any]:
    model.train()
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    micro_batches_per_epoch = len(dataloader)
    steps_per_epoch = (micro_batches_per_epoch + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    total_optimizer_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    step_durations: list[float] = []
    losses: list[float] = []
    grad_norms: list[float] = []
    learning_rates: list[float] = []
    tokens_seen = 0
    samples_seen = 0
    micro_steps = 0
    optimizer_steps = 0

    reset_peak_memory(device)
    sync_device(device)
    train_start = time.perf_counter()

    progress = tqdm(total=total_optimizer_steps, desc="train", leave=False)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulation_count = 0
        sync_device(device)
        step_start = time.perf_counter()

        for micro_batch_idx, batch in enumerate(dataloader, start=1):
            micro_steps += 1
            accumulation_count += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            tokens_seen += int(batch["attention_mask"].sum().item())
            samples_seen += int(batch["input_ids"].shape[0])

            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / args.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += float(raw_loss.detach().cpu().item())

            should_step = (
                accumulation_count == args.gradient_accumulation_steps
                or micro_batch_idx == micro_batches_per_epoch
            )
            if not should_step:
                continue

            grad_norm = clip_or_get_grad_norm(model, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            sync_device(device)
            duration = time.perf_counter() - step_start
            learning_rate = float(scheduler.get_last_lr()[0])
            logged_loss = accumulated_loss / accumulation_count
            step_durations.append(duration)
            losses.append(logged_loss)
            grad_norms.append(grad_norm)
            learning_rates.append(learning_rate)
            progress.update(1)
            progress.set_postfix(
                epoch=epoch,
                loss=f"{logged_loss:.4f}",
                lr=f"{learning_rate:.2e}",
                sec=f"{duration:.3f}",
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": optimizer_steps,
                        "train/epoch": epoch,
                        "train/loss": logged_loss,
                        "train/learning_rate": learning_rate,
                        "train/grad_norm": grad_norm,
                        "train/step_seconds": duration,
                        "train/tokens_seen": tokens_seen,
                        "train/samples_seen": samples_seen,
                        "train/micro_steps": micro_steps,
                        "train/tokens_per_second_step": tokens_seen / (time.perf_counter() - train_start),
                        "train/gpu_allocated_mb": current_allocated_mb(device),
                        "train/gpu_reserved_mb": current_reserved_mb(device),
                    }
                )

            accumulated_loss = 0.0
            accumulation_count = 0
            sync_device(device)
            step_start = time.perf_counter()

    progress.close()

    sync_device(device)
    train_seconds = time.perf_counter() - train_start
    memory = memory_stats(device)

    warmup = min(args.warmup_steps_to_ignore, len(step_durations))
    measured_durations = step_durations[warmup:] or step_durations
    return {
        "train_seconds": train_seconds,
        "avg_step_seconds": statistics.mean(step_durations),
        "median_step_seconds": statistics.median(step_durations),
        "avg_step_seconds_after_warmup": statistics.mean(measured_durations),
        "tokens_per_second": tokens_seen / train_seconds if train_seconds > 0 else 0.0,
        "samples_per_second": samples_seen / train_seconds if train_seconds > 0 else 0.0,
        "tokens_seen": tokens_seen,
        "samples_seen": samples_seen,
        "micro_steps": micro_steps,
        "optimizer_steps": optimizer_steps,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "epochs": args.epochs,
        "effective_batch_size": args.effective_batch_size,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "losses": losses,
        "grad_norm_first": grad_norms[0] if grad_norms else None,
        "grad_norm_last": grad_norms[-1] if grad_norms else None,
        "grad_norm_avg": statistics.mean(grad_norms) if grad_norms else None,
        "learning_rate_first": learning_rates[0] if learning_rates else None,
        "learning_rate_last": learning_rates[-1] if learning_rates else None,
        **memory,
    }


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
        return int(trainable), int(total)
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def save_adapter(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    method: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.save_adapters:
        return {
            "adapter_save_path": None,
            "adapter_save_seconds": 0.0,
            "adapter_file_size_mb": 0.0,
        }

    save_dir = output_dir / args.adapter_dir_name / method
    save_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    adapter_save_seconds = time.perf_counter() - start
    adapter_file_size = sum(path.stat().st_size for path in save_dir.rglob("*") if path.is_file())

    return {
        "adapter_save_path": str(save_dir),
        "adapter_save_seconds": adapter_save_seconds,
        "adapter_file_size_mb": adapter_file_size / 2**20,
    }


def load_tokenizer(args: argparse.Namespace) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer.model_max_length = args.max_seq_length
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
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
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    return model


def run_one_method(
    method: str,
    tokenizer: Any,
    dataset: Dataset,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    method_start = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    peft_config = build_peft_config(method, args)
    write_method_summary(
        output_dir,
        {
            "method": method,
            "status": "running",
            "started_at": started_at,
            "model_id": args.model_id,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "train_samples": args.train_samples,
            "num_train_examples": len(dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "effective_batch_size": args.effective_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "loraxs_root": args.loraxs_root if method == "lora-xs" else None,
            "loraxs_reconstruct_config": args.loraxs_reconstruct_config if method == "lora-xs" else None,
            "loraxs_reconstruction_type": args.loraxs_reconstruction_type if method == "lora-xs" else None,
            "adapter_save_path": str(output_dir / args.adapter_dir_name / method) if args.save_adapters else None,
            "peft_config": config_to_dict(peft_config),
        },
    )
    wandb_run = init_wandb_run(method, args, output_dir, peft_config)

    try:
        if args.dry_run:
            return {
                "method": method,
                "status": "dry_run",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "model_id": args.model_id,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "train_samples": args.train_samples,
                "num_train_examples": len(dataset),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "micro_batch_size": args.micro_batch_size,
                "effective_batch_size": args.effective_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_seq_length": args.max_seq_length,
                "optimizer": args.optimizer,
                "lr_scheduler": args.lr_scheduler,
                "warmup_ratio": args.warmup_ratio,
                "loraxs_root": args.loraxs_root if method == "lora-xs" else None,
                "loraxs_reconstruct_config": args.loraxs_reconstruct_config if method == "lora-xs" else None,
                "loraxs_reconstruction_type": args.loraxs_reconstruction_type if method == "lora-xs" else None,
                "peft_config": config_to_dict(peft_config),
            }

        dataloader = make_dataloader(dataset, tokenizer, args)

        sync_device(device)
        load_start = time.perf_counter()
        model = load_base_model(args, device)
        sync_device(device)
        model_load_seconds = time.perf_counter() - load_start

        sync_device(device)
        adapter_start = time.perf_counter()
        model = get_peft_model(model, peft_config)
        loraxs_metrics: dict[str, Any] = {}
        if method == "lora-xs":
            sync_device(device)
            loraxs_start = time.perf_counter()
            loraxs_metrics = apply_loraxs_initialization(model, args)
            sync_device(device)
            loraxs_metrics["loraxs_initialization_seconds"] = time.perf_counter() - loraxs_start
        sync_device(device)
        adapter_setup_seconds = time.perf_counter() - adapter_start

        trainable_params, total_params = count_parameters(model)
        if wandb_run is not None:
            wandb_run.config.update(
                {
                    "num_train_examples": len(dataset),
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
                },
                allow_val_change=True,
            )

        train_metrics = train_model(model, dataloader, device, args, wandb_run)
        save_metrics = save_adapter(
            model=model,
            tokenizer=tokenizer,
            method=method,
            output_dir=output_dir,
            args=args,
        )
        end_to_end_seconds = time.perf_counter() - method_start

        row = {
            "method": method,
            "status": "success",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "model_id": args.model_id,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "train_samples": args.train_samples,
            "num_train_examples": len(dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "effective_batch_size": args.effective_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "peft_config": config_to_dict(peft_config),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
            "model_load_seconds": model_load_seconds,
            "adapter_setup_seconds": adapter_setup_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            **loraxs_metrics,
            **train_metrics,
            **save_metrics,
        }
        if wandb_run is not None:
            wandb_run.log(
                {
                    "summary/train_seconds": row["train_seconds"],
                    "summary/avg_step_seconds_after_warmup": row["avg_step_seconds_after_warmup"],
                    "summary/tokens_per_second": row["tokens_per_second"],
                    "summary/optimizer_steps": row["optimizer_steps"],
                    "summary/loraxs_initialization_seconds": row.get("loraxs_initialization_seconds", 0.0),
                    "summary/loss_last": row["loss_last"],
                    "summary/grad_norm_last": row["grad_norm_last"],
                    "summary/peak_reserved_mb": row["peak_reserved_mb"],
                    "summary/adapter_file_size_mb": row["adapter_file_size_mb"],
                }
            )
            wandb_run.summary.update(
                {
                    "status": "success",
                    "train_seconds": row["train_seconds"],
                    "avg_step_seconds_after_warmup": row["avg_step_seconds_after_warmup"],
                    "tokens_per_second": row["tokens_per_second"],
                    "samples_per_second": row["samples_per_second"],
                    "optimizer_steps": row["optimizer_steps"],
                    "micro_steps": row["micro_steps"],
                    "warmup_steps": row["warmup_steps"],
                    "loss_first": row["loss_first"],
                    "loss_last": row["loss_last"],
                    "grad_norm_first": row["grad_norm_first"],
                    "grad_norm_last": row["grad_norm_last"],
                    "grad_norm_avg": row["grad_norm_avg"],
                    "learning_rate_last": row["learning_rate_last"],
                    "trainable_params": row["trainable_params"],
                    "trainable_percent": row["trainable_percent"],
                    "peak_allocated_mb": row["peak_allocated_mb"],
                    "peak_reserved_mb": row["peak_reserved_mb"],
                    "model_load_seconds": row["model_load_seconds"],
                    "adapter_setup_seconds": row["adapter_setup_seconds"],
                    "loraxs_initialization_seconds": row.get("loraxs_initialization_seconds"),
                    "adapter_save_path": row["adapter_save_path"],
                    "adapter_save_seconds": row["adapter_save_seconds"],
                    "adapter_file_size_mb": row["adapter_file_size_mb"],
                    "end_to_end_seconds": row["end_to_end_seconds"],
                }
            )
        return row
    except Exception as exc:
        if wandb_run is not None:
            wandb_run.summary["status"] = "failed"
            wandb_run.summary["error"] = str(exc)
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(__file__).resolve().parent / "results" / "math_time_compare" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), ensure_ascii=True) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "method",
        "status",
        "dataset_name",
        "train_samples",
        "num_train_examples",
        "epochs",
        "batch_size",
        "micro_batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "optimizer",
        "lr_scheduler",
        "warmup_ratio",
        "loraxs_initialization_seconds",
        "loraxs_latent_modules",
        "warmup_steps",
        "optimizer_steps",
        "micro_steps",
        "train_seconds",
        "avg_step_seconds",
        "avg_step_seconds_after_warmup",
        "tokens_per_second",
        "samples_per_second",
        "trainable_params",
        "trainable_percent",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "model_load_seconds",
        "adapter_setup_seconds",
        "adapter_save_path",
        "adapter_save_seconds",
        "adapter_file_size_mb",
        "end_to_end_seconds",
        "loss_first",
        "loss_last",
        "grad_norm_first",
        "grad_norm_last",
        "grad_norm_avg",
        "learning_rate_first",
        "learning_rate_last",
        "error",
    ]
    exists = path.exists()
    flat_row = {key: row.get(key) for key in fieldnames}
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(flat_row)


def print_summary(row: dict[str, Any]) -> None:
    method = row.get("method")
    status = row.get("status")
    if status != "success":
        print(f"[{method}] {status}: {row.get('error', '')}")
        return
    print(
        f"[{method}] train={row['train_seconds']:.2f}s "
        f"epochs={row['epochs']} "
        f"steps={row['optimizer_steps']} "
        f"avg_step={row['avg_step_seconds_after_warmup']:.3f}s "
        f"tok/s={row['tokens_per_second']:.1f} "
        f"trainable={row['trainable_params']:,d} "
        f"peak={row['peak_reserved_mb']:.0f}MB "
        f"adapter={row.get('adapter_save_path')}"
    )


def main() -> None:
    args = parse_args()
    args = apply_fixed_training_hyperparams(args)
    methods = normalize_methods(args.methods)
    output_dir = make_output_dir(args)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    args_path = output_dir / "args.json"

    with args_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(vars(args)), handle, indent=2, ensure_ascii=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    device = get_device()
    print(f"device={device} output_dir={output_dir}")
    print(f"methods={', '.join(methods)}")
    print(
        "fixed training: "
        f"dataset={args.dataset_name} split={args.dataset_split} full_data=True "
        f"epochs={args.epochs} effective_batch_size={args.effective_batch_size} "
        f"micro_batch_size={args.micro_batch_size} grad_accum={args.gradient_accumulation_steps} "
        f"max_seq_len={args.max_seq_length} optimizer={args.optimizer} "
        f"scheduler={args.lr_scheduler} warmup_ratio={args.warmup_ratio}"
    )
    print(f"current method summary: {output_dir / 'current_method_summary.json'}")

    if args.dry_run:
        tokenizer = None
        dataset = Dataset.from_dict({"input_ids": []})
    else:
        tokenizer = load_tokenizer(args)
        dataset = build_dataset(args, tokenizer)

    completed_rows: list[dict[str, Any]] = []
    for method in methods:
        row: dict[str, Any]
        try:
            print(f"\n=== {method} ===")
            row = run_one_method(method, tokenizer, dataset, device, output_dir, args)
        except UnsupportedMethodError as exc:
            row = {
                "method": method,
                "status": "unsupported",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
            }
        except Exception as exc:  # keep the suite moving unless requested otherwise
            row = {
                "method": method,
                "status": "failed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            if args.stop_on_error:
                append_jsonl(jsonl_path, row)
                append_csv(csv_path, row)
                write_method_summary(output_dir, row)
                completed_rows.append(row)
                write_overall_summary(output_dir, completed_rows)
                print_summary(row)
                raise
        finally:
            cleanup_device(device)

        append_jsonl(jsonl_path, row)
        append_csv(csv_path, row)
        write_method_summary(output_dir, row)
        completed_rows.append(row)
        write_overall_summary(output_dir, completed_rows)
        print_summary(row)

    print(f"\nSaved JSONL: {jsonl_path}")
    print(f"Saved CSV:   {csv_path}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
