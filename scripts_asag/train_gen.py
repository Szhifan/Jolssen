from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset

try:
    from accelerate import Accelerator

    ACCELERATE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - guarded for local validation
    Accelerator = None
    ACCELERATE_IMPORT_ERROR = exc

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable=None, **kwargs):
        return iterable

try:  # pragma: no cover - unavailable in the current sandbox by default
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        HfArgumentParser,
        get_cosine_schedule_with_warmup,
    )

    TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - guarded for local validation
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    HfArgumentParser = None
    get_cosine_schedule_with_warmup = None
    TRANSFORMERS_IMPORT_ERROR = exc

try:  # pragma: no cover - unavailable in the current sandbox by default
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    PEFT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - guarded for local validation
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PEFT_IMPORT_ERROR = exc

from scripts_asag.alice_label_remap import remap_predictions_to_original_alice_labels
from scripts_asag.data_processing.benchmark_meta import BENCHMARK_DESCRIPTIONS
from scripts_asag.data_processing.lasa_format import build_lasa_llm_input
from utils import clear_gpu_memory, per_qid_metrics, save_report, set_seed


SUPPORTED_BENCHMARKS = {
    "alice_lp",
    "asap_sas",
    "beetle",
    "istudio",
    "pt_asag",
    "scientsbank",
    "scientsbank2",
}
ALICE_UNSUPPORTED_BENCHMARKS = {"alice_ke", "alice_sk"}
NON_CAUSAL_MODEL_TYPES = {
    "albert",
    "bert",
    "deberta",
    "deberta-v2",
    "distilbert",
    "electra",
    "modernbert",
    "roberta",
    "xlm-roberta",
}
PREDICTION_PATTERN = re.compile(r"(?<!\d)-?\d+(?!\d)")
REPO_ROOT = Path(__file__).resolve().parent.parent


def require_transformers() -> None:
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "transformers is required to run train_gen.py. "
            f"Original import error: {TRANSFORMERS_IMPORT_ERROR}"
        ) from TRANSFORMERS_IMPORT_ERROR


def require_peft() -> None:
    if PEFT_IMPORT_ERROR is not None:
        raise ImportError(
            "peft is required when --use_lora is enabled. "
            f"Original import error: {PEFT_IMPORT_ERROR}"
        ) from PEFT_IMPORT_ERROR


def require_accelerate() -> None:
    if ACCELERATE_IMPORT_ERROR is not None:
        raise ImportError(
            "accelerate is required to run distributed generation training. "
            f"Original import error: {ACCELERATE_IMPORT_ERROR}"
        ) from ACCELERATE_IMPORT_ERROR


def normalize_cli_args(args: list[str]) -> list[str]:
    alias_map = {
        "--train-tasks": "--train_tasks",
        "--eval-tasks": "--eval_tasks",
        "--test-tasks": "--test_tasks",
        "--max-source-length": "--max_source_length",
        "--max-new-tokens": "--max_new_tokens",
        "--eval-batch-size": "--eval_batch_size",
        "--max-epoch": "--max_epoch",
        "--weight-decay": "--weight_decay",
        "--warmup-ratio": "--warmup_ratio",
        "--gradient-accumulation-steps": "--gradient_accumulation_steps",
        "--clip-norm": "--clip_norm",
        "--save-dir": "--save_dir",
        "--test-only": "--test_only",
        "--log-wandb": "--log_wandb",
        "--use-lora": "--use_lora",
        "--use-bnb": "--use_bnb",
        "--lora-rank": "--lora_rank",
        "--lora-alpha": "--lora_alpha",
        "--num-beams": "--num_beams",
        "--top-p": "--top_p",
        "--add-context": "--add_context",
        "--add-suffix": "--add_suffix",
        "--random-suffix": "--random_suffix",
        "--use-translated-prompts": "--use_translated_prompts",
        "--random-solution": "--random_solution",
        "--train-frac": "--train_frac",
        "--base-model": "--base_model",
    }
    return [alias_map.get(arg, arg) for arg in args]


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def stable_hash(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def parse_int_like(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or not float(value).is_integer():
            return None
        return int(value)

    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        fvalue = float(value_str)
    except ValueError:
        return None
    if not fvalue.is_integer():
        return None
    return int(fvalue)


def is_nan_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    value_str = str(value).strip().lower()
    return value_str in {"", "nan", "none", "null"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def format_rubric_text(value: Any) -> str:
    if isinstance(value, dict):
        rule = clean_text(value.get("rule"))
        description = clean_text(value.get("description"))
        if is_nan_like(description):
            description = ""
        if rule and description and description != rule:
            return f"{rule}. {description}"
        return rule or description
    return clean_text(value)


def select_sample_solution(value: Any, rng: random.Random, random_solution: bool, is_training: bool) -> str:
    if isinstance(value, list):
        candidates = [clean_text(item) for item in value if not is_nan_like(item)]
        if not candidates:
            return ""
        if random_solution and is_training:
            return rng.choice(candidates)
        return candidates[0]
    return clean_text(value)


def validate_task_names(task_names: list[str]) -> None:
    unknown = dedupe_keep_order([task for task in task_names if task not in BENCHMARK_DESCRIPTIONS])
    if unknown:
        valid = ", ".join(sorted(BENCHMARK_DESCRIPTIONS))
        raise ValueError(f"Unknown ASAG benchmark(s): {unknown}. Valid options: {valid}")
    unsupported = dedupe_keep_order([task for task in task_names if task in ALICE_UNSUPPORTED_BENCHMARKS])
    if unsupported:
        raise ValueError(
            "train_gen.py only supports alice_lp for ALICE. "
            f"Unsupported benchmark(s): {unsupported}"
        )
    outside_scope = dedupe_keep_order([task for task in task_names if task not in SUPPORTED_BENCHMARKS])
    if outside_scope:
        raise ValueError(
            "train_gen.py only supports these ASAG benchmarks: "
            f"{sorted(SUPPORTED_BENCHMARKS)}. Received: {outside_scope}"
        )


@dataclass
class TaskArguments:
    base_model: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "causal/instruction model to use"},
    )
    seed: int = field(default=114514, metadata={"help": "random seed"})
    train_frac: float = field(
        default=1.0,
        metadata={
            "help": (
                "fraction of training data to use when <= 1, "
                "or exact number of training instances when > 1"
            )
        },
    )
    benchmark: Optional[str] = field(
        default=None,
        metadata={"help": "single ASAG benchmark for mono-benchmark training"},
    )
    train_tasks: list[str] = field(
        default_factory=lambda: ["alice_lp"],
        metadata={"help": "ASAG datasets used for training"},
    )
    eval_tasks: list[str] = field(
        default_factory=list,
        metadata={"help": "ASAG datasets used for validation; defaults to train_tasks"},
    )
    test_tasks: list[str] = field(
        default_factory=list,
        metadata={"help": "ASAG datasets used for test evaluation; defaults to train_tasks"},
    )
    add_context: bool = field(default=True, metadata={"help": "include question/context fields"})
    add_suffix: bool = field(default=False, metadata={"help": "append the LASA instruction suffix"})
    random_suffix: bool = field(default=False, metadata={"help": "randomly sample LASA instruction suffixes"})
    use_translated_prompts: bool = field(
        default=False,
        metadata={"help": "translate fixed prompt labels when metadata supports it"},
    )
    random_solution: bool = field(
        default=False,
        metadata={"help": "randomly sample one reference solution during training"},
    )
    dry_run: bool = field(default=False, metadata={"help": "build data only and stop"})
    max_source_length: int = field(default=2048, metadata={"help": "prompt token budget"})
    max_new_tokens: int = field(default=8, metadata={"help": "generation budget"})

    def __post_init__(self) -> None:
        if self.train_frac <= 0:
            raise ValueError("train_frac must be > 0")
        if self.train_frac > 1 and not float(self.train_frac).is_integer():
            raise ValueError("train_frac > 1 must be an integer-valued exact number of training instances")
        if self.max_source_length <= 0:
            raise ValueError("max_source_length must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if self.benchmark is not None:
            if self.train_tasks not in (["alice_lp"], [self.benchmark]):
                raise ValueError("Use either --benchmark or --train_tasks, not both.")
            self.train_tasks = [self.benchmark]

        self.train_tasks = dedupe_keep_order(self.train_tasks)
        if not self.train_tasks:
            raise ValueError("train_tasks must contain at least one benchmark")
        self.eval_tasks = dedupe_keep_order(self.eval_tasks) if self.eval_tasks else list(self.train_tasks)
        self.test_tasks = dedupe_keep_order(self.test_tasks) if self.test_tasks else list(self.train_tasks)
        validate_task_names(self.train_tasks + self.eval_tasks + self.test_tasks)


@dataclass
class GenerationTrainingArguments:
    batch_size: int = field(default=4, metadata={"help": "train batch size"})
    eval_batch_size: Optional[int] = field(default=None, metadata={"help": "evaluation batch size"})
    max_epoch: int = field(default=3, metadata={"help": "number of training epochs"})
    lr: float = field(default=2e-5, metadata={"help": "learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "weight decay"})
    warmup_ratio: float = field(default=0.01, metadata={"help": "warmup ratio"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "gradient accumulation"})
    clip_norm: float = field(default=1.0, metadata={"help": "gradient clipping norm"})
    save_dir: Optional[str] = field(default=None, metadata={"help": "output directory (default: results_{benchmark}/llm_gen)"})
    cp_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint directory for test_only"})
    test_only: bool = field(default=False, metadata={"help": "skip training and run evaluation only"})
    bf16: bool = field(default=False, metadata={"help": "use bfloat16 on CUDA when available"})
    log_wandb: bool = field(default=False, metadata={"help": "log metrics to Weights & Biases"})
    use_lora: bool = field(default=False, metadata={"help": "train adapters with LoRA"})
    use_bnb: bool = field(default=False, metadata={"help": "load the model in 4-bit mode"})
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    num_beams: int = field(default=1, metadata={"help": "beam width for generation"})
    do_sample: bool = field(default=False, metadata={"help": "sample during generation"})
    temperature: float = field(default=1.0, metadata={"help": "sampling temperature"})
    top_p: float = field(default=1.0, metadata={"help": "nucleus sampling threshold"})

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if self.max_epoch <= 0:
            raise ValueError("max_epoch must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.test_only and not self.cp_dir:
            raise ValueError("cp_dir must be specified when test_only=True")


@dataclass
class TaskDatasetBundle:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: dict[str, list[dict[str, Any]]]


class SupervisedGenerationDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_source_length: int):
        self.records: list[dict[str, Any]] = []
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id for generation training.")

        for example in examples:
            target_text = f" {example['labels']}"
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [eos_token_id]
            max_prompt_length = max(8, max_source_length - len(target_ids))
            prompt_ids = tokenizer(
                example["prompt"],
                add_special_tokens=True,
                truncation=True,
                max_length=max_prompt_length,
            )["input_ids"]
            input_ids = prompt_ids + target_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + target_ids
            self.records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


class PromptOnlyDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_source_length: int):
        self.records: list[dict[str, Any]] = []
        for example in examples:
            encoded = tokenizer(
                example["prompt"],
                add_special_tokens=True,
                truncation=True,
                max_length=max_source_length,
            )
            self.records.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "meta": {
                        "id": example["id"],
                        "question_id": example["question_id"],
                        "benchmark": example["benchmark"],
                        "split": example["split"],
                        "labels": example.get("labels"),
                        "valid_level_ids": list(example["valid_level_ids"]),
                        "prompt": example["prompt"],
                    },
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def left_pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(([pad_value] * pad_len) + seq)
    return torch.tensor(padded, dtype=torch.long)


def train_collate_fn(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = left_pad_sequences([item["input_ids"] for item in batch], pad_token_id)
    attention_mask = left_pad_sequences([item["attention_mask"] for item in batch], 0)
    labels = left_pad_sequences([item["labels"] for item in batch], -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def eval_collate_fn(batch: list[dict[str, Any]], pad_token_id: int) -> tuple[dict[str, torch.Tensor], dict[str, list[Any]]]:
    input_ids = left_pad_sequences([item["input_ids"] for item in batch], pad_token_id)
    attention_mask = left_pad_sequences([item["attention_mask"] for item in batch], 0)
    meta_keys = batch[0]["meta"].keys()
    meta = {key: [item["meta"][key] for item in batch] for key in meta_keys}
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, meta


def build_prompt(example: dict[str, Any], task_args: TaskArguments, rng: random.Random | None = None) -> str:
    benchmark = example["benchmark"]
    benchmark_meta = BENCHMARK_DESCRIPTIONS[benchmark]
    prompt, _, _ = build_lasa_llm_input(
        example,
        benchmark_meta,
        add_context=task_args.add_context,
        add_suffix=task_args.add_suffix,
        random_suffix=task_args.random_suffix,
        random_solution=task_args.random_solution,
        use_translated_prompts=task_args.use_translated_prompts,
        rng=rng,
    )
    return prompt


def sample_training_rows(rows: list[dict[str, Any]], train_frac: float, seed: int) -> list[dict[str, Any]]:
    if train_frac == 1:
        return list(rows)

    rng = random.Random(seed)
    if train_frac < 1:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            qid = str(row.get("question_id", row.get("id", "")))
            grouped.setdefault(qid, []).append(row)
        all_qids = list(grouped.keys())
        n_sample = max(1, int(len(all_qids) * train_frac))
        sampled_qids = set(rng.sample(all_qids, n_sample))
        sampled_rows: list[dict[str, Any]] = []
        for qid in all_qids:
            if qid in sampled_qids:
                sampled_rows.extend(grouped[qid])
        return sampled_rows

    n_sample = int(train_frac)
    if n_sample > len(rows):
        raise ValueError(
            f"train_frac={train_frac} requests {n_sample} training instances, "
            f"but only {len(rows)} are available."
        )
    sampled_indices = rng.sample(range(len(rows)), n_sample)
    return [rows[idx] for idx in sampled_indices]


def split_train_val(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2:
        return list(rows), list(rows)

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * 0.1)))
    val_size = min(val_size, len(shuffled) - 1)
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:]
    return train_rows, val_rows


def sorted_level_items(rubric_map: dict[str, Any], label_map: dict[str, int] | None = None) -> list[tuple[int, Any]]:
    if label_map:
        pairs = [
            (label_id, rubric_map[label_name])
            for label_name, label_id in sorted(label_map.items(), key=lambda item: item[1])
            if label_name in rubric_map
        ]
        if pairs:
            return pairs

    pairs: list[tuple[int, Any]] = []
    for key, value in rubric_map.items():
        level_id = parse_int_like(key)
        if level_id is None:
            continue
        pairs.append((level_id, value))
    pairs.sort(key=lambda item: item[0])
    return pairs


def normalize_level(raw_level: Any, label_map: dict[str, int] | None = None) -> int | None:
    if label_map and isinstance(raw_level, str):
        normalized = raw_level.strip()
        if normalized in label_map:
            return int(label_map[normalized])
        lowered = normalized.lower()
        if lowered in label_map:
            return int(label_map[lowered])
    return parse_int_like(raw_level)


def parse_generated_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        return {}

    candidates = [response.strip()]
    if candidates[0].startswith("```"):
        lines = candidates[0].splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def apply_generated_rubric_meta(
    base_dir: Path,
    benchmark_meta: dict[str, Any],
    question_meta: dict[str, Any],
) -> dict[str, Any]:
    rubric_meta_path = benchmark_meta.get("rubric_meta_path")
    if not rubric_meta_path:
        return question_meta

    generated_path = base_dir / rubric_meta_path
    if not generated_path.exists():
        return question_meta

    generated_meta = load_json(generated_path)
    for qid, generated_entry in generated_meta.items():
        meta = question_meta.setdefault(str(qid), {})
        response = parse_generated_response(generated_entry.get("response", {}))
        if isinstance(response, dict) and isinstance(response.get("rubrics"), dict):
            meta["rubrics"] = response["rubrics"]
        if "reference_answer" in generated_entry:
            meta["sample_solution"] = generated_entry["reference_answer"]
    return question_meta


def build_general_examples(
    benchmark: str,
    rows: list[dict[str, Any]],
    question_meta: dict[str, Any],
    split_name: str,
    task_args: TaskArguments,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    is_training = split_name == "train"
    rng = random.Random(task_args.seed + stable_hash(f"{benchmark}:{split_name}"))
    benchmark_meta = BENCHMARK_DESCRIPTIONS.get(benchmark, {})
    label_map = benchmark_meta.get("label_map")

    for row in rows:
        question_id = str(row.get("question_id", ""))
        meta = question_meta.get(question_id, {})
        rubric_items = sorted_level_items(meta.get("rubrics", {}), label_map=label_map)
        if not rubric_items:
            continue

        valid_level_ids = [level_id for level_id, _ in rubric_items]
        rubrics = [format_rubric_text(value) for _, value in rubric_items]
        label = normalize_level(row.get("level", row.get("score_level")), label_map=label_map)
        sample_solution = select_sample_solution(
            meta.get("sample_solution"),
            rng=rng,
            random_solution=task_args.random_solution,
            is_training=is_training,
        )
        example = {
            "id": str(row.get("id", f"{benchmark}_{split_name}_{len(examples)}")),
            "question_id": question_id,
            "benchmark": benchmark,
            "split": split_name,
            "labels": label,
            "answer": clean_text(row.get("answer")),
            "valid_level_ids": valid_level_ids,
            "rubrics": rubrics,
            "rubric": rubrics,
        }
        if "question" in meta:
            example["question"] = clean_text(meta.get("question"))
        if "question_context" in meta:
            example["question_context"] = clean_text(meta.get("question_context"))
        if "sample_solution" in meta:
            example["sample_solution"] = sample_solution
        example["prompt"] = build_prompt(example, task_args, rng=rng if is_training else None)
        examples.append(example)
    return examples


def build_alice_lp_examples(
    rows: list[dict[str, Any]],
    question_meta: dict[str, Any],
    split_name: str,
    task_args: TaskArguments,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    is_training = split_name == "train"
    rng = random.Random(task_args.seed + stable_hash(f"alice_lp:{split_name}"))

    for row in rows:
        question_id = str(row.get("question_id", ""))
        meta = question_meta.get(question_id, {})
        lp_rubrics = sorted_level_items(meta.get("learning_performance", {}))
        if not lp_rubrics:
            continue

        valid_level_ids = [level_id for level_id, _ in lp_rubrics]
        rubrics = [format_rubric_text(value) for _, value in lp_rubrics]
        lp_entry = row.get("learning_performance", {}) or {}
        label_raw = next(iter(lp_entry.values()), 0)
        label = parse_int_like(label_raw)
        example = {
            "id": str(row.get("id", f"alice_lp_{split_name}_{len(examples)}")),
            "question_id": question_id,
            "benchmark": "alice_lp",
            "split": split_name,
            "labels": label,
            "answer": clean_text(row.get("answer")),
            "question": clean_text(meta.get("prompt")),
            "sample_solution": clean_text(meta.get("sample_solution", "")),
            "valid_level_ids": valid_level_ids,
            "rubrics": rubrics,
            "rubric": rubrics,
        }
        example["prompt"] = build_prompt(example, task_args, rng=rng if is_training else None)
        examples.append(example)
    return examples


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict(orient="records")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_general_task_bundle(benchmark: str, task_args: TaskArguments) -> TaskDatasetBundle:
    base_dir = REPO_ROOT / "asag_benchmarks" / benchmark
    train_rows = read_csv_records(base_dir / "train.csv")
    train_rows = sample_training_rows(train_rows, task_args.train_frac, task_args.seed + stable_hash(benchmark))

    val_path = base_dir / "val.csv"
    if val_path.exists():
        val_rows = read_csv_records(val_path)
    else:
        train_rows, val_rows = split_train_val(train_rows, task_args.seed + 42 + stable_hash(benchmark))

    question_meta = load_json(base_dir / "question_meta.json")
    question_meta = {str(key): value for key, value in question_meta.items()}
    question_meta = apply_generated_rubric_meta(
        base_dir,
        BENCHMARK_DESCRIPTIONS.get(benchmark, {}),
        question_meta,
    )

    train_examples = build_general_examples(benchmark, train_rows, question_meta, "train", task_args)
    val_examples = build_general_examples(benchmark, val_rows, question_meta, "val", task_args)

    test_examples: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(base_dir.glob("test*.csv")):
        split_name = path.stem
        split_rows = read_csv_records(path)
        test_examples[split_name] = build_general_examples(
            benchmark,
            split_rows,
            question_meta,
            split_name,
            task_args,
        )

    if not test_examples:
        raise ValueError(f"No test files found for benchmark {benchmark}")
    return TaskDatasetBundle(train=train_examples, val=val_examples, test=test_examples)


def load_alice_lp_bundle(task_args: TaskArguments) -> TaskDatasetBundle:
    alice_dir = REPO_ROOT / "asag_benchmarks" / "alice_data"
    question_meta = load_json(alice_dir / "question_meta.json")
    question_meta = {str(key): value for key, value in question_meta.items()}

    train_rows = load_json(alice_dir / "train.json")
    train_rows = sample_training_rows(train_rows, task_args.train_frac, task_args.seed + stable_hash("alice_lp"))
    train_rows, val_rows = split_train_val(train_rows, task_args.seed + 42 + stable_hash("alice_lp"))

    test_ua_rows = load_json(alice_dir / "test_ua.json")
    test_uq_rows = load_json(alice_dir / "test_uq.json")

    return TaskDatasetBundle(
        train=build_alice_lp_examples(train_rows, question_meta, "train", task_args),
        val=build_alice_lp_examples(val_rows, question_meta, "val", task_args),
        test={
            "test_ua": build_alice_lp_examples(test_ua_rows, question_meta, "test_ua", task_args),
            "test_uq": build_alice_lp_examples(test_uq_rows, question_meta, "test_uq", task_args),
        },
    )


def load_task_bundle(benchmark: str, task_args: TaskArguments) -> TaskDatasetBundle:
    if benchmark == "alice_lp":
        return load_alice_lp_bundle(task_args)
    return load_general_task_bundle(benchmark, task_args)


def load_all_task_bundles(task_args: TaskArguments) -> dict[str, TaskDatasetBundle]:
    task_names = dedupe_keep_order(task_args.train_tasks + task_args.eval_tasks + task_args.test_tasks)
    return {task_name: load_task_bundle(task_name, task_args) for task_name in task_names}


def combine_examples(
    task_args: TaskArguments,
    task_bundles: dict[str, TaskDatasetBundle],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    train_examples: list[dict[str, Any]] = []
    val_examples: list[dict[str, Any]] = []
    test_examples: dict[str, list[dict[str, Any]]] = {}

    train_sizes: dict[str, int] = {}
    val_sizes: dict[str, int] = {}
    test_sizes: dict[str, int] = {}

    for task_name in task_args.train_tasks:
        task_train = task_bundles[task_name].train
        train_examples.extend(task_train)
        train_sizes[task_name] = len(task_train)

    for task_name in task_args.eval_tasks:
        task_val = task_bundles[task_name].val
        val_examples.extend(task_val)
        val_sizes[task_name] = len(task_val)

    for task_name in task_args.test_tasks:
        for split_name, split_rows in task_bundles[task_name].test.items():
            dataset_key = f"{task_name}__{split_name}"
            test_examples[dataset_key] = split_rows
            test_sizes[dataset_key] = len(split_rows)

    size_summary = {
        "train_sizes": train_sizes,
        "val_sizes": val_sizes,
        "test_sizes": test_sizes,
    }
    return train_examples, val_examples, test_examples, size_summary


def compute_generation_metrics(pred_df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "response_inclusion_rate": 0.0,
        "accuracy": None,
        "f1": None,
        "qwk": None,
    }
    if pred_df.empty:
        return metrics

    metrics["response_inclusion_rate"] = float(pred_df["response_included"].astype(float).mean())

    if "labels" not in pred_df.columns:
        return metrics

    label_series = pd.to_numeric(pred_df["labels"], errors="coerce")
    pred_series = pd.to_numeric(pred_df["pred_id"], errors="coerce")
    valid_rows = label_series.notna() & pred_series.notna()
    if not valid_rows.any():
        return metrics

    y_true = label_series.loc[valid_rows].astype(int).tolist()
    y_pred = pred_series.loc[valid_rows].astype(int).tolist()
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    return metrics


def parse_prediction_text(text: str, valid_level_ids: list[int]) -> tuple[int, bool, bool]:
    valid_set = set(valid_level_ids)
    for match in PREDICTION_PATTERN.finditer(text):
        value = parse_int_like(match.group(0))
        if value is None:
            continue
        if value in valid_set:
            return value, True, False
    return -1, False, True


def maybe_cast_int_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    numeric = pd.to_numeric(df[col], errors="coerce")
    if numeric.isna().any():
        return
    if not ((numeric % 1) == 0).all():
        return
    df[col] = numeric.astype(int)


def init_wandb(train_args: GenerationTrainingArguments, task_args: TaskArguments):
    if not train_args.log_wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"Warning: wandb import failed, disabling logging. Error: {exc}")
        return None

    wandb.login()
    wandb.init(
        config={**asdict(train_args), **asdict(task_args)},
        dir=train_args.save_dir,
        project="span-align-gen",
    )
    return wandb


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for param in model.parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
    pct = 100.0 * trainable / total if total else 0.0
    print(f"Trainable parameters: {trainable:,d} / {total:,d} ({pct:.2f}%)")


def get_autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def validate_generation_model(config, base_model: str) -> None:
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError(
            f"{base_model} appears to be an encoder-decoder model. "
            "train_gen.py expects a decoder-only causal/instruction LM."
        )
    if getattr(config, "model_type", None) in NON_CAUSAL_MODEL_TYPES:
        raise ValueError(
            f"{base_model} appears to be an encoder-style model ({config.model_type}). "
            "Please choose a causal/instruction LM for generation training."
        )


class GenerationBaselineTrainer:
    def __init__(
        self,
        task_args: TaskArguments,
        train_args: GenerationTrainingArguments,
        train_examples: list[dict[str, Any]],
        val_examples: list[dict[str, Any]],
        test_examples: dict[str, list[dict[str, Any]]],
    ):
        self.task_args = task_args
        self.train_args = train_args
        self.train_examples = train_examples
        self.val_examples = val_examples
        self.test_examples = test_examples
        require_accelerate()
        self.accelerator = Accelerator(gradient_accumulation_steps=train_args.gradient_accumulation_steps)
        self.device = self.accelerator.device
        self.wandb = init_wandb(train_args, task_args) if self.accelerator.is_main_process else None

        self.tokenizer = self._load_tokenizer(task_args.base_model)
        self.model = self._load_model(task_args.base_model)
        self.best_val_metrics: Optional[dict[str, Any]] = None

        if not self.train_args.test_only and self.accelerator.is_main_process:
            self._save_training_args()
        self.accelerator.wait_for_everyone()

    def _load_tokenizer(self, base_model: str):
        require_transformers()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            else:
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _model_load_kwargs(self) -> dict[str, Any]:
        load_kwargs: dict[str, Any] = {}
        if self.train_args.use_bnb:
            if not torch.cuda.is_available():
                print("Warning: --use_bnb requested but CUDA is unavailable. Disabling 4-bit loading.")
                self.train_args.use_bnb = False
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                load_kwargs["device_map"] = {"": self.accelerator.process_index}
        if self.train_args.bf16 and torch.cuda.is_available() and not self.train_args.use_bnb:
            load_kwargs["torch_dtype"] = torch.bfloat16
        return load_kwargs

    def _load_model(self, model_path: str):
        require_transformers()
        config = AutoConfig.from_pretrained(model_path)
        validate_generation_model(config, model_path)

        load_kwargs = self._model_load_kwargs()
        is_adapter_checkpoint = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        if is_adapter_checkpoint:
            require_peft()
            base_model = AutoModelForCausalLM.from_pretrained(
                self.task_args.base_model,
                **load_kwargs,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        if len(self.tokenizer) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        if self.train_args.use_lora and not is_adapter_checkpoint:
            require_peft()
            if self.train_args.use_bnb:
                model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=self.train_args.lora_rank,
                lora_alpha=self.train_args.lora_alpha,
                lora_dropout=0.1,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        if not self.train_args.use_bnb:
            model = model.to(self.device)

        print_trainable_parameters(model)
        return model

    def _save_training_args(self) -> None:
        os.makedirs(self.train_args.save_dir, exist_ok=True)
        all_args = {**asdict(self.train_args), **asdict(self.task_args)}
        with open(os.path.join(self.train_args.save_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(all_args, f, indent=2)

    def _save_best_checkpoint(self, metrics: dict[str, Any]) -> None:
        os.makedirs(self.train_args.save_dir, exist_ok=True)
        model_to_save = self.accelerator.unwrap_model(self.model)
        model_to_save.save_pretrained(self.train_args.save_dir, save_function=self.accelerator.save)
        self.tokenizer.save_pretrained(self.train_args.save_dir)
        save_report(metrics, os.path.join(self.train_args.save_dir, "best_val_metrics.json"))

    def _train_dataloader(self):
        dataset = SupervisedGenerationDataset(
            self.train_examples,
            tokenizer=self.tokenizer,
            max_source_length=self.task_args.max_source_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: train_collate_fn(batch, self.tokenizer.pad_token_id),
        )

    def _eval_dataloader(self, examples: list[dict[str, Any]]):
        dataset = PromptOnlyDataset(
            examples,
            tokenizer=self.tokenizer,
            max_source_length=self.task_args.max_source_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: eval_collate_fn(batch, self.tokenizer.pad_token_id),
        )

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.task_args.max_new_tokens,
            "num_beams": self.train_args.num_beams,
            "do_sample": self.train_args.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.train_args.do_sample:
            kwargs["temperature"] = self.train_args.temperature
            kwargs["top_p"] = self.train_args.top_p
        return kwargs

    def train(self) -> None:
        if not self.train_examples:
            raise ValueError("No training examples were loaded.")

        train_loader = self._train_dataloader()
        optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.train_args.lr,
            weight_decay=self.train_args.weight_decay,
        )
        self.model, optimizer, train_loader = self.accelerator.prepare(
            self.model,
            optimizer,
            train_loader,
        )
        total_update_steps_per_epoch = math.ceil(len(train_loader) / self.train_args.gradient_accumulation_steps)
        total_training_steps = max(1, total_update_steps_per_epoch * self.train_args.max_epoch)
        warmup_steps = int(total_training_steps * self.train_args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        scheduler = self.accelerator.prepare(scheduler)

        best_accuracy = float("-inf")
        global_step = 0
        use_autocast = self.train_args.bf16 and self.device.type == "cuda"

        for epoch in range(self.train_args.max_epoch):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.train_args.max_epoch}",
                disable=not self.accelerator.is_local_main_process,
            )
            for step, batch in enumerate(iterator):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with self.accelerator.accumulate(self.model):
                    with get_autocast_context(use_autocast):
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    self.accelerator.backward(loss)
                    running_loss += loss.detach().float().item()

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.train_args.clip_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                        if self.wandb is not None and self.accelerator.is_main_process:
                            self.wandb.log(
                                {
                                    "train/loss": float(running_loss / max(1, global_step)),
                                    "train/lr": float(scheduler.get_last_lr()[0]),
                                    "train/epoch": epoch + 1,
                                }
                            )

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                val_predictions, val_metrics = self.predict_examples(self.val_examples)
                self.best_val_metrics = val_metrics
                val_accuracy = val_metrics.get("accuracy")
                if val_accuracy is None:
                    val_accuracy = float("-inf")

                if self.wandb is not None:
                    self.wandb.log({f"val/{key}": value for key, value in val_metrics.items() if value is not None})

                print(f"Epoch {epoch + 1} validation metrics: {val_metrics}")
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    self._save_best_checkpoint(val_metrics)
                    print(f"Saved new best checkpoint to {self.train_args.save_dir}")

                if not val_predictions.empty:
                    val_predictions.to_csv(
                        os.path.join(self.train_args.save_dir, f"val_epoch_{epoch + 1}_predictions.csv"),
                        index=False,
                    )
            self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def predict_examples(self, examples: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not examples:
            return pd.DataFrame(), compute_generation_metrics(pd.DataFrame())

        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        dataloader = self._eval_dataloader(examples)
        generation_kwargs = self._generation_kwargs()
        predictions: dict[str, list[Any]] = {
            "id": [],
            "question_id": [],
            "benchmark": [],
            "split": [],
            "labels": [],
            "pred_id": [],
            "generated_text": [],
            "response_included": [],
            "parse_failed": [],
            "prompt": [],
        }

        iterator = tqdm(
            dataloader,
            desc="Generating",
            disable=not self.accelerator.is_local_main_process,
        )
        use_autocast = self.train_args.bf16 and self.device.type == "cuda"
        for batch, meta in iterator:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with get_autocast_context(use_autocast):
                generated = model.generate(**batch, **generation_kwargs)

            prompt_width = batch["input_ids"].shape[1]
            generated_only = generated[:, prompt_width:]
            decoded = self.tokenizer.batch_decode(generated_only, skip_special_tokens=True)

            for idx, text in enumerate(decoded):
                pred_id, response_included, parse_failed = parse_prediction_text(
                    text=text.strip(),
                    valid_level_ids=meta["valid_level_ids"][idx],
                )
                predictions["id"].append(meta["id"][idx])
                predictions["question_id"].append(meta["question_id"][idx])
                predictions["benchmark"].append(meta["benchmark"][idx])
                predictions["split"].append(meta["split"][idx])
                predictions["labels"].append(meta["labels"][idx])
                predictions["pred_id"].append(pred_id)
                predictions["generated_text"].append(text.strip())
                predictions["response_included"].append(bool(response_included))
                predictions["parse_failed"].append(bool(parse_failed))
                predictions["prompt"].append(meta["prompt"][idx])

        pred_df = pd.DataFrame(predictions)
        pred_df, _ = remap_predictions_to_original_alice_labels(
            pred_df,
            benchmark=pred_df["benchmark"].iloc[0] if not pred_df.empty else None,
        )
        maybe_cast_int_column(pred_df, "labels")
        maybe_cast_int_column(pred_df, "pred_id")
        metrics = compute_generation_metrics(pred_df)

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return pred_df, metrics

    def evaluate_and_save(self) -> None:
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if self.train_args.test_only:
                self.model = self._load_model(str(self.train_args.cp_dir))
            else:
                model_path = self.train_args.save_dir
                if model_path and os.path.abspath(str(model_path)) != os.path.abspath(self.train_args.save_dir):
                    self.model = self._load_model(str(model_path))

            pred_dir = os.path.join(self.train_args.save_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            aggregate_frames: list[pd.DataFrame] = []

            for dataset_key, examples in self.test_examples.items():
                print(f"***** Running generation evaluation on {dataset_key} *****")
                pred_df, metrics = self.predict_examples(examples)
                pred_df.to_csv(os.path.join(pred_dir, f"{dataset_key}_predictions.csv"), index=False)
                save_report(metrics, os.path.join(pred_dir, f"{dataset_key}_metrics.json"))
                per_qid = per_qid_metrics(pred_df)
                if per_qid is not None:
                    save_report(per_qid, os.path.join(pred_dir, f"{dataset_key}_per_question_metrics.json"))
                aggregate_frames.append(pred_df)

                if self.wandb is not None:
                    self.wandb.log({f"test/{dataset_key}/{key}": value for key, value in metrics.items() if value is not None})

                print(f"{dataset_key} metrics: {metrics}")

            if aggregate_frames:
                aggregate_df = pd.concat(aggregate_frames, ignore_index=True)
                aggregate_df.to_csv(os.path.join(pred_dir, "aggregate_predictions.csv"), index=False)
                aggregate_metrics = compute_generation_metrics(aggregate_df)
                save_report(aggregate_metrics, os.path.join(pred_dir, "aggregate_metrics.json"))
                if self.wandb is not None:
                    self.wandb.log({f"aggregate/{key}": value for key, value in aggregate_metrics.items() if value is not None})
                print(f"Aggregate metrics: {aggregate_metrics}")

        self.accelerator.wait_for_everyone()


def parse_args(args: Optional[list[str]] = None):
    require_transformers()
    parser = HfArgumentParser((TaskArguments, GenerationTrainingArguments))
    normalized_args = normalize_cli_args(sys.argv[1:] if args is None else args)
    return parser.parse_args_into_dataclasses(args=normalized_args)


def load_args_from_checkpoint(
    cp_dir: str,
    current_train_args: GenerationTrainingArguments,
    current_task_args: TaskArguments,
) -> tuple[GenerationTrainingArguments, TaskArguments]:
    args_path = Path(cp_dir) / "training_args.json"
    if not args_path.exists():
        args_path = Path(cp_dir).parent / "training_args.json"
    if not args_path.exists():
        print(f"Warning: training_args.json not found in {cp_dir} or its parent. Using current arguments.")
        return current_train_args, current_task_args

    with args_path.open("r", encoding="utf-8") as f:
        saved_args = json.load(f)

    updated_train_args = deepcopy(current_train_args)
    updated_task_args = deepcopy(current_task_args)

    inference_specific_train_args = {"test_only", "cp_dir", "save_dir", "log_wandb"}
    preserve_task_args: set[str] = set()
    if current_task_args.eval_tasks and current_task_args.eval_tasks != current_task_args.train_tasks:
        preserve_task_args.add("eval_tasks")
    if current_task_args.test_tasks and current_task_args.test_tasks != current_task_args.train_tasks:
        preserve_task_args.add("test_tasks")

    for key, value in saved_args.items():
        if hasattr(updated_train_args, key) and key not in inference_specific_train_args:
            setattr(updated_train_args, key, value)
        elif hasattr(updated_task_args, key) and key not in preserve_task_args:
            setattr(updated_task_args, key, value)

    updated_train_args.__post_init__()
    updated_task_args.__post_init__()
    return updated_train_args, updated_task_args


def preview_prompts(train_examples: list[dict[str, Any]], val_examples: list[dict[str, Any]], test_examples: dict[str, list[dict[str, Any]]]) -> None:
    if train_examples:
        print("Sample training prompt:")
        print(train_examples[0]["prompt"])
        print(f"Gold label: {train_examples[0]['labels']}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Test splits: { {key: len(value) for key, value in test_examples.items()} }")


def main(task_args: TaskArguments, train_args: GenerationTrainingArguments) -> None:
    if train_args.test_only and train_args.cp_dir:
        print(f"Loading arguments from checkpoint directory: {train_args.cp_dir}")
        train_args, task_args = load_args_from_checkpoint(train_args.cp_dir, train_args, task_args)

    if train_args.save_dir is None:
        task_label = task_args.benchmark or "_".join(task_args.train_tasks)
        train_args.save_dir = f"results_{task_label}/llm_gen"

    set_seed(task_args.seed)
    os.makedirs(train_args.save_dir, exist_ok=True)

    task_bundles = load_all_task_bundles(task_args)
    train_examples, val_examples, test_examples, size_summary = combine_examples(task_args, task_bundles)

    print(f"Training tasks: {task_args.train_tasks}")
    print(f"Evaluation tasks: {task_args.eval_tasks}")
    print(f"Test tasks: {task_args.test_tasks}")
    print(f"Dataset sizes: {size_summary}")

    if task_args.dry_run:
        preview_prompts(train_examples, val_examples, test_examples)
        return

    trainer = GenerationBaselineTrainer(
        task_args=task_args,
        train_args=train_args,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
    )

    if not train_args.test_only:
        print("***** Running training *****")
        print(f"Num train examples = {len(train_examples)}")
        print(f"Num val examples = {len(val_examples)}")
        trainer.train()
        print("***** Training finished *****")

    trainer.evaluate_and_save()
    clear_gpu_memory()


if __name__ == "__main__":
    parsed_task_args, parsed_train_args = parse_args()
    main(parsed_task_args, parsed_train_args)
