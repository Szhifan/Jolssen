from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request


WORKSPACE_ROOT = Path(__file__).resolve().parent
load_dotenv(WORKSPACE_ROOT / ".env")

ASAG_BENCHMARK_OPTIONS: list[tuple[str, str]] = [
    ("alice_lp", "ALICE LP"),
    ("alice_ke", "ALICE KE"),
    ("alice_sk", "ALICE SK"),
    ("asap_sas", "ASAP SAS"),
    ("beetle", "BEETLE"),
    ("istudio", "iStudio"),
    ("pt_asag", "PT ASAG"),
    ("scientsbank", "Scientsbank"),
    ("scientsbank2", "Scientsbank2"),
]
ASAG_BENCHMARK_SET = {value for value, _ in ASAG_BENCHMARK_OPTIONS}

MODEL_OPTIONS: list[tuple[str, str]] = [
    ("markussagen/xlm-roberta-longformer-base-4096", "XLM-Roberta Long"),
    ("jhu-clsp/mmBERT-base", "mmBERT-base"),
    ("meta-llama/Llama-3.2-1B-instruct", "Llama 3.2 1B Instruct"),
    ("meta-llama/Llama-3.2-3B-instruct", "Llama 3.2 3B Instruct"),
    ("meta-llama/Llama-3.2-1B", "Llama 3.2 1B"),
    ("meta-llama/Llama-3.2-3B", "Llama 3.2 3B"),
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B v0.1"),
    ("nvidia/NV-Embed-v2", "NVIDIA NV-Embed v2"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct"),
]

SPAN_FUSE_OPTIONS = [
    "p-concat",
    "p-diff",
    "p-gate",
    "p-condiff",
    "p-bl",
    "p-only",
    "l-only",
    "t-bl",
    "t-concat",
    "t-diff",
    "tpl-concat",
]

MODEL_SHORTNAME = {
    "markussagen/xlm-roberta-longformer-base-4096": "xlm-roberta-long",
    "jhu-clsp/mmBERT-base": "mmBERT-base",
    "meta-llama/Llama-3.2-1B-instruct": "llama3.2-1B-instruct",
    "meta-llama/Llama-3.2-3B-instruct": "llama3.2-3B-instruct",
    "meta-llama/Llama-3.2-1B": "llama3.2-1B",
    "meta-llama/Llama-3.2-3B": "llama3.2-3B",
    "mistralai/Mistral-7B-v0.1": "mistral-7B-v0.1",
    "nvidia/NV-Embed-v2": "nv-embed-v2",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8B-instruct",
}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _parse_task_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        raw = [x for x in stripped.replace(",", " ").split(" ") if x]
    else:
        raise ValueError(f"task list must be list[str] or string, got: {type(value)}")
    return _dedupe_keep_order([str(x).strip() for x in raw if str(x).strip()])


def _coerce_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(float(value))


def _coerce_float(value, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    return float(value)


def _task_signature(tasks: list[str], max_items: int = 3) -> str:
    if not tasks:
        return "none"
    visible = tasks[:max_items]
    suffix = "-etc" if len(tasks) > max_items else ""
    return f"{'+'.join(visible)}{suffix}"


@dataclass
class MultiExperimentConfig:
    train_tasks: list[str]
    eval_tasks: list[str] = field(default_factory=list)
    test_tasks: list[str] = field(default_factory=list)

    base_model: str = "markussagen/xlm-roberta-longformer-base-4096"
    model_class: str = "span"
    span_fuse_type: str = "p-concat"
    span_pool_type: str = "last"
    pool_type: str = "last"
    layer_fuse_type: str = "avg"

    num_bidir_layers: int = 0
    num_prune_layers: int = 0
    num_fuse_layers: int = 0
    num_unsink_layers: int = 0

    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    train_frac: float = 1.0
    lr: float = 2e-4
    max_epoch: int = 4
    seed: int = 114514
    test_drop_rub: float = 0.0
    train_drop_rub: float = 0.0

    use_lora: bool = True
    use_bnb: bool = True
    bf16: bool = True
    log_wandb: bool = True
    add_suffix: bool = True
    add_context: bool = True
    random_suffix: bool = True
    use_translated_prompts: bool = True
    random_solution: bool = False
    rubric_independent_attn: bool = False
    reindex_rub: bool = False

    exp_name: str = ""

    def generate_exp_name(self) -> str:
        if self.exp_name and self.exp_name.strip():
            return self.exp_name.strip()

        short_model = MODEL_SHORTNAME.get(
            self.base_model,
            self.base_model.split("/")[-1].lower(),
        )
        parts = [
            "multi",
            short_model,
            self.span_fuse_type if self.model_class == "span" else self.model_class,
        ]
        if self.random_solution:
            parts.append("randsolu")
        if self.rubric_independent_attn:
            parts.append("rub-ind")
            if self.reindex_rub:
                parts.append("reindex")
        return "-".join(parts)


def _normalize_config_dict(config_dict: dict) -> dict:
    out = dict(config_dict)

    train_tasks = _parse_task_list(out.get("train_tasks"))
    eval_tasks = _parse_task_list(out.get("eval_tasks"))
    test_tasks = _parse_task_list(out.get("test_tasks"))
    if not train_tasks:
        raise ValueError("train_tasks must contain at least one benchmark")
    if not eval_tasks:
        eval_tasks = list(train_tasks)
    if not test_tasks:
        test_tasks = list(train_tasks)

    invalid = [
        x
        for x in (train_tasks + eval_tasks + test_tasks)
        if x not in ASAG_BENCHMARK_SET
    ]
    if invalid:
        valid = ", ".join(sorted(ASAG_BENCHMARK_SET))
        raise ValueError(f"unknown benchmark(s): {invalid}. valid options: {valid}")

    out["train_tasks"] = train_tasks
    out["eval_tasks"] = eval_tasks
    out["test_tasks"] = test_tasks

    out["model_class"] = str(out.get("model_class", "span")).strip()
    if out["model_class"] not in {"span", "xnet"}:
        raise ValueError("model_class must be one of: span, xnet")
    if out["model_class"] == "span":
        out["span_fuse_type"] = str(out.get("span_fuse_type", "p-concat")).strip() or "p-concat"
    else:
        out["span_fuse_type"] = "p-concat"

    out["base_model"] = str(
        out.get("base_model", "markussagen/xlm-roberta-longformer-base-4096")
    ).strip()
    if not out["base_model"]:
        raise ValueError("base_model must be non-empty")

    out["batch_size"] = _coerce_int(out.get("batch_size"), 8)
    out["gradient_accumulation_steps"] = _coerce_int(out.get("gradient_accumulation_steps"), 2)
    out["max_epoch"] = _coerce_int(out.get("max_epoch"), 4)
    out["seed"] = _coerce_int(out.get("seed"), 114514)
    out["num_bidir_layers"] = _coerce_int(out.get("num_bidir_layers"), 0)
    out["num_prune_layers"] = _coerce_int(out.get("num_prune_layers"), 0)
    out["num_fuse_layers"] = _coerce_int(out.get("num_fuse_layers"), 0)
    out["num_unsink_layers"] = _coerce_int(out.get("num_unsink_layers"), 0)

    out["train_frac"] = _coerce_float(out.get("train_frac"), 1.0)
    out["lr"] = _coerce_float(out.get("lr"), 2e-4)
    out["test_drop_rub"] = _coerce_float(out.get("test_drop_rub"), 0.0)
    out["train_drop_rub"] = _coerce_float(out.get("train_drop_rub"), 0.0)

    if out["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")
    if out["gradient_accumulation_steps"] <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if out["max_epoch"] <= 0:
        raise ValueError("max_epoch must be > 0")
    if out["train_frac"] <= 0:
        raise ValueError("train_frac must be > 0")
    if out["lr"] <= 0:
        raise ValueError("lr must be > 0")
    if not (0.0 <= out["test_drop_rub"] <= 1.0):
        raise ValueError("test_drop_rub must be in [0, 1]")
    if not (0.0 <= out["train_drop_rub"] <= 1.0):
        raise ValueError("train_drop_rub must be in [0, 1]")

    out["pool_type"] = str(out.get("pool_type", "last")).strip() or "last"
    if out["pool_type"] not in {"avg", "weightedavg", "cls", "last"}:
        raise ValueError("pool_type must be one of: avg, weightedavg, cls, last")
    out["layer_fuse_type"] = str(out.get("layer_fuse_type", "avg")).strip() or "avg"
    if out["layer_fuse_type"] not in {"avg", "weighted"}:
        raise ValueError("layer_fuse_type must be one of: avg, weighted")
    out["span_pool_type"] = str(out.get("span_pool_type", "last")).strip() or "last"
    if out["span_pool_type"] not in {"mean", "last"}:
        raise ValueError("span_pool_type must be one of: mean, last")

    bool_fields = [
        "use_lora",
        "use_bnb",
        "bf16",
        "log_wandb",
        "add_suffix",
        "add_context",
        "random_suffix",
        "use_translated_prompts",
        "random_solution",
        "rubric_independent_attn",
        "reindex_rub",
    ]
    defaults = {
        "use_lora": True,
        "use_bnb": True,
        "bf16": True,
        "log_wandb": True,
        "add_suffix": True,
        "add_context": True,
        "random_suffix": True,
        "use_translated_prompts": True,
        "random_solution": False,
        "rubric_independent_attn": False,
        "reindex_rub": False,
    }
    if out["rubric_independent_attn"] and out.get("span_fuse_type") != "l-only":
        raise ValueError("rubric_independent_attn is only compatible with span_fuse_type='l-only'.")
    if out["reindex_rub"] and not out["rubric_independent_attn"]:
        raise ValueError("reindex_rub requires rubric_independent_attn to be set.")
    for field_name in bool_fields:
        out[field_name] = _coerce_bool(out.get(field_name), defaults[field_name])

    out["exp_name"] = str(out.get("exp_name", "") or "").strip()
    return out


def _finalize_cmd_lines(lines: list[str]) -> list[str]:
    if lines and lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]
    return lines


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _extract_api_tokens(payload: dict | None) -> tuple[str, str]:
    payload = payload or {}
    wandb_api_key = str(payload.get("wandb_api_key") or "").strip()
    hf_token = str(payload.get("hf_token") or "").strip()

    if not wandb_api_key:
        wandb_api_key = str(os.getenv("WANDB_API_KEY", "")).strip()
    if not hf_token:
        hf_token = str(os.getenv("HF_TOKEN", "")).strip()

    return wandb_api_key, hf_token


def _write_api_token_exports(file_obj, wandb_api_key: str, hf_token: str):
    if wandb_api_key:
        file_obj.write(f"export WANDB_API_KEY={_shell_single_quote(wandb_api_key)}\n")
    if hf_token:
        file_obj.write(f"export HF_TOKEN={_shell_single_quote(hf_token)}\n")
    if wandb_api_key or hf_token:
        file_obj.write("\n")


def _build_command_lines(config: MultiExperimentConfig, save_dir: str) -> list[str]:
    lines = [
        "accelerate launch \\",
        "    scripts_asag/train_multi.py \\",
        f"    --save-dir {save_dir} \\",
        f"    --train-tasks {' '.join(config.train_tasks)} \\",
        f"    --eval-tasks {' '.join(config.eval_tasks)} \\",
        f"    --test-tasks {' '.join(config.test_tasks)} \\",
        f"    --base-model \"{config.base_model}\" \\",
        f"    --model-class {config.model_class} \\",
        f"    --batch-size {config.batch_size} \\",
        f"    --gradient-accumulation-steps {config.gradient_accumulation_steps} \\",
        f"    --train-frac {config.train_frac} \\",
        f"    --lr {config.lr} \\",
        f"    --max-epoch {config.max_epoch} \\",
        f"    --seed {config.seed} \\",
    ]

    if config.model_class == "span":
        lines.append(f"    --span-fuse-type {config.span_fuse_type} \\")
        if config.span_pool_type != "last":
            lines.append(f"    --span-pool-type {config.span_pool_type} \\")
        if config.rubric_independent_attn:
            lines.append("    --rubric-independent-attn \\")
            if config.reindex_rub:
                lines.append("    --reindex-rub \\")

    if config.num_bidir_layers > 0:
        lines.append(f"    --num-bidir-layers {config.num_bidir_layers} \\")
    if config.num_prune_layers > 0:
        lines.append(f"    --num-prune-layers {config.num_prune_layers} \\")
    if config.num_fuse_layers > 0:
        lines.append(f"    --num-fuse-layers {config.num_fuse_layers} \\")
        lines.append(f"    --fuse-type {config.layer_fuse_type} \\")
    if config.num_unsink_layers > 0:
        lines.append(f"    --num-unsink-layers {config.num_unsink_layers} \\")
    if config.pool_type != "last":
        lines.append(f"    --pool-type {config.pool_type} \\")

    if config.random_solution:
        lines.append("    --random-solution \\")
    if config.use_lora:
        lines.append("    --use-lora \\")
    if config.use_bnb:
        lines.append("    --use-bnb \\")
    if config.add_suffix:
        lines.append("    --add-suffix \\")
    if config.add_context:
        lines.append("    --add-context \\")
    if config.random_suffix:
        lines.append("    --random-suffix \\")
    if config.use_translated_prompts:
        lines.append("    --use_translated_prompts \\")
    if config.test_drop_rub > 0:
        lines.append(f"    --test-drop-rub {config.test_drop_rub} \\")
    if config.train_drop_rub > 0:
        lines.append(f"    --train-drop-rub {config.train_drop_rub} \\")
    if config.bf16:
        lines.append("    --bf16 \\")
    if config.log_wandb:
        lines.append("    --log-wandb \\")

    return _finalize_cmd_lines(lines)


def _to_config(config_dict: dict) -> MultiExperimentConfig:
    normalized = _normalize_config_dict(config_dict)
    filtered = {
        key: value
        for key, value in normalized.items()
        if key in MultiExperimentConfig.__dataclass_fields__
    }
    return MultiExperimentConfig(**filtered)


def _build_run_script(
    configs: list[MultiExperimentConfig],
    wandb_api_key: str,
    hf_token: str,
) -> tuple[Path, list[dict], list[str]]:
    batch_id = int(time.time())
    run_script_path = WORKSPACE_ROOT / f"run_batch_multi_{batch_id}.sh"

    results = []
    command_blocks: list[str] = []

    for i, config in enumerate(configs):
        exp_name = config.generate_exp_name()
        exp_root = WORKSPACE_ROOT / "results_multi" / exp_name

        block = [
            f"# Experiment {i + 1}: {exp_name}",
            f'EXP_ROOT="{exp_root}"',
            "mkdir -p ${EXP_ROOT}",
            f'export WANDB_NAME="{exp_name}"',
        ]

        cmd_lines = _build_command_lines(config, "${EXP_ROOT}")
        if cmd_lines:
            cmd_lines[-1] += " 2>&1 | tee ${EXP_ROOT}/out.log"
        block.extend(cmd_lines)
        block.append("")
        command_blocks.append("\n".join(block))

        results.append(
            {
                "idx": i + 1,
                "exp_name": exp_name,
                "save_dir": str(exp_root),
                "status": "configured",
            }
        )

    with open(run_script_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -e\n\n")
        _write_api_token_exports(f, wandb_api_key, hf_token)
        f.write(f"# Batch ID: {batch_id}\n")
        f.write(f"# Total experiments: {len(configs)}\n")
        f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(command_blocks))

    os.chmod(run_script_path, 0o755)
    return run_script_path, results, command_blocks


DEFAULT_TRAIN_TASKS = {"asap_sas", "beetle", "scientsbank"}
DEFAULT_EVAL_TASKS = {"istudio"}
DEFAULT_TEST_TASKS = {"alice_lp", "asap_sas", "beetle", "scientsbank", "istudio", "pt_asag"}


def _build_template() -> str:
    def options(items: list[tuple[str, str]], selected: str | None = None) -> str:
        chunks = []
        for value, label in items:
            is_selected = " selected" if selected is not None and value == selected else ""
            chunks.append(f'<option value="{value}"{is_selected}>{label}</option>')
        return "\n".join(chunks)

    def multi_options(items: list[tuple[str, str]], selected_set: set) -> str:
        chunks = []
        for value, label in items:
            is_selected = " selected" if value in selected_set else ""
            chunks.append(f'<option value="{value}"{is_selected}>{label}</option>')
        return "\n".join(chunks)

    train_benchmark_options_html = multi_options(ASAG_BENCHMARK_OPTIONS, DEFAULT_TRAIN_TASKS)
    eval_benchmark_options_html = multi_options(ASAG_BENCHMARK_OPTIONS, DEFAULT_EVAL_TASKS)
    test_benchmark_options_html = multi_options(ASAG_BENCHMARK_OPTIONS, DEFAULT_TEST_TASKS)
    model_options_html = options(MODEL_OPTIONS, selected="markussagen/xlm-roberta-longformer-base-4096")
    span_fuse_html = "\n".join(
        f'<option value="{name}"{" selected" if name == "p-concat" else ""}>{name}</option>'
        for name in SPAN_FUSE_OPTIONS
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASAG Multi-Task Config</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 22px;
            color: #222;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: #fff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 24px;
            text-align: center;
        }}
        .header h1 {{ margin: 0 0 8px; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; }}
        .content {{ padding: 26px; }}
        .section {{ margin-bottom: 26px; }}
        .section h2 {{
            margin: 0 0 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #667eea;
            font-size: 22px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }}
        .form-group {{ margin-bottom: 12px; }}
        label {{
            display: block;
            margin-bottom: 6px;
            font-weight: 700;
            font-size: 14px;
            color: #444;
        }}
        select, input[type="number"], input[type="text"] {{
            width: 100%;
            border: 1px solid #d8d8d8;
            border-radius: 6px;
            padding: 10px;
            font-size: 14px;
            background: #fff;
        }}
        select[multiple] {{ min-height: 115px; }}
        .select-tall {{ min-height: 165px !important; }}
        .help {{ color: #666; font-size: 12px; margin-top: 4px; }}
        .checkbox-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(190px, 1fr));
            gap: 8px;
        }}
        .checkbox-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .checkbox-item input {{ width: 16px; height: 16px; }}
        .alert {{
            background: #dbe9f7;
            color: #0b4a8e;
            padding: 10px 12px;
            border-left: 4px solid #4c9be8;
            border-radius: 6px;
            margin-bottom: 10px;
        }}
        .exp-item {{
            border: 1px solid #dedede;
            border-radius: 6px;
            background: #f9f9f9;
            padding: 10px;
            margin-bottom: 8px;
        }}
        .exp-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            color: #454545;
            margin-bottom: 8px;
        }}
        .btn-delete {{
            border: none;
            border-radius: 4px;
            background: #dc3545;
            color: #fff;
            font-size: 12px;
            padding: 4px 9px;
            cursor: pointer;
        }}
        .btn-delete:hover {{ background: #c52b3a; }}
        .summary {{
            border: 1px solid #dbdbdb;
            border-radius: 6px;
            background: #f7f7f7;
            padding: 12px;
        }}
        .button-row {{
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-top: 12px;
        }}
        button {{
            border: none;
            border-radius: 7px;
            padding: 10px 18px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
        }}
        .btn-secondary {{ background: #e5e7eb; color: #313131; }}
        .btn-primary {{ color: #fff; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .status {{
            margin-top: 14px;
            min-height: 20px;
            font-size: 14px;
            color: #2f4f8f;
            white-space: pre-wrap;
        }}
        .output {{
            margin-top: 14px;
            background: #1f1f1f;
            color: #d5d5d5;
            border-radius: 6px;
            padding: 12px;
            white-space: pre-wrap;
            display: none;
            font-family: "Courier New", monospace;
            font-size: 13px;
            max-height: 340px;
            overflow: auto;
        }}
        @media (max-width: 900px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .checkbox-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ASAG Multi-Task Configuration</h1>
            <p>Standalone config page for the ASAG multi-task runner.</p>
        </div>
        <div class="content">
            <form id="multiForm">
                <div class="section">
                    <h2>Benchmark Splits</h2>
                    <div class="grid">
                        <div class="form-group">
                            <label for="trainTasks">Train Benchmarks *</label>
                            <select id="trainTasks" multiple class="select-tall">{train_benchmark_options_html}</select>
                            <div class="help">Select one or more tasks for train.</div>
                        </div>
                        <div class="form-group">
                            <label for="evalTasks">Eval Benchmarks</label>
                            <select id="evalTasks" multiple class="select-tall">{eval_benchmark_options_html}</select>
                            <div class="help">If empty, eval uses train benchmarks.</div>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="testTasks">Test Benchmarks</label>
                        <select id="testTasks" multiple class="select-tall">{test_benchmark_options_html}</select>
                        <div class="help">If empty, test uses train benchmarks.</div>
                    </div>
                </div>

                <div class="section">
                    <h2>Model</h2>
                    <div class="grid">
                        <div class="form-group">
                            <label for="modelClass">Model Class</label>
                            <select id="modelClass" multiple>
                                <option value="span" selected>span</option>
                                <option value="xnet">xnet</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="baseModel">Base Model</label>
                            <select id="baseModel" multiple>{model_options_html}</select>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="form-group" id="spanFuseWrap">
                            <label for="spanFuseType">Span Fuse Type</label>
                            <select id="spanFuseType" multiple>{span_fuse_html}</select>
                        </div>
                        <div class="form-group">
                            <label for="spanPoolType">Span Pool Type</label>
                            <select id="spanPoolType">
                                <option value="mean">mean</option>
                                <option value="last" selected>last</option>
                            </select>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="form-group">
                            <label for="poolType">Pool Type</label>
                            <select id="poolType">
                                <option value="avg">avg</option>
                                <option value="weightedavg">weightedavg</option>
                                <option value="cls">cls</option>
                                <option value="last" selected>last</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="layerFuseType">Layer Fuse Type</label>
                            <select id="layerFuseType">
                                <option value="avg" selected>avg</option>
                                <option value="weighted">weighted</option>
                            </select>
                        </div>
                    </div>
                    <div class="grid" id="rubIndWrap">
                        <div class="form-group">
                            <label class="checkbox-item" style="font-weight:700;font-size:14px;color:#444;">
                                <input type="checkbox" id="rubricIndependentAttn">
                                rubric_independent_attn
                            </label>
                            <div class="help">Block-sparse attention: each rubric only attends to context + itself.</div>
                        </div>
                        <div class="form-group" id="reindexWrap">
                            <label class="checkbox-item" style="font-weight:700;font-size:14px;color:#444;">
                                <input type="checkbox" id="reindexRub" disabled>
                                reindex_rub
                            </label>
                            <div class="help">Reset position IDs per rubric (requires rubric_independent_attn).</div>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="form-group"><label for="numBidirLayers">Num Bidirectional Layers</label><input type="number" id="numBidirLayers" value="0" min="0" step="1"></div>
                        <div class="form-group"><label for="numPruneLayers">Num Pruned Layers</label><input type="number" id="numPruneLayers" value="0" min="0" step="1"></div>
                    </div>
                    <div class="grid">
                        <div class="form-group"><label for="numFuseLayers">Num Fused Layers</label><input type="number" id="numFuseLayers" value="0" min="0" step="1"></div>
                        <div class="form-group"><label for="numUnsinkLayers">Num Unsink Layers</label><input type="number" id="numUnsinkLayers" value="0" min="0" step="1"></div>
                    </div>
                </div>

                <div class="section">
                    <h2>Training</h2>
                    <div class="grid">
                        <div class="form-group"><label for="batchSize">Batch Size</label><input type="number" id="batchSize" value="8" min="1"></div>
                        <div class="form-group"><label for="gradAccum">Gradient Accumulation Steps</label><input type="number" id="gradAccum" value="2" min="1"></div>
                    </div>
                    <div class="grid">
                        <div class="form-group"><label for="trainFrac">Train Fraction</label><input type="number" id="trainFrac" value="1.0" min="0.1" step="0.1"></div>
                        <div class="form-group"><label for="lr">Learning Rate</label><input type="text" id="lr" value="2e-4"></div>
                    </div>
                    <div class="grid">
                        <div class="form-group"><label for="maxEpoch">Max Epoch</label><input type="number" id="maxEpoch" value="4" min="1"></div>
                        <div class="form-group"><label for="seed">Seed</label><input type="number" id="seed" value="114514"></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="testDropRub">Test Drop Rubric Probability</label>
                            <input type="number" id="testDropRub" value="0.0" min="0" max="1" step="0.1">
                        </div>
                        <div class="form-group">
                            <label for="trainDropRub">Train Drop Rubric Probability</label>
                            <input type="number" id="trainDropRub" value="0.0" min="0" max="1" step="0.1">
                        </div>
                    </div>
                    <div class="checkbox-grid">
                        <label class="checkbox-item"><input type="checkbox" id="useLora" checked>use_lora</label>
                        <label class="checkbox-item"><input type="checkbox" id="useBnb" checked>use_bnb</label>
                        <label class="checkbox-item"><input type="checkbox" id="bf16" checked>bf16</label>
                        <label class="checkbox-item"><input type="checkbox" id="logWandb" checked>log_wandb</label>
                        <label class="checkbox-item"><input type="checkbox" id="addSuffix" checked>add_suffix</label>
                        <label class="checkbox-item"><input type="checkbox" id="addContext" checked>add_context</label>
                        <label class="checkbox-item"><input type="checkbox" id="randomSuffix" checked>random_suffix</label>
                        <label class="checkbox-item"><input type="checkbox" id="useTranslated" checked>use_translated_prompts</label>
                        <label class="checkbox-item"><input type="checkbox" id="randomSolution">random_solution</label>
                    </div>
                </div>

                <div class="section">
                    <h2>Experiment Names</h2>
                    <div class="alert">Each experiment has its own name field. Press Tab to fill auto-name. Empty fields use auto-generated names.</div>
                    <div class="help">Auto-generated preview: <code id="autoExpName"></code></div>
                    <div id="experimentNamesContainer"><p>Select model parameters to generate experiment names.</p></div>
                </div>

                <div class="section">
                    <h2>Experiment Summary</h2>
                    <div id="batchSummary" class="summary"><p>Total experiments will be calculated based on selected combinations.</p></div>
                </div>

                <div class="button-row">
                    <button type="button" class="btn-secondary" id="btnReset">Reset</button>
                    <button type="button" class="btn-secondary" id="btnGenerate">Generate Commands</button>
                    <button type="button" class="btn-primary" id="btnStart">Start Experiments</button>
                </div>

                <div class="status" id="statusText"></div>
                <pre class="output" id="commandOutput"></pre>
            </form>
        </div>
    </div>

    <script>
        const MODEL2SHORT = {MODEL_SHORTNAME!r};
        let experimentConfigs = [];

        function setStatus(message, isError = false) {{
            const node = document.getElementById('statusText');
            if (node) {{
                node.textContent = String(message || '');
                node.style.color = isError ? '#b00020' : '#2f4f8f';
            }}
        }}

        function setOutput(message, show = true) {{
            const box = document.getElementById('commandOutput');
            if (!box) return;
            box.style.display = show ? 'block' : 'none';
            box.textContent = message || '';
        }}

        function selectedValues(id) {{
            const el = document.getElementById(id);
            if (!el) return [];
            const arr = Array.from(el.options || []).filter((x) => x.selected).map((x) => x.value);
            return arr;
        }}

        function selectedOrDefault(id) {{
            const vals = selectedValues(id);
            if (vals.length > 0) return vals;
            const el = document.getElementById(id);
            if (el && el.options && el.options.length > 0) {{
                el.options[0].selected = true;
                return [el.options[0].value];
            }}
            return [];
        }}

        function shortModel(model) {{
            return (MODEL2SHORT[model] || model.split('/').pop()).toLowerCase();
        }}

        function taskSig(tasks, maxItems = 3) {{
            if (!tasks || tasks.length === 0) return 'none';
            const visible = tasks.slice(0, maxItems);
            const suffix = tasks.length > maxItems ? '-etc' : '';
            return visible.join('+') + suffix;
        }}

        function computeAutoName(cfg) {{
            const parts = ['multi', shortModel(cfg.base_model)];
            parts.push(cfg.model_class === 'span' ? (cfg.span_fuse_type || 'p-concat') : cfg.model_class);
            if (cfg.random_solution) {{
                parts.push('randsolu');
            }}
            if (cfg.rubric_independent_attn) {{
                parts.push('rub-ind');
                if (cfg.reindex_rub) parts.push('reindex');
            }}
            return parts.join('-');
        }}

        function baseConfig() {{
            const trainTasks = selectedOrDefault('trainTasks');
            let evalTasks = selectedValues('evalTasks');
            let testTasks = selectedValues('testTasks');
            if (trainTasks.length > 0 && evalTasks.length === 0) evalTasks = [...trainTasks];
            if (trainTasks.length > 0 && testTasks.length === 0) testTasks = [...trainTasks];

            return {{
                train_tasks: trainTasks,
                eval_tasks: evalTasks,
                test_tasks: testTasks,
                span_pool_type: document.getElementById('spanPoolType').value,
                pool_type: document.getElementById('poolType').value,
                layer_fuse_type: document.getElementById('layerFuseType').value,
                num_bidir_layers: Number(document.getElementById('numBidirLayers').value || 0),
                num_prune_layers: Number(document.getElementById('numPruneLayers').value || 0),
                num_fuse_layers: Number(document.getElementById('numFuseLayers').value || 0),
                num_unsink_layers: Number(document.getElementById('numUnsinkLayers').value || 0),
                batch_size: Number(document.getElementById('batchSize').value || 8),
                gradient_accumulation_steps: Number(document.getElementById('gradAccum').value || 2),
                train_frac: Number(document.getElementById('trainFrac').value || 1.0),
                lr: Number(document.getElementById('lr').value || 2e-4),
                max_epoch: Number(document.getElementById('maxEpoch').value || 4),
                seed: Number(document.getElementById('seed').value || 114514),
                test_drop_rub: Number(document.getElementById('testDropRub').value || 0.0),
                train_drop_rub: Number(document.getElementById('trainDropRub').value || 0.0),
                use_lora: document.getElementById('useLora').checked,
                use_bnb: document.getElementById('useBnb').checked,
                bf16: document.getElementById('bf16').checked,
                log_wandb: document.getElementById('logWandb').checked,
                add_suffix: document.getElementById('addSuffix').checked,
                add_context: document.getElementById('addContext').checked,
                random_suffix: document.getElementById('randomSuffix').checked,
                use_translated_prompts: document.getElementById('useTranslated').checked,
                random_solution: document.getElementById('randomSolution').checked,
                rubric_independent_attn: document.getElementById('rubricIndependentAttn').checked,
                reindex_rub: document.getElementById('reindexRub').checked,
            }};
        }}

        function updateSpanUI() {{
            const hasSpan = selectedOrDefault('modelClass').includes('span');
            const wrap = document.getElementById('spanFuseWrap');
            const fuse = document.getElementById('spanFuseType');
            const spanPool = document.getElementById('spanPoolType');
            wrap.style.opacity = hasSpan ? '1' : '0.5';
            fuse.disabled = !hasSpan;
            spanPool.disabled = !hasSpan;

            const rubIndChk = document.getElementById('rubricIndependentAttn');
            const reindexChk = document.getElementById('reindexRub');
            const rubIndWrap = document.getElementById('rubIndWrap');
            rubIndWrap.style.opacity = hasSpan ? '1' : '0.5';
            rubIndChk.disabled = !hasSpan;

            const rubIndOn = hasSpan && rubIndChk.checked;
            reindexChk.disabled = !rubIndOn;
            document.getElementById('reindexWrap').style.opacity = rubIndOn ? '1' : '0.5';
            if (!rubIndOn) reindexChk.checked = false;

            if (rubIndOn) {{
                Array.from(fuse.options).forEach((o) => {{ o.selected = o.value === 'l-only'; }});
                fuse.disabled = true;
                wrap.style.opacity = '0.5';
            }}
        }}

        function buildExperimentConfigs() {{
            const base = baseConfig();
            const models = selectedOrDefault('baseModel');
            const classes = selectedOrDefault('modelClass');
            const spanFuses = selectedOrDefault('spanFuseType');

            const prev = new Map(experimentConfigs.map((x) => [x.key, x.customName || '']));
            const next = [];
            let idx = 0;

            models.forEach((model) => {{
                classes.forEach((modelClass) => {{
                    if (modelClass === 'span') {{
                        const fuses = spanFuses.length ? spanFuses : ['p-concat'];
                        fuses.forEach((fuse) => {{
                            const cfg = {{
                                ...base,
                                id: idx++,
                                key: `${{model}}|${{modelClass}}|${{fuse}}`,
                                base_model: model,
                                model_class: modelClass,
                                span_fuse_type: fuse,
                            }};
                            cfg.autoName = computeAutoName(cfg);
                            cfg.customName = prev.get(cfg.key) || '';
                            next.push(cfg);
                        }});
                    }} else {{
                        const cfg = {{
                            ...base,
                            id: idx++,
                            key: `${{model}}|${{modelClass}}|_`,
                            base_model: model,
                            model_class: modelClass,
                        }};
                        cfg.autoName = computeAutoName(cfg);
                        cfg.customName = prev.get(cfg.key) || '';
                        next.push(cfg);
                    }}
                }});
            }});

            experimentConfigs = next;
            renderExperimentNameFields();
            updateAutoExpNamePreview();
            updateBatchSummary();
            setStatus(next.length ? `Prepared ${{next.length}} experiment name(s).` : 'No valid experiment combinations.', next.length === 0);
        }}

        function renderExperimentNameFields() {{
            const container = document.getElementById('experimentNamesContainer');
            if (!experimentConfigs.length) {{
                container.innerHTML = '<p>Select model parameters to generate experiment names.</p>';
                return;
            }}

            let html = '';
            experimentConfigs.forEach((cfg) => {{
                const parts = [taskSig(cfg.train_tasks, 2), cfg.base_model.split('/').pop(), cfg.model_class];
                if (cfg.model_class === 'span') parts.push(cfg.span_fuse_type || 'p-concat');
                html += `
                    <div class="exp-item" id="exp-${{cfg.id}}">
                        <div class="exp-head">
                            <span>${{parts.join(' + ')}}</span>
                            <button type="button" class="btn-delete" onclick="deleteExperiment(${{cfg.id}})">delete</button>
                        </div>
                        <input type="text"
                               id="expName${{cfg.id}}"
                               placeholder="${{cfg.autoName}}"
                               value="${{cfg.customName || ''}}"
                               oninput="updateCustomName(${{cfg.id}}, this.value)"
                               onkeydown="handleTabComplete(event, ${{cfg.id}})">
                    </div>
                `;
            }});
            container.innerHTML = html;
        }}

        function updateCustomName(id, value) {{
            const cfg = experimentConfigs.find((x) => x.id === id);
            if (cfg) cfg.customName = value;
        }}

        function handleTabComplete(event, id) {{
            if (event.key !== 'Tab') return;
            event.preventDefault();
            const cfg = experimentConfigs.find((x) => x.id === id);
            if (!cfg) return;
            const input = document.getElementById(`expName${{id}}`);
            if (!input) return;
            input.value = cfg.autoName;
            cfg.customName = cfg.autoName;
        }}

        function deleteExperiment(id) {{
            experimentConfigs = experimentConfigs.filter((x) => x.id !== id);
            renderExperimentNameFields();
            updateAutoExpNamePreview();
            updateBatchSummary();
        }}

        function updateAutoExpNamePreview() {{
            const code = document.getElementById('autoExpName');
            if (!code) return;
            if (!experimentConfigs.length) {{
                code.textContent = '';
                return;
            }}
            const names = experimentConfigs.map((x) => x.autoName);
            code.textContent = names.length === 1 ? names[0] : `${{names[0]}} (+${{names.length - 1}} more)`;
        }}

        function updateBatchSummary() {{
            const node = document.getElementById('batchSummary');
            if (!node) return;
            if (!experimentConfigs.length) {{
                node.innerHTML = '<p>Total experiments will be calculated based on selected combinations.</p>';
                return;
            }}

            const models = [...new Set(experimentConfigs.map((x) => x.base_model.split('/').pop()))];
            const trainSets = [...new Set(experimentConfigs.map((x) => (x.train_tasks || []).join('+')))];
            const evalSets = [...new Set(experimentConfigs.map((x) => (x.eval_tasks || []).join('+')))];
            const testSets = [...new Set(experimentConfigs.map((x) => (x.test_tasks || []).join('+')))];
            const modelClasses = [...new Set(experimentConfigs.map((x) => x.model_class))];
            const fuseTypes = [...new Set(experimentConfigs.map((x) => x.span_fuse_type).filter(Boolean))];

            let html = `
                <p><strong>Selected Configurations:</strong></p>
                <ul>
                    <li>Models: ${{models.length}} (${{models.join(', ')}})</li>
                    <li>Train Benchmarks: ${{trainSets.length}} (${{trainSets.join(' | ')}})</li>
                    <li>Eval Benchmarks: ${{evalSets.length}} (${{evalSets.join(' | ')}})</li>
                    <li>Test Benchmarks: ${{testSets.length}} (${{testSets.join(' | ')}})</li>
                    <li>Model Classes: ${{modelClasses.length}} (${{modelClasses.join(', ')}})</li>`;
            if (fuseTypes.length) {{
                html += `<li>Fusion Types: ${{fuseTypes.length}} (${{fuseTypes.join(', ')}})</li>`;
            }}
            html += `
                </ul>
                <p><strong>Total Experiments: ${{experimentConfigs.length}}</strong></p>
                <p><strong>Preview:</strong> ${{experimentConfigs.slice(0, 3).map((x) => x.customName || x.autoName).join(', ')}}${{experimentConfigs.length > 3 ? '...' : ''}}</p>
            `;
            node.innerHTML = html;
        }}

        function getBatchConfigs() {{
            const currentBase = baseConfig();
            return experimentConfigs.map((cfg) => {{
                const out = {{ ...cfg, ...currentBase }};
                out.base_model = cfg.base_model;
                out.model_class = cfg.model_class;
                if (cfg.model_class === 'span') {{
                    out.span_fuse_type = cfg.span_fuse_type || 'p-concat';
                }} else {{
                    delete out.span_fuse_type;
                }}
                out.exp_name = (cfg.customName && cfg.customName.trim()) ? cfg.customName.trim() : cfg.autoName;
                delete out.id;
                delete out.key;
                delete out.autoName;
                delete out.customName;
                return out;
            }});
        }}

        function validateConfigs(configs) {{
            if (!configs.length) {{
                setStatus('Please select at least one base model and one model class.', true);
                alert('Please select at least one base model and one model class.');
                return false;
            }}
            if (!configs[0].train_tasks || !configs[0].train_tasks.length) {{
                setStatus('Please select at least one train benchmark.', true);
                alert('Please select at least one train benchmark.');
                return false;
            }}
            return true;
        }}

        async function generateCommands() {{
            const configs = getBatchConfigs();
            if (!validateConfigs(configs)) return;
            setStatus(`Generating ${{configs.length}} command(s)...`);
            try {{
                const generated = [];
                for (let i = 0; i < configs.length; i++) {{
                    const res = await fetch('/api/generate-command', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(configs[i]),
                    }});
                    const data = await res.json();
                    if (!data.success) {{
                        setStatus(`Failed on config ${{i + 1}}: ${{data.error}}`, true);
                        alert(`Failed on config ${{i + 1}}: ${{data.error}}`);
                        return;
                    }}
                    generated.push(`# Experiment ${{i + 1}}: ${{data.exp_name}}\\n${{data.command}}`);
                }}
                setOutput(generated.join('\\n\\n'), true);
                setStatus(`Prepared ${{generated.length}} command(s).`);
            }} catch (err) {{
                setStatus(`Failed to generate commands: ${{err}}`, true);
                alert(`Failed to generate commands: ${{err}}`);
            }}
        }}

        async function startBatchExperiments() {{
            const configs = getBatchConfigs();
            if (!validateConfigs(configs)) return;
            setStatus(`Submitting ${{configs.length}} experiment(s)...`);
            try {{
                const res = await fetch('/api/start-batch', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ configs }}),
                }});
                const data = await res.json();
                if (!data.success) {{
                    setStatus('Failed to create run script: ' + data.error, true);
                    alert('Failed to create run script: ' + data.error);
                    return;
                }}
                setStatus(`run script: ${{data.run_script}} (${{configs.length}} experiments)`);
                alert(`Batch script generated!\\nTotal experiments: ${{configs.length}}\\nScript: ${{data.run_script}}`);
                if (data.commands_preview) {{
                    setOutput(data.commands_preview, true);
                }}
            }} catch (err) {{
                setStatus(`Create run script failed: ${{err}}`, true);
                alert(`Create run script failed: ${{err}}`);
            }}
        }}

        function resetForm() {{
            const form = document.getElementById('multiForm');
            form.reset();
            experimentConfigs = [];
            setOutput('', false);
            selectedOrDefault('trainTasks');
            selectedOrDefault('baseModel');
            selectedOrDefault('modelClass');
            selectedOrDefault('spanFuseType');
            updateSpanUI();
            buildExperimentConfigs();
            setStatus('Form reset.');
        }}

        function bindEvents() {{
            document.getElementById('btnReset').addEventListener('click', resetForm);
            document.getElementById('btnGenerate').addEventListener('click', generateCommands);
            document.getElementById('btnStart').addEventListener('click', startBatchExperiments);
            document.querySelectorAll('#multiForm select, #multiForm input').forEach((el) => {{
                el.addEventListener('change', () => {{
                    updateSpanUI();
                    buildExperimentConfigs();
                }});
                if (el.type === 'number' || el.type === 'text') {{
                    el.addEventListener('input', () => {{
                        updateSpanUI();
                        buildExperimentConfigs();
                    }});
                }}
            }});
        }}

        window.updateCustomName = updateCustomName;
        window.handleTabComplete = handleTabComplete;
        window.deleteExperiment = deleteExperiment;
        window.startBatchExperiments = startBatchExperiments;
        window.generateCommands = generateCommands;
        window.resetForm = resetForm;

        window.addEventListener('error', (ev) => {{
            setStatus(`JS error: ${{ev.message}} @ ${{ev.filename || 'inline'}}:${{ev.lineno || 0}}`, true);
        }});

        document.addEventListener('DOMContentLoaded', () => {{
            selectedOrDefault('trainTasks');
            selectedOrDefault('baseModel');
            selectedOrDefault('modelClass');
            selectedOrDefault('spanFuseType');
            bindEvents();
            updateSpanUI();
            buildExperimentConfigs();
            setStatus('Ready.');
        }});
    </script>
</body>
</html>
"""


app = Flask(__name__)


@app.after_request
def add_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


HTML_TEMPLATE = _build_template()


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/generate-command", methods=["POST", "OPTIONS"])
def generate_command():
    if request.method == "OPTIONS":
        return "", 204
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "expected JSON body"})
        config = _to_config(request.get_json(silent=True) or {})
        exp_name = config.generate_exp_name()
        exp_root = WORKSPACE_ROOT / "results_multi" / exp_name
        command = "\n".join(_build_command_lines(config, str(exp_root)))
        return jsonify(
            {
                "success": True,
                "exp_name": exp_name,
                "save_dir": str(exp_root),
                "command": command,
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/start-batch", methods=["POST", "OPTIONS"])
def start_batch():
    if request.method == "OPTIONS":
        return "", 204
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "expected JSON body"})
        payload = request.get_json(silent=True) or {}
        raw_configs = payload.get("configs")
        if not isinstance(raw_configs, list) or not raw_configs:
            return jsonify({"success": False, "error": "configs must be a non-empty list"})
        wandb_api_key, hf_token = _extract_api_tokens(payload)

        parsed_configs: list[MultiExperimentConfig] = []
        parse_errors = []
        for idx, raw in enumerate(raw_configs, 1):
            try:
                parsed_configs.append(_to_config(raw))
            except Exception as exc:
                parse_errors.append(f"config #{idx}: {exc}")

        if not parsed_configs:
            return jsonify({"success": False, "error": "; ".join(parse_errors) or "no valid configs"})

        run_script_path, results, command_blocks = _build_run_script(
            parsed_configs,
            wandb_api_key,
            hf_token,
        )
        return jsonify(
            {
                "success": True,
                "run_script": str(run_script_path),
                "total_experiments": len(parsed_configs),
                "results": results,
                "errors": parse_errors,
                "commands_preview": "\n".join(command_blocks[:3]),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    print(f"Starting config_web_multitask.py at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
