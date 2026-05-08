from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_BENCHMARKS = {"alice_ke", "alice_sk"}
SPLITS = ("train.json", "test_ua.json", "test_uq.json")
ROW_ID_PATTERN = re.compile(r"^(?P<base>.+)_(?P<tag>k|s|ke|sk)(?P<idx>\d+)$")
TARGET_LABEL_COLUMNS = ("pred_id", "labels", "level")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _load_alice_resources() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    alice_data_dir = _repo_root() / "asag_benchmarks" / "alice_data"
    question_meta_path = alice_data_dir / "question_meta.json"

    if not alice_data_dir.exists():
        raise FileNotFoundError(f"Missing ALICE data directory: {alice_data_dir}")
    if not question_meta_path.exists():
        raise FileNotFoundError(f"Missing ALICE question meta file: {question_meta_path}")

    with question_meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    entry_index: dict[str, dict[str, Any]] = {}
    for split_name in SPLITS:
        split_path = alice_data_dir / split_name
        with split_path.open("r", encoding="utf-8") as f:
            split_data = json.load(f)

        for entry in split_data:
            entry_index[entry["id"]] = {
                "question_id": entry["question_id"],
                "ke_elements": list((entry.get("knowledge_elements") or {}).keys()),
                "sk_elements": list((entry.get("skills") or {}).keys()),
            }

    return meta, entry_index


def _parse_level_key(value: Any) -> int | None:
    try:
        fvalue = float(str(value))
    except (TypeError, ValueError):
        return None

    ivalue = int(round(fvalue))
    if abs(fvalue - ivalue) > 1e-9:
        return None
    return ivalue


def _parse_int_like(value: Any) -> int | None:
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

    try:
        fvalue = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not fvalue.is_integer():
        return None
    return int(fvalue)


def _maybe_cast_int_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return

    numeric = pd.to_numeric(df[col], errors="coerce")
    if numeric.isna().any():
        return
    if not ((numeric % 1) == 0).all():
        return

    df[col] = numeric.astype(int)


def _get_level_mapping_for_row(
    row_id: str,
    meta: dict[str, dict[str, Any]],
    entry_index: dict[str, dict[str, Any]],
    cache: dict[str, list[int] | None],
) -> list[int] | None:
    if row_id in cache:
        return cache[row_id]

    match = ROW_ID_PATTERN.match(str(row_id))
    if not match:
        cache[row_id] = None
        return None

    base_id = match.group("base")
    tag = match.group("tag")
    elem_idx = int(match.group("idx"))

    entry_info = entry_index.get(base_id)
    if entry_info is None:
        cache[row_id] = None
        return None

    qmeta = meta.get(entry_info["question_id"])
    if qmeta is None:
        cache[row_id] = None
        return None

    if tag in {"k", "ke"}:
        elements = entry_info["ke_elements"]
        rubric_section = qmeta.get("knowledge_elements") or {}
    else:
        elements = entry_info["sk_elements"]
        rubric_section = qmeta.get("skills") or {}

    if elem_idx >= len(elements):
        cache[row_id] = None
        return None

    element_name = elements[elem_idx]
    rubric = rubric_section.get(element_name)
    if not isinstance(rubric, dict) or not rubric:
        cache[row_id] = None
        return None

    level_values = sorted(
        {
            parsed
            for parsed in (_parse_level_key(key) for key in rubric.keys())
            if parsed is not None
        }
    )
    if not level_values:
        cache[row_id] = None
        return None

    cache[row_id] = level_values
    return level_values


def remap_predictions_to_original_alice_labels(
    pred_df: pd.DataFrame,
    benchmark: str | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Remap normalized ALICE KE/SK label ids back to original rubric ids.

    Returns a potentially updated DataFrame and a summary dictionary.
    """
    summary = {
        "processed_rows": 0,
        "changed_rows": 0,
        "changed_cells": 0,
    }

    if benchmark not in TARGET_BENCHMARKS:
        return pred_df, summary

    if "id" not in pred_df.columns:
        return pred_df, summary

    label_cols = [col for col in TARGET_LABEL_COLUMNS if col in pred_df.columns]
    if not label_cols:
        return pred_df, summary

    meta, entry_index = _load_alice_resources()

    corrected_df = pred_df.copy()
    row_level_cache: dict[str, list[int] | None] = {}
    changed_rows = set()

    for row_idx, row in corrected_df.iterrows():
        level_values = _get_level_mapping_for_row(
            row["id"],
            meta,
            entry_index,
            row_level_cache,
        )
        if level_values is None:
            continue

        summary["processed_rows"] += 1

        for col in label_cols:
            current_int = _parse_int_like(row[col])
            if current_int is None:
                continue
            if current_int < 0 or current_int >= len(level_values):
                # Already in original id space (or invalid). Keep as-is.
                continue

            remapped = level_values[current_int]
            if remapped != current_int:
                corrected_df.at[row_idx, col] = remapped
                changed_rows.add(row_idx)
                summary["changed_cells"] += 1

    for col in label_cols:
        _maybe_cast_int_column(corrected_df, col)

    summary["changed_rows"] = len(changed_rows)
    return corrected_df, summary
