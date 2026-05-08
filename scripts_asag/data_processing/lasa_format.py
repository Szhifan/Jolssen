from __future__ import annotations

import random
from typing import Any


def _choice(items: list[Any], rng: random.Random | None = None) -> Any:
    if rng is not None:
        return rng.choice(items)
    return random.choice(items)


def build_lasa_llm_input(
    example: dict[str, Any],
    benchmark_meta: dict[str, Any],
    *,
    add_context: bool = True,
    add_suffix: bool = False,
    random_suffix: bool = False,
    random_solution: bool = False,
    use_translated_prompts: bool = False,
    drop_all_rubrics: bool = False,
    rng: random.Random | None = None,
) -> tuple[str, tuple[int, int], list[tuple[int, int]]]:
    """Build the standard LASA LLM input string and its character spans."""
    context_cols = benchmark_meta["context_cols"]
    text_col = benchmark_meta["text_col"]
    label_semantics = benchmark_meta["label_semantics"]
    suffixes = benchmark_meta.get("suffixes", [])

    context_translate = benchmark_meta.get("context_translate", {}) if use_translated_prompts else {}
    text_col_translate = benchmark_meta.get("text_col_translate", {}) if use_translated_prompts else {}
    suffix_translate = benchmark_meta.get("suffix_translate", []) if use_translated_prompts else []

    rubrics = example[label_semantics] if isinstance(example[label_semantics], list) else [example[label_semantics]]
    answer = example[text_col]

    input_parts: list[str] = []
    rubric_spans_char: list[tuple[int, int]] = []

    if add_context:
        for col in context_cols:
            if col not in example:
                continue
            col_display = context_translate.get(col, col.replace("_", " "))
            if col == "sample_solution" and isinstance(example[col], list):
                if len(example[col]) > 0:
                    solution = _choice(example[col], rng) if random_solution else example[col][0]
                    input_parts.append(f"<{col_display}> {str(solution)} </{col_display}>")
            else:
                input_parts.append(f"<{col_display}> {str(example[col])} </{col_display}>")

    current_str = "\n".join(input_parts)
    answer_display = text_col_translate.get(text_col, text_col.replace("_", " "))
    answer_tagged = f"<{answer_display}> {str(answer)} </{answer_display}>"
    answer_start_char = len(current_str) + (1 if input_parts else 0)
    answer_end_char = answer_start_char + len(answer_tagged)
    input_parts.append(answer_tagged)

    if not drop_all_rubrics:
        for rubric in rubrics:
            current_str = "\n".join(input_parts)
            rubric_tagged = f"<{label_semantics}> {str(rubric)} </{label_semantics}>"
            start_char = len(current_str) + (1 if input_parts else 0)
            end_char = start_char + len(rubric_tagged)
            input_parts.append(rubric_tagged)
            rubric_spans_char.append((start_char, end_char))

    if add_suffix and suffixes:
        if use_translated_prompts and suffix_translate:
            suffix = _choice(suffix_translate, rng) if random_suffix else suffix_translate[0]
        else:
            suffix = _choice(suffixes, rng) if random_suffix else suffixes[0]
        input_parts.append(suffix)

    return "\n".join(input_parts), (answer_start_char, answer_end_char), rubric_spans_char
