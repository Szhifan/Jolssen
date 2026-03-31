import random
import torch
from typing import Callable
from functools import partial
from datasets import Dataset
from transformers import AutoTokenizer

from data_processing_other.benchmark_meta import BENCHMARK_DESCRIPTIONS
from data_processing_other import data_loader as other_loader


# LLM model identifiers
LLM_IDENTIFIERS = ("llama", "mistral", "gpt", "qwen", "phi")


def is_llm_model(model_name: str) -> bool:
    """Check if model is an LLM based on model name."""
    return any(identifier in model_name.lower() for identifier in LLM_IDENTIFIERS)


def get_tokenizer(base_model: str) -> AutoTokenizer:
    """Get tokenizer for the base model with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if "llama" in base_model.lower() or "mistral" in base_model.lower():
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.sep_token or tokenizer.eos_token
    return tokenizer


class OtherDataLoader:
    def __init__(self, benchmark: str, train_frac: float = 1.0):
        assert train_frac > 0, "train_frac must be > 0"
        assert train_frac <= 1 or float(train_frac).is_integer(), (
            "train_frac > 1 must be an integer-valued exact number of training instances"
        )
        self.benchmark = benchmark
        self.train_frac = train_frac
        self._load_datasets()

    def _load_datasets(self):
        meta = BENCHMARK_DESCRIPTIONS[self.benchmark]
        format_fn_name = meta.get("format_fn")
        if not format_fn_name:
            raise ValueError(f"No format_fn defined for benchmark {self.benchmark}")
        if not hasattr(other_loader, format_fn_name):
            raise ValueError(f"format_fn {format_fn_name} not found in data_processing_other.data_loader")
        format_fn = getattr(other_loader, format_fn_name)
        ds = format_fn()

        if "train" not in ds or "val" not in ds or "test" not in ds:
            raise ValueError(f"format_fn {format_fn_name} must return train/val/test splits")

        train_ds = ds["train"]
        if self.train_frac < 1:
            n_keep = max(1, int(len(train_ds) * self.train_frac))
            sampled_indices = random.sample(range(len(train_ds)), n_keep)
            train_ds = train_ds.select(sampled_indices)
        elif self.train_frac > 1:
            n_keep = int(self.train_frac)
            if n_keep > len(train_ds):
                raise ValueError(
                    f"train_frac={self.train_frac} requests {n_keep} training instances, "
                    f"but only {len(train_ds)} are available for benchmark '{self.benchmark}'."
                )
            sampled_indices = random.sample(range(len(train_ds)), n_keep)
            train_ds = train_ds.select(sampled_indices)

        self.train = train_ds
        self.val = ds["val"]
        self.test = ds["test"]


class DataPipelineOther:
    """
    Data pipeline for non-ASAG benchmarks.
    Provides encoding and collate functions for standard classification tasks.
    """

    def __init__(
        self,
        base_model: str,
        benchmark: str,
        train_frac: float = 1.0,
        add_context: bool = True,
        add_suffix: bool = False,
        random_suffix: bool = False,
        use_translated_prompts: bool = False,
        model_class: str = "span",
        add_options: bool = True,
    ):
        self.base_model = base_model
        self.benchmark = benchmark
        self.train_frac = train_frac
        self.model_class = model_class
        self.add_context = add_context
        self.add_suffix = add_suffix
        self.random_suffix = random_suffix
        self.use_translated_prompts = use_translated_prompts
        self.add_options = add_options
        if not self.add_options:
            raise ValueError("add_options must be True for span-based encoding in other datasets.")

        try:
            print(f"Loading benchmark metadata for {benchmark}...")
            self.benchmark_meta = BENCHMARK_DESCRIPTIONS[benchmark]
        except KeyError:
            raise ValueError(f"Benchmark {benchmark} not found in BENCHMARK_DESCRIPTIONS")

        self.num_labels = self.benchmark_meta.get("num_labels")
        self.is_llm = is_llm_model(base_model)
        self.tokenizer = get_tokenizer(base_model)
        self.pad_token_id = self.tokenizer.pad_token_id

        self._init_data_loader()

    def _init_data_loader(self):
        self.data_loader = OtherDataLoader(self.benchmark, train_frac=self.train_frac)

    def get_datasets(self, test_only: bool = True):
        enc_fn = self.get_encode_fn()
        if test_only:
            train_ds, val_ds = self.data_loader.train, self.data_loader.val
            test_ds = self.data_loader.test.map(lambda x: enc_fn(x))
        else:
            train_ds = self.data_loader.train.map(lambda x: enc_fn(x))
            val_ds = self.data_loader.val.map(lambda x: enc_fn(x))
            test_ds = self.data_loader.test.map(lambda x: enc_fn(x))
                    
            input_id = train_ds[0]["input_ids"]
            rubric_span = train_ds[0].get("rubric_spans", None)
            answer_span = train_ds[0].get("answer_span", None)
            sub_tokens = self.tokenizer.convert_ids_to_tokens(input_id)
            print(f"Sample decoded text: {self.tokenizer.decode(input_id)}")
            print("answer_span:", sub_tokens[answer_span[0]:answer_span[1]] if answer_span else None)
            for i, span in enumerate(rubric_span if rubric_span else []):
                print(f"rubric_span {i}:", sub_tokens[span[0]:span[1]])
        return train_ds, val_ds, test_ds

    def get_encode_fn(self) -> Callable:
        if self.is_llm:
            return partial(self.encode_spans_llm)
        return partial(self.encode_spans_bert)

    def get_collate_fn(self) -> Callable:
        return partial(self.span_collate_fn)

    def _get_label_semantics(self, example):
        label_semantics_key = self.benchmark_meta.get("label_semantics")
        lang = example.get("language")
        if self.use_translated_prompts:
            by_lang = self.benchmark_meta.get("label_semantics_by_lang", {})
            if lang in by_lang:
                return by_lang[lang]
        if label_semantics_key and label_semantics_key in example:
            return example[label_semantics_key]
        label_tags = self.benchmark_meta.get("label_tags", [])
        if isinstance(label_tags, dict):
            if lang in label_tags:
                return label_tags[lang]
            return label_tags.get("en", [])
        return label_tags

    def _get_suffix(self, example):
        suffixes = self.benchmark_meta.get("suffixes", [])
        if self.use_translated_prompts:
            by_lang = self.benchmark_meta.get("suffixes_by_lang", {})
            lang = example.get("language")
            if lang in by_lang:
                suffixes = by_lang[lang]
        if not suffixes:
            return None
        if self.random_suffix:
            import random
            return random.choice(suffixes)
        return suffixes[0]

    def encode_spans_bert(self, example):
        context_cols = self.benchmark_meta.get("context_cols", [])
        text_col = self.benchmark_meta["text_col"]
        label_col = self.benchmark_meta["label_col"]

        # Build input with span tracking
        input_parts = []
        span_indices_char = []
        answer_start_char = answer_end_char = None

        if self.add_context:
            for col in context_cols:
                if col in example:
                    input_parts.append(f"{col.replace('_', ' ')}: {str(example[col])}")

        current_str = self.tokenizer.sep_token.join(input_parts)
        answer_start_char = len(current_str) + (len(self.tokenizer.sep_token) if input_parts else 0)
        input_parts.append(f"{text_col.replace('_', ' ')}: {str(example.get(text_col, ''))}")
        new_str = self.tokenizer.sep_token.join(input_parts)
        answer_end_char = len(new_str)

        label_semantics = self._get_label_semantics(example)
        if isinstance(label_semantics, list) and label_semantics:
            for option in label_semantics:
                current_str = self.tokenizer.sep_token.join(input_parts)
                start_char = len(current_str) + (len(self.tokenizer.sep_token) if input_parts else 0)
                input_parts.append(f"{self.benchmark_meta.get('label_semantics', 'options')}: {str(option)}")
                new_str = self.tokenizer.sep_token.join(input_parts)
                end_char = len(new_str)
                span_indices_char.append((start_char, end_char))

        if self.add_suffix:
            suffix = self._get_suffix(example)
            if suffix:
                input_parts.append(suffix)

        input_str = self.tokenizer.sep_token.join(input_parts)
        encoding = self.tokenizer(
            input_str,
            return_offsets_mapping=True,
            max_length=2048,
            truncation=True,
            add_special_tokens=True,
        )

        offsets = encoding["offset_mapping"]
        rubric_spans = []
        for span_start_char, span_end_char in span_indices_char:
            token_start = token_end = None
            for i, (start, end) in enumerate(offsets):
                if start <= span_start_char < end:
                    token_start = i
                if start < span_end_char <= end:
                    token_end = i + 1
                    break
            if token_start is None:
                token_start = next((i for i, (s, _) in enumerate(offsets) if s >= span_start_char), 0)
            if token_end is None:
                token_end = next((i for i, (_, e) in enumerate(offsets) if e >= span_end_char), len(offsets) - 1) + 1
            rubric_spans.append((token_start, token_end))

        answer_token_start = answer_token_end = None
        for i, (start, end) in enumerate(offsets):
            if start <= answer_start_char < end:
                answer_token_start = i
            if start < answer_end_char <= end:
                answer_token_end = i + 1
                break
        if answer_token_start is None:
            answer_token_start = next((i for i, (s, _) in enumerate(offsets) if s >= answer_start_char), 0)
        if answer_token_end is None:
            answer_token_end = next((i for i, (_, e) in enumerate(offsets) if e >= answer_end_char), len(offsets) - 1) + 1
        encoding["answer_span"] = [answer_token_start, answer_token_end]

        encoding["rubric_spans"] = rubric_spans
        encoding["labels"] = int(example[label_col])
        encoding.pop("offset_mapping", None)
        return encoding

    def encode_spans_llm(self, example):
        context_cols = self.benchmark_meta.get("context_cols", [])
        text_col = self.benchmark_meta["text_col"]
        label_col = self.benchmark_meta["label_col"]

        input_parts = []
        span_indices_char = []
        answer_start_char = answer_end_char = None

        if self.add_context:
            for col in context_cols:
                if col in example:
                    tag = col.replace("_", " ")
                    input_parts.append(f"<{tag}> {str(example[col])} </{tag}>")

        current_str = "\n".join(input_parts)
        answer_display = text_col.replace("_", " ")
        answer_content = str(example.get(text_col, ""))
        answer_tagged = f"<{answer_display}> {answer_content} </{answer_display}>"
        base_len = len(current_str) + (1 if input_parts else 0)
        opening_tag_len = len(f"<{answer_display}>")
        answer_start_char = base_len + opening_tag_len
        answer_end_char = answer_start_char + len(answer_content)
        input_parts.append(answer_tagged)

        label_semantics = self._get_label_semantics(example)
        if isinstance(label_semantics, list) and label_semantics:
            label_tag = self.benchmark_meta.get("label_semantics", "options")
            for option in label_semantics:
                current_str = "\n".join(input_parts)
                option_content = str(option)
                option_tagged = f"<{label_tag}> {option_content} </{label_tag}>"
                base_len = len(current_str) + (1 if input_parts else 0)
                opening_tag_len = len(f"<{label_tag}>")
                start_char = base_len + opening_tag_len
                end_char = start_char + len(option_content)
                input_parts.append(option_tagged)
                span_indices_char.append((start_char, end_char))

        if self.add_suffix:
            suffix = self._get_suffix(example)
            if suffix:
                input_parts.append(suffix)

        input_str = "\n".join(input_parts)
        encoding = self.tokenizer(
            input_str,
            return_offsets_mapping=True,
            max_length=2048,
            truncation=True,
            add_special_tokens=True,
        )

        offsets = encoding["offset_mapping"]
        rubric_spans = []
        for span_start_char, span_end_char in span_indices_char:
            token_start = token_end = None
            for i, (start, end) in enumerate(offsets):
                if start <= span_start_char < end:
                    token_start = i
                if start < span_end_char <= end:
                    token_end = i + 1
                    break
            if token_start is None:
                token_start = next((i for i, (s, _) in enumerate(offsets) if s >= span_start_char), 0)
            if token_end is None:
                token_end = next((i for i, (_, e) in enumerate(offsets) if e >= span_end_char), len(offsets) - 1) + 1
            rubric_spans.append((token_start, token_end))

        answer_token_start = answer_token_end = None
        for i, (start, end) in enumerate(offsets):
            if start <= answer_start_char < end:
                answer_token_start = i
            if start < answer_end_char <= end:
                answer_token_end = i + 1
                break
        if answer_token_start is None:
            answer_token_start = next((i for i, (s, _) in enumerate(offsets) if s >= answer_start_char), 0)
        if answer_token_end is None:
            answer_token_end = next((i for i, (_, e) in enumerate(offsets) if e >= answer_end_char), len(offsets) - 1) + 1
        encoding["answer_span"] = [answer_token_start, answer_token_end]

        encoding["rubric_spans"] = rubric_spans
        encoding["labels"] = int(example[label_col])
        encoding.pop("offset_mapping", None)
        return encoding

    def span_collate_fn(self, input_batch, pad_id=None, return_meta=False):
        if pad_id is None:
            pad_id = self.pad_token_id

        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in input_batch]
        attention_masks = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in input_batch]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        max_rubrics = max(len(x["rubric_spans"]) for x in input_batch)
        batch_size = len(input_batch)
        rubric_spans_tensor = torch.zeros(batch_size, max_rubrics, 2, dtype=torch.long)
        rubric_mask_tensor = torch.zeros(batch_size, max_rubrics, dtype=torch.bool)
        for i, example in enumerate(input_batch):
            spans = torch.tensor(example["rubric_spans"], dtype=torch.long)
            num_spans = spans.shape[0]
            rubric_spans_tensor[i, :num_spans, :] = spans
            rubric_mask_tensor[i, :num_spans] = True
        answer_spans = torch.tensor([x["answer_span"] for x in input_batch], dtype=torch.long)

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "rubric_spans": rubric_spans_tensor,
            "answer_span": answer_spans,
            "rubric_mask": rubric_mask_tensor,
            "labels": torch.tensor([x["labels"] for x in input_batch], dtype=torch.long),
        }

        meta = {}
        if "id" in input_batch[0]:
            meta["id"] = [x.get("id", None) for x in input_batch]
        if "language" in input_batch[0]:
            meta["language"] = [x.get("language", None) for x in input_batch]

        if return_meta:
            return batch, meta
        return batch
