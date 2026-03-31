import torch
from typing import Literal, Callable, Optional
from functools import partial
from datasets import Dataset, concatenate_datasets
from data_processing_asag.benchmark_meta import BENCHMARK_DESCRIPTIONS
from transformers import AutoTokenizer


# LLM model identifiers
LLM_IDENTIFIERS = ("llama", "mistral", "gpt", "qwen", "phi")
XNET_MODEL_CLASSES = {"xnet"}


def dedupe_keep_order(items):
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


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


class DataPipeline:
    """
    Configuration class for selecting encoding functions and collate functions
    based on model class and model type (LLM vs MLM).
    """

    def __init__(
        self,
        base_model: str,
        benchmark: str = "alice_lp",
        train_frac: float = 1.0,
        add_context: bool = True,
        add_suffix: bool = False,
        random_suffix: bool = False,
        random_solution: bool = False,
        use_translated_prompts: bool = False,
        model_class: str = "span",
        random_drop_rub: float = 0.0,
    ):
        self.base_model = base_model
        self.benchmark = benchmark
        self.train_frac = train_frac
        self.model_class = model_class
        try:
            print(f"Loading benchmark metadata for {benchmark}...")
            print(benchmark)
            self.benchmark_meta = BENCHMARK_DESCRIPTIONS[benchmark]
        except KeyError:
            raise ValueError(f"Benchmark {benchmark} not found in BENCHMARK_DESCRIPTIONS")
        self.num_labels = self.benchmark_meta.get("num_labels", 3)
        self.add_context = add_context
        self.add_suffix = add_suffix
        self.random_suffix = random_suffix
        self.random_solution = random_solution
        self.use_translated_prompts = use_translated_prompts
        self.random_drop_rub = random_drop_rub

        self.is_llm = is_llm_model(base_model)
        self.tokenizer = get_tokenizer(base_model)
        self.pad_token_id = self.tokenizer.pad_token_id

        self._init_data_loader()

    def _init_data_loader(self):
        """Initialize the appropriate data loader based on benchmark."""
        if "alice" in self.benchmark:
            from data_processing_asag.alice_asag_loader import Alice_Loader
            task_type = self.benchmark.split("_")[-1] if "_" in self.benchmark else "lp"
            self.data_loader = Alice_Loader(train_frac=self.train_frac, task_type=task_type)
        else:
            from data_processing_asag.general_asag_loader import ASAG_Data_Loader
            self.data_loader = ASAG_Data_Loader(
                benchmark=self.benchmark,
                train_frac=self.train_frac,
            )

    def get_datasets(self, test_only=True):
        """
        Get train, validation, and test datasets.
        """
        if test_only:
            enc_fn = self.get_encode_fn()
            if isinstance(self.data_loader.test, dict):
                test_ds = {}
                for key, dataset in self.data_loader.test.items():
                    if self.needs_grouping():
                        dataset = self.expand_rubrics(dataset)
                    encoded_ds = dataset.map(lambda x: enc_fn(x))
                    if self.needs_grouping():
                        encoded_ds = self.group_by_id(encoded_ds)
                    test_ds[key] = encoded_ds
            else:
                if self.needs_grouping():
                    test_ds_raw = self.expand_rubrics(self.data_loader.test)
                else:
                    test_ds_raw = self.data_loader.test
                test_ds = test_ds_raw.map(lambda x: enc_fn(x))
                if self.needs_grouping():
                    test_ds = self.group_by_id(test_ds)
            train_ds, val_ds = self.data_loader.train, self.data_loader.val
        else:
            enc_fn = self.get_encode_fn()
            if self.needs_grouping():
                train_raw = self.expand_rubrics(self.data_loader.train)
                val_raw = self.expand_rubrics(self.data_loader.val)
            else:
                train_raw = self.data_loader.train
                val_raw = self.data_loader.val
            train_ds = train_raw.map(lambda x: enc_fn(x))
            val_ds = val_raw.map(lambda x: enc_fn(x))

            if self.needs_grouping():
                train_ds = self.group_by_id(train_ds)
                val_ds = self.group_by_id(val_ds)

            if not self.needs_grouping():
                input_id = train_ds[0]["input_ids"]
                rubric_span = train_ds[0].get("rubric_spans", None)
                answer_span = train_ds[0].get("answer_span", None)
                sub_tokens = self.tokenizer.convert_ids_to_tokens(input_id)
                print(f"Sample decoded text: {self.tokenizer.decode(input_id)}")
                print("answer_span:", sub_tokens[answer_span[0]:answer_span[1]] if answer_span else None)
                for i, span in enumerate(rubric_span if rubric_span else []):
                    print(f"rubric_span {i}:", sub_tokens[span[0]:span[1]])

            if isinstance(self.data_loader.test, dict):
                test_ds = {}
                for key, dataset in self.data_loader.test.items():
                    if self.needs_grouping():
                        dataset = self.expand_rubrics(dataset)
                    encoded_ds = dataset.map(lambda x: enc_fn(x))
                    if self.needs_grouping():
                        encoded_ds = self.group_by_id(encoded_ds)
                    test_ds[key] = encoded_ds
            else:
                if self.needs_grouping():
                    test_ds_raw = self.expand_rubrics(self.data_loader.test)
                else:
                    test_ds_raw = self.data_loader.test
                test_ds = test_ds_raw.map(lambda x: enc_fn(x))
                if self.needs_grouping():
                    test_ds = self.group_by_id(test_ds)

        return train_ds, val_ds, test_ds

    def get_encode_fn(self) -> Callable:
        if self.model_class in XNET_MODEL_CLASSES:
            if self.is_llm:
                return partial(self.encode_xnet_llm)
            return partial(self.encode_xnet_bert)
        if self.is_llm:
            return partial(self.encode_spans_llm)
        return partial(self.encode_spans_bert)

    def get_collate_fn(self) -> Callable:
        if self.model_class in XNET_MODEL_CLASSES:
            return partial(self.xnet_collate_fn)
        return partial(self.span_collate_fn)

    def needs_grouping(self) -> bool:
        return self.model_class in XNET_MODEL_CLASSES

    def expand_rubrics(self, dataset):
        """
        Explode each row (which has a list of rubrics) into one row per rubric.
        Sets 'rubric' to a single rubric string and 'rubric_level' to its index.
        The answer-level label (= index of the correct rubric) is preserved unchanged.
        """
        from datasets import Dataset
        expanded = []
        label_semantics = self.benchmark_meta["label_semantics"]
        for example in dataset:
            rubric_list = example[label_semantics]
            if not isinstance(rubric_list, list):
                rubric_list = [rubric_list]
            for idx, rubric_text in enumerate(rubric_list):
                row = dict(example)
                row[label_semantics] = rubric_text
                row["rubric_level"] = idx
                expanded.append(row)
        return Dataset.from_list(expanded)

    def group_by_id(self, dataset):
        from collections import defaultdict

        grouped = defaultdict(list)
        for example in dataset:
            example_id = example.get("id", None)
            if example_id is None:
                example_id = example.get("question_id", "unknown")
            grouped[example_id].append(example)

        grouped_examples = []
        for _, examples in grouped.items():
            grouped_example = {
                "input_ids": [ex["input_ids"] for ex in examples],
                "attention_mask": [ex["attention_mask"] for ex in examples],
                "labels": examples[0]["labels"],
                "num_rubrics": len(examples),
                "id": examples[0].get("id", None),
                "level": examples[0].get("level", None),
                "question_id": examples[0].get("question_id", None),
                "rubric_level": [ex.get("rubric_level", None) for ex in examples],
            }
            if "token_type_ids" in examples[0] and examples[0]["token_type_ids"] is not None:
                grouped_example["token_type_ids"] = [ex["token_type_ids"] for ex in examples]
            grouped_examples.append(grouped_example)

        return Dataset.from_list(grouped_examples)

    def encode_spans_bert(self, example):
        """
        Encode example with span information for BERT-like models.
        Tracks spans for rubrics and answer in the tokenized sequence.
        Output example:
        Instruction sep_token Context sep_token text sep_token label_semantic_1 sep_token label_semantic_2 ... sep_token suffix
        """
        import random
        
        # Get benchmark metadata
        context_cols = self.benchmark_meta["context_cols"]
        text_col = self.benchmark_meta["text_col"]  # "answer"
        label_semantics = self.benchmark_meta["label_semantics"]  # "rubric"
        suffixes = self.benchmark_meta.get("suffixes", [])
        
        # Get translations if needed
        context_translate = self.benchmark_meta.get("context_translate", {}) if self.use_translated_prompts else {}
        text_col_translate = self.benchmark_meta.get("text_col_translate", {}) if self.use_translated_prompts else {}
        suffix_translate = self.benchmark_meta.get("suffix_translate", []) if self.use_translated_prompts else []
        
        # Extract fields
        rubrics = example[label_semantics] if isinstance(example[label_semantics], list) else [example[label_semantics]]
        answer = example[text_col]

        # Step 1: Build the input string while tracking character spans
        input_parts = []
        span_indices_char = []
        answer_start_char = answer_end_char = None
        
        # Add context columns (question, sample_solution, etc.)
        if self.add_context:
            for col in context_cols:
                if col in example:
                    col_display = context_translate.get(col, col.replace("_", " "))
                    # Handle sample_solution: select one from list if it's a list
                    if col == "sample_solution" and isinstance(example[col], list):
                        if len(example[col]) > 0:
                            solution = random.choice(example[col]) if self.random_solution else example[col][0]
                            
                            input_parts.append(f"{col_display}: {str(solution)}")
                    else:
                        input_parts.append(f"{col_display}: {str(example[col])}")
        
        # Add answer with span tracking
        current_str = self.tokenizer.sep_token.join(input_parts)
        answer_start_char = len(current_str) + (len(self.tokenizer.sep_token) if input_parts else 0)
        answer_display = text_col_translate.get(text_col, text_col.replace("_", " "))
        input_parts.append(f"{answer_display}: {str(answer)}")
        new_str = self.tokenizer.sep_token.join(input_parts)
        answer_end_char = len(new_str)
        
        # Add rubrics with span tracking
        for rubric in rubrics:
                current_str = self.tokenizer.sep_token.join(input_parts)
                start_char = len(current_str) + (len(self.tokenizer.sep_token) if input_parts else 0)
                
                input_parts.append(f"{label_semantics}: {str(rubric)}")
                
                new_str = self.tokenizer.sep_token.join(input_parts)
                end_char = len(new_str)
                span_indices_char.append((start_char, end_char))
        
        # Add suffix if requested
        if self.add_suffix and suffixes:
            # Use translated suffixes if available, otherwise use original
            if self.use_translated_prompts and suffix_translate:
                suffix = random.choice(suffix_translate) if self.random_suffix else suffix_translate[0]
            else:
                suffix = random.choice(suffixes) if self.random_suffix else suffixes[0]
            input_parts.append(suffix)

        # Join all parts
        input_str = self.tokenizer.sep_token.join(input_parts)
        # Step 2: Tokenize with offset mapping
        encoding = self.tokenizer(
            input_str,
            return_offsets_mapping=True,
            truncation=True,
            max_length=2048,
        )

        offsets = encoding["offset_mapping"]
        rubric_spans = []

        # Step 3: Map rubric char spans to token spans
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

        # Step 4: Map answer span to token span
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

        # Step 5: Add to encoding
        encoding["rubric_spans"] = rubric_spans
        encoding["labels"] = int(example["level"])
        
        # Remove offset_mapping as it's not needed for training
        encoding.pop("offset_mapping", None)

        return encoding
    def encode_spans_llm(self, example):
        """
        Encode example with span information using tags for LLM models.
        Tracks spans for rubrics and answer in the tokenized sequence.
        Output example:
        Instruction\n<question>question_text</question>\n<answer>answer_text</answer>\n<rubric>rubric_1</rubric>\n<rubric>rubric_2</rubric>\n...
        """
        import random
        
        # Get benchmark metadata
        context_cols = self.benchmark_meta["context_cols"]
        text_col = self.benchmark_meta["text_col"]  # "answer"
        label_semantics = self.benchmark_meta["label_semantics"]  # "rubric"
        suffixes = self.benchmark_meta.get("suffixes", [])
        
        # Get translations if needed
        context_translate = self.benchmark_meta.get("context_translate", {}) if self.use_translated_prompts else {}
        text_col_translate = self.benchmark_meta.get("text_col_translate", {}) if self.use_translated_prompts else {}
        suffix_translate = self.benchmark_meta.get("suffix_translate", []) if self.use_translated_prompts else []
        
        # Extract fields
        rubrics = example[label_semantics] if isinstance(example[label_semantics], list) else [example[label_semantics]]
        answer = example[text_col]

        # Step 1: Build the input string while tracking character spans
        input_parts = []
        span_indices_char = []
        answer_start_char = answer_end_char = None
        
        # Add context columns (question, sample_solution, etc.) with column-specific tags
        if self.add_context:
            for col in context_cols:
                if col in example:
                    col_display = context_translate.get(col, col.replace("_", " "))
                    # Handle sample_solution: select one from list if it's a list
                    if col == "sample_solution" and isinstance(example[col], list):
                        if len(example[col]) > 0:
                            solution = random.choice(example[col]) if self.random_solution else example[col][0]
                   
                            input_parts.append(f"<{col_display}> {str(solution)} </{col_display}>")
                    else:
                        input_parts.append(f"<{col_display}> {str(example[col])} </{col_display}>")
        
        # Add answer with span tracking and column-specific tags
        current_str = "\n".join(input_parts)
        answer_display = text_col_translate.get(text_col, text_col.replace("_", " "))
        answer_content = str(answer)
        answer_tagged = f"<{answer_display}> {answer_content} </{answer_display}>"
        
        # Calculate answer span (excluding tags)
        base_len = len(current_str) + (1 if input_parts else 0)  # +1 for \n
        opening_tag_len = len(f"<{answer_display}>")
        answer_start_char = base_len + opening_tag_len
        answer_end_char = answer_start_char + len(answer_content)
        
        input_parts.append(answer_tagged)
        
        # Add rubrics with span tracking and column-specific tags
        for rubric in rubrics:
                current_str = "\n".join(input_parts)
                rubric_content = str(rubric)
                rubric_tagged = f"<{label_semantics}> {rubric_content} </{label_semantics}>"
                
                # Calculate rubric span (excluding tags)
                base_len = len(current_str) + (1 if input_parts else 0)  # +1 for \n
                opening_tag_len = len(f"<{label_semantics}>")
                start_char = base_len + opening_tag_len
                end_char = start_char + len(rubric_content)
                
                input_parts.append(rubric_tagged)
                span_indices_char.append((start_char, end_char))
        
        # Add suffix if requested
        if self.add_suffix and suffixes:
            # Use translated suffixes if available, otherwise use original
            if self.use_translated_prompts and suffix_translate:
                suffix = random.choice(suffix_translate) if self.random_suffix else suffix_translate[0]
            else:
                suffix = random.choice(suffixes) if self.random_suffix else suffixes[0]
            input_parts.append(suffix)

        # Join all parts with newlines
        input_str = "\n".join(input_parts)

        # Step 2: Tokenize with offset mapping
        encoding = self.tokenizer(
            input_str,
            return_offsets_mapping=True,
            truncation=True,
            max_length=2048,
        )

        offsets = encoding["offset_mapping"]
        rubric_spans = []

        # Step 3: Map rubric char spans to token spans
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

        # Step 4: Map answer span to token span
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

        # Step 5: Add to encoding
        encoding["rubric_spans"] = rubric_spans
        encoding["labels"] = int(example["level"])
        
        # Remove offset_mapping as it's not needed for training
        encoding.pop("offset_mapping", None)
        # print(self.tokenizer.convert_ids_to_tokens(encoding["input_ids"]))
        # for start, end in rubric_spans:
        #     print(self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][start:end]))
        return encoding

    def encode_xnet_bert(self, example):
        """
        Encode example for xnet with BERT-like models.
        Concatenates answer and rubric with SEP token.
        Output: answer sep_token rubric (optionally with question and sample_solution)
        """
        import random
        
        # Get benchmark metadata
        context_cols = self.benchmark_meta["context_cols"]
        text_col = self.benchmark_meta["text_col"]  # "answer"
        label_semantics = self.benchmark_meta["label_semantics"]  # "rubric"
        
        # Extract fields
        answer = example[text_col]
        rubric = example[label_semantics]
        
        input_parts = []
        
        # Add context columns if requested (question, sample_solution, etc.)
        if self.add_context:
            for col in context_cols:
                if col in example:
                    # Handle sample_solution: select one from list if it's a list
                    if col == "sample_solution" and isinstance(example[col], list):
                        if len(example[col]) > 0:
                            solution = random.choice(example[col]) if self.random_solution else example[col][0]
                            input_parts.append(str(solution))
                    else:
                        input_parts.append(str(example[col]))
        
        # Add answer and rubric
        input_parts.append(answer)
        input_parts.append(rubric)
        
        # Join with SEP token
        input_str = self.tokenizer.sep_token.join(input_parts)
        
        # Tokenize
        encoding = self.tokenizer(
            input_str,
            max_length=2048,
            truncation=True,
            add_special_tokens=True
        )
        
        # Add metadata
        encoding["labels"] = int(example["level"])
        encoding["id"] = example.get("id", None)
        encoding["level"] = example.get("level", None)
        encoding["question_id"] = example.get("question_id", None)
        encoding["rubric_level"] = example.get("rubric_level", None)
        
        return encoding

    def encode_xnet_llm(self, example):
        """
        Encode example for xnet with LLM models.
        Uses structured format with tags for answer and rubric.
        Output: <answer>answer_text</answer><rubric>rubric_text</rubric>
        """
        import random
        
        # Get benchmark metadata
        context_cols = self.benchmark_meta["context_cols"]
        text_col = self.benchmark_meta["text_col"]  # "answer"
        label_semantics = self.benchmark_meta["label_semantics"]  # "rubric"
        suffixes = self.benchmark_meta.get("suffixes", [])
        
        # Get translations if needed
        context_translate = self.benchmark_meta.get("context_translate", {}) if self.use_translated_prompts else {}
        text_col_translate = self.benchmark_meta.get("text_col_translate", {}) if self.use_translated_prompts else {}
        suffix_translate = self.benchmark_meta.get("suffix_translate", []) if self.use_translated_prompts else []
        
        # Extract fields
        answer = example[text_col]
        rubric = example[label_semantics]
        
        input_parts = []
        
        # Add context columns with tags if requested
        if self.add_context:
            for col in context_cols:
                if col in example:
                    col_display = context_translate.get(col, col.replace("_", " "))
                    # Handle sample_solution: select one from list if it's a list
                    if col == "sample_solution" and isinstance(example[col], list):
                        if len(example[col]) > 0:
                            solution = random.choice(example[col]) if self.random_solution else example[col][0]
                            input_parts.append(f"<{col_display}> {str(solution)} </{col_display}>")
                    else:
                        input_parts.append(f"<{col_display}> {str(example[col])} </{col_display}>")
        
        # Add answer and rubric with tags
        answer_display = text_col_translate.get(text_col, text_col.replace("_", " "))
        input_parts.append(f"<{answer_display}> {answer} </{answer_display}>")
        input_parts.append(f"<{label_semantics}> {rubric} </{label_semantics}>")
        
        # Add suffix if requested
        if self.add_suffix and suffixes:
            if self.use_translated_prompts and suffix_translate:
                suffix = random.choice(suffix_translate) if self.random_suffix else suffix_translate[0]
            else:
                suffix = random.choice(suffixes) if self.random_suffix else suffixes[0]
            input_parts.append(suffix)
        
        # Join with newlines
        input_str = "\n".join(input_parts)
        
        # Tokenize
        encoding = self.tokenizer(
            input_str,
            max_length=2048,
            truncation=True
        )
        
        # Add metadata
        encoding["labels"] = int(example["level"])
        encoding["id"] = example.get("id", None)
        encoding["level"] = example.get("level", None)
        encoding["question_id"] = example.get("question_id", None)
        encoding["rubric_level"] = example.get("rubric_level", None)
        
        return encoding

    def xnet_collate_fn(self, input_batch, pad_id=None, return_meta=False):
        """
        Collate function for xnet model that handles grouped examples.
        Each example contains multiple rubrics in [R, S] format.
        
        Args:
            input_batch: List of examples from grouped dataset
            pad_id: Padding token id
            return_meta: Whether to return metadata
            
        Returns:
            batch: Dict with tensors of shape [B, R, S] where B=batch_size, R=num_rubrics, S=seq_len
            meta: Optional metadata dict
        """
        if pad_id is None:
            pad_id = self.pad_token_id
        
        batch_size = len(input_batch)
        max_rubrics = max([x["num_rubrics"] for x in input_batch])
        
        # Initialize lists to collect padded sequences
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []
        
        for example in input_batch:
            num_rubrics = example["num_rubrics"]
            
            # Get input_ids and attention_mask as lists of tensors
            input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in example["input_ids"]]
            attention_mask_list = [torch.tensor(mask, dtype=torch.long) for mask in example["attention_mask"]]
            
            # Pad rubric dimension to max_rubrics
            while len(input_ids_list) < max_rubrics:
                input_ids_list.append(torch.tensor([pad_id], dtype=torch.long))
                attention_mask_list.append(torch.tensor([0], dtype=torch.long))
            
            # Pad sequence length within this example
            max_len = max(ids.shape[0] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_mask = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[0] < max_len:
                    pad_len = max_len - ids.shape[0]
                    ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
            
            # Stack to [R, S]
            batch_input_ids.append(torch.stack(padded_input_ids))
            batch_attention_mask.append(torch.stack(padded_attention_mask))
            
            # Handle token_type_ids if present
            if "token_type_ids" in example and example["token_type_ids"][0] is not None:
                token_type_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in example["token_type_ids"]]
                while len(token_type_ids_list) < max_rubrics:
                    token_type_ids_list.append(torch.tensor([0], dtype=torch.long))
                padded_token_type_ids = []
                for ids in token_type_ids_list:
                    if ids.shape[0] < max_len:
                        pad_len = max_len - ids.shape[0]
                        ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
                    padded_token_type_ids.append(ids)
                batch_token_type_ids.append(torch.stack(padded_token_type_ids))
            
            batch_labels.append(example["labels"])
        
        # Pad sequence length dimension across batch
        max_seq_len = max(x.shape[1] for x in batch_input_ids)
        
        final_input_ids = []
        final_attention_mask = []
        final_token_type_ids = []
        
        for i in range(batch_size):
            curr_seq_len = batch_input_ids[i].shape[1]
            if curr_seq_len < max_seq_len:
                pad_len = max_seq_len - curr_seq_len
                batch_input_ids[i] = torch.cat([
                    batch_input_ids[i],
                    torch.full((max_rubrics, pad_len), pad_id, dtype=torch.long)
                ], dim=1)
                batch_attention_mask[i] = torch.cat([
                    batch_attention_mask[i],
                    torch.zeros((max_rubrics, pad_len), dtype=torch.long)
                ], dim=1)
                if batch_token_type_ids:
                    batch_token_type_ids[i] = torch.cat([
                        batch_token_type_ids[i],
                        torch.zeros((max_rubrics, pad_len), dtype=torch.long)
                    ], dim=1)
            
            final_input_ids.append(batch_input_ids[i])
            final_attention_mask.append(batch_attention_mask[i])
            if batch_token_type_ids:
                final_token_type_ids.append(batch_token_type_ids[i])
        
        # Stack to create final batch tensors [B, R, S]
        batch = {
            "input_ids": torch.stack(final_input_ids),
            "attention_mask": torch.stack(final_attention_mask),
            "labels": torch.tensor(batch_labels),
            "num_rubrics": torch.tensor([x["num_rubrics"] for x in input_batch]),
        }
        
        if final_token_type_ids:
            batch["token_type_ids"] = torch.stack(final_token_type_ids)
        
        meta = {
            "id": [x["id"] for x in input_batch],
            "level": [x["level"] for x in input_batch],
            "question_id": [x["question_id"] for x in input_batch],
            "rubric_level": [x["rubric_level"] for x in input_batch],
        }
        
        if return_meta:
            return batch, meta
        return batch

    def span_collate_fn(self, input_batch, pad_id=None, return_meta=False):
        """
        Collate function for model that handles span information.
        Processes rubric spans, answer spans, and rubric masks.
        
        Args:
            input_batch: List of examples with span information
            pad_id: Padding token id (defaults to tokenizer's pad_token_id)
            return_meta: Whether to return metadata
            
        Returns:
            batch: Dict with input_ids, attention_mask, rubric_spans, answer_span, rubric_mask, labels
            meta: Optional metadata dict
        """
        # Use stored pad_token_id if not provided
        if pad_id is None:
            pad_id = self.pad_token_id
        
        # Pad input_ids and attention_mask
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in input_batch]
        attention_masks = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in input_batch]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        # Handle rubric spans - pad to max number of rubrics
        max_rubrics = max(len(x["rubric_spans"]) for x in input_batch)
        batch_size = len(input_batch)
        
        # Initialize tensors for spans and masks
        rubric_spans_tensor = torch.zeros(batch_size, max_rubrics, 2, dtype=torch.long)
        rubric_mask_tensor = torch.zeros(batch_size, max_rubrics, dtype=torch.bool)
        
        for i, example in enumerate(input_batch):
            spans = torch.tensor(example["rubric_spans"], dtype=torch.long)
            num_spans = spans.shape[0]
            rubric_spans_tensor[i, :num_spans, :] = spans
            rubric_mask_tensor[i, :num_spans] = True

        # Handle answer spans
        answer_spans = torch.tensor([x["answer_span"] for x in input_batch], dtype=torch.long)

        # Create batch dictionary
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "rubric_spans": rubric_spans_tensor,
            "answer_span": answer_spans,
            "rubric_mask": rubric_mask_tensor,
            "labels": torch.tensor([x["labels"] for x in input_batch], dtype=torch.long),
        }
        
        # Prepare metadata
        meta = {
            "id": [x.get("id", None) for x in input_batch],
            "question_id": [x.get("question_id", None) for x in input_batch]
        }
        if return_meta:
            return batch, meta
        return batch


class MultiTaskDataPipeline(DataPipeline):
    """Compose multiple ASAG pipelines into one train/eval interface."""

    def __init__(
        self,
        base_model: str,
        train_tasks: list[str],
        eval_tasks: Optional[list[str]] = None,
        test_tasks: Optional[list[str]] = None,
        train_frac: float = 1.0,
        add_context: bool = True,
        add_suffix: bool = False,
        random_suffix: bool = False,
        random_solution: bool = False,
        use_translated_prompts: bool = False,
        model_class: str = "span",
        random_drop_rub: float = 0.0,
    ):
        self.train_tasks = dedupe_keep_order(train_tasks)
        self.eval_tasks = dedupe_keep_order(eval_tasks) if eval_tasks else list(self.train_tasks)
        self.test_tasks = dedupe_keep_order(test_tasks) if test_tasks else list(self.train_tasks)
        if not self.train_tasks:
            raise ValueError("train_tasks must contain at least one benchmark.")
        if not self.eval_tasks:
            raise ValueError("eval_tasks must contain at least one benchmark.")
        if not self.test_tasks:
            raise ValueError("test_tasks must contain at least one benchmark.")

        exemplar_task = self.train_tasks[0]
        super().__init__(
            base_model=base_model,
            benchmark=exemplar_task,
            train_frac=train_frac,
            add_context=add_context,
            add_suffix=add_suffix,
            random_suffix=random_suffix,
            random_solution=random_solution,
            use_translated_prompts=use_translated_prompts,
            model_class=model_class,
            random_drop_rub=random_drop_rub,
        )

        self.task_pipelines = {
            task: DataPipeline(
                base_model=base_model,
                benchmark=task,
                train_frac=train_frac,
                add_context=add_context,
                add_suffix=add_suffix,
                random_suffix=random_suffix,
                random_solution=random_solution,
                use_translated_prompts=use_translated_prompts,
                model_class=model_class,
                random_drop_rub=random_drop_rub,
            )
            for task in dedupe_keep_order(self.train_tasks + self.eval_tasks + self.test_tasks)
        }

        exemplar_pipeline = self.task_pipelines[exemplar_task]
        self.tokenizer = exemplar_pipeline.tokenizer
        self.pad_token_id = exemplar_pipeline.pad_token_id
        self.is_llm = exemplar_pipeline.is_llm
        self.num_labels = max(pipeline.num_labels for pipeline in self.task_pipelines.values())

        self.train_sizes = {}
        self.val_sizes = {}
        self.test_sizes = {}
        self.eval_split_map = {}
        self.test_split_map = {}

    def _training_columns(self):
        if self.needs_grouping():
            return ["input_ids", "attention_mask", "labels", "num_rubrics", "token_type_ids"]
        return ["input_ids", "attention_mask", "rubric_spans", "answer_span", "labels"]

    def _trim_for_training(self, dataset):
        keep_columns = [col for col in self._training_columns() if col in dataset.column_names]
        remove_columns = [col for col in dataset.column_names if col not in keep_columns]
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)
        return dataset

    def _concat(self, datasets):
        if not datasets:
            return Dataset.from_list([])
        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)

    def get_datasets(self, test_only=True):
        train_datasets = []
        val_datasets = []
        cached_task_splits = {}

        if not test_only:
            self.train_sizes = {}
            self.val_sizes = {}
            all_tasks = dedupe_keep_order(self.train_tasks + self.eval_tasks + self.test_tasks)

            for task in all_tasks:
                train_ds, val_ds, test_ds = self.task_pipelines[task].get_datasets(test_only=False)
                cached_task_splits[task] = {"train": train_ds, "val": val_ds, "test": test_ds}

                if task in self.train_tasks:
                    self.train_sizes[task] = len(train_ds)
                    train_datasets.append(self._trim_for_training(train_ds))
                if task in self.eval_tasks:
                    self.val_sizes[task] = len(val_ds)
                    val_datasets.append(self._trim_for_training(val_ds))

            train_dataset = self._concat(train_datasets)
            val_dataset = self._concat(val_datasets)
        else:
            train_dataset = Dataset.from_list([])
            val_dataset = Dataset.from_list([])
            self.train_sizes = {}
            self.val_sizes = {}

        self.test_sizes = {}
        self.eval_split_map = {}
        self.test_split_map = {}
        test_datasets = {}

        for task in self.test_tasks:
            task_splits = cached_task_splits.get(task)
            task_test_ds = task_splits["test"] if task_splits is not None else None
            if task_test_ds is None:
                _, _, task_test_ds = self.task_pipelines[task].get_datasets(test_only=True)

            task_test_ds = task_test_ds if isinstance(task_test_ds, dict) else {"test": task_test_ds}
            self.test_sizes[task] = {}

            for split_name, split_dataset in task_test_ds.items():
                dataset_key = f"{task}_{split_name}"
                split_info = {"task": task, "split": split_name}
                self.eval_split_map[dataset_key] = split_info
                self.test_split_map[dataset_key] = split_info
                self.test_sizes[task][split_name] = len(split_dataset)
                test_datasets[dataset_key] = split_dataset

        return train_dataset, val_dataset, test_datasets

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer

    config = DataPipeline(
        base_model="meta-llama/Llama-3.2-1B-Instruct",
        benchmark="alice",
        add_context=True,
        add_suffix=True,
        random_suffix=False
    )

    # Example data
    example = {
        "question": "What is the capital of France?",
        "sample_solution": "The capital of France is Paris.",
        "answer": "Paris",
        "rubric": ["Correctly identifies Paris as the capital.", "Provides additional context about France."],
        "level": 2
    }