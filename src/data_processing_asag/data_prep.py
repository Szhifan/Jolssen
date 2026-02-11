import torch
from typing import Literal, Callable, Optional
from functools import partial
from data_processing_asag.benchmark_meta import BENCHMARK_DESCRIPTIONS
from transformers import AutoTokenizer


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
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set
    tokenizer.sep_token = tokenizer.sep_token or tokenizer.eos_token  # Ensure sep_token is set
    return tokenizer

class DataPipeline:
    """
    Configuration class for selecting encoding functions and collate functions
    based on model class and model type (LLM vs MLM).
    
    Usage:
        config = DataPipeline(
            base_model="bert-base-uncased",
            benchmark="alice",
            train_frac=1.0,
            add_context=True,
            add_suffix=False,
            random_suffix=False
        )
        train_ds, val_ds, test_ds = config.get_datasets()
        enc_fn = config.get_encode_fn()
        collate_fn = config.get_collate_fn()
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
        drop_rubric: bool = False,
        model_class: str = "span",
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
        self.num_labels = self.benchmark_meta.get("num_labels", 3)  # Default to 3 if not specified
        self.add_context = add_context
        self.add_suffix = add_suffix
        self.random_suffix = random_suffix
        self.random_solution = random_solution
        self.use_translated_prompts = use_translated_prompts
        self.drop_rubric = drop_rubric
        
        # Validate drop_rubric usage
        if drop_rubric and model_class != "span":
            raise ValueError("drop_rubric option is only available for model_class='span'")
        
        self.is_llm = is_llm_model(base_model)
        self.tokenizer = get_tokenizer(base_model)
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Initialize data loader
        self._init_data_loader()
    
    def _init_data_loader(self):
        """Initialize the appropriate data loader based on benchmark."""
        if "alice" in self.benchmark:
            from data_processing_asag.alice_asag_loader import Alice_Loader
            # Extract task_type from benchmark name if it contains suffix (e.g., "alice_lp")
            task_type = self.benchmark.split("_")[-1] if "_" in self.benchmark else "lp"
            self.data_loader = Alice_Loader(train_frac=self.train_frac, task_type=task_type)
        else:
            from data_processing_asag.general_asag_loader import ASAG_Data_Loader
            self.data_loader = ASAG_Data_Loader(
                benchmark=self.benchmark,
                train_frac=self.train_frac
            )
    
    def get_datasets(self, test_only=True):
        """
        Get train, validation, and test datasets.
        
        Args:
            test_only: If True, only encode the test datasets.
            
        Returns:
            tuple: (train_dataset, val_dataset, test_datasets)
                   test_datasets can be a dict or single dataset
        """
        if test_only:
            # Only encode test datasets
            enc_fn = self.get_encode_fn()
            if isinstance(self.data_loader.test, dict):
                test_ds = {}
                for key, dataset in self.data_loader.test.items():
                    encoded_ds = dataset.map(lambda x: enc_fn(x))
                    if self.needs_grouping():
                        encoded_ds = self.group_by_id(encoded_ds)
                    test_ds[key] = encoded_ds
            else:
                test_ds = self.data_loader.test.map(lambda x: enc_fn(x))
                if self.needs_grouping():
                    test_ds = self.group_by_id(test_ds)
            train_ds, val_ds = self.data_loader.train, self.data_loader.val
        else:
            # Encode train, validation, and test datasets
            enc_fn = self.get_encode_fn()
            train_ds = self.data_loader.train.map(lambda x: enc_fn(x))
            val_ds = self.data_loader.val.map(lambda x: enc_fn(x))
            
            # For xnet models, group by ID after encoding
            if self.needs_grouping():
                train_ds = self.group_by_id(train_ds)
                val_ds = self.group_by_id(val_ds)
            
            input_id = train_ds[0]["input_ids"]
            rubric_span = train_ds[0].get("rubric_spans", None)
            answer_span = train_ds[0].get("answer_span", None)
            sub_tokens = self.tokenizer.convert_ids_to_tokens(input_id)
            print(f"Sample decoded text: {self.tokenizer.decode(input_id)}")
            print("answer_span:", sub_tokens[answer_span[0]:answer_span[1]] if answer_span else None)
            for i, span in enumerate(rubric_span if rubric_span else []):
                print(f"rubric_span {i}:", sub_tokens[span[0]:span[1]])
            
            # Handle test datasets (can be dict or single dataset)
            if isinstance(self.data_loader.test, dict):
                test_ds = {}
                for key, dataset in self.data_loader.test.items():
                    encoded_ds = dataset.map(lambda x: enc_fn(x))
                    if self.needs_grouping():
                        encoded_ds = self.group_by_id(encoded_ds)
                    test_ds[key] = encoded_ds
            else:
                test_ds = self.data_loader.test.map(lambda x: enc_fn(x))
                if self.needs_grouping():
                    test_ds = self.group_by_id(test_ds)
        
        return train_ds, val_ds, test_ds
    
    def get_encode_fn(self) -> Callable:
        """
        Get the appropriate encoding function based on model class and type.
        
        Returns:
            Callable: Encoding function that takes (example) and returns encoded example
        """
        # For xnet models, use xnet encoding
        if self.model_class in ["xnet", "xnet-pwr", "xnet-contrastive"]:
            if self.is_llm:
                return partial(self.encode_xnet_llm)
            else:
                return partial(self.encode_xnet_bert)
        # Default to span encoding
        else:
            if self.is_llm:
                return partial(self.encode_spans_llm)
            else:
                return partial(self.encode_spans_bert)
    
    def get_collate_fn(self) -> Callable:
        """
        Get the collate function for batching.
        
        Returns:
            Callable: Collate function
        """
        if self.model_class in ["xnet", "xnet-pwr", "xnet-contrastive"]:
            return partial(self.xnet_collate_fn)
        else:
            return partial(self.span_collate_fn)
    
    def needs_grouping(self) -> bool:
        """Check if the dataset needs to be grouped by ID for xnet models."""
        return self.model_class in ["xnet", "xnet-pwr", "xnet-contrastive"]
    
    def group_by_id(self, dataset):
        """
        Group dataset examples by ID for xnet models.
        Each answer can have multiple rubrics, so we group them together.
        
        Returns a dataset where each example contains:
        - input_ids: list of input_ids for each rubric [R, S]
        - attention_mask: list of attention_masks for each rubric [R, S]
        - labels: the correct rubric index
        - num_rubrics: number of rubrics for this answer
        - id, level, question_id, rubric_level: metadata
        """
        from collections import defaultdict
        
        # Group by ID
        grouped = defaultdict(list)
        for example in dataset:
            example_id = example.get("id", None)
            if example_id is None:
                # If no ID, use question_id as fallback
                example_id = example.get("question_id", "unknown")
            grouped[example_id].append(example)
        
        # Create new dataset with grouped examples
        grouped_examples = []
        for group_id, examples in grouped.items():
            # All examples in a group should have the same answer but different rubrics
            # The label indicates which rubric is correct (index in the group)
            
            grouped_example = {
                "input_ids": [ex["input_ids"] for ex in examples],
                "attention_mask": [ex["attention_mask"] for ex in examples],
                "labels": examples[0]["labels"],  # Use the label from first example
                "num_rubrics": len(examples),
                "id": examples[0].get("id", None),
                "level": examples[0].get("level", None),
                "question_id": examples[0].get("question_id", None),
                "rubric_level": [ex.get("rubric_level", None) for ex in examples],
            }
            
            # Handle token_type_ids if present
            if "token_type_ids" in examples[0] and examples[0]["token_type_ids"] is not None:
                grouped_example["token_type_ids"] = [ex["token_type_ids"] for ex in examples]
            
            grouped_examples.append(grouped_example)
        
        # Convert back to HuggingFace dataset format
        from datasets import Dataset
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
        
        # Add rubrics with span tracking (unless drop_rubric is True)
        if not self.drop_rubric:
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
        
        # Add rubrics with span tracking and column-specific tags (unless drop_rubric is True)
        if not self.drop_rubric:
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