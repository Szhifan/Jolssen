import pandas as pd
import random
from collections import defaultdict
from pathlib import Path
from datasets import Dataset
import json
from scripts_asag.data_processing.benchmark_meta import BENCHMARK_DESCRIPTIONS


class ASAG_Data_Loader:
    def __init__(self, benchmark: str = None, train_frac=1):
        assert train_frac > 0, "train_frac must be > 0"
        assert train_frac <= 1 or float(train_frac).is_integer(), (
            "train_frac > 1 must be an integer-valued exact number of training instances"
        )
        self.train_frac = train_frac
        self.benchmark = benchmark
        if benchmark:
            self.task_name = Path("asag_benchmarks", benchmark)
        self.benchmark_meta = BENCHMARK_DESCRIPTIONS.get(benchmark, {})
        self.question_meta = self._load_question_meta()
        self.label_semantics = "rubric"
        self.text_col = "answer"
        self._score_level_map = self.benchmark_meta.get("label_map", {})
        
        self._load_datasets()

    def _load_datasets(self):
        """Load train, val, and test datasets. Can be overridden by subclasses."""
        train_df = pd.read_csv(self.task_name / "train.csv")
        
     

        # ----------- Group by question_id and sample questions -----------
        if self.train_frac < 1:
            question_to_entries = defaultdict(list)
            for _, row in train_df.iterrows():
                question_to_entries[str(row["question_id"])].append(row.to_dict())

            all_qids = list(question_to_entries.keys())
            n_sample = max(1, int(len(all_qids) * self.train_frac))
            sampled_qids = set(random.sample(all_qids, n_sample))
            train_df = pd.DataFrame(
                [e for qid in sampled_qids for e in question_to_entries[qid]]
            )
        elif self.train_frac > 1:
            n_sample = int(self.train_frac)
            if n_sample > len(train_df):
                raise ValueError(
                    f"train_frac={self.train_frac} requests {n_sample} training instances, "
                    f"but only {len(train_df)} are available for benchmark '{self.benchmark}'."
                )
            sampled_indices = random.sample(range(len(train_df)), n_sample)
            train_df = train_df.iloc[sampled_indices].reset_index(drop=True)
        # ----------- Sampling complete -----------

        val_path = self.task_name / "val.csv"
        if val_path.exists():
            val_df = pd.read_csv(val_path)
            train_dataset = Dataset.from_pandas(train_df)
            self.train = train_dataset.map(lambda x: self._retrieve_meta(x, is_training=True))
            self.val = Dataset.from_pandas(val_df).map(lambda x: self._retrieve_meta(x, is_training=False))
        else:
            train_dataset = Dataset.from_pandas(train_df)
            split = train_dataset.train_test_split(test_size=0.1, seed=42)
            self.train = split["train"].map(lambda x: self._retrieve_meta(x, is_training=True))
            self.val = split["test"].map(lambda x: self._retrieve_meta(x, is_training=False))
        test_dfs = {}
        for path in sorted(self.task_name.glob("test*.csv")):
            if path.name == "train.csv":
                continue
            test_dts = Dataset.from_pandas(pd.read_csv(path))
            test_dts = test_dts.map(lambda x: self._retrieve_meta(x, is_training=False))
            test_dfs[path.stem] = test_dts
        if not test_dfs:
            raise ValueError(f"No test files found in {self.task_name}")
        self.test = test_dfs



    def _load_question_meta(self):
        meta_path = self.task_name / "question_meta.json"
        if not meta_path.exists():
            return {}
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta = {str(k): v for k, v in meta.items()}

        rubric_meta_path = self.benchmark_meta.get("rubric_meta_path")
        if rubric_meta_path:
            generated_path = self.task_name / rubric_meta_path
            if generated_path.exists():
                with open(generated_path, "r") as f:
                    generated_meta = json.load(f)
                for qid, generated_entry in generated_meta.items():
                    question_meta = meta.setdefault(str(qid), {})
                    response = generated_entry.get("response", {})
                    if isinstance(response, str):
                        try:
                            response = json.loads(response)
                        except json.JSONDecodeError:
                            response = self._parse_fenced_json(response)
                    if isinstance(response, dict) and isinstance(response.get("rubrics"), dict):
                        question_meta["rubrics"] = response["rubrics"]
                    if "reference_answer" in generated_entry:
                        question_meta["sample_solution"] = generated_entry["reference_answer"]
        return meta

    def _parse_fenced_json(self, text):
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {}

    def _normalize_level(self, raw_level):
        if raw_level is None:
            return 0
        if isinstance(raw_level, str):
            normalized = raw_level.strip()
            if normalized in self._score_level_map:
                return int(self._score_level_map[normalized])
            lowered = normalized.lower()
            if lowered in self._score_level_map:
                return int(self._score_level_map[lowered])
            raw_level = normalized
        return int(raw_level)

    def _rubric_items(self, rubrics):
        if not self._score_level_map:
            return sorted(rubrics.items(), key=lambda item: float(item[0]))
        label_order = sorted(self._score_level_map.items(), key=lambda item: item[1])
        if all(label in rubrics for label, _ in label_order):
            return [(label, rubrics[label]) for label, _ in label_order]
        return sorted(rubrics.items(), key=lambda item: float(item[0]))

    def _retrieve_meta(self, example, is_training=False):
        """Retrieve and enrich example with metadata. Can be overridden by subclasses."""
        meta = self.question_meta.get(str(example.get("question_id", "")), {})

        # Always use metadata from question_meta.json if available
        if "question" in meta:
            example["question"] = meta["question"]
        if "sample_solution" in meta:
            example["sample_solution"] = meta["sample_solution"][:5]
        if "question_context" in meta:
            example["question_context"] = meta["question_context"]


        rubrics = meta["rubrics"]
        rubric_list = [
            v for _, v in self._rubric_items(rubrics)
        ]
        
        level = self._normalize_level(example.get("level", example.get("score_level", 0)))

        example["rubric"] = rubric_list
        example["level"] = level
        example["num_rubrics"] = len(rubric_list)
        return example


if __name__ == "__main__":
    data_loader = ASAG_Data_Loader("scientsbank", train_frac=0.5)
    print(data_loader.train[0])
