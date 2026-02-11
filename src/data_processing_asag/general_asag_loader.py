import pandas as pd
import random
from collections import defaultdict
from pathlib import Path
from datasets import Dataset
import json


class ASAG_Data_Loader:
    def __init__(self, benchmark: str = None, train_frac=1, random_solution=False, random_drop_rub=0.0):
        assert train_frac <= 1 and train_frac > 0, "train_frac must be in (0, 1]"
        self.train_frac = train_frac
        self.random_solution = random_solution
        self.random_drop_rub = random_drop_rub
        self.benchmark = benchmark
        if benchmark:
            self.task_name = Path("asag_benchmarks", benchmark)
        self.question_meta = self._load_question_meta()
        self.label_semantics = "rubric"
        self.text_col = "answer"
        self._score_level_map = None
        
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
            n_sample = int(len(all_qids) * self.train_frac)
            sampled_qids = set(random.sample(all_qids, n_sample))
            train_df = pd.DataFrame(
                [e for qid in sampled_qids for e in question_to_entries[qid]]
            )
        # ----------- Sampling complete -----------

        val_path = self.task_name / "val.csv"
        if val_path.exists():
            val_df = pd.read_csv(val_path)
            train_dataset = Dataset.from_pandas(train_df)
            self.train = train_dataset.map(lambda x: self._retrieve_meta(x))
            self.val = Dataset.from_pandas(val_df).map(lambda x: self._retrieve_meta(x))
        else:
            train_dataset = Dataset.from_pandas(train_df)
            split = train_dataset.train_test_split(test_size=0.1, seed=42)
            self.train = split["train"].map(lambda x: self._retrieve_meta(x))
            self.val = split["test"].map(lambda x: self._retrieve_meta(x))
        test_dfs = {}
        for path in sorted(self.task_name.glob("test*.csv")):
            if path.name == "train.csv":
                continue
            test_dts = Dataset.from_pandas(pd.read_csv(path))
            test_dts = test_dts.map(lambda x: self._retrieve_meta(x))
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
        return {str(k): v for k, v in meta.items()}

    def _retrieve_meta(self, example):
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
            v for _, v in sorted(rubrics.items(), key=lambda item: float(item[0]))
        ]

        example["rubric"] = rubric_list
        example["num_rubrics"] = len(rubric_list)
        return example


if __name__ == "__main__":
    data_loader = ASAG_Data_Loader("scientsbank", train_frac=0.5)
    print(data_loader.train[0])
