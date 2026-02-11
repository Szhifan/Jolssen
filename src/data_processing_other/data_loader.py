import pandas as pd
import random
from collections import defaultdict
from pathlib import Path
from datasets import Dataset, load_dataset
import json
from typing import Iterable, List, Dict, Any, Optional


def format_winogrande():
    def _process_split(split):
        data = []
        for item in split:
            sentence = item["sentence"]
            options = [item["option1"], item["option2"]]
            answer = item["answer"]
            if not answer:  # Check if answer is empty
                print(f"Empty answer in item: {item}")
                continue
            label = int(answer[-1]) - 1  # Convert "option1"/"option2" to 0/1
            data.append({"sentence": sentence, "options": options, "label": label})

        return data
    ds = load_dataset("allenai/winogrande", "winogrande_l")
    train_data = _process_split(ds["train"])
    test_data = _process_split(ds["validation"])
    train_val_split = int(0.9 * len(train_data))
    val_data = train_data[train_val_split:]
    train_data = train_data[:train_val_split]
    # test_data = _process_split(ds["test"])
    ds = {
        "train": Dataset.from_pandas(pd.DataFrame(train_data)),
        "val": Dataset.from_pandas(pd.DataFrame(val_data)),
        "test": Dataset.from_pandas(pd.DataFrame(test_data))
    }
    return ds

def format_piqa():
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _read_labels(path: Path) -> List[int]:
        with path.open("r", encoding="utf-8") as f:
            return [int(line.strip()) for line in f if line.strip() != ""]

    def _to_examples(records: Iterable[Dict[str, Any]], labels: Iterable[int]) -> List[Dict[str, Any]]:
        examples = []
        for idx, (record, label) in enumerate(zip(records, labels)):
            goal = record.get("goal", "")
            sol1 = record.get("sol1", "")
            sol2 = record.get("sol2", "")
            examples.append(
                {
                    "id": idx,
                    "goal": goal,
                    "sol1": sol1,
                    "sol2": sol2,
                    "options": [sol1, sol2],
                    "label": int(label),
                }
            )
        return examples

    base = Path("other_benchmarks/piqa")  # Updated path
    train_path = base / "train.jsonl"
    train_labels_path = base / "train-labels.lst"
    valid_path = base / "valid.jsonl"
    valid_labels_path = base / "valid-labels.lst"

    train_records = _read_jsonl(train_path)
    train_labels = _read_labels(train_labels_path)
    if len(train_records) != len(train_labels):
        raise ValueError(
            f"PIQA train records/labels mismatch: {len(train_records)} != {len(train_labels)} "
            f"({train_path}, {train_labels_path})"
        )

    test_records = _read_jsonl(valid_path)
    test_labels = _read_labels(valid_labels_path)
    if len(test_records) != len(test_labels):
        raise ValueError(
            f"PIQA valid records/labels mismatch: {len(test_records)} != {len(test_labels)} "
            f"({valid_path}, {valid_labels_path})"
        )

    full_train = Dataset.from_pandas(pd.DataFrame(_to_examples(train_records, train_labels)))
    split = full_train.train_test_split(test_size=0.1, seed=42)

    ds = {
        "train": split["train"],
        "val": split["test"],
        "test": Dataset.from_pandas(pd.DataFrame(_to_examples(test_records, test_labels))),
    }
    return ds


def format_xstance():
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_questions() -> Dict[tuple, Dict[str, Any]]:
        questions = {}
        q_files = {
            "de": "questions.de.jsonl",
            "en": "questions.en.jsonl",
            "fr": "questions.fr.jsonl",
            "it": "questions.it.jsonl",
        }
        base = Path("other_benchmarks/xstance")  # Updated path
        for lang, fname in q_files.items():
            path = base / fname
            if not path.exists():
                continue
            for item in _read_jsonl(path):
                qid = item.get("id")
                if qid is None:
                    continue
                questions[(lang, qid)] = item
        return questions

    label_map = {"AGAINST": 0, "FAVOR": 1}
    label_semantics = [
        "Against: The text expresses an unfavorable opinion, opposition, or disagreement with the claim.",
        "Favor: The text expresses a favorable opinion, support, or agreement with the claim.",
    ]

    questions = _load_questions()

    def _process(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        data = []
        for record in records:
            label_text = record.get("label")
            if label_text not in label_map:
                raise ValueError(f"Unexpected label in xstance record: {label_text}")
            example = dict(record)
            lang = record.get("language")
            qid = record.get("question_id")
            q_info = questions.get((lang, qid))
            if q_info is not None:
                example["question"] = q_info.get("text", example.get("question"))
            example["label_text"] = label_text
            example["label"] = label_map[label_text]
            example["options"] = label_semantics
            data.append(example)
        return data

    base = Path("other_benchmarks/xstance")
    train_records = _read_jsonl(base / "train.jsonl")
    val_records = _read_jsonl(base / "valid.jsonl")
    test_records = _read_jsonl(base / "test.jsonl")

    ds = {
        "train": Dataset.from_pandas(pd.DataFrame(_process(train_records))),
        "val": Dataset.from_pandas(pd.DataFrame(_process(val_records))),
        "test": Dataset.from_pandas(pd.DataFrame(_process(test_records))),
    }
    return ds


def format_semeval2016():
    import csv

    def _read_tsv(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="latin-1") as f:
            reader = csv.DictReader(f, delimiter="\t")
            return [row for row in reader]

    label_map = {"AGAINST": 0, "FAVOR": 1, "NONE": 2}
    label_semantics = [
        "Against: The text expresses opposition or disagreement with the target.",
        "Favor: The text expresses support or agreement with the target.",
        "None: The text does not express a clear stance toward the target.",
    ]

    def _process(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        data = []
        for record in records:
            stance = record.get("Stance")
            if stance not in label_map:
                raise ValueError(f"Unexpected stance label in SemEval-2016: {stance}")
            example = {
                "id": int(record["ID"]) if record.get("ID") else None,
                "target": record.get("Target", ""),
                "tweet": record.get("Tweet", ""),
                "label_text": stance,
                "label": label_map[stance],
                "options": label_semantics,
                "opinion_towards": record.get("Opinion towards", ""),
                "sentiment": record.get("Sentiment", ""),
            }
            data.append(example)
        return data

    base = Path("other_benchmarks/semeval2016")  # Updated path
    train_records = _read_tsv(base / "trainingdata-all-annotations.txt")
    val_records = _read_tsv(base / "trialdata-all-annotations.txt")
    test_a_records = _read_tsv(base / "testdata-taskA-all-annotations.txt")
    test_b_records = _read_tsv(base / "testdata-taskB-all-annotations.txt")

    ds = {
        "train": Dataset.from_pandas(pd.DataFrame(_process(train_records))),
        "val": Dataset.from_pandas(pd.DataFrame(_process(val_records))),
        "test_a": Dataset.from_pandas(pd.DataFrame(_process(test_a_records))),
        "test_b": Dataset.from_pandas(pd.DataFrame(_process(test_b_records))),
    }
    return ds

def format_figqa():
    def _process_split(split):
        data = []
        for item in split:
            if "valid" in item and item.get("valid") != 1:
                continue
            sentence = item["startphrase"]
            options = [item["ending1"], item["ending2"]]
            data.append({
                "sentence": sentence,
                "options": options,
                "label": int(item["labels"]) if item.get("labels") is not None else None
            })
        return data

    ds = load_dataset("nightingal3/fig-qa")

    train_data = _process_split(ds["train"])
    val_data = _process_split(ds["validation"])
    test_data = _process_split(ds["test"])
    return {
        "train": Dataset.from_pandas(pd.DataFrame(train_data)),
        "val": Dataset.from_pandas(pd.DataFrame(val_data)),
        "test": Dataset.from_pandas(pd.DataFrame(test_data))
    }

def format_yelp():
    def _read_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            header=None,
            names=["label_raw", "review"],
            dtype={"label_raw": int, "review": str},
            quotechar='"',
            escapechar="\\",
            engine="python"
        )
        return df

    def _to_examples(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label"] = df["label_raw"].map({1: 0, 2: 1})
        df["label_text"] = df["label"].map({0: "Negative", 1: "Positive"})
        df["options"] = [
            [
                "Negative: The review expresses unfavorable sentiment.",
                "Positive: The review expresses favorable sentiment."
            ]
        ] * len(df)
        return df[["review", "label", "label_text", "options"]]

    base = Path("other_benchmarks/yelp")  # Updated path
    train_df = _read_csv(base / "train.csv")
    test_df = _read_csv(base / "test.csv")

    if train_df["label_raw"].isin([1, 2]).mean() < 1.0:
        raise ValueError("Unexpected label values in Yelp train.csv (expected 1/2).")
    if test_df["label_raw"].isin([1, 2]).mean() < 1.0:
        raise ValueError("Unexpected label values in Yelp test.csv (expected 1/2).")

    full_train = Dataset.from_pandas(_to_examples(train_df))
    split = full_train.train_test_split(test_size=0.1, seed=42)

    ds = {
        "train": split["train"],
        "val": split["test"],
        "test": Dataset.from_pandas(_to_examples(test_df)),
    }
    return ds

def format_eic():
    label_tags = ["Claim", "Clarity", "Fact/Evidence", "Grammar", "Other"]
    label_map = {label: idx for idx, label in enumerate(label_tags)}
    label_semantics = [
        "Claim: Edits that introduce or modify claims or subjective statements.",
        "Clarity: Edits that improve clarity or readability without changing meaning.",
        "Fact/Evidence: Edits that add, remove, or adjust factual information or evidence.",
        "Grammar: Edits that fix grammar, spelling, or punctuation.",
        "Other: Edits that do not fit the above categories."
    ]

    def _read_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            dtype={
                "edit_index": str,
                "id": str,
                "doc_name": str,
                "node_ix_src": str,
                "node_ix_tgt": str,
                "text_src": str,
                "text_tgt": str,
                "label": str,
            },
            keep_default_na=False,
        )
        return df

    def _process(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not df["label"].isin(label_map).all():
            bad = sorted(set(df["label"]) - set(label_map))
            raise ValueError(f"Unexpected labels in edit_intent_classification: {bad}")
        df = df.rename(columns={"text_src": "old", "text_tgt": "new"})
        df["label_text"] = df["label"]
        df["label"] = df["label"].map(label_map)
        df["options"] = [label_semantics] * len(df)
        return df

    base = Path("other_benchmarks/eic")  # Updated path
    train_df = _read_csv(base / "train.csv")
    val_df = _read_csv(base / "val.csv")
    test_df = _read_csv(base / "test.csv")

    ds = {
        "train": Dataset.from_pandas(_process(train_df)),
        "val": Dataset.from_pandas(_process(val_df)),
        "test": Dataset.from_pandas(_process(test_df)),
    }
    return ds
