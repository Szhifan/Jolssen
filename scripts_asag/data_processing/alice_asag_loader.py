from datasets import Dataset
import json
import random
from collections import defaultdict
from scripts_asag.data_processing.general_asag_loader import ASAG_Data_Loader

"""
Alice-specific data loader that extends the general ASAG data loader.
Supports multiple task types: lp (learning_performance), ke (knowledge_elements), sk (skills)
"""

path_train = "asag_benchmarks/alice_data/train.json"
path_ua = "asag_benchmarks/alice_data/test_ua.json"
path_uq = "asag_benchmarks/alice_data/test_uq.json"
path_meta = "asag_benchmarks/alice_data/question_meta.json"

# Load the single question_meta file
with open(path_meta, "r") as f:
    question_meta = json.load(f)


class Alice_Loader(ASAG_Data_Loader):
    """
    Load the splits of Alice dataset with support for multiple task types.
    Task types:
        - lp: Learning Performance
        - ke: Knowledge Elements
        - sk: Skills
    """
    
    def __init__(self, train_frac=1, task_type="lp", remap=True):
        assert train_frac > 0, "train_frac must be > 0"
        assert train_frac <= 1 or float(train_frac).is_integer(), (
            "train_frac > 1 must be an integer-valued exact number of training instances"
        )
        assert task_type in ["lp", "ke", "sk"], "task_type must be one of ['lp','ke','sk']"
        
        self.task_type = task_type
        self.remap = remap
        self.train_frac = train_frac
        
        # Set Alice-specific attributes
        self.task_name = None  # Not used for Alice, using file paths instead
        self.benchmark = "alice"
        self.question_meta = question_meta
        self.context_cols = ["question", "sample_solution"]
        self.label_semantics = "rubric"
        self.text_col = "answer"
        
        # Load Alice datasets
        self._load_datasets()

    def _load_datasets(self):
        """Load Alice train and test datasets from JSON files."""
        with open(path_train, "r") as f:
            train_data = json.load(f)
        with open(path_ua, "r") as f:
            test_ua_data = json.load(f)
        with open(path_uq, "r") as f:
            test_uq_data = json.load(f)
        
        # Process datasets
        train_data = self._process_data(train_data, is_training=True)
        test_ua_data = self._process_data(test_ua_data, is_training=False)
        test_uq_data = self._process_data(test_uq_data, is_training=False)
        
        # Sample training data if needed
        if self.train_frac < 1:
            question_to_entries = defaultdict(list)
            for entry in train_data:
                question_to_entries[entry["question_id"]].append(entry)
            all_qids = list(question_to_entries.keys())
            n_sample = max(1, int(len(all_qids) * self.train_frac))
            sampled_qids = set(random.sample(all_qids, n_sample))
            train_data = [e for qid in sampled_qids for e in question_to_entries[qid]]
        elif self.train_frac > 1:
            n_sample = int(self.train_frac)
            if n_sample > len(train_data):
                raise ValueError(
                    f"train_frac={self.train_frac} requests {n_sample} training instances, "
                    f"but only {len(train_data)} are available for Alice {self.task_type}."
                )
            train_data = random.sample(train_data, n_sample)
        
        # Create train/val split
        train_dataset = Dataset.from_list(train_data)
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        self.train = split["train"]
        self.val = split["test"]
        
        # Set test datasets
        self.test = {
            "test_ua": Dataset.from_list(test_ua_data),
            "test_uq": Dataset.from_list(test_uq_data)
        }

    def _process_data(self, data_list, is_training=False):
        """Process raw data entries based on task type."""
        processed = []
        for entry in data_list:
            if self.task_type == "lp":
                processed_entry = self._retrieve_meta_lp(entry, is_training=is_training)
                processed.append(processed_entry)
            else:
                # For ke and sk, one entry may expand to multiple
                expanded_entries = self._retrieve_meta_ke_sk(entry, is_training=is_training)
                processed.extend(expanded_entries)
        return processed

    def _retrieve_meta_lp(self, entry: dict, is_training=False):
        """Retrieve metadata for learning_performance task type."""
        question_id = entry["question_id"]
        meta_info = question_meta.get(question_id, {})
        
        entry["question"] = meta_info["prompt"]
        entry["sample_solution"] = meta_info["sample_solution"][:5]  
        
        rubric = meta_info.get("learning_performance", {})
        rubric_list = list(rubric.values())
        level = int(next(iter(entry.get("learning_performance", {}).values()), 0))
        
        
        entry["rubric"] = rubric_list
        entry["level"] = level
        entry["num_rubrics"] = len(rubric_list)
        
        return entry

    def _retrieve_meta_ke_sk(self, entry: dict, is_training=False):
        """Retrieve metadata for knowledge_elements or skills task types. Returns list of entries."""
        question_id = entry["question_id"]
        meta_info = question_meta.get(question_id, {})
        
        expanded_entries = []
        
        # Determine which field to process (ke or sk)
        field_name = "knowledge_elements" if self.task_type == "ke" else "skills"
        
        if not entry.get(field_name):
            return []
        
        for i, item_key in enumerate(entry[field_name]):
            new_entry = entry.copy()
            new_entry["id"] = f"{entry['id']}_{field_name[0]}{i}"
            new_entry["question"] = meta_info["prompt"]
            new_entry["sample_solution"] = meta_info["sample_solution"]
            
            # Get rubric for this specific item
            item_rubric = meta_info.get(field_name, {}).get(item_key, {})
            if len(item_rubric) == 0:
                continue
            
            level_range = set(item_rubric.keys())
            
            # Remap levels if needed (e.g., {0,1,3} -> {0,1,2})
            if self.remap:
                level_remap = {k: i for i, k in enumerate(sorted(level_range, key=float))}
                level = level_remap[str(entry[field_name][item_key])]
            else:
                level = int(entry[field_name][item_key])
            
            # Format rubric
            item_rubric_list = [f"{item_key}: {v['description']}" for v in item_rubric.values()]
            
            
            new_entry["rubric"] = item_rubric_list
            new_entry["level"] = level
            new_entry["num_rubrics"] = len(item_rubric_list)
            new_entry[field_name[:-1]] = item_key
            
            expanded_entries.append(new_entry)
        
        return expanded_entries



if __name__ == "__main__":
    loader = Alice_Loader(train_frac=1, task_type="ke", remap=True)
    print("Train set size:", len(loader.train))
    print("Val set size:", len(loader.val))
    print("Sample train entry:", loader.train[0])