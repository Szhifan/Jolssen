import os
import sys
from pathlib import Path
import wandb
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from trainer import AsagTrainer, AsagTrainingArguments
from utils import (
    set_seed,
    eval_report,
    save_report,
    evaluate,
    clear_gpu_memory,
    per_qid_metrics,
)
from scripts_asag.alice_label_remap import remap_predictions_to_original_alice_labels
from scripts_asag.data_processing.data_prep import MultiTaskDataPipeline, dedupe_keep_order
from scripts_asag.data_processing.benchmark_meta import BENCHMARK_DESCRIPTIONS
from modelling.modelling_utils import BackwardSupportedArguments


VALID_MODEL_CLASSES = ["span", "xnet"]


def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def normalize_cli_args(args):
    alias_map = {
        "--train-tasks": "--train_tasks",
        "--eval-tasks": "--eval_tasks",
        "--test-tasks": "--test_tasks",
        "--flip-levels": "--flip_levels",
        "--rub-shuffle": "--rub_shuffle",
    }
    return [alias_map.get(arg, arg) for arg in args]


def load_args_from_checkpoint(cp_dir, current_train_args, current_task_args):
    """
    Load training and task arguments from checkpoint directory.
    Preserves inference-specific flags and explicitly provided eval/test task overrides.
    """
    import json
    from copy import deepcopy

    args_path = os.path.join(cp_dir, "training_args.json")
    if not os.path.exists(args_path):
        parent_dir = os.path.dirname(cp_dir)
        args_path = os.path.join(parent_dir, "training_args.json")

    if not os.path.exists(args_path):
        print(f"Warning: training_args.json not found in {cp_dir} or its parent. Using current arguments.")
        return current_train_args, current_task_args

    print(f"Loading training arguments from {args_path}")
    with open(args_path, "r") as f:
        saved_args = json.load(f)

    updated_train_args = deepcopy(current_train_args)
    updated_task_args = deepcopy(current_task_args)

    inference_specific_train_args = {
        "test_only",
        "cp_dir",
        "save_dir",
        "save_attweights",
        "log_wandb",
        "dry_run",
        "multi_gpu",
    }
    preserve_task_args = {"test_drop_rub", "use_translated_prompts"}
    if current_task_args.eval_tasks and current_task_args.eval_tasks != current_task_args.train_tasks:
        preserve_task_args.add("eval_tasks")
    if current_task_args.test_tasks and current_task_args.test_tasks != current_task_args.train_tasks:
        preserve_task_args.add("test_tasks")

    for key, value in saved_args.items():
        if hasattr(updated_train_args, key) and key not in inference_specific_train_args:
            setattr(updated_train_args, key, value)
        elif hasattr(updated_task_args, key) and key not in preserve_task_args:
            setattr(updated_task_args, key, value)

    updated_task_args.__post_init__()
    return updated_train_args, updated_task_args


@dataclass
class TaskArguments:
    """Task/experiment related arguments dataclass for multi-dataset ASAG training."""

    base_model: str = field(default="bert-base-uncased", metadata={"help": "base model to use"})
    model_class: str = field(default="span", metadata={"help": "model class to use: span, xnet"})
    seed: int = field(default=114514, metadata={"help": "random seed for reproducibility"})
    train_frac: float = field(
        default=1.0,
        metadata={
            "help": (
                "training data usage per dataset: fraction when <= 1, "
                "or exact number of training instances when > 1 (must be integer-valued)"
            )
        },
    )
    train_tasks: list[str] = field(default_factory=lambda: ["alice_lp"], metadata={"help": "datasets to use for joint training"})
    eval_tasks: list[str] = field(default_factory=list, metadata={"help": "datasets to use for validation (eval); defaults to train_tasks"})
    test_tasks: list[str] = field(default_factory=list, metadata={"help": "datasets to use for test evaluation (e.g. zero-shot transfer); defaults to train_tasks"})
    dry_run: bool = field(default=False, metadata={"help": "whether to do a dry run for debugging"})
    add_suffix: bool = field(default=False, metadata={"help": "whether to add suffix to the input"})
    add_context: bool = field(default=False, metadata={"help": "whether to add context columns to the input"})
    random_suffix: bool = field(default=False, metadata={"help": "whether to randomly select suffix when multiple are available"})
    use_translated_prompts: bool = field(default=False, metadata={"help": "whether to use translated prompts (e.g., German for Alice)"})
    random_solution: bool = field(default=False, metadata={"help": "whether to use random sample solution from other questions"})
    test_drop_rub: float = field(default=0.0, metadata={"help": "probability of randomly dropping one non-gold rubric during span-model test evaluation"})
    train_drop_rub: float = field(default=0.0, metadata={"help": "probability of randomly dropping one non-gold rubric per training example (data-level regularization for span model)"})
    flip_levels: bool = field(default=False, metadata={"help": "reverse rubric level order for LASA/span inputs and restore outputs before saving"})
    drop_all_rubrics: bool = field(default=False, metadata={"help": "drop all rubrics from input during both training and testing (no-rubric baseline). Only compatible with model_class='span' and span_fuse_type='p-only'."})
    rub_shuffle: bool = field(default=False, metadata={"help": "sanity check: shuffle rubric texts while keeping level labels fixed, breaking rubric-label alignment. Only compatible with model_class='span'."})

    def __post_init__(self):
        assert self.train_frac > 0, "train_frac must be > 0"
        assert self.train_frac <= 1.0 or float(self.train_frac).is_integer(), (
            "train_frac > 1 must be an integer-valued exact number of training instances"
        )
        assert 0 <= self.test_drop_rub <= 1.0, "test_drop_rub must be between 0 and 1"
        assert 0 <= self.train_drop_rub <= 1.0, "train_drop_rub must be between 0 and 1"
        if not self.add_suffix:
            self.random_suffix = False
        assert self.model_class in VALID_MODEL_CLASSES, f"model_class must be one of {VALID_MODEL_CLASSES}"
        if self.flip_levels and self.model_class != "span":
            raise ValueError("flip_levels is only implemented for the LASA span model (model_class='span').")
        if self.rub_shuffle and self.model_class != "span":
            raise ValueError("rub_shuffle is only compatible with model_class='span'.")

        self.train_tasks = dedupe_keep_order(self.train_tasks)
        if not self.train_tasks:
            raise ValueError("train_tasks must contain at least one benchmark.")

        self.eval_tasks = dedupe_keep_order(self.eval_tasks) if self.eval_tasks else list(self.train_tasks)
        self.test_tasks = dedupe_keep_order(self.test_tasks) if self.test_tasks else list(self.train_tasks)

        invalid_tasks = [
            task
            for task in self.train_tasks + self.eval_tasks + self.test_tasks
            if task not in BENCHMARK_DESCRIPTIONS
        ]
        if invalid_tasks:
            valid_tasks = ", ".join(sorted(BENCHMARK_DESCRIPTIONS))
            raise ValueError(f"Unknown ASAG benchmark(s): {invalid_tasks}. Valid options: {valid_tasks}")


def validate_multitask_configuration(task_args, datappl, custom_model_args):
    if task_args.model_class != "span":
        return

    if custom_model_args.span_fuse_type != "p-only":
        return

    task_to_num_labels = {
        task: datappl.task_pipelines[task].num_labels
        for task in dedupe_keep_order(task_args.train_tasks + task_args.eval_tasks + task_args.test_tasks)
    }
    unique_num_labels = set(task_to_num_labels.values())
    if len(unique_num_labels) > 1:
        raise ValueError(
            "span_fuse_type='p-only' is incompatible with mixed label counts in train_multi.py. "
            f"Observed: {task_to_num_labels}"
        )
    datappl.num_labels = next(iter(unique_num_labels))


def parse_args(args=None):
    parser = HfArgumentParser((TaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    parsed_args = normalize_cli_args(sys.argv[1:] if args is None else args)
    return parser.parse_args_into_dataclasses(args=parsed_args)


def main(task_args: TaskArguments, train_args: AsagTrainingArguments, custom_model_args: BackwardSupportedArguments):
    if train_args.test_only and train_args.cp_dir:
        print(f"Loading arguments from checkpoint directory: {train_args.cp_dir}")
        train_args, task_args = load_args_from_checkpoint(train_args.cp_dir, train_args, task_args)
        print(f"After loading - Base model: {task_args.base_model}")
        print(f"After loading - Train tasks: {task_args.train_tasks}")
        print(f"After loading - Eval tasks: {task_args.eval_tasks}")
        print(f"After loading - Test tasks: {task_args.test_tasks}")

    if task_args.test_drop_rub > 0.0 and task_args.model_class != "span":
        raise ValueError("Test-time rubric dropping is only implemented for model_class='span'.")
    if task_args.train_drop_rub > 0.0 and task_args.model_class != "span":
        raise ValueError("Training-time rubric dropping is only implemented for model_class='span'.")

    if task_args.drop_all_rubrics and custom_model_args.span_fuse_type != "p-only":
        raise ValueError("drop_all_rubrics requires span_fuse_type='p-only' (rubric spans are not produced).")

    set_seed(task_args.seed)
    os.makedirs(train_args.save_dir, exist_ok=True)

    if train_args.log_wandb and is_main_process():
        wandb.login()
        wandb.init(
            config={**vars(train_args), **vars(task_args)},
            dir=train_args.save_dir,
            project="span-align-multi",
        )
    else:
        wandb.init(mode="disabled")

    print(f"Training arguments: {train_args}")
    print(f"Task arguments: {task_args}")

    datappl = MultiTaskDataPipeline(
        base_model=task_args.base_model,
        train_tasks=task_args.train_tasks,
        eval_tasks=task_args.eval_tasks,
        test_tasks=task_args.test_tasks,
        train_frac=task_args.train_frac,
        add_context=task_args.add_context,
        add_suffix=task_args.add_suffix,
        random_suffix=task_args.random_suffix,
        use_translated_prompts=task_args.use_translated_prompts,
        random_solution=task_args.random_solution,
        model_class=task_args.model_class,
        span_fuse_type=custom_model_args.span_fuse_type,
        test_drop_rub=task_args.test_drop_rub,
        train_drop_rub=task_args.train_drop_rub,
        flip_levels=task_args.flip_levels,
        rub_shuffle=task_args.rub_shuffle,
        seed=task_args.seed,
    )
    validate_multitask_configuration(task_args, datappl, custom_model_args)

    train_ds, val_ds, test_datasets = datappl.get_datasets(test_only=train_args.test_only)

    if datappl.train_sizes:
        print(f"Train dataset sizes by task: {datappl.train_sizes}")
        print(f"Validation dataset sizes by task (eval_tasks): {datappl.val_sizes}")
        print(f"Combined train dataset size: {len(train_ds)}")
        print(f"Combined val dataset size: {len(val_ds)}")
    print(f"Test dataset sizes by task (test_tasks): {datappl.test_sizes}")

    collate_fn = datappl.get_collate_fn()
    trainer = AsagTrainer(
        train_args,
        task_args,
        train_ds,
        val_ds,
        custom_model_args=custom_model_args,
        multi_gpu=train_args.multi_gpu,
        data_pipeline=datappl,
    )
    trainer.set_collate_fn(collate_fn)

    if task_args.dry_run:
        return

    if not train_args.test_only:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_ds)}")
        print(f"  Num Epochs = {train_args.max_epoch}")
        print(f"  Instantaneous batch size per GPU = {train_args.batch_size}")
        trainer.train()
        print("***** Training finished *****")

    if not is_main_process():
        return

    test_model = trainer.load_model()
    pred_dir = os.path.join(train_args.save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for dataset_key, test_ds in test_datasets.items():
        split_map = datappl.test_split_map if hasattr(datappl, "test_split_map") else datappl.eval_split_map
        task_name = split_map[dataset_key]["task"]
        split_name = split_map[dataset_key]["split"]

        print(f"***** Running evaluation on {task_name} / {split_name} *****")
        print(f"Num examples = {len(test_ds)}")

        eval_results = evaluate(
            test_model,
            test_ds,
            batch_size=train_args.batch_size,
            collate_fn=lambda x: trainer.collate_fn(x, return_meta=True),
            save_attweights=train_args.save_attweights,
        )

        if train_args.save_attweights and len(eval_results) == 3:
            test_predictions, test_loss, attention_weights = eval_results
        else:
            test_predictions, test_loss = eval_results
            attention_weights = None

        test_predictions, remap_summary = remap_predictions_to_original_alice_labels(
            test_predictions,
            benchmark=task_name,
        )
        if remap_summary["changed_rows"] > 0:
            print(
                f"Applied ALICE label remap for {dataset_key}: "
                f"rows={remap_summary['changed_rows']}, cells={remap_summary['changed_cells']}"
            )

        test_predictions.insert(0, "benchmark", task_name)
        test_predictions.insert(1, "split", split_name)

        pred_path_prefix = os.path.join(pred_dir, dataset_key)
        test_predictions.to_csv(f"{pred_path_prefix}_predictions.csv", index=False)

        if attention_weights is not None:
            attn_weights_path = os.path.join(train_args.save_dir, f"{dataset_key}_attention_weights.pt")
            torch.save(attention_weights, attn_weights_path)

        test_metrics = eval_report(test_predictions)
        test_metrics["loss"] = float(test_loss)
        save_report(test_metrics, f"{pred_path_prefix}_metrics.json")

        per_qid_results = per_qid_metrics(test_predictions)
        if per_qid_results is not None:
            save_report(per_qid_results, f"{pred_path_prefix}_per_question_metrics.json")

        wandb.log({dataset_key: test_metrics})

        print(f"***** {task_name} / {split_name} Results *****")
        for key, value in test_metrics.items():
            print(f"{key} = {value:.4f}")

    print("***** Training and evaluation completed *****")
    clear_gpu_memory()


if __name__ == "__main__":
    task_args, train_args, custom_model_args = parse_args()
    main(task_args, train_args, custom_model_args)
