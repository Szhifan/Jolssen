import os
import wandb
from dataclasses import dataclass, field
import torch
import torch.distributed as dist
from transformers import HfArgumentParser

from trainer import AsagTrainer, AsagTrainingArguments
from utils import (
    set_seed,
    eval_report,
    save_report,
    evaluate,
    clear_gpu_memory,
)
from data_processing_other.data_prep import DataPipelineOther
from modelling.modelling_utils import BackwardSupportedArguments


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def load_args_from_checkpoint(cp_dir, current_train_args, current_task_args):
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
        "test_only", "cp_dir", "save_dir", "save_attweights", "log_wandb", "dry_run"
    }

    for key, value in saved_args.items():
        if hasattr(updated_train_args, key) and key not in inference_specific_train_args:
            setattr(updated_train_args, key, value)
        elif hasattr(updated_task_args, key):
            setattr(updated_task_args, key, value)

    return updated_train_args, updated_task_args


@dataclass
class TaskArguments:
    base_model: str = field(default="bert-base-uncased", metadata={"help": "base model to use"})
    model_class: str = field(default="span", metadata={"help": "model class to use: span"})
    seed: int = field(default=114514, metadata={"help": "random seed for reproducibility"})
    train_frac: float = field(default=1.0, metadata={"help": "fraction of training data to use"})
    benchmark: str = field(default="xstance", metadata={"help": "name of the benchmark"})
    dry_run: bool = field(default=False, metadata={"help": "whether to do a dry run for debugging"})
    pairwise_margin: float = field(default=0.1, metadata={"help": "margin for pairwise ranking loss (unused for other)"})
    add_suffix: bool = field(default=False, metadata={"help": "whether to add suffix to the input"})
    add_context: bool = field(default=False, metadata={"help": "whether to add context columns to the input"})
    random_suffix: bool = field(default=False, metadata={"help": "whether to randomly select suffix when multiple are available"})
    use_translated_prompts: bool = field(default=False, metadata={"help": "use translated prompts when available (e.g., xstance)"})
    add_options: bool = field(default=True, metadata={"help": "include label options in the input (required for span model)"})

    def __post_init__(self):
        assert 0 < self.train_frac <= 1.0, "train_frac must be between 0 and 1"
        if not self.add_suffix:
            self.random_suffix = False
        if self.model_class != "span":
            raise ValueError("train_other currently supports model_class='span' only.")
        if not self.add_options:
            raise ValueError("add_options must be True for span-based encoding.")


def main(task_args: TaskArguments, train_args: AsagTrainingArguments, custom_model_args: BackwardSupportedArguments):
    if train_args.test_only and train_args.cp_dir:
        print(f"Loading arguments from checkpoint directory: {train_args.cp_dir}")
        train_args, task_args = load_args_from_checkpoint(train_args.cp_dir, train_args, task_args)
        print(f"After loading - Base model: {task_args.base_model}")
        print(f"After loading - Benchmark: {task_args.benchmark}")

    set_seed(task_args.seed)
    os.makedirs(train_args.save_dir, exist_ok=True)

    if train_args.log_wandb and is_main_process():
        wandb.login()
        wandb.init(
            config={**vars(train_args), **vars(task_args)},
            dir=train_args.save_dir,
            project="span-align-other",
        )
    else:
        wandb.init(mode="disabled")

    print(f"Training arguments: {train_args}")
    print(f"Task arguments: {task_args}")

    datappl = DataPipelineOther(
        base_model=task_args.base_model,
        benchmark=task_args.benchmark,
        train_frac=task_args.train_frac,
        add_context=task_args.add_context,
        add_suffix=task_args.add_suffix,
        random_suffix=task_args.random_suffix,
        use_translated_prompts=task_args.use_translated_prompts,
        model_class=task_args.model_class,
        add_options=task_args.add_options,
    )

    train_ds, val_ds, test_ds = datappl.get_datasets(test_only=train_args.test_only)
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")

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
    test_datasets = test_ds if isinstance(test_ds, dict) else {"test": test_ds}

    for test_name, split_ds in test_datasets.items():
        print(f"***** Running evaluation on {test_name} *****")
        print(f"Num examples = {len(split_ds)}")

        eval_results = evaluate(
            test_model,
            split_ds,
            batch_size=train_args.batch_size,
            collate_fn=lambda x: trainer.collate_fn(x, return_meta=True),
            save_attweights=train_args.save_attweights,
        )

        if train_args.save_attweights and len(eval_results) == 3:
            test_predictions, test_loss, attention_weights = eval_results
        else:
            test_predictions, test_loss = eval_results
            attention_weights = None

        pred_dir = os.path.join(train_args.save_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        test_predictions.to_csv(os.path.join(pred_dir, f"{test_name}_predictions.csv"), index=False)

        if attention_weights is not None:
            attn_weights_path = os.path.join(pred_dir, f"{test_name}_attention_weights.pt")
            torch.save(attention_weights, attn_weights_path)

        test_metrics = eval_report(test_predictions)
        save_report(test_metrics, os.path.join(pred_dir, f"{test_name}_metrics.json"))

        wandb.log({f"{test_name}": test_metrics})
        print(f"***** {test_name} Results *****")
        for key, value in test_metrics.items():
            print(f"{key} = {value:.4f}")

    print("***** Training and evaluation completed *****")
    clear_gpu_memory()


if __name__ == "__main__":
    parser = HfArgumentParser((TaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    task_args, train_args, custom_model_args = parser.parse_args_into_dataclasses()
    main(task_args, train_args, custom_model_args)
