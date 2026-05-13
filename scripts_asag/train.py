import time 
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
import wandb
import torch
from dataclasses import dataclass, field
from trainer import (
    AsagTrainer,
    AsagTrainingArguments,
)
from utils import (
    set_seed,
    eval_report,
    save_report,
    evaluate, 
    clear_gpu_memory,
    per_qid_metrics
)

from scripts_asag.alice_label_remap import remap_predictions_to_original_alice_labels
from scripts_asag.data_processing.data_prep import DataPipeline
from modelling.modelling_utils import BackwardSupportedArguments
from transformers import HfArgumentParser
import torch.distributed as dist


def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def normalize_cli_args(args):
    alias_map = {
        "--flip-levels": "--flip_levels",
        "--rub-shuffle": "--rub_shuffle",
        "--label-names-only": "--label_names_only",
    }
    return [alias_map.get(arg, arg) for arg in args]

def load_args_from_checkpoint(cp_dir, current_train_args, current_task_args):
    """
    Load training and task arguments from checkpoint directory.
    Preserves some current arguments that are specific to inference.
    """
    import json
    from copy import deepcopy
    
    # First try to find training_args.json in the checkpoint directory
    args_path = os.path.join(cp_dir, "training_args.json")
    
    # If not found, try the parent directory (common case for HuggingFace checkpoints)
    if not os.path.exists(args_path):
        parent_dir = os.path.dirname(cp_dir)
        args_path = os.path.join(parent_dir, "training_args.json")
    
    if not os.path.exists(args_path):
        print(f"Warning: training_args.json not found in {cp_dir} or its parent. Using current arguments.")
        return current_train_args, current_task_args
    
    print(f"Loading training arguments from {args_path}")
    
    with open(args_path, "r") as f:
        saved_args = json.load(f)
    
    # Create new instances of the argument classes
    updated_train_args = deepcopy(current_train_args)
    updated_task_args = deepcopy(current_task_args)
    
    # Update train_args with saved values, but preserve some inference-specific settings
    inference_specific_train_args = {
        'test_only', 'cp_dir', 'save_dir', 'save_attweights', 'attn_layer_idx',
        'attn_max_examples', 'multi_gpu', 'log_wandb', 'dry_run'
    }
    preserve_task_args = {'test_drop_rub'}
    
    for key, value in saved_args.items():
        # Check if this key belongs to train_args
        if hasattr(updated_train_args, key) and key not in inference_specific_train_args:
            setattr(updated_train_args, key, value)
        # Check if this key belongs to task_args  
        elif hasattr(updated_task_args, key) and key not in preserve_task_args:
            setattr(updated_task_args, key, value)

    updated_task_args.__post_init__()
    return updated_train_args, updated_task_args
@dataclass
class TaskArguments:
    """Task/experiment related arguments dataclass"""
    base_model: str = field(default='bert-base-uncased', metadata={"help": "base model to use"})
    model_class: str = field(default="span", metadata={"help": "model class to use: span, xnet"})
    seed: int = field(default=114514, metadata={"help": "random seed for reproducibility"})
    train_frac: float = field(
        default=1.0,
        metadata={
            "help": (
                "fraction of training data to use when <= 1, "
                "or exact number of training instances when > 1 (must be integer-valued)"
            )
        },
    )
    benchmark : str = field(default="alice_lp", metadata={"help": "name of the task (lp, ke, sk)"})
    dry_run: bool = field(default=False, metadata={"help": "whether to do a dry run for debugging"})
    # ---- dataset parameters ----
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
    label_names_only: bool = field(default=False, metadata={"help": "replace full rubric descriptions with unified label names only (e.g. Incorrect, Partially Correct, Correct). Not supported for asap_sas or pt_asag."})
    def __post_init__(self):
        """Validation checks after initialization"""
        assert self.train_frac > 0, "train_frac must be > 0"
        assert self.train_frac <= 1.0 or float(self.train_frac).is_integer(), (
            "train_frac > 1 must be an integer-valued exact number of training instances"
        )
        assert 0 <= self.test_drop_rub <= 1.0, "test_drop_rub must be between 0 and 1"
        assert 0 <= self.train_drop_rub <= 1.0, "train_drop_rub must be between 0 and 1"
        if not self.add_suffix:
            self.random_suffix = False
        valid_model_classes = ["span", "xnet"]
        assert self.model_class in valid_model_classes, f"model_class must be one of {valid_model_classes}"
        if self.flip_levels and self.model_class != "span":
            raise ValueError("flip_levels is only implemented for the LASA span model (model_class='span').")
        if self.rub_shuffle and self.model_class != "span":
            raise ValueError("rub_shuffle is only compatible with model_class='span'.")
        if self.label_names_only and self.benchmark in {"asap_sas", "pt_asag"}:
            raise ValueError(
                "label_names_only is not supported for asap_sas or pt_asag because their "
                "levels cannot be captured by the unified label-name scheme."
            )

def main(task_args: TaskArguments, train_args: AsagTrainingArguments, custom_model_args: BackwardSupportedArguments):
    # If test_only mode and cp_dir is specified, load training args from checkpoint BEFORE anything else
    if train_args.test_only and train_args.cp_dir:
        print(f"Loading arguments from checkpoint directory: {train_args.cp_dir}")
        train_args, task_args = load_args_from_checkpoint(train_args.cp_dir, train_args, task_args)
        print(f"After loading - Base model: {task_args.base_model}")
        print(f"After loading - Benchmark: {task_args.benchmark}")
    
    if task_args.test_drop_rub > 0.0 and task_args.model_class != "span":
        raise ValueError("Test-time rubric dropping is only implemented for model_class='span'.")
    if task_args.train_drop_rub > 0.0 and task_args.model_class != "span":
        raise ValueError("Training-time rubric dropping is only implemented for model_class='span'.")

    if task_args.drop_all_rubrics and custom_model_args.span_fuse_type != "p-only":
        raise ValueError("drop_all_rubrics requires span_fuse_type='p-only' (rubric spans are not produced).")

    set_seed(task_args.seed)
    if not os.path.exists(train_args.save_dir):
        os.makedirs(train_args.save_dir, exist_ok=True)

    
    if train_args.log_wandb and is_main_process():
        wandb.login()
        wandb.init(
            config={**vars(train_args), **vars(task_args)},
            dir=train_args.save_dir,
            project="span-align",
        )
    else:
        wandb.init(mode="disabled")
    
    print(f"Training arguments: {train_args}")
    print(f"Task arguments: {task_args}")
    
    # Initialize data pipeline with all configurations
    datappl = DataPipeline(
        base_model=task_args.base_model,
        benchmark=task_args.benchmark,
        train_frac=task_args.train_frac,
        add_context=task_args.add_context,
        add_suffix=task_args.add_suffix,
        random_suffix=task_args.random_suffix,
        use_translated_prompts=task_args.use_translated_prompts,
        random_solution=task_args.random_solution,
        model_class=task_args.model_class,
        test_drop_rub=task_args.test_drop_rub,
        train_drop_rub=task_args.train_drop_rub,
        flip_levels=task_args.flip_levels,
        drop_all_rubrics=task_args.drop_all_rubrics,
        rub_shuffle=task_args.rub_shuffle,
        label_names_only=task_args.label_names_only,
        seed=task_args.seed,
    )
    
    # Get datasets with encoding applied
    train_ds, val_ds, test_datasets = datappl.get_datasets(test_only=train_args.test_only
    )
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")

    # Get collate function
    collate_fn = datappl.get_collate_fn()
    train_args.save_dir = os.path.join(train_args.save_dir)
    trainer = AsagTrainer(train_args, 
                          task_args, 
                          train_ds, 
                          val_ds, 
                          custom_model_args=custom_model_args, 
                          multi_gpu=train_args.multi_gpu,
                          data_pipeline=datappl)
    
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
    
    # Evaluate on test datasets
    if not is_main_process():
        return
    test_model = trainer.load_model()
    
    # Handle both dict and direct attribute access
    test_datasets = test_datasets if isinstance(test_datasets, dict) else {"test": test_datasets}
    
    # Evaluate on all test splits
    for test_name, test_ds in test_datasets.items():
        print(f"***** Running evaluation on {test_name} *****")
        print(f"Num examples = {len(test_ds)}")
        
        # Call evaluate function with attention weights saving option
        eval_results = evaluate(
            test_model,
            test_ds,
            batch_size=train_args.batch_size,
            collate_fn=lambda x: trainer.collate_fn(x, return_meta=True),
            save_attweights=train_args.save_attweights,
            layer_idx=train_args.attn_layer_idx,
            attn_max_examples=train_args.attn_max_examples,
        )
        
        # Handle different return values based on whether attention weights were saved
        if train_args.save_attweights and len(eval_results) == 3:
            test_predictions, test_loss, attention_weights = eval_results
        else:
            test_predictions, test_loss = eval_results
            attention_weights = None
        
        test_predictions, remap_summary = remap_predictions_to_original_alice_labels(
            test_predictions,
            benchmark=task_args.benchmark,
        )
        if remap_summary["changed_rows"] > 0:
            print(
                f"Applied ALICE label remap for {test_name}: "
                f"rows={remap_summary['changed_rows']}, cells={remap_summary['changed_cells']}"
            )

        # Save predictions
        pred_dir = os.path.join(train_args.save_dir, "predictions")
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        test_predictions.to_csv(os.path.join(pred_dir, f"{test_name}_predictions.csv"), index=False)
        
        # Save attention weights if available
        if attention_weights is not None:
            print(f"Saving attention weights for {test_name}...")
            attn_weights_path = os.path.join(pred_dir, f"{test_name}_attention_weights.pt")
            torch.save(attention_weights, attn_weights_path)
            print(f"Attention weights saved to {attn_weights_path}")
        
        # Calculate and save metrics
        test_metrics = eval_report(test_predictions)
        save_report(test_metrics, os.path.join(pred_dir, f"{test_name}_metrics.json"))
        
        # Calculate per question ID metrics
        per_qid_results = per_qid_metrics(test_predictions)
        save_report(per_qid_results, os.path.join(pred_dir, f"{test_name}_per_question_metrics.json"))
        
        # Log metrics to wandb
        metrics_wandb = {f"{test_name}": test_metrics}
        wandb.log(metrics_wandb)
        
        print(f"***** {test_name} Results *****")
        for key, value in test_metrics.items():
            print(f"{key} = {value:.4f}")
    
    print("***** Training and evaluation completed *****")
    clear_gpu_memory()
if __name__ == "__main__":
    parser = HfArgumentParser((TaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    task_args, train_args, custom_model_args = parser.parse_args_into_dataclasses(
        args=normalize_cli_args(sys.argv[1:])
    )
    main(task_args, train_args, custom_model_args)
