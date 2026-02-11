import time 
import os
import wandb
from dataclasses import dataclass, field
from typing import List
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

from data_processing_asag.data_prep import DataPipeline
from modelling.modelling_utils import BackwardSupportedArguments
from transformers import HfArgumentParser
import torch.distributed as dist
from datasets import concatenate_datasets


def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


@dataclass
class MultiTaskArguments:
    """Task/experiment related arguments dataclass for multi-task learning"""
    base_model: str = field(default='bert-base-uncased', metadata={"help": "base model to use"})
    seed: int = field(default=114514, metadata={"help": "random seed for reproducibility"})
    train_frac: float = field(default=1.0, metadata={"help": "fraction of training data to use"})
    benchmarks: str = field(default="alice_lp,beetle", metadata={"help": "comma-separated list of benchmarks to train on jointly"})
    dry_run: bool = field(default=False, metadata={"help": "whether to do a dry run for debugging"})
    # ---- dataset parameters ----
    add_suffix: bool = field(default=False, metadata={"help": "whether to add suffix to the input"})
    add_context: bool = field(default=True, metadata={"help": "whether to add context columns to the input"})
    random_suffix: bool = field(default=False, metadata={"help": "whether to randomly select suffix when multiple are available"})
    use_translated_prompts: bool = field(default=False, metadata={"help": "whether to use translated prompts (e.g., German for Alice)"})
    random_solution: bool = field(default=False, metadata={"help": "whether to use random sample solution from other questions"})
    
    def __post_init__(self):
        """Validation checks after initialization"""
        assert 0 < self.train_frac <= 1.0, "train_frac must be between 0 and 1"
        if not self.add_suffix:
            self.random_suffix = False
        # Parse benchmarks into a list
        self.benchmark_list = [b.strip() for b in self.benchmarks.split(",")]
        assert len(self.benchmark_list) > 0, "At least one benchmark must be specified"


class MultiTaskDataPipeline:
    """
    Multi-task data pipeline that combines datasets from multiple benchmarks.
    Uses a single shared encoding pipeline for consistency.
    """
    
    def __init__(
        self,
        base_model: str,
        benchmark_list: List[str],
        train_frac: float = 1.0,
        add_context: bool = True,
        add_suffix: bool = False,
        random_suffix: bool = False,
        random_solution: bool = False,
        use_translated_prompts: bool = False,
    ):
        self.base_model = base_model
        self.benchmark_list = benchmark_list
        self.train_frac = train_frac
        self.add_context = add_context
        self.add_suffix = add_suffix
        self.random_suffix = random_suffix
        self.random_solution = random_solution
        self.use_translated_prompts = use_translated_prompts
        
        # Initialize individual pipelines for each benchmark
        self.pipelines = {}
        for benchmark in benchmark_list:
            self.pipelines[benchmark] = DataPipeline(
                base_model=base_model,
                benchmark=benchmark,
                train_frac=train_frac,
                add_context=add_context,
                add_suffix=add_suffix,
                random_suffix=random_suffix,
                use_translated_prompts=use_translated_prompts,
                random_solution=random_solution
            )
        
        # Use the first pipeline's tokenizer and collate function
        self.tokenizer = self.pipelines[benchmark_list[0]].tokenizer
        self.pad_token_id = self.pipelines[benchmark_list[0]].pad_token_id
    
    def get_datasets(self, apply_encoding=True):
        """
        Get combined train/val datasets and individual test datasets.
        
        Args:
            apply_encoding: Whether to apply encoding function to datasets
            
        Returns:
            tuple: (combined_train_dataset, combined_val_dataset, test_datasets_dict)
                   test_datasets_dict maps benchmark -> {test_name -> test_dataset}
        """
        all_train_datasets = []
        all_val_datasets = []
        all_test_datasets = {}  # benchmark -> {test_name -> dataset}
        
        for benchmark in self.benchmark_list:
            pipeline = self.pipelines[benchmark]
            train_ds, val_ds, test_ds = pipeline.get_datasets(apply_encoding=apply_encoding)
            
            # Add benchmark identifier to each example for tracking
            def add_benchmark_id(example, benchmark_name):
                example["benchmark"] = benchmark_name
                return example
            
            train_ds = train_ds.map(lambda x: add_benchmark_id(x, benchmark))
            val_ds = val_ds.map(lambda x: add_benchmark_id(x, benchmark))
            
            all_train_datasets.append(train_ds)
            all_val_datasets.append(val_ds)
            
            # Store test datasets per benchmark
            if isinstance(test_ds, dict):
                all_test_datasets[benchmark] = {
                    name: ds.map(lambda x: add_benchmark_id(x, benchmark)) 
                    for name, ds in test_ds.items()
                }
            else:
                all_test_datasets[benchmark] = {
                    "test": test_ds.map(lambda x: add_benchmark_id(x, benchmark))
                }
        
        # Concatenate train and val datasets
        combined_train = concatenate_datasets(all_train_datasets)
        combined_val = concatenate_datasets(all_val_datasets)
        
        # Shuffle the combined datasets
        combined_train = combined_train.shuffle(seed=42)
        combined_val = combined_val.shuffle(seed=42)
        
        return combined_train, combined_val, all_test_datasets
    
    def get_collate_fn(self):
        """Get the collate function from the first pipeline."""
        return self.pipelines[self.benchmark_list[0]].get_collate_fn()


def main(task_args: MultiTaskArguments, train_args: AsagTrainingArguments, custom_model_args: BackwardSupportedArguments):
    set_seed(task_args.seed)
    if not os.path.exists(train_args.save_dir):
        os.makedirs(train_args.save_dir)

    if train_args.log_wandb and is_main_process():
        wandb.login()
        wandb.init(
            config={**vars(train_args), **vars(task_args)},
            dir=train_args.save_dir,
            project="span-align-multitask",
        )
    else:
        wandb.init(mode="disabled")
    
    print(f"Training arguments: {train_args}")
    print(f"Task arguments: {task_args}")
    print(f"Benchmarks for joint training: {task_args.benchmark_list}")
    
    # Initialize multi-task data pipeline
    multitask_pipeline = MultiTaskDataPipeline(
        base_model=task_args.base_model,
        benchmark_list=task_args.benchmark_list,
        train_frac=task_args.train_frac,
        add_context=task_args.add_context,
        add_suffix=task_args.add_suffix,
        random_suffix=task_args.random_suffix,
        use_translated_prompts=task_args.use_translated_prompts,
        random_solution=task_args.random_solution
    )
    
    # Get combined datasets for training and individual test datasets
    train_ds, val_ds, test_datasets_by_benchmark = multitask_pipeline.get_datasets(
        apply_encoding=not train_args.test_only
    )
    
    print(f"Combined train dataset size: {len(train_ds)}")
    print(f"Combined val dataset size: {len(val_ds)}")
    for benchmark, test_dict in test_datasets_by_benchmark.items():
        for test_name, test_ds in test_dict.items():
            print(f"  {benchmark}/{test_name}: {len(test_ds)} examples")
    
    # Get collate function
    collate_fn = multitask_pipeline.get_collate_fn()
    
    # Create a mock task_args with benchmark field for trainer compatibility
    class TrainerTaskArgs:
        def __init__(self, multi_task_args):
            self.base_model = multi_task_args.base_model
            self.seed = multi_task_args.seed
            self.train_frac = multi_task_args.train_frac
            self.benchmark = "multitask_" + "_".join(multi_task_args.benchmark_list)
            self.dry_run = multi_task_args.dry_run
            self.add_suffix = multi_task_args.add_suffix
            self.add_context = multi_task_args.add_context
            self.random_suffix = multi_task_args.random_suffix
            self.use_translated_prompts = multi_task_args.use_translated_prompts
            self.random_solution = multi_task_args.random_solution
    
    trainer_task_args = TrainerTaskArgs(task_args)
    
    train_args.save_dir = os.path.join(train_args.save_dir)
    trainer = AsagTrainer(
        train_args, 
        trainer_task_args, 
        train_ds, 
        val_ds, 
        custom_model_args=custom_model_args, 
        multi_gpu=train_args.multi_gpu
    )
    trainer.set_collate_fn(collate_fn)
    
    if task_args.dry_run:
        return
    
    if not train_args.test_only:
        print("***** Running multi-task training *****")
        print(f"  Num examples = {len(train_ds)}")
        print(f"  Num Epochs = {train_args.max_epoch}")
        print(f"  Instantaneous batch size per GPU = {train_args.batch_size}")
        trainer.train()
        print("***** Training finished *****")
    
    # Evaluate on test datasets
    if not is_main_process():
        return
    
    test_model = trainer.load_model()
    
    # Evaluate on all test splits for each benchmark
    all_metrics = {}
    for benchmark, test_dict in test_datasets_by_benchmark.items():
        print(f"\n***** Evaluating benchmark: {benchmark} *****")
        benchmark_metrics = {}
        
        for test_name, test_ds in test_dict.items():
            print(f"***** Running evaluation on {benchmark}/{test_name} *****")
            print(f"Num examples = {len(test_ds)}")
            
            test_predictions, test_loss = evaluate(
                test_model,
                test_ds,
                batch_size=train_args.batch_size,
                collate_fn=lambda x: trainer.collate_fn(x, return_meta=True)
            )
            
            # Save predictions
            pred_dir = os.path.join(train_args.save_dir, "predictions", benchmark)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            test_predictions.to_csv(os.path.join(pred_dir, f"{test_name}_predictions.csv"), index=False)
            
            # Calculate and save metrics
            test_metrics = eval_report(test_predictions)
            save_report(test_metrics, os.path.join(pred_dir, f"{test_name}_metrics.json"))
            
            # Calculate per question ID metrics
            per_qid_results = per_qid_metrics(test_predictions)
            save_report(per_qid_results, os.path.join(pred_dir, f"{test_name}_per_question_metrics.json"))
            
            # Store metrics for logging
            benchmark_metrics[test_name] = test_metrics
            
            print(f"***** {benchmark}/{test_name} Results *****")
            for key, value in test_metrics.items():
                print(f"{key} = {value:.4f}")
        
        all_metrics[benchmark] = benchmark_metrics
    
    # Log all metrics to wandb
    wandb_metrics = {}
    for benchmark, test_dict in all_metrics.items():
        for test_name, metrics in test_dict.items():
            for metric_name, value in metrics.items():
                wandb_metrics[f"{benchmark}/{test_name}/{metric_name}"] = value
    wandb.log(wandb_metrics)
    
    # Save combined metrics summary
    summary_path = os.path.join(train_args.save_dir, "predictions", "all_metrics_summary.json")
    save_report(all_metrics, summary_path)
    
    print("\n***** Multi-task training and evaluation completed *****")
    print(f"Results saved to {train_args.save_dir}/predictions/")
    clear_gpu_memory()


if __name__ == "__main__":
    parser = HfArgumentParser((MultiTaskArguments, AsagTrainingArguments, BackwardSupportedArguments))
    task_args, train_args, custom_model_args = parser.parse_args_into_dataclasses()
    main(task_args, train_args, custom_model_args)
