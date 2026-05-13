from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoConfig,
    Trainer,
    TrainingArguments
)
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import torch
import os 
from dataclasses import dataclass, field
from peft import LoraConfig
import evaluate
import numpy as np
from accelerate import PartialState
from functools import partial
import tempfile
import shutil
import torch.distributed as dist
from modelling.modelling_span import SpanAlignmentModel
from scripts_asag.data_processing.data_prep import get_tokenizer

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# logger = logging.getLogger(__name__)
print("Using device:", DEFAULT_DEVICE)

# f1 = evaluate.load("f1")
acc = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # return f1.compute(predictions=predictions, references=labels, average="macro")
    return acc.compute(predictions=predictions, references=labels)

@dataclass
class AsagTrainingArguments:
    """Training arguments dataclass"""
    batch_size: int = field(default=16, metadata={"help": "maximum number of sentences in a batch"})
    max_epoch: int = field(default=3, metadata={"help": "force stop training at specified epoch"})
    clip_norm: float = field(default=1.0, metadata={"help": "clip threshold of gradients"})
    lr: float = field(default=None, metadata={"help": "global learning rate shortcut; if set, overrides lr1"})
    lr1: float = field(default=2e-5, metadata={"help": "learning rate for base model parameters (encoder.*)"})
    lr2: float = field(default=5e-5, metadata={"help": "learning rate for non-main modules (classification head, fusion layers, etc.)"})
    patience: int = field(default=3, metadata={"help": "number of epochs without improvement on validation set before early stopping"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "number of updates steps to accumulate before performing a backward/update pass"})
    weight_decay: float = field(default=0.01, metadata={"help": "weight decay for Adam"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "epsilon for Adam optimizer"})
    warmup_ratio: float = field(default=0.01, metadata={"help": "proportion of warmup steps"})
    save_dir: str = field(default="results/checkpoints", metadata={"help": "path to save checkpoints"})
    no_save: bool = field(default=False, metadata={"help": "don't save models or checkpoints"})
    cp_dir: str = field(default=None, metadata={"help": "path to the model checkpoint to load"})
    cp_dir_init: str = field(default=None, metadata={"help": "path to the model checkpoint to initialize from"})
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    test_only: bool = field(default=False, metadata={"help": "test model only"})
    bf16: bool = field(default=False, metadata={"help": "use 16-bit float precision instead of 32-bit"})
    log_wandb: bool = field(default=False, metadata={"help": "log experiment to wandb"})
    use_lora: bool = field(default=False, metadata={"help": "use LoRA for training"})
    use_bnb: bool = field(default=False, metadata={"help": "use 4-bit quantization for training"})
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    multi_gpu: bool = field(default=True, metadata={"help": "whether to use multiple GPUs"})
    save_attweights: bool = field(default=False, metadata={"help": "save attention weights during inference"})
    attn_layer_idx: int = field(default=-1, metadata={"help": "transformer layer index to save attention weights from; -1 means last layer"})
    attn_max_examples: int = field(default=0, metadata={"help": "maximum number of examples to save attention weights for; 0 means all examples"})
    def __post_init__(self):
        """Validation checks after initialization"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_epoch > 0, "max_epoch must be positive"
        if self.lr is not None:
            self.lr1 = self.lr
            self.lr2 = self.lr
        assert self.lr1 > 0, "lr1 must be positive"
        assert self.lr2 > 0, "lr2 must be positive"
        assert self.patience >= 0, "patience must be non-negative"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert 0 <= self.warmup_ratio <= 1, "warmup_ratio must be between 0 and 1"
        assert self.attn_max_examples >= 0, "attn_max_examples must be non-negative"
        if self.test_only:
            assert self.cp_dir is not None, "cp_dir must be specified in test_only mode"
        if self.multi_gpu:
            assert dist.is_available(), "Distributed package is not available, cannot use multi_gpu."
            dist.init_process_group(backend="nccl")



def print_trainable_parameters(model, use_4bit=False):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    
    # 添加量化状态检查
    quantized_layers = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        if hasattr(param, 'quant_state'):
            quantized_layers += 1

    if use_4bit:
        trainable_params /= 2
    trainable_params_int = int(trainable_params)
    
    print(f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params_int:,d} || Trainable Parameters %: {100 * trainable_params / all_param:.2f}")
    
    if quantized_layers > 0:
        print(f"Quantized layers detected: {quantized_layers}")
class ModelLoader:
    """Model loading and initialization utilities."""

    def __init__(self, task_args, train_args, custom_model_args=None, device_map="auto", data_pipeline=None):
        self.task_args = task_args
        self.train_args = train_args
        self.custom_model_args = custom_model_args
        self.device_map = device_map
        self.data_pipeline = data_pipeline
        
        self.lora_config = LoraConfig(
            r=self.train_args.lora_rank,
            lora_alpha=self.train_args.lora_alpha,
            lora_dropout=0.1,
            bias='none',
            target_modules="all-linear",
        )
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, # Activate 4-bit precision base model loading
            bnb_4bit_use_double_quant = True, # Activate nested quantization for 4-bit base models (double quantization)
            bnb_4bit_quant_type = "nf4",# Quantization type (fp4 or nf4)
            bnb_4bit_compute_dtype = torch.bfloat16, 
            )
        self.use_custom_model = "llama" in self.task_args.base_model
    def _update_with_custom_config(self, config):
        """
        Update autoconfig with backwardsarguments 
        """
        if self.custom_model_args is not None:
            # Skip custom config update in test_only mode to avoid serialization issues
            if self.train_args.test_only:
                print("Skipping custom_model_args update in test_only mode to avoid serialization issues")
            else:

                custom_dict = self.custom_model_args.to_dict()
                for key, value in custom_dict.items():
                    setattr(config, key, value)
        
        # Add num_labels from data pipeline if available
        if self.data_pipeline is not None:
            config.num_labels = self.data_pipeline.num_labels

        # Random rubric drop is applied in model-side listwise masking.
        config.test_drop_rub = float(getattr(self.task_args, "test_drop_rub", 0.0))

        return config
    def _init_model(self, use_lora=False, use_bnb=False, config=None):
        """
        Helping function to initialize the model
        """
        model = None
        
        config.base_model_name_or_path = self.task_args.base_model
        
        # Get model class type
        model_class = getattr(self.task_args, 'model_class', 'span')
        
        # Import the appropriate model class
        if model_class == "xnet":
            from modelling.modelling_xnet import AsagXnet
            
            # Set model-specific config
            config.model_class = model_class
            config.pool_type = getattr(self.custom_model_args, 'pool_type', 'avg') if self.custom_model_args else 'avg'
            
            model = AsagXnet(config,
                            lora_config=self.lora_config if use_lora else None,
                            bnb_config=self.bnb_config if use_bnb else None)
        else:
            # Default to SpanAlignmentModel
            from modelling.modelling_span import SpanAlignmentModel
            model = SpanAlignmentModel(config,
                                lora_config=self.lora_config if use_lora else None,
                                bnb_config=self.bnb_config if use_bnb else None)
        
        device_value = self.device_map['']
        if isinstance(device_value, int):
            device = torch.device(f"cuda:{device_value}" if torch.cuda.is_available() else DEFAULT_DEVICE)
        elif isinstance(device_value, str) and device_value != 'cpu':
            device = torch.device(device_value if torch.cuda.is_available() else DEFAULT_DEVICE)
        else:
            device = torch.device(DEFAULT_DEVICE)
        model = model.to(device)
            
     
        os.makedirs(self.train_args.save_dir, exist_ok=True)
        config.save_pretrained(self.train_args.save_dir)
        return model

    def init_model(self):
        """
        The main function to initialize the model 
        """
        # If we're in test_only mode and have a checkpoint directory, load config from there
        if self.train_args.test_only and self.train_args.cp_dir:
            print(f"Loading config from checkpoint: {self.train_args.cp_dir}")
            config = AutoConfig.from_pretrained(self.train_args.cp_dir, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(self.task_args.base_model, trust_remote_code=True)

        config = self._update_with_custom_config(config)
        self.train_args.use_bnb = (self.train_args.use_bnb and torch.cuda.is_available()) 
        self.train_args.use_lora = (self.train_args.use_lora and torch.cuda.is_available()) 
        
        if self.train_args.cp_dir_init:
            print(f"Initializing model from checkpoint: {self.train_args.cp_dir_init}")
            config = AutoConfig.from_pretrained(self.train_args.cp_dir_init, trust_remote_code=True)
            assert self.task_args.base_model.lower() == config.base_model_name_or_path.lower(), \
                "Base model in training arguments must match the model used in the checkpoint."
            model = self._init_model_from_cp(self.train_args.cp_dir_init)
        else:
            model = self._init_model(
                use_lora=self.train_args.use_lora,
                use_bnb=self.train_args.use_bnb,
                config=config
            )
        # save config for future reference
        os.makedirs(self.train_args.save_dir, exist_ok=True)
        model.config.save_pretrained(self.train_args.save_dir)
        model = self._init_peft_model(model)
        model = self._apply_freeze_policy(model)
        return model
            
    def _init_peft_model_from_cp(self, cp_path: str):
        """
        Initialize a PEFT model from a checkpoint.
        1. Load the pretrained model from model id
        2. Wrap it with Peft and apply merge_and_unload.
        3. If quantization is requested:
            3a. Save the full merged model in a temporary directory.
            3b. Reload the model with quantization.
        """
        print(f"Initializing quantized PEFT model from checkpoint: {cp_path}")
        
        # Read config from checkpoint path
        config = AutoConfig.from_pretrained(cp_path, trust_remote_code=True)

        base_model = self._init_model(
            use_lora=False,
            use_bnb=False,
            config=config
        )
        # Step 2: Load PEFT weights and merge
        print("Loading PEFT adapter and merging with base model...")
        peft_model = base_model._load_peft_adapter(str(cp_path) + '/')
        peft_model.encoder = peft_model.encoder.merge_and_unload()

        
        # Step 3: If quantization is requested, save and reload with quantization
        if self.train_args.use_bnb:
            print("Applying quantization to merged model...")
            
            # Create temporary directory to save merged model
            temp_dir = tempfile.mkdtemp(dir="/pfss/mlde/workspaces/mlde_wsp_ALICE_ASAS")
            try:
                # Save the merged model temporarily
                peft_model.save_pretrained(temp_dir)
                # Also save the config
                config.save_pretrained(temp_dir)
                
                # Reload with quantization
                final_model = self._init_model(
                    use_lora=False,
                    use_bnb=True,
                    config=config
                )
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
        else:
            final_model = peft_model

        return final_model
       
    def _init_model_from_cp(self, cp_path: str):
        """Initialize model from checkpoint path."""
        is_peft = os.path.exists(os.path.join(cp_path, "adapter_config.json")) 
        if is_peft:
            model = self._init_peft_model_from_cp(
                cp_path=cp_path
            )
        else:
            model = self.load_model(cp_path, use_lora=self.train_args.use_lora)
        return model
    def _init_peft_model(self, model):
        """Wrap the model with LoRA."""
        if not self.train_args.use_lora:
            return model
        model.init_peft(self.lora_config)
        print_trainable_parameters(model, use_4bit=self.train_args.use_bnb)
        return model

    def _apply_freeze_policy(self, model):
        """Apply encoder-freezing policy after optional LoRA wrapping."""
        freeze_base = bool(getattr(self.custom_model_args, 'freeze_base_encoder', False))
        frozen_layers = float(getattr(self.custom_model_args, 'frozen_layers', 0.0) or 0.0)
        if not freeze_base and frozen_layers <= 0.0:
            return model

        if freeze_base and frozen_layers > 0.0:
            print("freeze_base_encoder=True overrides frozen_layers; freezing the full encoder.")

        if freeze_base and hasattr(model, 'freeze_encoder'):
            model.freeze_encoder()
        elif freeze_base and hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif frozen_layers > 0.0:
            if not hasattr(model, 'freeze_lm_backbone_layers'):
                raise ValueError(f"Model type {type(model).__name__} does not support frozen_layers.")
            num_layers, num_embedding_modules = model.freeze_lm_backbone_layers(frozen_layers)
            print(
                f"frozen_layers={frozen_layers}: froze {num_embedding_modules} embedding module(s) "
                f"and {num_layers} bottom transformer layer(s)."
            )

        if freeze_base:
            print("freeze_base_encoder=True: froze all encoder parameters (including LoRA adapters if present).")
        print_trainable_parameters(model, use_4bit=self.train_args.use_bnb)
        return model

    def load_model(self, cp_path: str, use_lora=False):
        """
        Load a model from a checkpoint path, with or without LoRA (PEFT).
        :param cp_path: Path to the model checkpoint.
        :param use_lora: Whether to load the model with LoRA (PEFT).
        :return: Loaded model.
        """
        cp_path = str(cp_path)
        config = AutoConfig.from_pretrained(cp_path, trust_remote_code=True)
        bnb_config = self.bnb_config if self.train_args.use_bnb else None
        model_class = getattr(self.task_args, 'model_class', 'span')
        if model_class == "xnet":
            from modelling.modelling_xnet import AsagXnet
            model = AsagXnet.from_pretrained(
                cp_path,
                config=config,
                lora_config=self.lora_config if use_lora else None,
                bnb_config=bnb_config
            )
        else:
            model = SpanAlignmentModel.from_pretrained(
                cp_path,
                config=config,
                lora_config=self.lora_config if use_lora else None,
                bnb_config=bnb_config
            )
        model = self._apply_freeze_policy(model)
        return model

class LoraAwareTrainer(Trainer):
    def _load_best_model(self):
        best_ckpt = self.state.best_model_checkpoint
        if best_ckpt is None:
            return
        if os.path.exists(os.path.join(best_ckpt, "adapter_config.json")):
            from peft import PeftModel
            encoder = self.model.encoder
            if not isinstance(encoder, PeftModel):
                raise RuntimeError("Expected encoder to be a PeftModel but got: " + type(encoder).__name__)
            encoder.load_adapter(best_ckpt, adapter_name="default")
            encoder.set_adapter("default")

            non_peft_file = os.path.join(best_ckpt, "non_peft_params.bin")
            if os.path.exists(non_peft_file):
                non_peft_state = torch.load(non_peft_file, map_location="cpu")
                self.model.load_state_dict(non_peft_state, strict=False)
            else:
                print(f"[LoRA Load] non_peft_params.bin not found in best checkpoint: {best_ckpt}")
            return
        super()._load_best_model()


class AsagTrainer:
    """
    Trainer class for training and evaluating the AsagXNet, AsagSNet, or AsagXNetLlama models.
    """
    def __init__(self, train_args, task_args, train_dataset, validation_dataset=None, custom_model_args=None, multi_gpu=False, data_pipeline=None):
        self.train_args = train_args
        self.task_args = task_args
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        
        if multi_gpu:
            device_string = PartialState().process_index
            device_map = {'': device_string}
        else:
            device_map = {'': 0} if torch.cuda.is_available() else {'': 'cpu'}
        self.model_loader = ModelLoader(self.task_args, self.train_args, custom_model_args=custom_model_args, device_map=device_map, data_pipeline=data_pipeline)
        self.model = self.model_loader.init_model()
        self.tokenizer = get_tokenizer(self.task_args.base_model)
        self.multi_gpu = multi_gpu
        self.is_llm = "llama" in self.task_args.base_model or "mistral" in self.task_args.base_model or "mmbert" in self.task_args.base_model.lower()

        # Only save args if not in test_only mode
        if not self.train_args.test_only:
            all_args = {**vars(self.train_args), **vars(self.task_args)}
            with open(os.path.join(self.train_args.save_dir, "training_args.json"), "w") as f:
                json.dump(all_args, f, indent=4)
    
    def load_model(self):
        cp_path = self.train_args.cp_dir if self.train_args.cp_dir else self.train_args.save_dir
        print(f"Loading model from checkpoint: {cp_path}")
        if not cp_path:
            return self.model
        return self.model_loader.load_model(cp_path, use_lora=self.train_args.use_lora)
    def set_collate_fn(self, collate_fn, fc_kwargs=None):
        """Set the data collate function."""
        collate_fn = partial(collate_fn, **(fc_kwargs or {}))
        self.collate_fn = collate_fn
        
    def train(self):
        print("Starting training...")
        effective_lr_main = self.train_args.lr1
        effective_lr_other = self.train_args.lr2
        print(f"Using split learning rates: lr1(base encoder)={effective_lr_main}, lr2(other modules)={effective_lr_other}")

        train_args = TrainingArguments(
            # optimization parameters
            num_train_epochs=self.train_args.max_epoch,
            per_device_train_batch_size=self.train_args.batch_size,
            gradient_accumulation_steps=self.train_args.gradient_accumulation_steps,
            learning_rate=effective_lr_main,
            weight_decay=self.train_args.weight_decay,
            max_grad_norm=self.train_args.clip_norm,
            warmup_ratio=self.train_args.warmup_ratio,
            bf16=self.train_args.bf16,
            lr_scheduler_type="cosine",
            optim="paged_adamw_32bit" if self.is_llm else "adamw_torch",
            remove_unused_columns=False,
            gradient_checkpointing=True if self.is_llm else False,
            gradient_checkpointing_kwargs = {"use_reentrant": False} if self.is_llm else None,
            save_total_limit=2,
            # logging and saving parameters
            label_names=["labels"],
            greater_is_better=True,
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            logging_dir=os.path.join(self.train_args.save_dir, "logs"),
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            output_dir=self.train_args.save_dir,
        )

        trainer = LoraAwareTrainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            data_collator=self.collate_fn,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(self.train_args.save_dir)

        return
