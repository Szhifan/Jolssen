from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
import torch
from torch import nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import AutoModel
from torch import Tensor
import os
import logging
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)
logger.setLevel("INFO")

DECODER_MODEL_TYPES = tuple(['gpt2', 'llama', 'mistral', 'qwen2', 'phi3', 'olmo'])
ARCHITECTURES = tuple(['NONE', 'INPLACE', 'EXTEND', 'INTER', 'EXTRA'])

class Pooler:
    def __init__(self, pool_type, include_prompt=False):
        self.pool_type = pool_type
        self.include_prompt = include_prompt or self.pool_type in ("cls", "last")

    def __call__(
        self, 
        last_hidden_states: Tensor,
        attention_mask: Tensor,
        prompt_length: int = None,
    ) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1)
        batch_size = last_hidden_states.shape[0]
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        device = last_hidden_states.device
        
        if not self.include_prompt and prompt_length is not None:
            if left_padding:
                prompt_mask = torch.ones_like(attention_mask)
                range_tensor = torch.arange(attention_mask.size(1), 0, -1, device=device).unsqueeze(0)
                prompt_mask = (range_tensor > (sequence_lengths-prompt_length).unsqueeze(1))
                attention_mask[prompt_mask] = 0
            else:
                attention_mask[:, :prompt_length] = 0
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.pool_type == "avg":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
            attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            emb = s / d
        elif self.pool_type == "cls":
            emb = last_hidden[:, 0]
        elif self.pool_type == "last":
            if left_padding:
                emb = last_hidden[:, -1]
            else:
                emb = last_hidden[torch.arange(batch_size, device=device), sequence_lengths-1]
        else:
            raise ValueError(f"pool_type {self.pool_type} not supported")

        return emb
@dataclass
class BackwardSupportedArguments:
    # --- Model Architecture Arguments ---
    num_unsink_layers: float = field(
        default=0, 
        metadata={"help": "Number of layers to change to unsink attention."}
    )
    num_bidir_layers: float = field(
        default=0,
        metadata={"help": "Number of layers to change to bidirectional attention."}
    )

    # --- Model Pooling and Pruning Arguments ---
    pool_type: str = field(
        default="last", 
        metadata={"help": "Pooling type for the model output. Options: avg, weightedavg, cls, last"}
    )
    num_prune_layers: Optional[float] = field(
        default=0,
        metadata={"help": "Delete the top n layers of the model."}
    )
    num_fuse_layers: Optional[float] = field(
        default=0,
        metadata={"help": "Fuse the top n layers of the model into one embedding layer."}
    )
    fuse_type: Optional[str] = field(
        default="avg",
        metadata={"help": "Pooling type for the fused layer. Options: avg, weighted"}
    )
    span_pool_type: Optional[str] = field(
        default="last",
        metadata={"help": "Type of span pooling to use. Options: mean, last"}
    )
    span_fuse_type: Optional[str] = field(
        default="p-concat",
        metadata={"help": "Type of span fuser to use. Options: attention, bilinear"}
    ),
    num_labels: int = field(
        default=3,
        metadata={"help": "Number of labels for classification. Only used for p-only."}
    )



    def __post_init__(self):

        if self.pool_type not in ("avg", "weightedavg", "cls", "last"):
            self.pool_type = "last"
            print(f"Invalid pool_type {self.pool_type}, using default 'avg' pooling.")
        assert self.fuse_type in ("avg", "weighted"), "fuse_type must be one of ('avg', 'weighted')"
    def to_dict(self):
        return asdict(self)


def get_noncausal_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    if dtype is None:
        dtype = self.dtype

    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

def get_noncausal_attention_mask_0(self, attention_mask, input_shape, device=None, dtype=None):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    # assert attention_mask[:, 0].sum() == attention_mask.shape[0]
    assert self.config._attn_implementation != "flash_attention_2"
    # attention_mask[:, 0] = 0
    
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    if dtype is None:
        dtype = self.dtype

    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    
    extended_attention_mask = extended_attention_mask.repeat(1, 1, extended_attention_mask.shape[-1], 1)
    extended_attention_mask[:, :, 1:, 0] = 0

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

    return extended_attention_mask

def get_backward_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    output_attentions: bool,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask.flip(dims=(-1,))
        return None
    
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1]

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)

    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.flip(dims=(-2,-1))

    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    return causal_mask

def flip_tensor(tensor, flag=True):
    if flag:
        return tensor.flip(dims=(1,))
    else:
        return tensor



class BaseAsagModel(PreTrainedModel):
    """
    Base class for ASAG models with shared functionality:
    - LoRA support
    - Quantization support  
    - Custom save/load logic
    - Gradient checkpointing
    """
    
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config)
        self.config = config
        self.lora_config = lora_config
        self.bnb_config = bnb_config
        
        # Initialize encoder
        self._init_encoder()
        
    def _init_encoder(self):
        """Initialize the transformer encoder with optional quantization"""
        if self.bnb_config is not None:
            self.encoder = AutoModel.from_pretrained(
                self.config.base_model_name_or_path,
                quantization_config=self.bnb_config,
                config=self.config
            )
        else:
            self.encoder = AutoModel.from_pretrained(
                self.config.base_model_name_or_path,
                config=self.config
            )

    def init_peft(self, lora_config=None):
        """Initialize PEFT model"""
        from peft import get_peft_model
        
        if lora_config:
            self.lora_config = lora_config
        
        if self.lora_config and not hasattr(self.encoder, 'peft_config'):
            self.encoder = get_peft_model(self.encoder, self.lora_config)
            logger.info("Successfully initialized PEFT model")

    def _load_peft_adapter(self, ckpt_dir: str):
        """Load PEFT adapter weights"""
        from peft import PeftModel
        self.encoder = PeftModel.from_pretrained(self.encoder, ckpt_dir)

    @classmethod
    def from_pretrained(cls, model_path, config=None, lora_config=None, bnb_config=None, **kwargs):
        """
        Custom model loading logic that supports:
        1) Pure model (pytorch_model.bin)
        2) LoRA: adapter_model + non_peft_params.bin
        """
        # Create model instance
        model = cls(config, lora_config=lora_config, bnb_config=bnb_config)

        # Check for adapter files
        adapter_file_pt = os.path.join(model_path, "adapter_model.bin")
        adapter_file_st = os.path.join(model_path, "adapter_model.safetensors")
        has_adapter = os.path.exists(adapter_file_pt) or os.path.exists(adapter_file_st)

        non_peft_file = os.path.join(model_path, "non_peft_params.bin")
        full_file = os.path.join(model_path, "pytorch_model.bin")

        if lora_config:
            # LoRA model loading
            if has_adapter:
                model._load_peft_adapter(model_path)
                logger.info(f"[LoRA Load] Successfully loaded LoRA adapter from {model_path}")
            else:
                logger.warning(f"[LoRA Load] Adapter model not found in {model_path}")
            
            # Load non-PEFT parameters
            if os.path.exists(non_peft_file):
                non_peft_state = torch.load(non_peft_file, map_location="cpu")
                missing, unexpected = model.load_state_dict(non_peft_state, strict=False)
                if missing:
                    logger.warning(f"[LoRA Load] Missing non_peft parameters: {missing}")
                if unexpected:
                    logger.warning(f"[LoRA Load] Unexpected non_peft parameters: {unexpected}")
            else:
                logger.warning(f"[LoRA Load] non_peft_params.bin not found in {model_path}")
        else:
            # Non-LoRA: Load the full state dict
            if os.path.exists(full_file):
                full_state = torch.load(full_file, map_location="cpu")
                missing, unexpected = model.load_state_dict(full_state, strict=False)
                if missing:
                    logger.warning(f"[Full Load] Missing parameters: {missing}")
                if unexpected:
                    logger.warning(f"[Full Load] Unexpected parameters: {unexpected}")
            else:
                logger.error(f"[Full Load] {full_file} not found")
                
        return model

    def save_pretrained(self, save_path, **kwargs):
        """
        Custom save logic that handles both LoRA and full model saving
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_path)

        if hasattr(self.encoder, 'save_pretrained') and hasattr(self.encoder, 'peft_config'):
            # This is a PEFT model
            self.encoder.save_pretrained(save_path)
            
            # Save non-PEFT parameters
            full_state = self.state_dict()
            to_remove = [k for k in full_state.keys() if k.startswith("encoder.")]
            for k in to_remove:
                full_state.pop(k, None)
            torch.save(full_state, os.path.join(save_path, "non_peft_params.bin"))
            logger.info(f"Saved LoRA adapter and non-PEFT parameters to {save_path}")
        else:
            # Save full model
            torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            logger.info(f"Saved full model to {save_path}")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing"""
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.encoder.gradient_checkpointing_disable()

    def listwise_loss(
        self,
        logits: torch.Tensor,          # [B, R]
        rubric_mask: torch.Tensor,     # [B, R] in {0,1}
        pos_idx: torch.Tensor,         # [B] index of correct rubric
        tau: float = 1.0
    ) -> torch.Tensor:
        """
        Shared listwise loss for ranking rubrics.
        """
        from torch.nn import CrossEntropyLoss
        # Sanity: at least one valid rubric per example
        assert (rubric_mask.sum(dim=1) > 0).all(), "Every sample needs at least one valid rubric."

        # Temperature scaling
        scaled_logits = logits / tau

        # Ensure pos_idx refers to valid rubrics
        pos_mask = rubric_mask.gather(1, pos_idx.view(-1, 1)).squeeze(1)
        assert (pos_idx == -1).any() or (pos_mask == 1).all(), "pos_idx must refer to valid rubrics or be -1."

        return CrossEntropyLoss()(scaled_logits, pos_idx)

    

    def get_encoder_outputs(self, input_ids, attention_mask, token_type_ids=None):
        """Get encoder outputs with optional token type ids"""
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
            
        return self.encoder(**inputs)