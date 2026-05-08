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

def normalize_layer_count(value, total_layers, name, min_ratio_layers=1):
    """
    Interpret layer-count options consistently.

    Values in (0, 1) are ratios of total_layers. Values >= 1 must be integral
    counts, including floats parsed from CLI/config values such as 4.0.
    """
    if value is None:
        return 0

    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    if value == 0:
        return 0
    if value < 1:
        return max(min_ratio_layers, int(total_layers * value))
    if float(value).is_integer():
        return int(value)

    raise ValueError(
        f"{name} must be 0, a ratio in (0, 1), or an integer layer count; got {value!r}."
    )

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
    frozen_layers: float = field(
        default=0.0,
        metadata={
            "help": (
                "Freeze the bottom part of the LM backbone. Values between 0 and 1 freeze that "
                "portion of transformer layers; values >= 1 freeze that many bottom transformer "
                "layers. Embeddings are also frozen when this is > 0."
            )
        }
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
    )
    use_proj: bool = field(
        default=False,
        metadata={"help": "Whether to use projection layers for span embeddings."}
    )
    num_labels: int = field(
        default=3,
        metadata={"help": "Number of labels for classification. Only used for p-only."}
    )
    cl_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for rubric contrastive loss. 0 means no contrastive loss."}
    )
    rubric_independent_attn: bool = field(
        default=False,
        metadata={
            "help": (
                "Block-sparse attention: rubric tokens attend to context and their own tokens only, "
                "not to other rubrics."
            )
        },
    )
    reindex_rub: bool = field(
        default=False,
        metadata={
            "help": (
                "Reset position IDs per rubric so every rubric sees the same distance to context "
                "(matches xnet's per-pair positional distribution). Only valid when "
                "--rubric-independent-attn is set."
            )
        },
    )

    def __post_init__(self):
        if self.rubric_independent_attn and self.span_fuse_type != "l-only":
            raise ValueError("--rubric-independent-attn is only compatible with --span-fuse-type l-only.")
        if self.reindex_rub and not self.rubric_independent_attn:
            raise ValueError("--reindex-rub requires --rubric-independent-attn to be set.")

        if self.pool_type not in ("avg", "weightedavg", "cls", "last"):
            self.pool_type = "last"
            print(f"Invalid pool_type {self.pool_type}, using default 'avg' pooling.")
        if self.span_pool_type not in ("mean", "last"):
            raise ValueError("span_pool_type must be one of ('mean', 'last').")
        if self.num_labels < 1:
            raise ValueError("num_labels must be >= 1.")
        if self.cl_weight < 0.0:
            raise ValueError("cl_weight must be >= 0.")
        if self.frozen_layers < 0.0:
            raise ValueError("frozen_layers must be >= 0.")

    def to_dict(self):
        """Convert to dictionary using vars() to avoid deepcopy issues."""
        return {k: v for k, v in vars(self).items()}


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
    if attention_mask is None:
        return None

    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    if dtype is None:
        dtype = self.dtype

    is_decoder = getattr(self.config, "is_decoder", True)

    if not (attention_mask.dim() == 2 and is_decoder):
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
    if attention_mask is None:
        return None

    # assert attention_mask[:, 0].sum() == attention_mask.shape[0]
    assert self.config._attn_implementation != "flash_attention_2"
    # attention_mask[:, 0] = 0
    
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    if dtype is None:
        dtype = self.dtype

    is_decoder = getattr(self.config, "is_decoder", True)

    if not (attention_mask.dim() == 2 and is_decoder):
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


def _rubric_spans_for_example(
    rubric_spans: torch.Tensor,
    batch_idx: int,
    rubric_mask: Optional[torch.Tensor] = None,
    rubric_example_indices: Optional[torch.Tensor] = None,
    rubric_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return valid rubric spans for one example from flat or legacy padded layout."""
    if rubric_spans.dim() == 3:
        if rubric_mask is None:
            return rubric_spans[batch_idx]
        return rubric_spans[batch_idx][rubric_mask[batch_idx].bool()]

    if rubric_spans.dim() != 2:
        raise ValueError("rubric_spans must be [N, 2] or [B, R, 2].")
    if rubric_example_indices is None:
        raise ValueError("rubric_example_indices is required when rubric_spans is flat.")

    valid = rubric_example_indices == batch_idx
    spans = rubric_spans[valid]
    if rubric_indices is not None and spans.shape[0] > 1:
        order = torch.argsort(rubric_indices[valid])
        spans = spans[order]
    return spans


def build_rubric_block_attention_mask(
    attention_mask: torch.Tensor,   # [B, T]  binary padding mask (1=real, 0=pad)
    rubric_spans: torch.Tensor,     # [N, 2] or [B, R, 2] token spans, end is exclusive
    rubric_mask: Optional[torch.Tensor] = None,  # legacy [B, R]  1=valid rubric
    dtype: Optional[torch.dtype] = None,
    causal: bool = True,
    rubric_example_indices: Optional[torch.Tensor] = None,
    rubric_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:                  # [B, 1, T, T]  additive attention mask
    """
    Build a block-sparse 4-D attention mask so that each rubric's tokens attend
    only to the context (everything before the first rubric) and to their own
    tokens — never to other rubrics.

    By default, the mask preserves autoregressive behavior: context tokens can
    attend only to previous context tokens, and rubric tokens can attend to
    previous context tokens plus previous tokens in their own rubric span.
    Set causal=False for the older bidirectional block behavior.

    Returned tensor is additive (0.0 for allowed pairs, finfo.min for blocked),
    ready to be added directly to raw attention scores.
    """
    if isinstance(rubric_mask, torch.dtype) and dtype is None:
        dtype = rubric_mask
        rubric_mask = None
    if dtype is None:
        dtype = attention_mask.dtype if attention_mask.dtype.is_floating_point else torch.float32

    B, T = attention_mask.shape
    device = attention_mask.device
    min_val = torch.finfo(dtype).min

    # Build per-sample 2-D boolean allow-masks, then stack.
    masks = []
    for b in range(B):
        spans = _rubric_spans_for_example(
            rubric_spans,
            b,
            rubric_mask=rubric_mask,
            rubric_example_indices=rubric_example_indices,
            rubric_indices=rubric_indices,
        )

        # Context = everything before the first valid rubric span.
        if spans.shape[0] > 0:
            ctx_end = int(spans[0, 0])  # start of first rubric
        else:
            ctx_end = T  # no rubrics → full attention

        allow = torch.zeros(T, T, dtype=torch.bool, device=device)

        # Context tokens attend within context.
        if causal:
            allow[:ctx_end, :ctx_end] = torch.tril(
                torch.ones(ctx_end, ctx_end, dtype=torch.bool, device=device)
            )
        else:
            allow[:ctx_end, :ctx_end] = True

        # Each valid rubric: attend to context + own tokens, never other rubrics.
        for span in spans:
            s, e = int(span[0]), int(span[1])
            allow[s:e, :ctx_end] = True  # rubric → context
            if causal:
                rub_len = e - s
                allow[s:e, s:e] = torch.tril(
                    torch.ones(rub_len, rub_len, dtype=torch.bool, device=device)
                )
            else:
                allow[s:e, s:e] = True   # rubric → self (bidirectional within rubric)

        # Zero out columns/rows of padding tokens.
        pad = attention_mask[b].bool()   # [T]
        allow &= pad.unsqueeze(0)        # zero-out padded key positions
        allow &= pad.unsqueeze(1)        # zero-out padded query positions

        masks.append(allow)

    # [B, T, T] bool → [B, 1, T, T] additive float
    block_mask = torch.stack(masks, dim=0).unsqueeze(1)   # [B, 1, T, T]
    additive = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    additive[~block_mask] = min_val
    return additive


def build_rubric_block_position_ids(
    attention_mask: torch.Tensor,   # [B, T]  binary padding mask
    rubric_spans: torch.Tensor,     # [N, 2] or [B, R, 2]
    rubric_mask: Optional[torch.Tensor] = None,  # legacy [B, R]
    rubric_example_indices: Optional[torch.Tensor] = None,
    rubric_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:                  # [B, T]  position ids
    """
    Reset each rubric's position IDs so that every rubric starts at the same
    offset from the context end, matching the per-pair positional distribution
    of xnet's separate forward passes.

    Context tokens keep their natural positions 0..ctx_end-1.
    Rubric i gets positions ctx_end..ctx_end+(e_i-s_i)-1 regardless of where
    it actually sits in the flat sequence.
    Padding positions receive position 0 (ignored by the attention mask).
    """
    B, T = attention_mask.shape
    device = attention_mask.device

    pos_ids = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        spans = _rubric_spans_for_example(
            rubric_spans,
            b,
            rubric_mask=rubric_mask,
            rubric_example_indices=rubric_example_indices,
            rubric_indices=rubric_indices,
        )
        if spans.shape[0] > 0:
            ctx_end = int(spans[0, 0])
        else:
            # No rubrics: normal positions.
            pos_ids[b] = torch.arange(T, device=device)
            continue

        # Context: positions 0..ctx_end-1.
        pos_ids[b, :ctx_end] = torch.arange(ctx_end, device=device)

        # Each rubric: reset to start at ctx_end.
        for span in spans:
            s, e = int(span[0]), int(span[1])
            rub_len = e - s
            pos_ids[b, s:e] = torch.arange(ctx_end, ctx_end + rub_len, device=device)

    return pos_ids


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
        
    @staticmethod
    def _patch_remote_model_classes(model_name_or_path):
        """Add all_tied_weights_keys to custom model classes that omit it (e.g. NVEmbedModel)."""
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            for cls_ref in getattr(cfg, 'auto_map', {}).values():
                try:
                    cls = get_class_from_dynamic_module(cls_ref, model_name_or_path)
                    if not hasattr(cls, 'all_tied_weights_keys'):
                        cls.all_tied_weights_keys = {}
                except Exception:
                    pass
        except Exception:
            pass

    def _init_encoder(self):
        """Initialize the transformer encoder with optional quantization"""
        self._patch_remote_model_classes(self.config.base_model_name_or_path)
        if self.bnb_config is not None:
            self.encoder = AutoModel.from_pretrained(
                self.config.base_model_name_or_path,
                quantization_config=self.bnb_config,
                config=self.config,
                trust_remote_code=True
            )
        else:
            self.encoder = AutoModel.from_pretrained(
                self.config.base_model_name_or_path,
                config=self.config,
                trust_remote_code=True
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

    def freeze_encoder(self):
        """Freeze all encoder parameters, including adapter parameters if present."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _get_lm_backbone(self):
        """Return the underlying LM backbone, unwrapping PEFT containers when needed."""
        backbone = self.encoder
        if hasattr(backbone, "get_base_model"):
            backbone = backbone.get_base_model()
        if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model"):
            backbone = backbone.base_model.model
        return backbone

    @staticmethod
    def _find_transformer_layers(backbone):
        """Find the ordered transformer block list for common HF backbone layouts."""
        candidate_paths = (
            ("layers",),
            ("encoder", "layer"),
            ("encoder", "layers"),
            ("transformer", "h"),
            ("h",),
            ("model", "layers"),
            ("model", "encoder", "layers"),
            ("model", "encoder", "layer"),
            ("bert", "encoder", "layer"),
            ("roberta", "encoder", "layer"),
            ("deberta", "encoder", "layer"),
            ("deberta_v2", "encoder", "layer"),
        )
        for path in candidate_paths:
            module = backbone
            for attr in path:
                if not hasattr(module, attr):
                    module = None
                    break
                module = getattr(module, attr)
            if module is not None and isinstance(module, (nn.ModuleList, list, tuple)) and len(module) > 0:
                return module
        return None

    @staticmethod
    def _find_embedding_modules(backbone):
        """Find embedding modules that sit below transformer blocks."""
        candidate_paths = (
            ("embed_tokens",),
            ("embeddings",),
            ("wte",),
            ("model", "embed_tokens"),
            ("model", "embeddings"),
            ("transformer", "wte"),
            ("bert", "embeddings"),
            ("roberta", "embeddings"),
            ("deberta", "embeddings"),
            ("deberta_v2", "embeddings"),
        )
        modules = []
        seen = set()
        for path in candidate_paths:
            module = backbone
            for attr in path:
                if not hasattr(module, attr):
                    module = None
                    break
                module = getattr(module, attr)
            if isinstance(module, nn.Module) and id(module) not in seen:
                modules.append(module)
                seen.add(id(module))
        return modules

    def freeze_lm_backbone_layers(self, frozen_layers):
        """
        Freeze embeddings and the bottom transformer layers of the LM backbone.

        Args:
            frozen_layers: 0 disables freezing; 0 < x < 1 freezes that fraction of
                transformer layers; x >= 1 freezes int(x) bottom transformer layers.
        """
        frozen_layers = float(frozen_layers or 0.0)
        if frozen_layers <= 0.0:
            return 0, 0

        backbone = self._get_lm_backbone()
        layers = self._find_transformer_layers(backbone)
        if layers is None:
            raise ValueError(f"Could not find transformer layers on backbone type {type(backbone).__name__}.")

        total_layers = len(layers)
        if 0.0 < frozen_layers < 1.0:
            num_layers = max(1, int(total_layers * frozen_layers))
        else:
            num_layers = int(frozen_layers)
        num_layers = min(num_layers, total_layers)

        frozen_embedding_modules = 0
        for module in self._find_embedding_modules(backbone):
            for param in module.parameters():
                param.requires_grad = False
            frozen_embedding_modules += 1

        for layer in layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        return num_layers, frozen_embedding_modules

    def listwise_loss(
        self,
        logits: torch.Tensor,          # [B, R]
        rubric_mask: torch.Tensor,     # [B, R] in {0,1}
        pos_idx: torch.Tensor,         # [B] index of correct rubric
        tau: float = 1.0,
        label_smoothing: float = 0.0,
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

        return CrossEntropyLoss(label_smoothing=label_smoothing)(scaled_logits, pos_idx)

    

    def get_encoder_outputs(self, input_ids, attention_mask, token_type_ids=None, position_ids=None):
        """Get encoder outputs with optional token type ids and position ids"""
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        if position_ids is not None:
            inputs["position_ids"] = position_ids

        return self.encoder(**inputs)


    def pairwise_ranking_loss(
        self,
        rubric_embs: torch.Tensor,     # [B, R, H]
        rubric_mask: torch.Tensor,     # [B, R] in {0,1}
        pos_idx: torch.Tensor,         # [B] index of correct rubric
        temperature: float = 0.07,
        margin: float = 0.2,
        scale_margin: bool = True,
        rank_gold_only: bool = False
    ) -> torch.Tensor:
        """
        Embedding-based pairwise ranking loss that enforces gold rubric embedding
        to be closer to rubrics near it, and farther from distant rubrics.
        
        This loss operates in embedding space using cosine similarity.
        
        Args:
            rubric_embs: [B, R, H] - embeddings for each rubric
            rubric_mask: [B, R] - mask for valid rubrics
            pos_idx: [B] - gold label indices (which rubric is correct)
            temperature: temperature for scaling similarities
            margin: minimum margin between closer and farther rubrics
            scale_margin: whether to scale margin based on distance
            rank_gold_only: if True, only compare the gold rubric against others
        """
        import torch.nn.functional as F
        
        B, R, H = rubric_embs.shape
        
        # Normalize embeddings for cosine similarity
        rubric_embs_norm = F.normalize(rubric_embs, p=2, dim=-1)  # [B, R, H]
        
        pair_loss = 0.0
        num_pairs = 0
        
        for b in range(B):
            gold_idx = int(pos_idx[b])
            gold_emb = rubric_embs_norm[b, gold_idx]  # [H]
            
            # Compute cosine similarities between gold and all rubrics
            similarities = torch.matmul(rubric_embs_norm[b], gold_emb)  # [R]
            scaled_sims = similarities / temperature
            
            # For all pairs (i, j), enforce ranking based on distance to gold
            for i in range(R):
                for j in range(R):
                    if rank_gold_only and i != gold_idx:
                        continue
                    if rubric_mask[b, i] == 0 or rubric_mask[b, j] == 0:
                        continue
                    if i != j:
                        dist_i = abs(i - gold_idx)
                        dist_j = abs(j - gold_idx)
                        
                        if dist_i == dist_j:
                            # Same distance to gold, no ranking constraint needed
                            continue
                        
                        scale = 1.0 if not scale_margin else (dist_j - dist_i)
                        sim_diff = scaled_sims[i] - scaled_sims[j]
                        
                        if dist_i < dist_j:
                            # i is closer to gold than j, so sim[i] should be > sim[j]
                            pair_loss += F.relu(margin * scale - sim_diff)
                            num_pairs += 1
                        elif dist_i > dist_j:
                            # j is closer to gold than i, so sim[j] should be > sim[i]
                            pair_loss += F.relu(margin * scale + sim_diff)
                            num_pairs += 1
    
        if num_pairs > 0:
            pair_loss = pair_loss / num_pairs
        else:
            pair_loss = torch.tensor(0.0, device=rubric_embs.device, dtype=rubric_embs.dtype)
        
        return pair_loss
