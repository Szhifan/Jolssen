# adapted from 
import logging
from typing import  Optional, Tuple, Union
import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutput, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel, 
    LlamaDecoderLayer, 
    LlamaRMSNorm, 
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaMLP
    
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from .modelling_utils import (
    get_noncausal_attention_mask,
    Pooler,
)

from transformers.utils import TransformersKwargs, logging, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.utils.generic import check_model_inputs


logger = logging.get_logger(__name__)

class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        print("Using custom LlamaModel with extended architecture")
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Initialize standard Llama components
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # -----------------------------------------------------------------------------------------------------------------

        self.mask_type = getattr(config, 'mask_type', "MASK0")
        self.num_bidir_layers = getattr(config, 'num_bidir_layers', 0)
        self.num_prune_layers = getattr(config, 'num_prune_layers', 0)
        self.num_fuse_layers = getattr(config, 'num_fuse_layers', 0)
        self.fuse_type = getattr(config, 'fuse_type', 'avg')  # avg or weighted

        # If num_prune_layers is between 0 and 1 (exclusive), treat it as a ratio
        if 0 < self.num_prune_layers < 1:
            self.num_prune_layers = max(0, int(config.num_hidden_layers * self.num_prune_layers))
        
        self.num_hidden_layers = config.num_hidden_layers - self.num_prune_layers

        # If num_fuse_layers is between 0 and 1 (exclusive), treat it as a ratio
        if 0 < self.num_fuse_layers < 1:
            self.num_fuse_layers = max(1, int(self.num_hidden_layers * self.num_fuse_layers))

        assert self.num_hidden_layers > 0
        assert self.num_hidden_layers >= self.num_fuse_layers
        
        # If num_bidir_layers is between 0 and 1 (exclusive), treat it as a ratio
        if 0 < self.num_bidir_layers < 1:
            self.num_bidir_layers = max(1, int(self.num_hidden_layers * self.num_bidir_layers))
        
        assert self.num_bidir_layers <= self.num_hidden_layers
        self.bidir_layers = {layer for layer in range(self.num_hidden_layers - self.num_bidir_layers, self.num_hidden_layers)}
        for i in range(self.num_bidir_layers):
            self.layers[self.num_hidden_layers - self.num_bidir_layers + i].self_attn.is_causal = False

        self.fuse_layers = {layer for layer in range(self.num_hidden_layers - self.num_fuse_layers, self.num_hidden_layers)}
        # -----------------------------------------------------------------------------------------------------------------
        self.fuse_weights = nn.Parameter(torch.rand(self.num_fuse_layers))
        # Initialize weights and apply final processing
        self.post_init()
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )


        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

        # Get input shape from either input_ids or inputs_embeds
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        
        bidir_attention_mask = get_noncausal_attention_mask(self, attention_mask, input_shape)
        
        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # decoder layers
        fuse_layers = () 
        for i in range(self.num_hidden_layers):
            decoder_layer = self.layers[i]
            is_bidir = i in self.bidir_layers
            layer_mask = bidir_attention_mask if is_bidir else causal_mask

            hidden_states, attn_weights = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            fuse_layers = fuse_layers + (hidden_states,) if i in self.fuse_layers else fuse_layers
        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if fuse_layers and self.fuse_type == "avg":
            hidden_states = torch.stack(fuse_layers, dim=0).mean(dim=0)
        elif fuse_layers and self.fuse_type == "weighted":
            fuse_weights = torch.softmax(self.fuse_weights, dim=0)
            hidden_states = torch.stack(fuse_layers, dim=0)
            hidden_states = (fuse_weights.view(-1, 1, 1, 1) * hidden_states).sum(dim=0)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,

        )



class LlamaForSequenceClassification(LlamaPreTrainedModel):
    """
    Simple wrapper that adds a classification head to the custom LlamaModel
    """
    def __init__(self, config):
        print("Using custom LlamaForSequenceClassification")
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Use the custom LlamaModel as backbone
        self.model = LlamaModel(config)
        
        # Simple classification head - ensure float32 precision to avoid quantization issues
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pooler = Pooler(pool_type=getattr(config, 'pool_type', 'last'))

        # Initialize weights
        self.post_init()
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get hidden states
        hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)

        pooled_output = self.pooler(hidden_states, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
