import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutput
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.mistral.modeling_mistral import (
    MistralPreTrainedModel, 
    MistralDecoderLayer, 
    MistralRMSNorm, 
    MistralRotaryEmbedding,
    MistralConfig,
    MistralAttention,
    MistralMLP
)
from .modelling_utils import (
    get_noncausal_attention_mask,
    Pooler,
)

from transformers.utils import TransformersKwargs, logging, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.masking_utils import create_causal_mask
from functools import partial

logger = logging.get_logger(__name__)

# Utility function for residual connections
def use_res_connect(connect_layers, layer_idx):
    """Determine if a layer should use residual connection."""
    if connect_layers is None:
        return False
    if isinstance(connect_layers, (list, tuple)):
        return layer_idx in connect_layers
    elif isinstance(connect_layers, bool):
        return connect_layers
    return False
class MistralDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config=config, layer_idx=layer_idx)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

class MistralModel(MistralPreTrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MistralRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # -----------------------------------------------------------------------------------------------------------------
        self.architecture = getattr(config, 'architecture', 'NONE')
        self.mask_type = getattr(config, 'mask_type', "MASK0")
        self.num_bidir_layers = getattr(config, 'num_bidir_layers', 0)
        self.connect_layers = getattr(config, 'res_connect', None)
        self.use_res_connect = partial(use_res_connect, self.connect_layers)
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
        self.fuse_weights = nn.Parameter(torch.rand(self.num_fuse_layers)) if self.num_fuse_layers > 0 else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        
        # decoder layers
        h1, h2 = None, 0
        fuse_layers = ()
        for i in range(self.num_hidden_layers):
            decoder_layer = self.layers[i]
            is_bidir = i in self.bidir_layers
            layer_mask = bidir_attention_mask if is_bidir else causal_mask



            hidden_states, attn_weights = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            fuse_layers = fuse_layers + (hidden_states,) if i in self.fuse_layers else fuse_layers

        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if fuse_layers and self.fuse_type == "avg":
            hidden_states = torch.stack(fuse_layers, dim=0).mean(dim=0)
        elif fuse_layers and self.fuse_type == "weighted" and self.fuse_weights is not None:
            fuse_weights = torch.softmax(self.fuse_weights, dim=0)
            hidden_states = torch.stack(fuse_layers, dim=0)
            hidden_states = (fuse_weights.view(-1, 1, 1, 1) * hidden_states).sum(dim=0)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def freeze_model(self, config=None):
        if config.freeze_type == "all":
            for param in self.parameters():
                param.requires_grad = False

        elif config.freeze_type == "backbone":
            self.embed_tokens.weight.requires_grad = False

            for param in self.layers[:self.num_hidden_layers - config.num_unfreeze_layers].parameters():
                param.requires_grad = False

            if config.num_unfreeze_layers == 0:
                self.norm.weight.requires_grad = False

    def model_init(self):
        first_new_layer = self.config.num_hidden_layers
        for layer in range(first_new_layer, len(self.layers)):
            self.layers[layer].load_state_dict(self.layers[first_new_layer - 1].state_dict())


class MistralForSequenceClassification(MistralPreTrainedModel):
    """
    Simple wrapper that adds a classification head to the custom MistralModel
    """
    def __init__(self, config):
        print("Using custom MistralForSequenceClassification")
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Use the custom MistralModel as backbone
        self.model = MistralModel(config)
        
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
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None) 