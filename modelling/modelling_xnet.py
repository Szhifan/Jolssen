from typing import Optional
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging
from modelling.modelling_utils import Pooler, BaseAsagModel

logger = logging.get_logger(__name__)


class AsagXnet(BaseAsagModel):
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config, lora_config=lora_config, bnb_config=bnb_config)
        
        self.pooler = Pooler(pool_type=config.pool_type)
        self.use_token_type_ids = getattr(config, 'use_token_type_ids', True)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        self.test_drop_rub = float(getattr(config, "test_drop_rub", 0.0))
        


    @torch.no_grad()
    def _build_rubric_mask(self, num_rubrics: Optional[torch.Tensor], B: int, R: int, device, dtype=torch.long):
        """
        Vectorized mask builder: 1 for valid rubrics, 0 for padding.
        num_rubrics: [B] or None
        """
        if num_rubrics is None:
            mask = torch.ones(B, R, device=device, dtype=dtype)
        else:
            nr = num_rubrics.to(device=device)
            ar = torch.arange(R, device=device).unsqueeze(0).expand(B, R)
            mask = (ar < nr.unsqueeze(1)).to(dtype)

        return mask

    def _apply_random_negative_rubric_mask(
        self,
        rubric_mask: torch.Tensor,
        labels: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """Randomly mask one non-gold valid rubric per sample during training."""
        if (not self.training) or labels is None or self.test_drop_rub <= 0.0:
            return rubric_mask

        loss_mask = rubric_mask.clone()
        valid_mask = loss_mask.bool()
        B, R = valid_mask.shape

        labels = labels.to(device=loss_mask.device, dtype=torch.long)
        all_idx = torch.arange(R, device=loss_mask.device).unsqueeze(0).expand(B, R)
        candidate_mask = valid_mask & (all_idx != labels.unsqueeze(1))

        has_candidate = candidate_mask.any(dim=1)
        apply_drop = (torch.rand(B, device=loss_mask.device) < self.test_drop_rub) & has_candidate
        if not apply_drop.any():
            return loss_mask

        sampled = torch.rand(B, R, device=loss_mask.device).masked_fill(~candidate_mask, -1.0).argmax(dim=1)
        rows = torch.arange(B, device=loss_mask.device)[apply_drop]
        cols = sampled[apply_drop]
        loss_mask[rows, cols] = 0

        return loss_mask

    def forward_sce(
        self,
        input_ids: torch.LongTensor,        # [B, R, S]
        attention_mask: torch.Tensor,       # [B, R, S]
        token_type_ids: Optional[torch.Tensor] = None,  # [B, R, S]
        num_rubrics: Optional[torch.Tensor] = None,     # [B]
        labels: Optional[torch.LongTensor] = None,      # [B], index of correct rubric
        tau: float = 1.0,
    ) -> SequenceClassifierOutput:
        B, R, S = input_ids.shape

        # Build rubric mask
        rubric_mask = self._build_rubric_mask(
            num_rubrics, B, R, input_ids.device,
            dtype=torch.long
        )

        # Flatten batch+rubric for the encoder: [B*R, S]
        flat_input_ids = input_ids.reshape(B * R, S)
        flat_attention_mask = attention_mask.reshape(B * R, S)
        flat_token_type_ids = None
        
        if self.use_token_type_ids and (token_type_ids is not None):
            flat_token_type_ids = token_type_ids.reshape(B * R, S)

        # ---- Encode ----
        transformer_outputs = self.get_encoder_outputs(
            flat_input_ids, 
            flat_attention_mask, 
            flat_token_type_ids
        )

        # ---- Pool ----
        if hasattr(transformer_outputs, "pooler_output") and transformer_outputs.pooler_output is not None:
            pooled_output = transformer_outputs.pooler_output  # [B*R, H]
        else:
            pooled_output = self.pooler(transformer_outputs.last_hidden_state, flat_attention_mask)  # [B*R, H]

        # Reshape pooled output to [B, R, H]
        H = pooled_output.shape[-1]
        pooled_reshaped = pooled_output.reshape(B, R, H)  # [B, R, H]

        # ---- Score ----
        logits = self.score(pooled_output).squeeze(-1).reshape(B, R)  # [B, R]
        logits = logits.masked_fill(rubric_mask == 0, float('-inf'))

        loss = None
        if labels is not None:
            loss_rubric_mask = self._apply_random_negative_rubric_mask(rubric_mask, labels)
            loss_logits = logits.masked_fill(loss_rubric_mask == 0, float("-inf"))
            loss = self.listwise_loss(loss_logits, loss_rubric_mask, labels, tau=tau)
    
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    def forward_cl(
        self,
        input_ids: torch.LongTensor,        # [B, S]
        attention_mask: torch.Tensor,       # [B, S]
        token_type_ids: Optional[torch.Tensor] = None,  # [B, S]
        labels: Optional[torch.LongTensor] = None,      # [B]
    ) -> SequenceClassifierOutput:
        B, S = input_ids.shape

        # ---- Encode ----
        transformer_outputs = self.get_encoder_outputs(
            input_ids, 
            attention_mask, 
            token_type_ids
        )

        # ---- Pool ----
        if hasattr(transformer_outputs, "pooler_output") and transformer_outputs.pooler_output is not None:
            pooled_output = transformer_outputs.pooler_output  # [B, H]
        else:
            pooled_output = self.pooler(transformer_outputs.last_hidden_state, attention_mask)  # [B, H]

        # ---- Classify ----
        logits = self.score(pooled_output)  # [B, 2]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
        
    def forward(
        self,
        input_ids: torch.LongTensor,        # [B, R, S] or [B, S]
        attention_mask: torch.Tensor,       # [B, R, S] or [B, S]
        token_type_ids: Optional[torch.Tensor] = None,  # [B, R, S] or [B, S]
        num_rubrics: Optional[torch.Tensor] = None,     # [B]
        labels: Optional[torch.LongTensor] = None,      # [B]
        tau: float = 1.0,
    ) -> SequenceClassifierOutput:
        if input_ids.dim() == 3:
            return self.forward_sce(
                input_ids,
                attention_mask,
                token_type_ids,
                num_rubrics,
                labels,
                tau
            )
        else:
            return self.forward_cl(
                input_ids,
                attention_mask,
                token_type_ids,
                labels
            )