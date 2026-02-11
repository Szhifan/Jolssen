from modelling.modelling_utils import BaseAsagModel, Pooler
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from typing import Optional
import math
class SpanFuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        Span fuser module to fuse span representations. 
        The foward method will output scores of the labels based on the selected fusing type.
        t: text span
        p: pooled embedding of the entire input
        l: label span
        available span fusing types: 
        t-bl: text-label bilinear fusing
        p-bl: pooled_embed-label bilinear fusing
        
        t-concat: text-label concatenation fusing
        p-concat: pooled_embed-label concatenation fusing
        tpl-concat: text-pooled_embed-label concatenation fusing

        t-diff: text-label difference fusing
        p-diff: pooled_embed-label difference fusing
        t-diff-bl-p: text-label difference then pooled_embed bilinear fusing

        p-only: no span fusing. Use pool embedding directly for classification in the old-fasioned way.
        l-only: use label embeddings without fusing
        """
        self.span_fuse_type = getattr(config, 'span_fuse_type', 'p-concat')
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.num_labels = getattr(config, 'num_labels', 3)
        if self.span_fuse_type == 'p-only':
            self.to_score = nn.Linear(config.hidden_size, self.num_labels)
            return
        elif self.span_fuse_type == 'l-only':
            self.to_score = nn.Linear(config.hidden_size, 1)
            return
        elif "con" in self.span_fuse_type:
            input_size = 0
            if self.span_fuse_type == 't-concat':
                input_size = 2 * config.hidden_size
            elif self.span_fuse_type == 'p-concat':
                input_size = 2 * config.hidden_size
            elif self.span_fuse_type == 'p-condiff':
                input_size = 3 * config.hidden_size
            elif self.span_fuse_type == 'tpl-concat':
                input_size = 3 * config.hidden_size
            self.to_score = nn.Linear(input_size, 1)
            # For difference-based fusion types
        elif self.span_fuse_type == 't-diff' or self.span_fuse_type == 'p-diff':
                # Difference + linear scoring
                self.to_score = nn.Linear(config.hidden_size, 1)

        if "bl" in self.span_fuse_type:
            # Bilinear layer for bilinear fusing types
            self.bilinear = nn.Bilinear(config.hidden_size, config.hidden_size, 1)
            
    def forward(self, text_emb, label_embs, pooled_emb=None):
        """
        Forward pass for span fuser.
        
        Args:
            text_emb: [B, H] text span embeddings
            label_embs: [B, R, H] label span embeddings
            pooled_emb: [B, H] pooled embedding of the entire input (optional), but required for p-l and t-p-l fusing types.
        """
        B, R, H = label_embs.shape
        if self.span_fuse_type == 'p-only':
            # No span fusing, use pooled embedding directly
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for 'p-only' fusing type")
            scores = self.to_score(self.dropout(pooled_emb))  # [B, num_labels]
            return scores
        if self.span_fuse_type == 'l-only':
            # Use label embeddings only
            scores = self.to_score(self.dropout(label_embs)).squeeze(-1)  # [B, R]
            return scores
        if self.span_fuse_type == 't-bl':
            # Text-label bilinear fusing
            text_exp = text_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            scores = self.bilinear(text_exp, label_embs).squeeze(-1)  # [B, R]
            
        elif self.span_fuse_type == 'p-bl':
            # Pooled-label bilinear fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-bl fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            scores = self.bilinear(pooled_exp, label_embs).squeeze(-1)  # [B, R]

            
        elif self.span_fuse_type == 't-concat':
            # Text-label concatenation fusing
            text_exp = text_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            concat_emb = torch.cat([text_exp, label_embs], dim=-1)  # [B, R, 2H]
            scores = self.to_score(self.dropout(concat_emb)).squeeze(-1)  # [B, R]
            
        elif self.span_fuse_type == 'p-concat':
            # Pooled-label concatenation fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-concat fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            concat_emb = torch.cat([pooled_exp, label_embs], dim=-1)  # [B, R, 2H]
            scores = self.to_score(self.dropout(concat_emb)).squeeze(-1)  # [B, R]
        elif self.span_fuse_type == 'p-condiff':
            # Pooled-label concatenation + difference fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-condiff fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            diff = pooled_exp - label_embs  # [B, R, H]
            concat_emb = torch.cat([pooled_exp, label_embs, diff], dim=-1)  # [B, R, 3H]
            scores = self.to_score(self.dropout(concat_emb)).squeeze(-1)  # [B, R]
        elif self.span_fuse_type == 'tpl-concat':
            # Text-pooled-label concatenation fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for tpl-concat fusing type")
            text_exp = text_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            concat_emb = torch.cat([text_exp, pooled_exp, label_embs], dim=-1)  # [B, R, 3H]
            scores = self.to_score(self.dropout(concat_emb)).squeeze(-1)  # [B, R]
            
        elif self.span_fuse_type == 't-diff':
            # Text-label difference fusing
            # Compute element-wise difference between text and each label
            text_exp = text_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            diff = text_exp - label_embs  # [B, R, H]
            scores = self.to_score(self.dropout(diff)).squeeze(-1)  # [B, R]
            
        elif self.span_fuse_type == 'p-diff':
            # Pooled-label difference fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-diff fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            diff = pooled_exp - label_embs  # [B, R, H]
            scores = self.to_score(self.dropout(diff)).squeeze(-1)  # [B, R]
            
            
        else:
            raise ValueError(f"Unknown span_fuse_type: {self.span_fuse_type}")
            
        return scores

class SpanAlignmentModel(BaseAsagModel):
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config, lora_config=lora_config, bnb_config=bnb_config)
        self.pooler = Pooler(pool_type=config.pool_type) 
        self.span_fuser = SpanFuser(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.span_pool_type = getattr(config, 'span_pool_type', 'last')
    def _get_span_alignments(self, hidden_states, rubric_spans, answer_span, rubric_mask, span_pool_type='last'):
            """
            Extract and align span embeddings for rubrics and answers.
            
            
            """
            B, R = rubric_spans.shape[:2]
            H = hidden_states.shape[-1]
            
            # Process rubric spans
            rubric_embs = []
            for i in range(R):
                starts = rubric_spans[:, i, 0]
                ends = rubric_spans[:, i, 1]
                emb = torch.stack([
                    (hidden_states[b, starts[b]:ends[b]].mean(dim=0) 
                     if span_pool_type == 'mean' 
                     else hidden_states[b, ends[b]])
                    if rubric_mask[b, i]
                    else torch.zeros(H, device=hidden_states.device, dtype=hidden_states.dtype)
                    for b in range(B)
                ])
                rubric_embs.append(emb)
            rubric_embs = torch.stack(rubric_embs, dim=1)  # [B, R, H]

            # Process answer spans
            a_starts = answer_span[:, 0]
            a_ends = answer_span[:, 1]
            answer_emb = torch.stack([
                (hidden_states[b, a_starts[b]:a_ends[b]].mean(dim=0)
                 if span_pool_type == 'mean'
                 else hidden_states[b, a_ends[b]])
                for b in range(B)
            ])  # [B, H]
            
            return rubric_embs, answer_emb
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        rubric_spans: torch.Tensor,
        answer_span: torch.Tensor,
        rubric_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        tau: float = 1.0,
    ) -> SequenceClassifierOutput:
        # Encode
        outputs = self.get_encoder_outputs(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        B, R = rubric_spans.shape[:2]
        H = hidden_states.shape[-1]

        # Get span alignments
        rubric_embs, answer_emb = self._get_span_alignments(
            hidden_states, rubric_spans, answer_span, rubric_mask, span_pool_type=self.span_pool_type
        )

        # Compute pointer scores
        pooled_emb = self.pooler(self.dropout(outputs.last_hidden_state), attention_mask) \
                    if "p" in self.span_fuser.span_fuse_type else None
        logits = self.span_fuser(
            text_emb=answer_emb,
            label_embs=rubric_embs,
            pooled_emb=pooled_emb
        )  # [B, R]
        logits = logits.masked_fill(~rubric_mask.bool(), float('-inf'))
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.listwise_loss(logits, rubric_mask, labels, tau=tau)
        
        return SequenceClassifierOutput(loss=loss, logits=logits)
