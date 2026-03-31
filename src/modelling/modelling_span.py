from modelling.modelling_utils import BaseAsagModel, Pooler
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from typing import Optional
from torch.nn import functional as F


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
        self.span_fuse_type = getattr(config, "span_fuse_type", "p-concat")
        self.dropout = nn.Dropout(0.1)
        self.num_labels = getattr(config, "num_labels", 3)
        # self.cl_weight = getattr(config, "cl_weight", 0.0)

        if getattr(config, "use_proj", False):
            if self.span_fuse_type in ["t-bl", "t-concat", "t-diff", "tpl-concat"]:
                self.text_proj = nn.Linear(config.hidden_size, config.hidden_size)
            if self.span_fuse_type in ["p-bl", "p-concat", "p-diff", "tpl-concat"]:
                self.pooled_proj = nn.Linear(config.hidden_size, config.hidden_size)
            if self.span_fuse_type in ["t-bl", "p-bl", "t-concat", "p-concat", "t-diff", "p-diff", "tpl-concat"]:
                self.label_proj = nn.Linear(config.hidden_size, config.hidden_size)

        if self.span_fuse_type == "p-only":
            self.to_score = nn.Linear(config.hidden_size, self.num_labels)
            return
        elif self.span_fuse_type == "l-only":
            self.to_score = nn.Linear(config.hidden_size, 1)
            return
        elif "con" in self.span_fuse_type:
            input_size = 0
            if self.span_fuse_type == "t-concat":
                input_size = 2 * config.hidden_size
            elif self.span_fuse_type == "p-concat":
                input_size = 2 * config.hidden_size
            elif self.span_fuse_type == "p-condiff":
                input_size = 3 * config.hidden_size
            elif self.span_fuse_type == "tpl-concat":
                input_size = 3 * config.hidden_size
            self.to_score = nn.Linear(input_size, 1)
            # For difference-based fusion types
        elif self.span_fuse_type == "t-diff" or self.span_fuse_type == "p-diff":
            # Difference + linear scoring
            self.to_score = nn.Linear(config.hidden_size, 1)
        elif self.span_fuse_type == "p-gate":
            # Gated fusion with residual connection
            self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.layer_norm = nn.LayerNorm(config.hidden_size)
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

        # # Apply projections if enabled
        # if hasattr(self, "text_proj"):
        #     text_emb = self.text_proj(text_emb)
        # if hasattr(self, "label_proj"):
        #     label_embs = self.label_proj(label_embs)
        # if pooled_emb is not None and hasattr(self, "pooled_proj"):
        #     pooled_emb = self.pooled_proj(pooled_emb)

        if self.span_fuse_type == "p-only":
            # No span fusing, use pooled embedding directly
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for 'p-only' fusing type")
            scores = self.to_score(self.dropout(pooled_emb))  # [B, num_labels]
            return scores
        if self.span_fuse_type == "l-only":
            # Use label embeddings only
            scores = self.to_score(self.dropout(label_embs)).squeeze(-1)  # [B, R]
        if self.span_fuse_type == "t-bl":
            # Text-label bilinear fusing
            text_exp = text_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            scores = self.bilinear(text_exp, label_embs).squeeze(-1)  # [B, R]

        elif self.span_fuse_type == "p-bl":
            # Pooled-label bilinear fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-bl fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            scores = self.bilinear(pooled_exp, label_embs).squeeze(-1)  # [B, R]

        elif self.span_fuse_type == "t-concat":
            # Text-label concatenation fusing
            text_exp = text_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            concat_emb = torch.cat([text_exp, label_embs], dim=-1)  # [B, R, 2H]
            scores = self.to_score(self.dropout(concat_emb)).squeeze(-1)  # [B, R]

        elif self.span_fuse_type == "p-concat":
            # Pooled-label concatenation fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-concat fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            concat_emb = torch.cat([pooled_exp, label_embs], dim=-1)  # [B, R, 2H]
            scores = self.to_score(self.dropout(concat_emb)).squeeze(-1)  # [B, R]
        elif self.span_fuse_type == "p-condiff":
            # Pooled-label concatenation + difference fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-condiff fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            diff = pooled_exp - label_embs  # [B, R, H]
            concat_emb = torch.cat([pooled_exp, label_embs, diff], dim=-1)  # [B, R, 3H]
            scores = self.to_score(self.dropout(concat_emb)).squeeze(-1)  # [B, R]
        elif self.span_fuse_type == "tpl-concat":
            # Text-pooled-label concatenation fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for tpl-concat fusing type")
            text_exp = text_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            concat_emb = torch.cat([text_exp, pooled_exp, label_embs], dim=-1)  # [B, R, 3H]
            scores = self.to_score(self.dropout(concat_emb)).squeeze(-1)  # [B, R]

        elif self.span_fuse_type == "t-diff":
            # Text-label difference fusing
            # Compute element-wise difference between text and each label
            text_exp = text_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            diff = text_exp - label_embs  # [B, R, H]
            scores = self.to_score(self.dropout(diff)).squeeze(-1)  # [B, R]

        elif self.span_fuse_type == "p-diff":
            # Pooled-label difference fusing
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-diff fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]
            diff = pooled_exp - label_embs  # [B, R, H]
            scores = self.to_score(self.dropout(diff)).squeeze(-1)  # [B, R]

        elif self.span_fuse_type == "p-gate":
            # Gated fusion with residual: z_tilde = LayerNorm(z + z ⊙ σ(W·r_k))
            if pooled_emb is None:
                raise ValueError("pooled_emb is required for p-gate fusing type")
            pooled_exp = pooled_emb.unsqueeze(1).expand(-1, R, -1)  # [B, R, H]

            # Step 1: Compute gate from label embeddings: σ(W·r_k)
            gate = torch.sigmoid(self.gate_proj(label_embs))  # [B, R, H]

            # Step 2: Gated fusion: z ⊙ σ(W·r_k)
            gated = pooled_exp * gate  # [B, R, H]

            # Step 3: Residual connection + LayerNorm: LayerNorm(z + z ⊙ σ(W·r_k))
            fused = self.layer_norm(pooled_exp + gated)  # [B, R, H]

            scores = self.to_score(self.dropout(fused)).squeeze(-1)  # [B, R]

        else:
            raise ValueError(f"Unknown span_fuse_type: {self.span_fuse_type}")

        return scores


class SpanAlignmentModel(BaseAsagModel):
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config, lora_config=lora_config, bnb_config=bnb_config)
        self.pooler = Pooler(pool_type=config.pool_type)
        self.span_fuser = SpanFuser(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.1)
        self.span_pool_type = getattr(config, "span_pool_type", "last")
        self.cl_weight = getattr(config, "cl_weight", 0.0)
        self.label_smoothing = 0.0
        self.random_drop_rub = float(getattr(config, "random_drop_rub", 0.0))

    def _pool_single_span(
        self,
        hidden_states: torch.Tensor,
        batch_idx: int,
        start_idx: int,
        end_idx: int,
        span_pool_type: str,
    ) -> torch.Tensor:
        """Pool one span with bounds checks while preserving current 'last' semantics."""
        seq_len = hidden_states.shape[1]

        if span_pool_type == "mean":
            start = max(0, min(int(start_idx), seq_len))
            end = max(0, min(int(end_idx), seq_len))
            if end <= start:
                fallback = max(0, min(int(end_idx), seq_len - 1))
                return hidden_states[batch_idx, fallback]
            return hidden_states[batch_idx, start:end].mean(dim=0)


        index = max(0, min(int(end_idx), seq_len - 1))
        return hidden_states[batch_idx, index]

    def _get_span_alignments(
        self,
        hidden_states,
        rubric_spans,
        answer_span,
        rubric_mask,
        span_pool_type="last",
    ):
        """Extract and align span embeddings for rubrics and answer spans."""
        B, R = rubric_spans.shape[:2]
        H = hidden_states.shape[-1]

        rubric_embs = []
        for i in range(R):
            starts = rubric_spans[:, i, 0]
            ends = rubric_spans[:, i, 1]
            emb = torch.stack(
                [
                    self._pool_single_span(hidden_states, b, starts[b], ends[b], span_pool_type)
                    if rubric_mask[b, i]
                    else torch.zeros(H, device=hidden_states.device, dtype=hidden_states.dtype)
                    for b in range(B)
                ]
            )
            rubric_embs.append(emb)
        rubric_embs = torch.stack(rubric_embs, dim=1)  # [B, R, H]

        a_starts = answer_span[:, 0]
        a_ends = answer_span[:, 1]
        answer_emb = torch.stack(
            [
                self._pool_single_span(hidden_states, b, a_starts[b], a_ends[b], span_pool_type)
                for b in range(B)
            ]
        )  # [B, H]

        return rubric_embs, answer_emb

    def _apply_random_negative_rubric_mask(
        self,
        rubric_mask: torch.Tensor,
        labels: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """Randomly mask one non-gold valid rubric per sample during training."""
        if (not self.training) or labels is None or self.random_drop_rub <= 0.0:
            return rubric_mask

        loss_mask = rubric_mask.clone()
        valid_mask = loss_mask.bool()
        B, R = valid_mask.shape

        labels = labels.to(device=loss_mask.device, dtype=torch.long)
        all_idx = torch.arange(R, device=loss_mask.device).unsqueeze(0).expand(B, R)
        candidate_mask = valid_mask & (all_idx != labels.unsqueeze(1))

        has_candidate = candidate_mask.any(dim=1)
        apply_drop = (torch.rand(B, device=loss_mask.device) < self.random_drop_rub) & has_candidate
        if not apply_drop.any():
            return loss_mask

        sampled = torch.rand(B, R, device=loss_mask.device).masked_fill(~candidate_mask, -1.0).argmax(dim=1)
        rows = torch.arange(B, device=loss_mask.device)[apply_drop]
        cols = sampled[apply_drop]

        if loss_mask.dtype == torch.bool:
            loss_mask[rows, cols] = False
        else:
            loss_mask[rows, cols] = 0

        return loss_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        rubric_spans: torch.Tensor,
        answer_span: torch.Tensor,
        rubric_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> SequenceClassifierOutput:
        # Encode with base model.
        outputs = self.get_encoder_outputs(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        # Get span alignments from token embeddings.
        rubric_embs, answer_emb = self._get_span_alignments(
            hidden_states,
            rubric_spans,
            answer_span,
            rubric_mask,
            span_pool_type=self.span_pool_type,
        )

        # Compute pointer scores.
        pooled_emb = self.pooler(self.dropout(hidden_states), attention_mask) if "p" in self.span_fuser.span_fuse_type else None

        logits = self.span_fuser(
            text_emb=answer_emb,
            label_embs=rubric_embs,
            pooled_emb=pooled_emb,
        )  # [B, R]
        logits = logits.masked_fill(~rubric_mask.bool(), float("-inf"))

        # Compute loss if labels are provided.
        loss = None
        if labels is not None:
            # Main listwise loss with optional train-time random negative rubric masking.
            loss_rubric_mask = self._apply_random_negative_rubric_mask(rubric_mask, labels)
            loss_logits = logits.masked_fill(~loss_rubric_mask.bool(), float("-inf"))
            loss = self.listwise_loss(loss_logits, loss_rubric_mask, labels, label_smoothing=self.label_smoothing)

            # Add contrastive loss if enabled.
            if self.cl_weight > 0.0:
                contrastive_loss = self.contrastive_rubric_loss(
                    seq_emb=pooled_emb if pooled_emb is not None else answer_emb,
                    rubric_embs=rubric_embs,
                    rubric_mask=rubric_mask,
                    pos_idx=labels,
                    temperature=0.07,
                )
                loss = loss + self.cl_weight * contrastive_loss

            # Add pairwise ranking loss (example usage)
            # pairwise_loss = self.pairwise_ranking_loss(
            #     seq_emb=pooled_emb if pooled_emb is not None else answer_emb,
            #     rubric_embs=rubric_embs,
            #     rubric_mask=rubric_mask,
            #     pos_idx=labels,
            #     temperature=0.07,
            #     margin=0.2,
            # )
            # loss = loss + pairwise_weight * pairwise_loss

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def contrastive_rubric_loss(
        self,
        seq_emb: torch.Tensor,  # [B, H]   -> z
        rubric_embs: torch.Tensor,  # [B, R, H]
        rubric_mask: torch.Tensor,  # [B, R] in {0,1}
        pos_idx: torch.Tensor,  # [B]
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Contrastive alignment loss between sequence embedding and rubric embeddings.

        Implements:
            L_b = -log( exp(sim(z, r_y)/tau) / sum_k exp(sim(z, r_k)/tau) )

        Args:
            seq_emb: [B, H] sequence/global representation
            rubric_embs: [B, R, H] rubric span embeddings
            rubric_mask: [B, R] mask for valid rubric entries
            pos_idx: [B] gold rubric indices
            temperature: scaling temperature

        Returns:
            Scalar loss
        """

        # Normalize embeddings for cosine similarity
        seq_emb = F.normalize(seq_emb, dim=-1)  # [B, H]
        rubric_embs = F.normalize(rubric_embs, dim=-1)  # [B, R, H]

        # Compute cosine similarities
        # [B, R] = batch matmul
        similarities = torch.einsum("bh,brh->br", seq_emb, rubric_embs)

        # Temperature scaling
        similarities = similarities / temperature

        # Mask invalid rubric entries
        similarities = similarities.masked_fill(rubric_mask == 0, -1e9)

        # Compute cross-entropy
        loss = F.cross_entropy(similarities, pos_idx)

        return loss
