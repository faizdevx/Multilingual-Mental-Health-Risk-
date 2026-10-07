"""Multi-task models: BiLSTM and multilingual-transformer encoders sharing a representation with per-task heads.

                 text
                  |
        encoder (BiLSTM | transformer)
                  |
          shared representation  (masked pooling -> Linear -> GELU -> Dropout)
        /         |          \\
   sentiment    emotion      risk          one linear head per task that has labelled data
   (softmax-3)  (6 sigmoid)  (softmax-2)

Why share? Sentiment and emotion are both affect-related, so lexical/semantic features can transfer between them.
Risk (Dreaddit stress) is a different construct drawn from a different population, so it keeps its own head and
its own loss; the shared layer lets it use affect features without forcing a common decision boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TASK_LABELS


def masked_mean_max(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
    mx = h.masked_fill(m == 0, -1e4).max(1).values
    return torch.cat([mean, mx], dim=-1)


class MultiTaskBase(nn.Module):
    """Shared trunk + heads. Subclasses implement `encode_tokens`."""

    def __init__(self, rep_dim: int, shared_dim: int, dropout: float, tasks=tuple(TASK_LABELS)):
        super().__init__()
        self.tasks = tuple(tasks)
        self.shared = nn.Sequential(nn.Linear(rep_dim, shared_dim), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({t: nn.Linear(shared_dim, len(TASK_LABELS[t])) for t in self.tasks})

    def encode_tokens(self, input_ids, attention_mask) -> torch.Tensor:  # (B, rep_dim)
        raise NotImplementedError

    def represent(self, input_ids, attention_mask) -> torch.Tensor:
        return self.shared(self.encode_tokens(input_ids, attention_mask))

    def forward(self, input_ids, attention_mask, task: str | None = None):
        z = self.represent(input_ids, attention_mask)
        if task is not None:
            return self.heads[task](z)
        return {t: head(z) for t, head in self.heads.items()}

    def forward_from_embeddings(self, embeds, attention_mask, task):  # used by attribution
        raise NotImplementedError

    def embedding_layer(self) -> nn.Module:
        raise NotImplementedError

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MultiTaskBiLSTM(MultiTaskBase):
    kind = "bilstm"

    def __init__(self, vocab_size: int, emb_dim=200, hidden=192, layers=1, shared_dim=256, dropout=0.3, tasks=tuple(TASK_LABELS)):
        super().__init__(rep_dim=4 * hidden, shared_dim=shared_dim, dropout=dropout, tasks=tasks)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.config = dict(vocab_size=vocab_size, emb_dim=emb_dim, hidden=hidden, layers=layers, shared_dim=shared_dim,
                           dropout=dropout, tasks=list(tasks))

    def _run(self, e, attention_mask):
        e = self.emb_drop(e)
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=e.size(1))
        return masked_mean_max(out, attention_mask)

    def encode_tokens(self, input_ids, attention_mask):
        return self._run(self.emb(input_ids), attention_mask)

    def forward_from_embeddings(self, embeds, attention_mask, task):
        return self.heads[task](self.shared(self._run(embeds, attention_mask)))

    def embedding_layer(self):
        return self.emb


class MultiTaskTransformer(MultiTaskBase):
    kind = "transformer"

    def __init__(self, encoder, name: str, shared_dim=256, dropout=0.1, tasks=tuple(TASK_LABELS)):
        hid = encoder.config.hidden_size
        super().__init__(rep_dim=2 * hid, shared_dim=shared_dim, dropout=dropout, tasks=tasks)
        self.encoder = encoder
        self.config = dict(name=name, shared_dim=shared_dim, dropout=dropout, tasks=list(tasks))

    @classmethod
    def from_pretrained(cls, name: str, **kw):
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(name), name, **kw)

    def encode_tokens(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return masked_mean_max(h, attention_mask)

    def forward_from_embeddings(self, embeds, attention_mask, task):
        h = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        return self.heads[task](self.shared(masked_mean_max(h, attention_mask)))

    def embedding_layer(self):
        return self.encoder.get_input_embeddings()


# ---- losses -------------------------------------------------------------
def task_loss(task: str, logits: torch.Tensor, y: torch.Tensor, class_weight=None, pos_weight=None) -> torch.Tensor:
    """softmax CE for multiclass tasks; mask-aware BCE for the multi-label emotion task (y == -1 -> ignored)."""
    if task == "emotion":
        valid = (y >= 0).float()
        l = F.binary_cross_entropy_with_logits(logits, y.clamp(min=0).float(), pos_weight=pos_weight, reduction="none")
        return (l * valid).sum() / valid.sum().clamp(min=1)
    return F.cross_entropy(logits, y, weight=class_weight)
