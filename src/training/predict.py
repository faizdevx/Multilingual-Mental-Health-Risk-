"""Batched raw-logit prediction used by the trainer, evaluation and the inference service."""
from __future__ import annotations

import numpy as np
import torch

from src.config import MAX_LEN


@torch.no_grad()
def predict_logits(model, tokenizer, texts: list[str], task: str, device="cpu", batch_size: int = 64, max_len: int | None = None) -> np.ndarray:
    model.eval()
    max_len = max_len or MAX_LEN[task]
    order = np.argsort([len(t) for t in texts])  # length-sorted batches -> less padding
    out = [None] * len(texts)
    for i in range(0, len(texts), batch_size):
        idx = order[i : i + batch_size]
        enc = tokenizer.encode([texts[j] for j in idx], max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        lg = model(enc["input_ids"], enc["attention_mask"], task).float().cpu().numpy()
        for j, row in zip(idx, lg):
            out[j] = row
    return np.stack(out) if out else np.zeros((0, 0))
