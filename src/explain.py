"""Model attribution via Integrated Gradients over input embeddings (Sundararajan et al., 2017).

This shows which input tokens most influenced the model's *score* for a class. It is a statistical attribution of one
model's behaviour - NOT clinical reasoning, NOT evidence of any condition, and attributions can be unstable/misleading.
"""
from __future__ import annotations

import numpy as np
import torch

from src.config import MAX_LEN, TASK_LABELS

CAVEAT = ("Model attribution (Integrated Gradients): shows which tokens influenced this model's score for the target class. "
          "It is not clinical reasoning and does not indicate that any condition is present. Attributions can be noisy.")


def _clean(tok: str) -> str:
    return tok.replace("##", "").replace("▁", "").strip() or tok


def integrated_gradients(model, tokenizer, text: str, task: str, target: int | None = None, n_steps: int = 16, top_k: int = 8,
                         device: str = "cpu") -> dict:
    model.eval()
    enc = tokenizer.encode([text], MAX_LEN[task])
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    emb_layer = model.embedding_layer()
    with torch.no_grad():
        emb = emb_layer(ids)
        if target is None:
            target = int(model(ids, mask, task).argmax(-1).item())
    baseline = torch.zeros_like(emb)
    total = torch.zeros_like(emb)
    for k in range(1, n_steps + 1):
        x = (baseline + (k / n_steps) * (emb - baseline)).clone().requires_grad_(True)
        out = model.forward_from_embeddings(x, mask, task)
        out[0, target].backward()
        total += x.grad
    attr = ((emb - baseline) * total / n_steps).sum(-1)[0].detach().cpu().numpy()
    toks = [_clean(t) for t in (tokenizer.tok.convert_ids_to_tokens(ids[0].tolist()) if hasattr(tokenizer, "tok") else
                                [tokenizer.itos[i] for i in ids[0].tolist()])]
    special = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    keep = [i for i, t in enumerate(toks) if t not in special]
    scale = float(np.abs(attr[keep]).max()) or 1.0
    items = [{"token": toks[i], "attribution": float(attr[i] / scale)} for i in keep]
    top = sorted(items, key=lambda d: -abs(d["attribution"]))[:top_k]
    return {"method": "integrated_gradients", "task": task, "target_label": TASK_LABELS[task][target], "n_steps": n_steps,
            "tokens": items, "top_tokens": top, "caveat": CAVEAT,
            "note": "Positive values push the score toward the target label; values are normalised to [-1, 1] for display."}
