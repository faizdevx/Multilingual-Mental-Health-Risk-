"""Checkpoint save/load. Checkpoints are self-contained (weights + config + tokenizer) so serving works offline."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config import MODEL_VERSION, TASK_LABELS
from src.models.multitask import MultiTaskBiLSTM, MultiTaskTransformer
from src.models.tokenization import HFTokenizer, WordTokenizer


def save_checkpoint(out_dir, model, tokenizer, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    info = {"kind": model.kind, "config": model.config, "labels": TASK_LABELS, "model_version": MODEL_VERSION, **meta}
    if model.kind == "bilstm":
        (out / "vocab.json").write_text(tokenizer.to_json())
    else:
        model.encoder.config.save_pretrained(out / "encoder_config")
        tokenizer.tok.save_pretrained(out / "tokenizer")
    (out / "meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def load_checkpoint(ckpt_dir, device="cpu"):
    d = Path(ckpt_dir)
    meta = json.loads((d / "meta.json").read_text())
    cfg = dict(meta["config"])
    tasks = tuple(cfg.pop("tasks"))
    if meta["kind"] == "bilstm":
        tok = WordTokenizer.from_json((d / "vocab.json").read_text())
        model = MultiTaskBiLSTM(tasks=tasks, **cfg)
    else:
        from transformers import AutoConfig, AutoModel

        enc = AutoModel.from_config(AutoConfig.from_pretrained(d / "encoder_config"))
        name = cfg.pop("name")
        model = MultiTaskTransformer(enc, name, tasks=tasks, **cfg)
        tok = HFTokenizer(str(d / "tokenizer"))
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    return model.to(device).eval(), tok, meta


def detect_device(prefer: str | None = None) -> str:
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
