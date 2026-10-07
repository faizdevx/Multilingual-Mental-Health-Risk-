"""Uniform interface over the three model families so evaluation / serving code is model-agnostic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import CKPT_DIR
from src.models.baseline import TfidfBaseline
from src.models.io import detect_device, load_checkpoint
from src.training.predict import predict_logits

ARCH = {
    "baseline": "TF-IDF (word 1-2g + char 2-5g) + Logistic Regression, one model per task",
    "bilstm": "Multi-task BiLSTM (random-init embeddings, masked mean+max pooling, 3 heads)",
    "transformer": "Multi-task MuRIL (google/muril-base-cased) fine-tuned, shared layer + 3 heads",
}
DISPLAY = {"baseline": "TF-IDF + LogReg", "bilstm": "BiLSTM", "transformer": "Multilingual Transformer (MuRIL)"}


class Adapter:
    name: str
    kind: str
    device: str = "cpu"

    def predict_logits(self, task: str, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def n_parameters(self) -> int:
        raise NotImplementedError

    def calibration(self) -> dict:
        p = self.ckpt_dir / "calibration.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @property
    def ckpt_dir(self) -> Path:
        return CKPT_DIR / self.name


class BaselineAdapter(Adapter):
    kind = "baseline"

    def __init__(self, name="baseline", ckpt_dir=None):
        self.name = name
        self._dir = Path(ckpt_dir) if ckpt_dir else CKPT_DIR / name
        self.model = TfidfBaseline.load(self._dir / "model.joblib")

    ckpt_dir = property(lambda self: self._dir)

    def predict_logits(self, task, texts):
        return self.model.predict_logits(task, texts)

    def n_parameters(self):
        return self.model.n_parameters()


class NeuralAdapter(Adapter):
    def __init__(self, name: str, ckpt_dir=None, device: str | None = None):
        self.name = name
        self._dir = Path(ckpt_dir) if ckpt_dir else CKPT_DIR / name
        self.device = detect_device(device)
        self.model, self.tok, self.meta = load_checkpoint(self._dir, self.device)
        self.kind = self.meta["kind"]

    ckpt_dir = property(lambda self: self._dir)

    def predict_logits(self, task, texts):
        return predict_logits(self.model, self.tok, texts, task, self.device)

    def n_parameters(self):
        return self.model.n_parameters()


def load_adapter(name: str, device: str | None = None) -> Adapter:
    d = CKPT_DIR / name
    if (d / "model.joblib").exists():
        return BaselineAdapter(name)
    return NeuralAdapter(name, device=device)


def available_models() -> list[str]:
    return [n for n in ("baseline", "bilstm", "transformer") if (CKPT_DIR / n / "model.joblib").exists() or (CKPT_DIR / n / "meta.json").exists()]
