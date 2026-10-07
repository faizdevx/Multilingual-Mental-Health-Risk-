"""Shared fixtures.

IMPORTANT: everything here is a SYNTHETIC TEST FIXTURE used only to exercise code paths (shapes, I/O, routing, privacy).
It is never used for training or evaluation, and the tiny models are randomly initialised - their predictions are meaningless.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMOTION_LABELS  # noqa: E402
from src.evaluation.adapters import NeuralAdapter  # noqa: E402
from src.inference import ModelRegistry, Predictor  # noqa: E402
from src.models.io import save_checkpoint  # noqa: E402
from src.models.multitask import MultiTaskBiLSTM, MultiTaskTransformer  # noqa: E402
from src.models.tokenization import HFTokenizer, WordTokenizer  # noqa: E402

SYN = ["i went to the market today", "the weather is nice", "he was very tired and upset", "what a lovely day", "i am so worried about work",
       "yaar aaj mood off hai", "kuch samajh nahi aa raha", "bahut mast din tha", "यह एक परीक्षण वाक्य है", "आज मौसम अच्छा है"]


def synthetic_tables(tmp: Path) -> Path:
    """Write tiny SYNTHETIC processed tables in the real schema."""
    d = tmp / "processed"
    d.mkdir()
    for task in ("sentiment", "emotion", "risk"):
        rows = []
        for i in range(30):
            sp = "train" if i < 18 else "val" if i < 24 else "test"
            r = {"text_id": f"{task}-{sp[:2]}-{i:05d}", "text": SYN[i % len(SYN)] + f" {i}", "language": ["en-Latn", "hi-Latn", "hi-Deva"][i % 3],
                 "source": "synthetic-fixture", "split": sp, "group": ""}
            if task == "emotion":
                for j, e in enumerate(EMOTION_LABELS):
                    r[e] = -1 if (e == "disgust" and i % 3 == 0) else int((i + j) % 4 == 0)
            else:
                r["label"] = i % (3 if task == "sentiment" else 2)
            rows.append(r)
        pd.DataFrame(rows).to_csv(d / f"{task}.csv", index=False)
    return d


@pytest.fixture
def processed_dir(tmp_path, monkeypatch):
    d = synthetic_tables(tmp_path)
    monkeypatch.setattr("src.data.datasets.PROCESSED_DIR", d)
    return d


def _tiny_hf_tokenizer(tmp: Path) -> HFTokenizer:
    from transformers import BertTokenizerFast

    words = sorted({w for s in SYN for w in s.lower().split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words + list("abcdefghijklmnopqrstuvwxyz0123456789.,!?")
    vf = tmp / "vocab.txt"
    vf.write_text("\n".join(vocab))
    t = HFTokenizer.__new__(HFTokenizer)
    t.name, t.tok = "tiny-test", BertTokenizerFast(str(vf), do_lower_case=True)
    return t


@pytest.fixture
def tiny_bilstm(tmp_path):
    tok = WordTokenizer.build(SYN, min_freq=1)
    return MultiTaskBiLSTM(len(tok), emb_dim=16, hidden=16, shared_dim=16), tok


@pytest.fixture
def tiny_transformer(tmp_path):
    from transformers import BertConfig, BertModel

    tok = _tiny_hf_tokenizer(tmp_path)
    enc = BertModel(BertConfig(vocab_size=len(tok.tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, max_position_embeddings=256))
    return MultiTaskTransformer(enc, "tiny-random", shared_dim=16), tok


@pytest.fixture
def registry(tmp_path, tiny_bilstm, tiny_transformer):
    """Registry holding two untrained tiny models loaded through the real checkpoint save/load path."""
    adapters = {}
    for name, (m, tok) in (("bilstm", tiny_bilstm), ("transformer", tiny_transformer)):
        d = tmp_path / "ckpt" / name
        save_checkpoint(d, m, tok, {"name": name})
        (d / "calibration.json").write_text(json.dumps({"temperature": {"sentiment": 1.1, "emotion": 1.0, "risk": 0.9}, "emotion_thresholds": [0.5] * 6}))
        adapters[name] = NeuralAdapter(name, d, device="cpu")
    return ModelRegistry(adapters=adapters)


@pytest.fixture
def predictor(registry):
    return Predictor(registry)
