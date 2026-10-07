import numpy as np
import pandas as pd

from src.data import datasets
from src.data.prepare import _group_split, _norm_key, dedupe_splits
from src.preprocessing import clean_text, length_bucket, simple_tokenize


def test_dataset_loading(processed_dir):
    tr = datasets.load_task("sentiment", "train")
    assert len(tr) == 18 and set(tr.labels) <= {0, 1, 2}
    emo = datasets.load_task("emotion", "val")
    assert emo.labels.shape[1] == 6 and (emo.labels == -1).any()  # masked (unannotated) labels preserved
    assert datasets.processed_available() is True
    assert len(datasets.load_task("risk", "train", limit=5)) == 5
    assert set(datasets.load_all("test")) == {"sentiment", "emotion", "risk"}


def test_clean_text_masks_identifiers():
    t = clean_text("Mail me at jane.doe@example.com or +1 (555) 123-4567, see https://x.co/a @someone  hi")
    assert "example.com" not in t and "555" not in t and "https" not in t and "@someone" not in t
    assert "<email>" in t and "<phone>" in t and "<url>" in t and "@user" in t
    assert clean_text(None) == "" and clean_text("a​  b") == "a b"


def test_tokenize_handles_devanagari_and_latin():
    assert simple_tokenize("Yaar, mood off!") == ["yaar", ",", "mood", "off", "!"]
    assert "करता" in simple_tokenize("नहीं करता।") or any("कर" in x for x in simple_tokenize("नहीं करता।"))
    assert length_bucket(3) == "short(<8)" and length_bucket(100) == "long(40+)"


def test_norm_key_keeps_devanagari_vowel_signs():
    # regression: a \W-based key stripped matras and made different Hindi sentences collide
    assert _norm_key("मन नहीं") != _norm_key("मान नही")
    assert _norm_key("Hello, World!") == _norm_key("hello world")


def test_dedupe_removes_cross_split_leakage():
    df = pd.DataFrame({"text": ["same sentence here", "same sentence here!", "a different one", "unique train text", "train only"],
                       "split": ["train", "test", "test", "train", "train"], "label": 0})
    out, rep = dedupe_splits(df)
    assert "same sentence here" not in out[out.split == "train"].text.tolist()  # train copy removed, test kept
    assert (out.split == "test").sum() == 2 and rep["train_vs_val_test"] >= 1


def test_group_split_never_splits_a_group():
    g = pd.Series([f"g{i % 7}" for i in range(70)])
    assign = _group_split(g)
    assert set(assign.values()) <= {"train", "val", "test"} and len(assign) == 7
    sp = g.map(assign)
    assert all(sp[g == k].nunique() == 1 for k in assign)
