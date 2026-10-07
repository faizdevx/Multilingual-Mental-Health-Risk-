"""Tokenizers: a trainable word-level tokenizer for the BiLSTM and a thin HF wrapper for the transformer.

Both expose `encode(texts, max_len) -> dict(input_ids, attention_mask)` (torch tensors, dynamic padding).
"""
from __future__ import annotations

import json
from collections import Counter

import torch

from src.preprocessing import simple_tokenize

PAD, UNK = "<pad>", "<unk>"


class WordTokenizer:
    def __init__(self, vocab: list[str] | None = None):
        self.itos = vocab or [PAD, UNK]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2, max_size: int = 30000) -> "WordTokenizer":
        c = Counter(tok for t in texts for tok in simple_tokenize(t))
        words = [w for w, n in c.most_common(max_size) if n >= min_freq]
        return cls([PAD, UNK] + words)

    def __len__(self) -> int:
        return len(self.itos)

    def tokenize(self, text: str) -> list[str]:
        return simple_tokenize(text)

    def encode_one(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(t, 1) for t in simple_tokenize(text)[:max_len]]
        return ids or [1]

    def encode(self, texts: list[str], max_len: int) -> dict:
        seqs = [self.encode_one(t, max_len) for t in texts]
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def to_json(self) -> str:
        return json.dumps(self.itos, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "WordTokenizer":
        return cls(json.loads(s))


class HFTokenizer:
    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self.name = name_or_path
        self.tok = AutoTokenizer.from_pretrained(name_or_path)

    def tokenize(self, text: str) -> list[str]:
        return self.tok.tokenize(text)

    def encode(self, texts: list[str], max_len: int) -> dict:
        b = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
