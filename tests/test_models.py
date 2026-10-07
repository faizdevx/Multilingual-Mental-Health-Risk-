import numpy as np
import torch

from src.config import TASK_LABELS
from src.models.multitask import task_loss
from src.models.tokenization import WordTokenizer


def test_word_tokenizer_roundtrip_and_padding():
    tok = WordTokenizer.build(["hello world", "hello again"], min_freq=1)
    enc = tok.encode(["hello world", "hello"], max_len=5)
    assert enc["input_ids"].shape == (2, 2) and enc["attention_mask"].sum().item() == 3
    assert tok.encode(["zzzunknown"], 4)["input_ids"][0, 0].item() == 1  # <unk>
    assert WordTokenizer.from_json(tok.to_json()).itos == tok.itos
    assert tok.encode([""], 4)["attention_mask"].sum().item() == 1  # empty text still yields a valid sequence


def test_hf_tokenizer_wrapper(tiny_transformer):
    _, tok = tiny_transformer
    enc = tok.encode(["the weather is nice", "hi"], 8)
    assert enc["input_ids"].shape[0] == 2 and enc["input_ids"].shape[1] <= 8


def _check_multitask(model, tok):
    enc = tok.encode(["the weather is nice", "i went to the market today and it was fine"], 16)
    out = model(enc["input_ids"], enc["attention_mask"])  # all heads
    assert set(out) == {"sentiment", "emotion", "risk"}
    for t, lg in out.items():
        assert lg.shape == (2, len(TASK_LABELS[t]))
    assert model(enc["input_ids"], enc["attention_mask"], "risk").shape == (2, 2)
    model.zero_grad()
    sum(v.sum() for v in out.values()).backward()
    assert all(p.grad is not None for p in model.heads.parameters())


def test_bilstm_forward_and_multitask_outputs(tiny_bilstm):
    _check_multitask(*tiny_bilstm)


def test_transformer_forward_and_multitask_outputs(tiny_transformer):
    _check_multitask(*tiny_transformer)


def test_padding_does_not_change_output(tiny_bilstm):
    model, tok = tiny_bilstm
    model.eval()
    a = tok.encode(["the weather is nice"], 16)
    b = tok.encode(["the weather is nice", "i went to the market today and it was fine"], 16)
    la = model(a["input_ids"], a["attention_mask"], "sentiment")[0]
    lb = model(b["input_ids"], b["attention_mask"], "sentiment")[0]
    assert torch.allclose(la, lb, atol=1e-5)


def test_emotion_loss_ignores_masked_labels():
    logits = torch.zeros(2, 6)
    y = torch.tensor([[1, -1, 0, 0, 0, 0], [0, -1, 0, 0, 0, 1]])
    base = task_loss("emotion", logits, y)
    logits2 = logits.clone(); logits2[:, 1] = 50.0  # changing the masked column must not matter
    assert torch.isclose(base, task_loss("emotion", logits2, y))
    assert task_loss("sentiment", torch.zeros(2, 3), torch.tensor([0, 2])).item() > 0


def test_micro_training_loop_runs_and_saves(processed_dir, tmp_path, tiny_bilstm):
    from src.data.datasets import load_all
    from src.models.io import load_checkpoint
    from src.training.trainer import TrainConfig, train_multitask

    model, tok = tiny_bilstm
    cfg = TrainConfig(kind="bilstm", epochs=2, batch_size=8, patience=2, smoke=True)
    res = train_multitask(model, tok, load_all("train"), load_all("val"), cfg, tmp_path / "out")
    assert len(res["history"]) >= 1 and (tmp_path / "out" / "model.pt").exists()
    h = res["history"][0]
    assert {"sentiment", "emotion", "risk"} <= set(h["train_loss"]) and "total_weighted" in h["val_loss"]
    m2, _, meta = load_checkpoint(tmp_path / "out")
    assert meta["kind"] == "bilstm"
