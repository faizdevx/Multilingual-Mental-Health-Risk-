"""Shared entry logic for scripts/train_lstm.py and scripts/train_transformer.py."""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from src.config import CKPT_DIR, METRICS_DIR, MAX_LEN, TASK_LABELS, TRANSFORMER_NAME
from src.data.datasets import load_all, processed_available
from src.models.io import detect_device
from src.models.multitask import MultiTaskBiLSTM, MultiTaskTransformer
from src.models.tokenization import HFTokenizer, WordTokenizer
from src.training.trainer import TrainConfig, set_seed, train_multitask


def build_parser(kind: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"Train the multi-task {kind} model")
    ap.add_argument("--smoke", action="store_true", help="tiny, fast run on a data subset (output goes to checkpoints/smoke_*; NOT a real model)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=None, help="override max sequence length for all tasks")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task and split")
    ap.add_argument("--loss-weights", default="1,1,1", help="lambda for sentiment,emotion,risk")
    ap.add_argument("--out", default=None)
    if kind == "transformer":
        ap.add_argument("--model-name", default=TRANSFORMER_NAME)
        ap.add_argument("--encoder-lr", type=float, default=None)
    return ap


def main(kind: str, argv=None) -> dict:
    args = build_parser(kind).parse_args(argv)
    if not processed_available():
        raise SystemExit("Processed data not found. Run: python scripts/download_data.py && python scripts/prepare_data.py")
    dev = detect_device(args.device)
    torch.set_num_threads(os.cpu_count() or 1)
    set_seed(args.seed)
    lw = dict(zip(("sentiment", "emotion", "risk"), (float(x) for x in args.loss_weights.split(","))))
    limit = args.limit or (64 if args.smoke else None)
    train, val = load_all("train", limit, seed=args.seed), load_all("val", limit, seed=args.seed)

    defaults = {"bilstm": dict(epochs=12, batch_size=32, lr=2e-3, patience=3),
                "transformer": dict(epochs=3, batch_size=16, lr=1e-3, patience=2)}[kind]
    cfg = TrainConfig(kind=kind, seed=args.seed, device=dev, smoke=args.smoke, loss_weights=lw, **defaults)
    if args.smoke:
        cfg.epochs, cfg.patience = 2, 2
    for k, v in (("epochs", args.epochs), ("batch_size", args.batch_size), ("lr", args.lr), ("patience", args.patience)):
        if v is not None:
            setattr(cfg, k, v)
    if kind == "transformer" and args.encoder_lr:
        cfg.encoder_lr = args.encoder_lr
    if args.max_len:
        cfg.max_len = {t: args.max_len for t in TASK_LABELS}

    if kind == "bilstm":
        tok = WordTokenizer.build([t for d in train.values() for t in d.texts], min_freq=1 if args.smoke else 2)
        model = MultiTaskBiLSTM(len(tok), **(dict(emb_dim=32, hidden=32, shared_dim=32) if args.smoke else {}))
        name = "bilstm"
    else:
        tok = HFTokenizer(args.model_name)
        if args.smoke:  # tiny randomly-initialised encoder: validates the pipeline offline, learns nothing useful
            from transformers import BertConfig, BertModel

            enc = BertModel(BertConfig(vocab_size=len(tok.tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                       intermediate_size=128, max_position_embeddings=256))
            model = MultiTaskTransformer(enc, args.model_name + " (tiny random smoke encoder)", shared_dim=32)
        else:
            model = MultiTaskTransformer.from_pretrained(args.model_name)
        name = "transformer"
    out = args.out or str(CKPT_DIR / (f"smoke_{name}" if args.smoke else name))
    print(f"device={dev} params={model.n_parameters():,} train={ {t: len(d) for t, d in train.items()} } out={out}", flush=True)
    t0 = time.time()
    res = train_multitask(model, tok, train, val, cfg, out, extra_meta={"n_parameters": model.n_parameters(), "name": name})
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        (METRICS_DIR / f"class_balance.json").write_text(json.dumps(res["class_balance"], indent=2))
    print(f"done in {time.time() - t0:.0f}s; best mean val macro-F1 = {res['best']:.4f}; checkpoint -> {out}", flush=True)
    return res
