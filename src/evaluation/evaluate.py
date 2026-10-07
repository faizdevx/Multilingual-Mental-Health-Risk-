"""Evaluate one model: calibration fitting (on validation), test metrics (overall / per language / per source),
reliability data and per-example test predictions (no raw text).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import EMOTION_LABELS, TASK_LABELS
from src.data.datasets import TaskData
from src.training.calibration import binary_brier, fit_temperature, reliability_curve, sigmoid, softmax
from src.training.metrics import compute_task_metrics, tune_emotion_thresholds

PROBE_TEXT = "I went to the market and bought some vegetables today."  # neutral, fixed probe for latency timing


def _probs(task, logits, T=1.0):
    return sigmoid(logits, T) if task == "emotion" else softmax(logits, T)


def fit_calibration(adapter, val: dict[str, TaskData]) -> dict:
    """Temperature per task fit on validation NLL; emotion thresholds tuned on validation F1 (calibrated probs)."""
    cal = {"temperature": {}, "emotion_thresholds": None, "fit_on": "validation split only"}
    for task, d in val.items():
        lg = adapter.predict_logits(task, d.texts)
        y = d.labels
        cal["temperature"][task] = fit_temperature(lg, y, "sigmoid" if task == "emotion" else "softmax")
        if task == "emotion":
            thr = tune_emotion_thresholds(y, sigmoid(lg, cal["temperature"][task]))
            cal["emotion_thresholds"] = thr.tolist()
    return cal


def _group_metrics(task, y, probs, groups, thr):
    out = {}
    for g in sorted(set(groups)):
        m = groups == g
        out[g] = compute_task_metrics(task, y[m], probs[m], thr)
    return out


def evaluate_adapter(adapter, val: dict[str, TaskData], test: dict[str, TaskData]) -> tuple[dict, pd.DataFrame, dict]:
    cal = fit_calibration(adapter, val)
    res: dict = {"model": adapter.name, "n_parameters": adapter.n_parameters(), "calibration": cal, "tasks": {}}
    preds = []
    for task, d in test.items():
        lg = adapter.predict_logits(task, d.texts)
        y, T = d.labels, cal["temperature"][task]
        thr = np.array(cal["emotion_thresholds"]) if task == "emotion" else None
        p_raw, p_cal = _probs(task, lg), _probs(task, lg, T)
        # headline metrics use CALIBRATED probabilities (temperature fit on validation); raw is reported for comparison
        m = compute_task_metrics(task, y, p_cal, thr)
        m_raw = compute_task_metrics(task, y, p_raw, thr if task == "emotion" else None)
        m["uncalibrated"] = {k: m_raw.get(k) for k in ("ece", "brier", "macro_f1", "accuracy", "pr_auc", "roc_auc") if k in m_raw}
        m["temperature"] = T
        langs = d.languages
        m["by_language"] = _group_metrics(task, y, p_cal, langs, thr)
        m["by_source"] = _group_metrics(task, y, p_cal, d.df.source.to_numpy(), thr)
        if task == "risk":
            m["reliability_calibrated"] = reliability_curve(p_cal[:, 1], y)
            m["reliability_uncalibrated"] = reliability_curve(p_raw[:, 1], y)
            m["binary_brier_calibrated"] = binary_brier(p_cal[:, 1], y)
            m["binary_brier_uncalibrated"] = binary_brier(p_raw[:, 1], y)
        res["tasks"][task] = m
        df = d.df[["text_id", "language", "source"]].copy()
        df["task"], df["model"] = task, adapter.name
        df["n_words"] = d.df.text.str.split().str.len().to_numpy()
        if task == "emotion":
            pred = (p_cal >= thr).astype(int)
            df["true"] = ["|".join(e for e, v in zip(EMOTION_LABELS, r) if v == 1) or "none" for r in y]
            df["pred"] = ["|".join(e for e, v in zip(EMOTION_LABELS, r) if v == 1) or "none" for r in pred]
            # only compare labels that were annotated for the row
            err = [((r_p != r_t) & (r_t >= 0)).any() for r_p, r_t in zip(pred, y)]
            df["error"] = np.array(err)
            df["confidence"] = p_cal.max(1)
            df["risk_probability"] = np.nan
        else:
            labels = TASK_LABELS[task]
            df["true"] = [labels[i] for i in y]
            df["pred"] = [labels[i] for i in p_cal.argmax(1)]
            df["error"] = (p_cal.argmax(1) != y)
            df["confidence"] = p_cal.max(1)
            df["risk_probability"] = p_cal[:, 1] if task == "risk" else np.nan
        preds.append(df)
    res["inference_latency_ms"] = measure_latency(adapter)
    return res, pd.concat(preds, ignore_index=True), cal


def measure_latency(adapter, n: int = 15) -> dict:
    """Median single-request latency (all three tasks on one short text), CPU/accelerator as configured."""
    for _ in range(2):
        for t in TASK_LABELS:
            adapter.predict_logits(t, [PROBE_TEXT])
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        for t in TASK_LABELS:
            adapter.predict_logits(t, [PROBE_TEXT])
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median": float(np.median(ts)), "p95": float(np.percentile(ts, 95)), "device": getattr(adapter, "device", "cpu"), "n_runs": n}
