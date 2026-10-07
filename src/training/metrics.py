"""Task metrics. All functions take numpy arrays and return plain-python dicts (JSON-serialisable)."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

from src.config import EMOTION_LABELS, RISK_LABELS, SENTIMENT_LABELS
from src.training.calibration import brier_score, expected_calibration_error


def _f(x) -> float:
    return float(x)


def multiclass_metrics(y_true, probs, labels: list[str]) -> dict:
    y_true = np.asarray(y_true)
    pred = probs.argmax(1)
    idx = list(range(len(labels)))
    kw = dict(labels=idx, zero_division=0)
    p, r, f, s = skm.precision_recall_fscore_support(y_true, pred, **kw)
    out = {
        "n": int(len(y_true)),
        "accuracy": _f(skm.accuracy_score(y_true, pred)),
        "macro_f1": _f(f.mean()),
        "weighted_f1": _f(skm.f1_score(y_true, pred, average="weighted", **kw)),
        "macro_precision": _f(p.mean()),
        "macro_recall": _f(r.mean()),
        "per_class": {l: {"precision": _f(p[i]), "recall": _f(r[i]), "f1": _f(f[i]), "support": int(s[i])} for i, l in enumerate(labels)},
        "confusion_matrix": skm.confusion_matrix(y_true, pred, labels=idx).tolist(),
        "ece": expected_calibration_error(probs, y_true),
        "brier": brier_score(probs, y_true),
    }
    return out


def sentiment_metrics(y_true, probs) -> dict:
    return multiclass_metrics(y_true, probs, SENTIMENT_LABELS)


def risk_metrics(y_true, probs) -> dict:
    """Binary risk task. Reports positive-class P/R/F1, ROC-AUC, PR-AUC and calibration."""
    y_true = np.asarray(y_true)
    out = multiclass_metrics(y_true, probs, RISK_LABELS)
    pos = probs[:, 1]
    pred = (pos >= 0.5).astype(int)
    out["positive_label"] = RISK_LABELS[1]
    out["precision"] = _f(skm.precision_score(y_true, pred, zero_division=0))
    out["recall"] = _f(skm.recall_score(y_true, pred, zero_division=0))
    out["f1"] = _f(skm.f1_score(y_true, pred, zero_division=0))
    out["decision_threshold"] = 0.5
    if len(set(y_true.tolist())) == 2:
        out["roc_auc"] = _f(skm.roc_auc_score(y_true, pos))
        out["pr_auc"] = _f(skm.average_precision_score(y_true, pos))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    return out


def emotion_metrics(y_true, probs, thresholds=None) -> dict:
    """Multi-label emotion. y_true: (n, L) in {0,1,-1}; -1 = label not annotated for that row (masked)."""
    y_true = np.asarray(y_true)
    L = y_true.shape[1]
    thr = np.full(L, 0.5) if thresholds is None else np.asarray(thresholds)
    pred = (probs >= thr).astype(int)
    per, f1s, sup = {}, [], []
    P, R = [], []
    for j, name in enumerate(EMOTION_LABELS):
        m = y_true[:, j] >= 0
        if m.sum() == 0:
            per[name] = {"f1": None, "precision": None, "recall": None, "support": 0, "n_annotated": 0}
            continue
        yt, yp = y_true[m, j], pred[m, j]
        f = skm.f1_score(yt, yp, zero_division=0)
        pr = skm.precision_score(yt, yp, zero_division=0)
        rc = skm.recall_score(yt, yp, zero_division=0)
        per[name] = {"f1": _f(f), "precision": _f(pr), "recall": _f(rc), "support": int(yt.sum()), "n_annotated": int(m.sum()),
                     "threshold": _f(thr[j]),
                     "pr_auc": _f(skm.average_precision_score(yt, probs[m, j])) if yt.sum() > 0 else None}
        f1s.append(f); sup.append(yt.sum()); P.append(pr); R.append(rc)
    sup = np.array(sup)
    # micro over annotated cells only
    valid = y_true >= 0
    out = {
        "n": int(len(y_true)),
        "macro_f1": _f(np.mean(f1s)) if f1s else None,
        "weighted_f1": _f(np.average(f1s, weights=sup)) if sup.sum() > 0 else None,
        "macro_precision": _f(np.mean(P)) if P else None,
        "macro_recall": _f(np.mean(R)) if R else None,
        "micro_f1": _f(skm.f1_score(y_true[valid], pred[valid], zero_division=0)),
        "per_class": per,
        "brier": _f(np.mean(((probs - np.clip(y_true, 0, 1)) ** 2)[valid])),
        "ece": expected_calibration_error(np.stack([1 - probs[valid], probs[valid]], 1), np.clip(y_true, 0, 1)[valid]),
    }
    out["decision_thresholds"] = {n: _f(t) for n, t in zip(EMOTION_LABELS, thr)}
    return out


def tune_emotion_thresholds(y_true, probs, grid=np.arange(0.1, 0.91, 0.05)) -> np.ndarray:
    """Per-label threshold maximising F1 on *validation* data (annotated cells only)."""
    thr = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        m = y_true[:, j] >= 0
        if m.sum() == 0 or y_true[m, j].sum() == 0:
            continue
        scores = [skm.f1_score(y_true[m, j], (probs[m, j] >= t).astype(int), zero_division=0) for t in grid]
        thr[j] = grid[int(np.argmax(scores))]
    return thr


def compute_task_metrics(task: str, y_true, probs, thresholds=None) -> dict:
    if task == "sentiment":
        return sentiment_metrics(y_true, probs)
    if task == "risk":
        return risk_metrics(y_true, probs)
    return emotion_metrics(y_true, probs, thresholds)


def primary_score(task: str, m: dict) -> float:
    """Validation score used for early stopping / checkpoint selection (macro F1)."""
    return float(m["macro_f1"]) if m.get("macro_f1") is not None else 0.0
