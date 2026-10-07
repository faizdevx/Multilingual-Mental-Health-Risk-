"""Calibration utilities: temperature scaling, ECE, Brier score, reliability diagram.

Calibration makes model probabilities *statistically* consistent with observed frequencies on the evaluation
population. It does NOT make them clinical probabilities.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / T))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: weighted mean |accuracy - confidence| over equal-width confidence bins."""
    probs = np.asarray(probs)
    conf = probs.max(1)
    correct = (probs.argmax(1) == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - 1[y=k])^2 (binary => 2x the usual binary Brier)."""
    probs = np.asarray(probs)
    onehot = np.eye(probs.shape[1])[np.asarray(y_true)]
    return float(((probs - onehot) ** 2).sum(1).mean())


def binary_brier(p_pos: np.ndarray, y_true: np.ndarray) -> float:
    return float(((np.asarray(p_pos) - np.asarray(y_true)) ** 2).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, kind: str = "softmax") -> float:
    """Single scalar temperature minimising validation NLL (Guo et al., 2017). Returns 1.0 if fitting fails."""
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def nll():
        opt.zero_grad()
        T = logT.exp()
        if kind == "softmax":
            loss = F.cross_entropy(lg / T, yt)
        else:
            m = (yt >= 0).double()  # -1 marks labels that were not annotated
            l = F.binary_cross_entropy_with_logits(lg / T, yt.clamp(min=0).double(), reduction="none")
            loss = (l * m).sum() / m.sum().clamp(min=1)
        loss.backward()
        return loss

    try:
        opt.step(nll)
        T = float(logT.exp().item())
        return T if np.isfinite(T) and 0.05 < T < 20 else 1.0
    except Exception:
        return 1.0


def reliability_curve(p_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """Binary reliability data: mean predicted prob vs observed frequency per bin."""
    p_pos, y_true = np.asarray(p_pos), np.asarray(y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    out = {"bin_lower": [], "bin_upper": [], "mean_predicted": [], "fraction_positive": [], "count": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_pos >= lo) & ((p_pos < hi) if hi < 1 else (p_pos <= hi))
        out["bin_lower"].append(float(lo)); out["bin_upper"].append(float(hi)); out["count"].append(int(m.sum()))
        out["mean_predicted"].append(float(p_pos[m].mean()) if m.any() else None)
        out["fraction_positive"].append(float(y_true[m].mean()) if m.any() else None)
    return out


def plot_reliability(curves: dict[str, dict], path, title: str = "Reliability diagram (risk task, test set)") -> None:
    """curves: {"label": reliability_curve(...)}. Saves a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=130)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    for name, c in curves.items():
        xs = [x for x, y in zip(c["mean_predicted"], c["fraction_positive"]) if x is not None]
        ys = [y for x, y in zip(c["mean_predicted"], c["fraction_positive"]) if x is not None]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5, label=name)
    ax.set_xlabel("mean predicted probability (stress class)")
    ax.set_ylabel("observed fraction positive")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
