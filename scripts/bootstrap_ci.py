"""Bootstrap 95% CIs (1000 resamples of test examples) for headline metrics, plus paired differences vs the baseline.
Reads reports/metrics/predictions_<model>.csv (anonymised ids, no text). Writes reports/metrics/bootstrap_ci.json.
Caveat: resamples *examples* only (not training seeds), so it understates total run-to-run variance."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import METRICS_DIR  # noqa: E402

B, rng = 1000, np.random.default_rng(13)
models = [p.stem.replace("predictions_", "") for p in sorted(METRICS_DIR.glob("predictions_*.csv"))]
P = {m: pd.read_csv(METRICS_DIR / f"predictions_{m}.csv", keep_default_na=False) for m in models}


def get(m, task):
    d = P[m][P[m].task == task].reset_index(drop=True)
    return d


def metrics_for(task, d, idx):
    s = d.iloc[idx]
    if task == "risk":
        y = (s.true == "stress").astype(int)
        return {"risk_f1": f1_score(y, (s.pred == "stress").astype(int), zero_division=0),
                "risk_pr_auc": average_precision_score(y, s.risk_probability.astype(float)) if y.nunique() == 2 else np.nan}
    return {"sentiment_macro_f1": f1_score(s.true, s.pred, average="macro", zero_division=0)}


out = {"n_resamples": B, "ci": {}, "paired_diff_vs_baseline": {}}
samples = {}
for task in ("risk", "sentiment"):
    n = len(get(models[0], task))
    idxs = [rng.integers(0, n, n) for _ in range(B)]  # same resamples for every model -> paired
    for m in models:
        d = get(m, task)
        samples[(m, task)] = pd.DataFrame([metrics_for(task, d, i) for i in idxs])
for (m, task), df in samples.items():
    for c in df:
        out["ci"].setdefault(m, {})[c] = {"mean": float(df[c].mean()), "lo": float(df[c].quantile(.025)), "hi": float(df[c].quantile(.975))}
for m in models:
    if m == "baseline":
        continue
    for task in ("risk", "sentiment"):
        for c in samples[(m, task)]:
            diff = samples[(m, task)][c] - samples[("baseline", task)][c]
            out["paired_diff_vs_baseline"].setdefault(m, {})[c] = {"mean": float(diff.mean()), "lo": float(diff.quantile(.025)), "hi": float(diff.quantile(.975)),
                                                                   "ci_excludes_zero": bool(diff.quantile(.025) > 0 or diff.quantile(.975) < 0)}
(METRICS_DIR / "bootstrap_ci.json").write_text(json.dumps(out, indent=2))
for m, v in out["ci"].items():
    print(m, {k: f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]" for k, x in v.items()})
for m, v in out["paired_diff_vs_baseline"].items():
    print("diff", m, {k: f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] sig={x['ci_excludes_zero']}" for k, x in v.items()})
