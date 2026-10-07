"""Print markdown result tables generated from reports/metrics/*.json (used to fill README.md - never typed by hand)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import METRICS_DIR  # noqa: E402

f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
comp = json.loads((METRICS_DIR / "comparison.json").read_text())
print("### Model comparison (test splits)\n")
print("| Model | Params | Sentiment macro-F1 | Emotion macro-F1 | Risk F1 (stress) | Risk macro-F1 | Risk PR-AUC | Risk ROC-AUC | Risk ECE (cal.) | Risk ECE (raw) | Latency ms (CPU) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in comp["models"]:
    print(f"| {m['display_name']} | {m['parameters']:,} | {f(m['sentiment_macro_f1'])} | {f(m['emotion_macro_f1'])} | {f(m['risk_f1'])} | {f(m['risk_macro_f1'])} | {f(m['risk_pr_auc'])} | {f(m['risk_roc_auc'])} | {f(m['risk_ece'])} | {f(m['risk_ece_uncalibrated'])} | {f(m['inference_latency_ms'],1)} |")
bl = json.loads((METRICS_DIR / "by_language.json").read_text())
print("\n### Macro-F1 by language / script (test)\n")
names = list(bl["by_language"])
print("| Task | Language (n) | " + " | ".join(names) + " |")
print("|---|---|" + "---|" * len(names))
for task in ("sentiment", "emotion", "risk"):
    langs = sorted({l for n in names for l in bl["by_language"][n][task]})
    for l in langs:
        n_ = next(bl["by_language"][n][task][l]["n"] for n in names if l in bl["by_language"][n][task])
        print(f"| {task} | {l} ({n_}) | " + " | ".join(f(bl["by_language"][n][task].get(l, {}).get("macro_f1")) for n in names) + " |")
print("\n### Sentiment macro-F1 by source (test)\n")
srcs = sorted({s for n in names for s in bl["by_source"][n]["sentiment"]})
print("| Source | " + " | ".join(names) + " |"); print("|---|" + "---|" * len(names))
for s in srcs:
    print(f"| {s} | " + " | ".join(f(bl["by_source"][n]["sentiment"].get(s, {}).get("macro_f1")) for n in names) + " |")
print("\n### Emotion per-class F1 (test; English / Hindi)\n")
print("| Model | Language | " + " | ".join(["anger", "disgust", "fear", "joy", "sadness", "surprise"]) + " |"); print("|---|---|" + "---|" * 6)
for n in names:
    m = json.loads((METRICS_DIR / f"{n}.json").read_text())["tasks"]["emotion"]["by_language"]
    for l, v in m.items():
        print(f"| {n} | {l} | " + " | ".join(f(v['per_class'][e]['f1']) for e in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]) + " |")
print("\n### Calibration (temperature scaling fitted on validation)\n")
print("| Model | Task | T | ECE raw → calibrated | Brier raw → calibrated |"); print("|---|---|---|---|---|")
for n in names:
    t = json.loads((METRICS_DIR / f"{n}.json").read_text())["tasks"]
    for task in ("sentiment", "emotion", "risk"):
        u = t[task]["uncalibrated"]
        print(f"| {n} | {task} | {f(t[task]['temperature'],2)} | {f(u.get('ece'))} → {f(t[task]['ece'])} | {f(u.get('brier'))} → {f(t[task]['brier'])} |")
