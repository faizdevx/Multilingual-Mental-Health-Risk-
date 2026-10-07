"""Build every evaluation artefact under reports/ from trained checkpoints.

Outputs
  reports/metrics/<model>.json             full metrics per model (overall, per language, per source, calibration)
  reports/metrics/comparison.json          model comparison table
  reports/metrics/by_language.json         per-language metrics for every model/task
  reports/metrics/error_summary.json       error counts, confusion matrices, length buckets, cross-task agreement
  reports/metrics/predictions_<model>.csv  per-example test predictions (anonymised ids, no text)
  reports/error_analysis.csv               misclassified test examples (anonymised ids, no text)
  reports/figures/*.png
Checkpoint dirs additionally receive calibration.json (temperatures + emotion thresholds fit on validation).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import EMOTION_LABELS, FIGURES_DIR, METRICS_DIR, REPORTS_DIR, SENTIMENT_LABELS, TASK_LABELS
from src.data.datasets import load_all
from src.evaluation.adapters import ARCH, DISPLAY, available_models, load_adapter
from src.evaluation.evaluate import evaluate_adapter
from src.preprocessing import length_bucket
from src.training.calibration import plot_reliability, softmax

ERR_COLS = ["text_id", "language", "true_sentiment", "predicted_sentiment", "true_emotion", "predicted_emotion",
            "true_risk", "predicted_risk", "risk_probability"]


def _j(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def error_rows(preds: pd.DataFrame) -> pd.DataFrame:
    e = preds[preds.error].copy()
    out = pd.DataFrame({c: "" for c in ERR_COLS}, index=e.index)
    out["text_id"], out["language"] = e.text_id, e.language
    for task, key in (("sentiment", "sentiment"), ("emotion", "emotion"), ("risk", "risk")):
        m = e.task == task
        out.loc[m, f"true_{key}"] = e.loc[m, "true"]
        out.loc[m, f"predicted_{key}"] = e.loc[m, "pred"]
    out["risk_probability"] = e.risk_probability.round(4)
    out["task"], out["model"], out["source"] = e.task, e.model, e.source
    out["length_bucket"] = e.n_words.map(length_bucket)
    out["confidence"] = e.confidence.round(4)
    if len(e):
        is_risk = e.task == "risk"
        out["error_type"] = np.where(is_risk & (e.pred == "stress"), "false_positive",
                                     np.where(is_risk & (e.pred == "non_stress"), "false_negative", "misclassified"))
    else:
        out["error_type"] = ""
    return out


def summarise_errors(name: str, preds: pd.DataFrame, res: dict, adapter, test, ) -> dict:
    s: dict = {"by_task": {}}
    for task in TASK_LABELS:
        p = preds[preds.task == task]
        p = p.assign(bucket=p.n_words.map(length_bucket))
        s["by_task"][task] = {
            "n": int(len(p)), "errors": int(p.error.sum()), "error_rate": float(p.error.mean()),
            "by_language": {l: {"n": int(len(g)), "errors": int(g.error.sum()), "error_rate": float(g.error.mean())} for l, g in p.groupby("language")},
            "by_source": {l: {"n": int(len(g)), "errors": int(g.error.sum()), "error_rate": float(g.error.mean())} for l, g in p.groupby("source")},
            "by_length_bucket": {b: {"n": int(len(g)), "errors": int(g.error.sum()), "error_rate": float(g.error.mean())} for b, g in p.groupby("bucket")},
            "confusion_matrix": res["tasks"][task].get("confusion_matrix"),
            "labels": TASK_LABELS[task],
        }
        if task == "risk":
            s["by_task"][task]["false_positives"] = int(((p.pred == "stress") & (p.true == "non_stress")).sum())
            s["by_task"][task]["false_negatives"] = int(((p.pred == "non_stress") & (p.true == "stress")).sum())
        if task == "emotion":
            per = {}
            for e in EMOTION_LABELS:
                m = p.true.str.contains(e) ^ p.pred.str.contains(e)
                per[e] = int(m.sum())
            s["by_task"][task]["label_disagreements"] = per
    # exploratory: sentiment head vs risk head on the *same* Dreaddit test texts (sentiment has no ground truth there)
    rt = test["risk"]
    sl = adapter.predict_logits("sentiment", rt.texts)
    rp = softmax(adapter.predict_logits("risk", rt.texts) / res["calibration"]["temperature"]["risk"])
    sp = softmax(sl / res["calibration"]["temperature"]["sentiment"]).argmax(1)
    ctab = pd.crosstab(pd.Series([SENTIMENT_LABELS[i] for i in sp], name="pred_sentiment"),
                       pd.Series(np.where(rp.argmax(1) == 1, "stress", "non_stress"), name="pred_risk"))
    s["sentiment_vs_risk_on_risk_test_texts"] = {
        "note": "Exploratory only: the sentiment head was never validated on Dreaddit texts (different population/domain).",
        "crosstab": json.loads(ctab.to_json()),
        "share_true_stress_predicted_negative_sentiment": float(((sp == 0) & (rt.labels == 1)).sum() / max(1, (rt.labels == 1).sum())),
        "share_true_nonstress_predicted_negative_sentiment": float(((sp == 0) & (rt.labels == 0)).sum() / max(1, (rt.labels == 0).sum())),
    }
    return s


def build_reports(models: list[str] | None = None, device: str | None = None) -> dict:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    val, test = load_all("val"), load_all("test")
    models = models or available_models()
    results, summaries, err_frames, curves = {}, {}, [], {}
    for name in models:
        ad = load_adapter(name, device)
        res, preds, cal = evaluate_adapter(ad, val, test)
        (ad.ckpt_dir / "calibration.json").write_text(json.dumps(cal, indent=2))
        res["architecture"] = ARCH[ad.kind]
        res["display_name"] = DISPLAY[ad.kind]
        res["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results[name] = res
        preds.drop(columns=[]).to_csv(METRICS_DIR / f"predictions_{name}.csv", index=False)
        err_frames.append(error_rows(preds))
        summaries[name] = summarise_errors(name, preds, _j(res) | {"tasks": {t: {**res["tasks"][t], "confusion_matrix": res["tasks"][t].get("confusion_matrix")} for t in res["tasks"]}}, ad, test)
        (METRICS_DIR / f"{name}.json").write_text(json.dumps(_j(res), indent=2))
        curves[f"{DISPLAY[ad.kind]} (calibrated)"] = res["tasks"]["risk"]["reliability_calibrated"]
        curves[f"{DISPLAY[ad.kind]} (raw)"] = res["tasks"]["risk"]["reliability_uncalibrated"]
        print(f"evaluated {name}", flush=True)

    comparison = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "note": "All numbers are measured on held-out test splits; the three tasks come from different datasets/populations. "
                          "Sentiment F1 pools English + Hinglish test sets; emotion F1 pools English + Hindi; risk is English only.",
                  "models": []}
    for name, r in results.items():
        t = r["tasks"]
        comparison["models"].append({
            "name": name, "display_name": r["display_name"], "architecture": r["architecture"], "parameters": r["n_parameters"],
            "sentiment_macro_f1": t["sentiment"]["macro_f1"], "sentiment_weighted_f1": t["sentiment"]["weighted_f1"],
            "emotion_macro_f1": t["emotion"]["macro_f1"], "emotion_weighted_f1": t["emotion"]["weighted_f1"],
            "risk_f1": t["risk"]["f1"], "risk_macro_f1": t["risk"]["macro_f1"], "risk_recall": t["risk"]["recall"],
            "risk_precision": t["risk"]["precision"], "risk_pr_auc": t["risk"]["pr_auc"], "risk_roc_auc": t["risk"]["roc_auc"],
            "risk_ece": t["risk"]["ece"], "risk_brier": t["risk"]["binary_brier_calibrated"],
            "risk_ece_uncalibrated": t["risk"]["uncalibrated"]["ece"],
            "inference_latency_ms": r["inference_latency_ms"]["median"], "inference_device": r["inference_latency_ms"]["device"],
            "checkpoint_status": "available"})
    (METRICS_DIR / "comparison.json").write_text(json.dumps(_j(comparison), indent=2))

    by_lang = {n: {task: {l: {k: v for k, v in m.items() if k in ("n", "macro_f1", "weighted_f1", "accuracy", "f1", "pr_auc", "roc_auc", "ece", "per_class")}
                            for l, m in r["tasks"][task]["by_language"].items()} for task in r["tasks"]} for n, r in results.items()}
    by_src = {n: {task: {l: {k: v for k, v in m.items() if k in ("n", "macro_f1", "accuracy")} for l, m in r["tasks"][task]["by_source"].items()}
                  for task in r["tasks"]} for n, r in results.items()}
    (METRICS_DIR / "by_language.json").write_text(json.dumps(_j({"by_language": by_lang, "by_source": by_src}), indent=2))
    (METRICS_DIR / "error_summary.json").write_text(json.dumps(_j(summaries), indent=2))
    errs = pd.concat(err_frames, ignore_index=True)
    errs.to_csv(REPORTS_DIR / "error_analysis.csv", index=False)
    plot_reliability(curves, FIGURES_DIR / "reliability_risk.png")
    _plot_language(by_lang, FIGURES_DIR / "f1_by_language.png")
    return comparison


def _plot_language(by_lang: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=130)
    for ax, (task, key) in zip(axes, (("sentiment", "macro_f1"), ("emotion", "macro_f1"), ("risk", "macro_f1"))):
        langs = sorted({l for m in by_lang.values() for l in m[task]})
        w = 0.8 / max(1, len(by_lang))
        for i, (name, m) in enumerate(by_lang.items()):
            ax.bar([j + i * w for j in range(len(langs))], [m[task].get(l, {}).get(key) or 0 for l in langs], w, label=DISPLAY.get(name if name != "transformer" else "transformer", name))
        ax.set_xticks([j + 0.4 - w / 2 for j in range(len(langs))]); ax.set_xticklabels(langs, fontsize=7)
        ax.set_title(f"{task} macro-F1 (test)", fontsize=9); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
