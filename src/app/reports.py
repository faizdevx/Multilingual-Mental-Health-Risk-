"""Read-only access to evaluation artefacts. Everything the UI shows is loaded from these files; if a file is missing
the helper returns None and the UI prints "Not available" - nothing is hard-coded."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import CKPT_DIR, EVALUATED_LANGUAGES, LANGUAGE_NAMES, METRICS_DIR, PROCESSED_DIR, REPORTS_DIR, TASK_LANGUAGES


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class Reports:
    def __init__(self, metrics_dir: Path = METRICS_DIR, reports_dir: Path = REPORTS_DIR, processed_dir: Path = PROCESSED_DIR, ckpt_dir: Path = CKPT_DIR):
        self.metrics_dir, self.reports_dir, self.processed_dir, self.ckpt_dir = metrics_dir, reports_dir, processed_dir, ckpt_dir

    def comparison(self):
        return _json(self.metrics_dir / "comparison.json")

    def model_metrics(self, name: str):
        return _json(self.metrics_dir / f"{name}.json")

    def by_language(self):
        return _json(self.metrics_dir / "by_language.json")

    def error_summary(self):
        return _json(self.metrics_dir / "error_summary.json")

    def langid(self):
        return _json(self.metrics_dir / "langid.json")

    def dataset_stats(self):
        return _json(self.processed_dir / "stats.json")

    def history(self, name: str):
        return _json(self.ckpt_dir / name / "history.json")

    def error_table(self) -> pd.DataFrame | None:
        p = self.reports_dir / "error_analysis.csv"
        return pd.read_csv(p, keep_default_na=False) if p.exists() else None

    def best_model_entry(self):
        c = self.comparison()
        if not c or not c.get("models"):
            return None
        order = {"transformer": 0, "bilstm": 1, "baseline": 2}
        return sorted(c["models"], key=lambda m: order.get(m["name"], 9))[0]

    def calibration_status(self, name: str) -> str:
        cal = _json(self.ckpt_dir / name / "calibration.json")
        if not cal:
            return "Not available"
        return "Temperature-scaled on validation split (T = " + ", ".join(f"{k}: {v:.2f}" for k, v in cal["temperature"].items()) + ")"

    def artifact_status(self) -> dict:
        names = ["comparison.json", "by_language.json", "error_summary.json", "langid.json"]
        d = {n: (self.metrics_dir / n).exists() for n in names}
        d["error_analysis.csv"] = (self.reports_dir / "error_analysis.csv").exists()
        d["reliability_risk.png"] = (self.reports_dir / "figures" / "reliability_risk.png").exists()
        return d

    def language_table(self) -> dict:
        """{task: {language: {model: {...}}}} restricted to languages that actually have evaluation data."""
        bl = self.by_language()
        if not bl:
            return {}
        out: dict = {}
        for model, tasks in bl["by_language"].items():
            for task, langs in tasks.items():
                for lang, m in langs.items():
                    out.setdefault(task, {}).setdefault(lang, {})[model] = m
        return out


SUPPORTED_LANGUAGES = [{"tag": t, "name": LANGUAGE_NAMES[t], "tasks": [k for k, v in TASK_LANGUAGES.items() if t in v]} for t in EVALUATED_LANGUAGES]
