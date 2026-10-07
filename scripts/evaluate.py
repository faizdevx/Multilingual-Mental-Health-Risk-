"""Evaluate all available checkpoints on validation (calibration) and test (metrics) and write reports/."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.report import build_reports  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="subset of: baseline bilstm transformer")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    comp = build_reports(a.models, a.device)
    for m in comp["models"]:
        print(m["display_name"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k.startswith(("sentiment_macro", "emotion_macro", "risk_f1", "risk_pr", "parameters"))})
