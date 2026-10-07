"""Train the TF-IDF + Logistic Regression baseline (fast, CPU)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CKPT_DIR  # noqa: E402
from src.data.datasets import load_all  # noqa: E402
from src.models.baseline import TfidfBaseline  # noqa: E402

if __name__ == "__main__":
    train, val = load_all("train"), load_all("val")
    m = TfidfBaseline().fit(train, val)
    out = CKPT_DIR / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / "model.joblib")
    (out / "meta.json").write_text(json.dumps({"kind": "baseline", "name": "baseline", "n_parameters": m.n_parameters(), "best_C": m.best_C}, indent=2))
    print("baseline saved; params:", m.n_parameters(), "best C:", m.best_C)
