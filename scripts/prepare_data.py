"""Build processed per-task tables with leakage-safe splits (data/processed/, git-ignored)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.prepare import main  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
