"""Train the multi-task transformer model.  python scripts/train_transformer.py [--smoke] [--epochs N] ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.run import main  # noqa: E402

if __name__ == "__main__":
    main("transformer")
