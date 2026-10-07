"""Download raw datasets into data/raw/ (git-ignored)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.download import download_all  # noqa: E402

if __name__ == "__main__":
    m = download_all()
    for k, v in m["datasets"].items():
        print(k, "->", len(v["files"]), "files", v.get("revision", ""))
