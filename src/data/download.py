"""Download the public datasets used by the project into data/raw (git-ignored).

Sources (see DATASET.md):
  * Dreaddit            -> authors' site (Columbia University)
  * tweet sentiment     -> Hugging Face: cardiffnlp/tweet_sentiment_multilingual (English, Hindi files)
  * Hinglish sentiment  -> Hugging Face: shae2977/hinglish-youtube-sentiments-dataset
  * BRIGHTER emotion    -> Hugging Face: brighter-dataset/BRIGHTER-emotion-categories (eng, hin)
A manifest with retrieval date, revisions and file hashes is written to data/raw/MANIFEST.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import date

from src.config import RAW_DIR

DREADDIT_URL = "http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip"
HF = {
    "tweets": "cardiffnlp/tweet_sentiment_multilingual",
    "hinglish": "shae2977/hinglish-youtube-sentiments-dataset",
    "brighter": "brighter-dataset/BRIGHTER-emotion-categories",
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(force: bool = False) -> dict:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"retrieved": date.today().isoformat(), "datasets": {}}
    api = HfApi()

    # Dreaddit (original source)
    d = RAW_DIR / "dreaddit"
    if force or not (d / "dreaddit-train.csv").exists():
        d.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DREADDIT_URL, timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        z.extractall(d)
    manifest["datasets"]["dreaddit"] = {
        "url": DREADDIT_URL,
        "files": {p.name: _sha256(p) for p in sorted(d.glob("*.csv"))},
    }

    # tweet sentiment (only the English and Hindi jsonl files are needed)
    t = RAW_DIR / "tweets"
    for lang in ("english", "hindi"):
        for split in ("train", "validation", "test"):
            hf_hub_download(HF["tweets"], f"data/{lang}/{split}.jsonl", repo_type="dataset", local_dir=t)
    manifest["datasets"]["tweets"] = {
        "hf_id": HF["tweets"],
        "revision": api.dataset_info(HF["tweets"]).sha,
        "files": {str(p.relative_to(t)): _sha256(p) for p in sorted(t.rglob("*.jsonl"))},
    }

    # Hinglish YouTube sentiment + BRIGHTER: snapshot their data files
    for key, allow in (("hinglish", None), ("brighter", ["eng/*", "hin/*"])):
        dd = RAW_DIR / key
        snapshot_download(HF[key], repo_type="dataset", local_dir=dd, allow_patterns=allow)
        files = [p for p in sorted(dd.rglob("*")) if p.is_file() and ".cache" not in p.parts]
        manifest["datasets"][key] = {
            "hf_id": HF[key],
            "revision": api.dataset_info(HF[key]).sha,
            "files": {str(p.relative_to(dd)): _sha256(p) for p in files},
        }
    (RAW_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest
