"""Build per-task processed tables with leakage-safe splits.

Output (data/processed/, git-ignored): sentiment.csv, emotion.csv, risk.csv, stats.json
Common columns: text_id, text, language, source, split, group  (+ task labels)

Split policy
  * official splits are used where the dataset has them (tweets, BRIGHTER, Dreaddit train/test)
  * emotion labels: -1 marks a label that was not annotated (BRIGHTER English has no 'disgust' column)
  * Hinglish YouTube comments have no official split -> split by video_id (group split)
  * Dreaddit validation is carved from train, grouped by post_id (segments of one post never straddle splits)
  * BRIGHTER dev/test files list every text twice (Track A + Track C copies, identical labels) -> collapsed by exact dedup
  * exact + near duplicates (TF-IDF cosine >= 0.9) are removed from the *earlier-priority* split
    (train loses rows that duplicate val/test; val loses rows that duplicate test). Test is never altered
    except for exact in-split duplicates.
"""
from __future__ import annotations

import json
import unicodedata

import numpy as np
import pandas as pd

from src.config import EMOTION_LABELS, PROCESSED_DIR, RAW_DIR
from src.preprocessing import clean_text

SEED = 13


def _norm_key(t: str) -> str:
    """Case/punctuation-insensitive key. Keeps letters, combining marks (Devanagari matras) and digits."""
    return "".join(c for c in t.lower() if unicodedata.category(c)[0] in "LMN")


def _near_key(t: str) -> str:
    """Like _norm_key but keeps single spaces, so word-boundary n-grams ignore punctuation/case differences."""
    return " ".join("".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in t.lower()).split())


def _near_dup_mask(a: pd.Series, b: pd.Series, thr: float = 0.9) -> np.ndarray:
    """Boolean mask over `a`: True if row is exact/near duplicate of any row in `b`."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    a, b = a.map(_near_key), b.map(_near_key)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=1, sublinear_tf=True)
    vec.fit(pd.concat([a, b]))
    A, B = vec.transform(a), vec.transform(b)
    mask = np.zeros(len(a), dtype=bool)
    for i in range(0, A.shape[0], 512):
        sims = (A[i : i + 512] @ B.T).max(axis=1).toarray().ravel()
        mask[i : i + 512] = sims >= thr
    return mask


def dedupe_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove in-split duplicates and cross-split leakage; test has priority over val over train."""
    rep = {"in_split_exact": 0, "train_vs_val_test": 0, "val_vs_test": 0}
    df = df.copy()
    df["_k"] = df.text.map(_norm_key)
    before = len(df)
    df = df[df.text.str.strip().astype(bool) & (df._k.str.len() > 0)]
    df = df.drop_duplicates(subset=["split", "_k"])
    rep["in_split_exact"] = before - len(df)
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    m = _near_dup_mask(parts["val"].text, parts["test"].text)
    rep["val_vs_test"] = int(m.sum())
    parts["val"] = parts["val"][~m]
    held = pd.concat([parts["val"].text, parts["test"].text])
    m = _near_dup_mask(parts["train"].text, held)
    rep["train_vs_val_test"] = int(m.sum())
    parts["train"] = parts["train"][~m]
    return pd.concat(parts.values()).drop(columns="_k").reset_index(drop=True), rep


def _group_split(groups: pd.Series, fracs=(0.70, 0.15, 0.15), seed=SEED) -> dict:
    """Assign whole groups to train/val/test, filling the largest groups first to hit target sizes."""
    rng = np.random.default_rng(seed)
    sizes = groups.value_counts()
    order = sorted(sizes.index, key=lambda g: (-sizes[g], rng.random()))
    total = sizes.sum()
    targets = dict(zip(("train", "val", "test"), (f * total for f in fracs)))
    filled = {k: 0 for k in targets}
    assign = {}
    for g in order:
        s = max(targets, key=lambda k: targets[k] - filled[k])
        assign[g] = s
        filled[s] += sizes[g]
    return assign


def _finalize(df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df.text.map(clean_text)
    df = df.reset_index(drop=True)
    split_short = {"train": "tr", "val": "va", "test": "te"}
    counters: dict[str, int] = {}
    ids = []
    for s in df.split:
        counters[s] = counters.get(s, 0) + 1
        ids.append(f"{task}-{split_short[s]}-{counters[s]:05d}")
    df.insert(0, "text_id", ids)  # anonymised, index-based id (no source ids exposed)
    return df


def build_sentiment() -> tuple[pd.DataFrame, dict]:
    rows = []
    for lang_cfg, tag in (("english", "en-Latn"), ("hindi", "hi-Latn")):
        for sp, name in (("train", "train"), ("validation", "val"), ("test", "test")):
            d = pd.read_json(RAW_DIR / f"tweets/data/{lang_cfg}/{sp}.jsonl", lines=True, dtype={"label": int})
            rows.append(pd.DataFrame({"text": d.text, "label": d.label.astype(int), "language": tag,
                                      "source": f"tweets-{lang_cfg}", "split": name, "group": ""}))
    h = pd.read_csv(RAW_DIR / "hinglish/yt_hinglish_comments_dataset.csv")
    h = h.dropna(subset=["comment", "sentiment"])
    h = h[h.comment.str.strip().astype(bool)].drop_duplicates("comment")
    lab = {"Negative": 0, "Neutral": 1, "Positive": 2}
    assign = _group_split(h.video_id)
    rows.append(pd.DataFrame({"text": h.comment, "label": h.sentiment.map(lab), "language": "hi-Latn",
                              "source": "youtube-hinglish", "split": h.video_id.map(assign), "group": h.video_id}))
    df = pd.concat(rows, ignore_index=True)
    out, rep = dedupe_splits(df)
    return _finalize(out, "sentiment"), rep


def build_emotion() -> tuple[pd.DataFrame, dict]:
    rows = []
    for cfg, tag in (("eng", "en-Latn"), ("hin", "hi-Deva")):
        for sp, name in (("train", "train"), ("dev", "val"), ("test", "test")):
            d = pd.read_parquet(next((RAW_DIR / f"brighter/{cfg}").glob(f"{sp}-*.parquet")))
            part = pd.DataFrame({"text": d.text, "language": tag, "source": f"brighter-{cfg}", "split": name, "group": ""})
            for e in EMOTION_LABELS:
                part[e] = d[e].fillna(-1).astype(int).values  # -1 = not annotated (English has no 'disgust')
            rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    df = df[df.text.str.strip().str.lower() != "#name?"]  # corrupted spreadsheet cells in the source files
    # official dev sets are tiny (230 / 200): enlarge validation with a seeded 12% of train
    rng = np.random.default_rng(SEED)
    is_tr = (df.split == "train").values
    move = is_tr & (rng.random(len(df)) < 0.12)
    df.loc[move, "split"] = "val"
    out, rep = dedupe_splits(df)
    return _finalize(out, "emotion"), rep


def build_risk() -> tuple[pd.DataFrame, dict]:
    tr = pd.read_csv(RAW_DIR / "dreaddit/dreaddit-train.csv")
    te = pd.read_csv(RAW_DIR / "dreaddit/dreaddit-test.csv")
    assign = _group_split(tr.post_id, fracs=(0.85, 0.15, 0.0))
    tr = tr.copy()
    tr["split"] = tr.post_id.map(assign).replace({"test": "val"})
    te = te.assign(split="test")
    d = pd.concat([tr, te], ignore_index=True)
    df = pd.DataFrame({"text": d.text, "label": d.label.astype(int), "language": "en-Latn", "source": "dreaddit",
                       "split": d.split, "group": d.subreddit + "/" + d.post_id.astype(str)})
    out, rep = dedupe_splits(df)
    out["group"] = out.group.str.split("/").str[0]  # keep only the subreddit (coarse domain), drop post ids
    return _finalize(out, "risk"), rep


def main() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    for task, fn in (("sentiment", build_sentiment), ("emotion", build_emotion), ("risk", build_risk)):
        df, rep = fn()
        df.to_csv(PROCESSED_DIR / f"{task}.csv", index=False)
        stats[task] = {
            "dedup": rep,
            "n": int(len(df)),
            "by_split": df.split.value_counts().to_dict(),
            "by_language_split": {f"{l}|{s}": int(n) for (l, s), n in df.groupby(["language", "split"]).size().items()},
            "by_source": df.source.value_counts().to_dict(),
        }
        if task == "emotion":
            tr = df[df.split == "train"]
            stats[task]["label_positive_rate_train_by_language"] = {
                lang: {e: round(float((g[e] == 1).sum() / max(1, (g[e] >= 0).sum())), 4) for e in EMOTION_LABELS if (g[e] >= 0).any()}
                for lang, g in tr.groupby("language")
            }
            stats[task]["no_emotion_rate_train_by_language"] = {
                lang: round(float(((g[EMOTION_LABELS] == 1).sum(axis=1) == 0).mean()), 4) for lang, g in tr.groupby("language")
            }
            stats[task]["unannotated_labels"] = {lang: [e for e in EMOTION_LABELS if (g[e] < 0).all()] for lang, g in df.groupby("language")}
        else:
            stats[task]["label_dist_train"] = {str(k): int(v) for k, v in df[df.split == "train"].label.value_counts().sort_index().items()}
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
