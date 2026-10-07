"""Train + honestly evaluate the English vs Romanized-Hindi (hi-Latn) classifier used for Latin-script text.

Training data (all public, see DATASET.md):
  hi-Latn  : youtube-hinglish comments (TRAIN split only)
  en-Latn  : English text from tweets-english, brighter-eng, dreaddit (TRAIN splits only), subsampled to balance
Evaluation: TEST splits of the same sources + the Devanagari script rule on BRIGHTER Hindi + a *noisy probe* on the
"hindi" tweet config (Romanized but contains plain-English tweets, so it cannot be treated as clean ground truth).
Known confound: hi-Latn comes from a single source/domain (YouTube); English from tweets/Reddit/short stories.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from src.config import CKPT_DIR, METRICS_DIR, PROCESSED_DIR  # noqa: E402
from src.langid import detect_language, latin_features_text  # noqa: E402

if __name__ == "__main__":
    S = pd.read_csv(PROCESSED_DIR / "sentiment.csv", keep_default_na=False)
    E = pd.read_csv(PROCESSED_DIR / "emotion.csv", keep_default_na=False)
    R = pd.read_csv(PROCESSED_DIR / "risk.csv", keep_default_na=False)
    en_pool = {"tweets-english": S[S.source == "tweets-english"], "brighter-eng": E[E.source == "brighter-eng"], "dreaddit": R}
    hi = {sp: S[(S.source == "youtube-hinglish") & (S.split == sp)].text for sp in ("train", "test")}
    rng = np.random.default_rng(13)

    def en_split(sp, n_total):
        parts = [d[d.split == sp].text for d in en_pool.values()]
        k = n_total // len(parts)
        return pd.concat([p.sample(min(k, len(p)), random_state=13) for p in parts])

    Xtr_hi, Xtr_en = hi["train"], en_split("train", len(hi["train"]))
    X = [latin_features_text(t) for t in list(Xtr_hi) + list(Xtr_en)]
    y = ["hi-Latn"] * len(Xtr_hi) + ["en-Latn"] * len(Xtr_en)
    keep = [i for i, t in enumerate(X) if t]
    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True),
                        LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))
    clf.fit([X[i] for i in keep], [y[i] for i in keep])
    out = CKPT_DIR / "langid"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "latn_en_hi.joblib")

    # ---- evaluation (end-to-end detect_language, so script rule + classifier + lingua are all exercised)
    def tags(texts):
        return [detect_language(t)["language_tag"] for t in texts]

    ev = {}
    te_hi, te_en = hi["test"], en_split("test", 3 * len(hi["test"]))
    for name, texts, want in (("youtube-hinglish test (hi-Latn)", te_hi, "hi-Latn"), ("english tests (tweets/brighter/dreaddit)", te_en, "en-Latn"),
                              ("brighter-hin test (hi-Deva)", E[(E.source == "brighter-hin") & (E.split == "test")].text, "hi-Deva")):
        p = tags(texts)
        ev[name] = {"n": len(p), "accuracy": float(np.mean([t == want for t in p])), "predicted_distribution": pd.Series(p).value_counts().to_dict()}
    noisy = S[(S.source == "tweets-hindi") & (S.split == "test")].text
    p = tags(noisy)
    ev["tweets-hindi test (NOISY PROBE: Romanized, includes plain-English tweets)"] = {"n": len(p), "share_predicted_hi-Latn": float(np.mean([t == "hi-Latn" for t in p])), "predicted_distribution": pd.Series(p).value_counts().to_dict()}
    res = {"training": {"hi-Latn": len(Xtr_hi), "en-Latn": len(Xtr_en)}, "evaluation": ev,
           "caveats": ["hi-Latn positives come from a single YouTube-comment corpus; English negatives from other platforms: style confound is possible.",
                       "Only en-Latn, hi-Latn and script-level detection were evaluated. Other languages are NOT evaluated."]}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "langid.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
