"""Classical baseline: TF-IDF (word 1-2gram + char_wb 2-5gram) + Logistic Regression, one model per task.

Establishes whether the neural models add value. C is chosen on the validation split only.
"""
from __future__ import annotations

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import EMOTION_LABELS, TASK_LABELS

C_GRID = (0.5, 2.0, 8.0)


class TfidfBaseline:
    kind = "baseline"

    def __init__(self):
        self.vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        self.vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=True, max_features=150000)
        self.models: dict = {}
        self.best_C: dict = {}

    def _X(self, texts, fit=False):
        if fit:
            return hstack([self.vec_w.fit_transform(texts), self.vec_c.fit_transform(texts)]).tocsr()
        return hstack([self.vec_w.transform(texts), self.vec_c.transform(texts)]).tocsr()

    def fit(self, train: dict, val: dict) -> "TfidfBaseline":
        """One shared vectoriser (fit on all training texts), separate LR per task (no label sharing)."""
        self._X([t for d in train.values() for t in d.texts], fit=True)
        for task, d in train.items():
            Xtr, Xva = self._X(d.texts), self._X(val[task].texts)
            ytr, yva = d.labels, val[task].labels
            best = (-1, None, None)
            for C in C_GRID:
                m = self._fit_one(task, Xtr, ytr, C)
                s = self._val_score(task, m, Xva, yva)
                if s > best[0]:
                    best = (s, C, m)
            self.best_C[task], self.models[task] = best[1], best[2]
        return self

    def _fit_one(self, task, X, y, C):
        if task == "emotion":
            ms = []
            for j in range(y.shape[1]):
                keep = y[:, j] >= 0  # masked labels (e.g. English 'disgust') are excluded
                ms.append(LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X[keep], y[keep, j]) if keep.sum() and len(set(y[keep, j])) > 1 else None)
            return ms
        return LogisticRegression(C=C, max_iter=2000).fit(X, y)

    def _logits(self, task, m, X):
        if task == "emotion":
            return np.stack([mm.decision_function(X) if mm is not None else np.full(X.shape[0], -10.0) for mm in m], 1)
        # log-probabilities act as logits: softmax(log p) == p, so temperature scaling is well defined
        return np.log(np.clip(m.predict_proba(X), 1e-9, 1.0))

    def _val_score(self, task, m, X, y):
        lg = self._logits(task, m, X)
        if task == "emotion":
            fs = []
            for j in range(y.shape[1]):
                k = y[:, j] >= 0
                if k.sum() and y[k, j].sum():
                    fs.append(f1_score(y[k, j], (lg[k, j] > 0).astype(int), zero_division=0))
            return float(np.mean(fs))
        return float(f1_score(y, lg.argmax(1), average="macro"))

    def predict_logits(self, task: str, texts: list[str]) -> np.ndarray:
        return self._logits(task, self.models[task], self._X(texts))

    def n_parameters(self) -> int:
        n = 0
        for task, m in self.models.items():
            for mm in (m if task == "emotion" else [m]):
                if mm is not None:
                    n += mm.coef_.size + mm.intercept_.size
        return int(n)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "TfidfBaseline":
        return joblib.load(path)
