"""Load processed task tables and expose them as (texts, labels) for each split."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import EMOTION_LABELS, PROCESSED_DIR, TASK_LABELS


@dataclass
class TaskData:
    task: str
    df: pd.DataFrame  # one split

    @property
    def texts(self) -> list[str]:
        return self.df.text.tolist()

    @property
    def labels(self) -> np.ndarray:
        """multiclass: int vector (n,) ; multilabel: int matrix (n, L) with -1 = not annotated."""
        if self.task == "emotion":
            return self.df[EMOTION_LABELS].to_numpy(dtype=np.int64)
        return self.df.label.to_numpy(dtype=np.int64)

    @property
    def languages(self) -> np.ndarray:
        return self.df.language.to_numpy()

    def __len__(self) -> int:
        return len(self.df)


def processed_available() -> bool:
    return all((PROCESSED_DIR / f"{t}.csv").exists() for t in TASK_LABELS)


def load_task(task: str, split: str, limit: int | None = None, seed: int = 0) -> TaskData:
    df = pd.read_csv(PROCESSED_DIR / f"{task}.csv", keep_default_na=False)
    df = df[df.split == split].reset_index(drop=True)
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
    return TaskData(task, df)


def load_all(split: str, limit: int | None = None, tasks=None, seed: int = 0) -> dict[str, TaskData]:
    return {t: load_task(t, split, limit, seed) for t in (tasks or TASK_LABELS)}
