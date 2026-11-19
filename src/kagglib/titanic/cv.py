"""Честная кросс-валидация для Titanic: признаки (включая групповые) пересчитываются внутри каждого фолда."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from kagglib.titanic import features as F


@dataclass
class Fold:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_valid: pd.DataFrame
    y_valid: np.ndarray
    valid_idx: np.ndarray


def make_folds(train: pl.DataFrame, test: pl.DataFrame | None = None, n_splits: int = 5, n_repeats: int = 10,
               seed: int = 42) -> list[Fold]:
    everyone = pl.concat([train.select("PassengerId", "Ticket")] +
                         ([test.select("PassengerId", "Ticket")] if test is not None else []))
    y = train["Survived"].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = []
    for tr_idx, va_idx in cv.split(np.zeros(len(y)), y):
        tr_df, va_df = train[tr_idx], train[va_idx]
        ctx = F.fold_context(tr_df, everyone)
        st = F.fit(tr_df, ctx)
        folds.append(Fold(F.transform(tr_df, st, ctx).select(F.FEATURES).to_pandas(), y[tr_idx],
                          F.transform(va_df, st, ctx).select(F.FEATURES).to_pandas(), y[va_idx], va_idx))
    return folds


def oof_proba(model, folds: list[Fold], n: int) -> np.ndarray:
    """Out-of-fold вероятности, усреднённые по повторам."""
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for f in folds:
        acc[f.valid_idx] += clone(model).fit(f.X_train, f.y_train).predict_proba(f.X_valid)[:, 1]
        cnt[f.valid_idx] += 1
    return acc / np.maximum(cnt, 1)
