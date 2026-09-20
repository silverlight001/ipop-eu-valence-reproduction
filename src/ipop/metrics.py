from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the standard regression metrics for a single evaluation fold."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def aggregate_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    """Summarize audit-preserving fold metrics by experiment identifiers."""
    grouping = [
        column
        for column in ("protocol", "feature_set", "model", "training_mode", "Eu valence")
        if column in folds
    ]
    grouped = folds.groupby(grouping, dropna=False) if grouping else [((), folds)]
    rows: list[dict[str, object]] = []
    for keys, frame in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(grouping, keys, strict=True))
        row["fold_count"] = len(frame)
        for metric in ("mae", "rmse", "r2"):
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)
