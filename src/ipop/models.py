from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the fixed, reproducible benchmark configuration."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Experiment configuration must be a mapping")

    required = {
        "seed",
        "outer_folds",
        "inner_folds",
        "target",
        "feature_sets",
        "protocols",
        "models",
        "xgboost",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing experiment configuration keys: {missing}")

    xgboost_settings = config["xgboost"]
    if not isinstance(xgboost_settings, dict) or len(
        xgboost_settings.get("parameter_grid", [])
    ) != 4:
        raise ValueError("The approved benchmark requires four bounded XGBoost candidates")
    return config


def build_dummy_pipeline() -> Pipeline:
    """Build the median baseline with train-fold-only imputation."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", DummyRegressor(strategy="median")),
        ]
    )


def build_xgb_search(config: dict[str, Any], inner_cv: object) -> GridSearchCV:
    """Build a leakage-safe, inner-CV XGBoost hyperparameter search."""
    settings = config["xgboost"]
    fixed = {key: settings[key] for key in ("objective", "tree_method", "n_jobs")}
    fixed["random_state"] = int(config["seed"])
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", XGBRegressor(**fixed)),
        ]
    )
    candidates = [
        {f"regressor__{key}": [value] for key, value in candidate.items()}
        for candidate in settings["parameter_grid"]
    ]
    return GridSearchCV(
        pipeline,
        param_grid=candidates,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
