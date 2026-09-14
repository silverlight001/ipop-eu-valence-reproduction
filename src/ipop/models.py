from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

_TOP_LEVEL_KEYS = {
    "seed",
    "outer_folds",
    "inner_folds",
    "target",
    "feature_sets",
    "protocols",
    "models",
    "xgboost",
}
_XGBOOST_KEYS = {"objective", "tree_method", "n_jobs", "parameter_grid"}
_CANDIDATE_KEYS = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
}
_APPROVED_CANDIDATES = [
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    },
]


def _require_exact_mapping(value: object, expected_keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        raise ValueError(f"{name} keys must be exactly {sorted(expected_keys)}")
    return value


def _is_approved_candidate(candidate: dict[str, Any]) -> bool:
    if set(candidate) != _CANDIDATE_KEYS:
        return False
    for approved in _APPROVED_CANDIDATES:
        if all(
            type(candidate[key]) is type(approved[key]) and candidate[key] == approved[key]
            for key in _CANDIDATE_KEYS
        ):
            return True
    return False


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the fixed, reproducible benchmark configuration."""
    config = _require_exact_mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), _TOP_LEVEL_KEYS, "experiment"
    )
    if (
        type(config["seed"]) is not int
        or config["seed"] != 42
        or type(config["outer_folds"]) is not int
        or config["outer_folds"] != 5
        or type(config["inner_folds"]) is not int
        or config["inner_folds"] != 3
        or config["target"] != "Emission max. (nm)"
        or config["feature_sets"] != ["AF", "AF+T", "AF+ES", "AF+T+ES"]
        or config["protocols"]
        != ["random_row", "group_formula", "group_host", "group_reference"]
        or config["models"] != ["median", "xgboost"]
    ):
        raise ValueError("Experiment settings do not match the approved benchmark contract")

    xgboost_settings = _require_exact_mapping(config["xgboost"], _XGBOOST_KEYS, "xgboost")
    if (
        xgboost_settings["objective"] != "reg:squarederror"
        or xgboost_settings["tree_method"] != "hist"
        or type(xgboost_settings["n_jobs"]) is not int
        or xgboost_settings["n_jobs"] != 1
    ):
        raise ValueError("XGBoost settings do not match the approved benchmark contract")

    candidates = xgboost_settings["parameter_grid"]
    if type(candidates) is not list or len(candidates) != len(_APPROVED_CANDIDATES):
        raise ValueError("The approved benchmark requires four bounded XGBoost candidates")
    if any(type(candidate) is not dict or not _is_approved_candidate(candidate) for candidate in candidates):
        raise ValueError("XGBoost candidates must match the approved parameter sets")
    if len({tuple(sorted(candidate.items())) for candidate in candidates}) != len(_APPROVED_CANDIDATES):
        raise ValueError("XGBoost candidates must not contain duplicates")
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
