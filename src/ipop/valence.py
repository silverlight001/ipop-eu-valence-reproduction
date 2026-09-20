"""Fair pooled-versus-valence-separated Eu emission comparison."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ipop.experiment import INCOMPLETE_MARKER_NAME, _inner_cv
from ipop.features import select_features
from ipop.metrics import aggregate_metrics, regression_metrics
from ipop.models import build_dummy_pipeline, build_xgb_search
from ipop.splits import GROUP_COLUMNS, audit_split_overlap, build_outer_splits

EU_VALENCE_COLUMN = "Eu valence"
TRAINING_MODES = ("pooled", "separate")


@dataclass(frozen=True)
class ValenceRunArtifacts:
    splits: Path
    overlap_audit: Path
    predictions: Path
    fold_metrics: Path
    summary_metrics: Path
    valence_fold_metrics: Path
    valence_summary_metrics: Path
    best_params: Path
    run_metadata: Path


def _artifact_paths(root: Path) -> ValenceRunArtifacts:
    return ValenceRunArtifacts(
        splits=root / "splits.csv",
        overlap_audit=root / "overlap_audit.csv",
        predictions=root / "predictions.csv",
        fold_metrics=root / "fold_metrics.csv",
        summary_metrics=root / "summary_metrics.csv",
        valence_fold_metrics=root / "valence_fold_metrics.csv",
        valence_summary_metrics=root / "valence_summary_metrics.csv",
        best_params=root / "best_params.json",
        run_metadata=root / "run_metadata.json",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _begin_run(root: Path, paths: ValenceRunArtifacts) -> Path:
    marker = root / INCOMPLETE_MARKER_NAME
    marker.write_text("Experiment run is incomplete.\n", encoding="utf-8")
    for path in (*paths.__dict__.values(), root / "findings_zh.md", root / "final_results.csv"):
        Path(path).unlink(missing_ok=True)
    figures = root / "figures"
    if figures.exists():
        shutil.rmtree(figures)
    return marker


def _validated_valence(frame: pd.DataFrame) -> pd.Series:
    if EU_VALENCE_COLUMN not in frame:
        raise ValueError(f"missing required column: {EU_VALENCE_COLUMN}")
    numeric = pd.to_numeric(frame[EU_VALENCE_COLUMN], errors="coerce")
    if numeric.isna().any():
        rows = frame.index[numeric.isna()].tolist()
        raise ValueError(f"Eu valence is missing or non-numeric at rows: {rows[:20]}")
    observed = set(numeric.astype(int))
    if observed != {2, 3} or not numeric.isin([2, 3]).all():
        raise ValueError(f"expected exactly Eu valences 2 and 3, got {sorted(observed)}")
    return numeric.astype(int)


def _build_estimator(
    model_name: str,
    config: dict[str, object],
    protocol: str,
):
    if model_name == "median":
        return build_dummy_pipeline()
    if model_name == "xgboost":
        return build_xgb_search(
            config,
            _inner_cv(protocol, int(config["inner_folds"]), int(config["seed"])),
        )
    raise KeyError(f"Unknown model: {model_name}")


def _fit_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    model_name: str,
    protocol: str,
    config: dict[str, object],
) -> tuple[np.ndarray, dict[str, object] | None]:
    target = str(config["target"])
    group_column = GROUP_COLUMNS.get(protocol)
    estimator = _build_estimator(model_name, config, protocol)
    fit_kwargs: dict[str, object] = {}
    if model_name == "xgboost" and group_column is not None:
        groups = train[group_column].to_numpy()
        if len(set(groups)) < int(config["inner_folds"]):
            raise ValueError(
                f"{protocol} has fewer than {config['inner_folds']} inner training groups"
            )
        fit_kwargs["groups"] = groups
    estimator.fit(
        select_features(train, feature_set),
        train[target].to_numpy(dtype=float),
        **fit_kwargs,
    )
    predicted = estimator.predict(select_features(test, feature_set))
    params = dict(estimator.best_params_) if hasattr(estimator, "best_params_") else None
    return np.asarray(predicted, dtype=float), params


def run_valence_comparison(
    frame: pd.DataFrame, config: dict[str, object], output_dir: str | Path
) -> ValenceRunArtifacts:
    """Rerun pooled and valence-separated models on identical outer folds."""
    working = frame.copy()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(root)
    incomplete_marker = _begin_run(root, paths)
    working[EU_VALENCE_COLUMN] = _validated_valence(working)
    seed = int(config["seed"])
    outer_folds = int(config["outer_folds"])
    target = str(config["target"])

    split_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    for protocol_value in config["protocols"]:
        protocol = str(protocol_value)
        assignments = build_outer_splits(working, protocol, outer_folds, seed)
        split_frames.append(assignments)
        audit_frames.append(audit_split_overlap(working, assignments, protocol))
    pd.concat(split_frames, ignore_index=True).sort_values(
        ["protocol", "fold", "row_id"]
    ).to_csv(paths.splits, index=False)
    pd.concat(audit_frames, ignore_index=True).sort_values(
        ["protocol", "fold", "audit_column"]
    ).to_csv(paths.overlap_audit, index=False)
    _write_json(
        paths.run_metadata,
        {
            "seed": seed,
            "config": config,
            "training_modes": list(TRAINING_MODES),
            "valence_counts": {
                str(key): int(value)
                for key, value in working[EU_VALENCE_COLUMN].value_counts().sort_index().items()
            },
            "versions": {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in ("numpy", "pandas", "scikit-learn", "xgboost")
                },
            },
        },
    )

    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    valence_fold_rows: list[dict[str, object]] = []
    parameter_records: dict[str, dict[str, object]] = {}
    for protocol_value, assignments in zip(config["protocols"], split_frames, strict=True):
        protocol = str(protocol_value)
        eligible = working.loc[working["row_id"].isin(assignments["row_id"])].copy()
        group_column = GROUP_COLUMNS.get(protocol)
        for feature_set_value in config["feature_sets"]:
            feature_set = str(feature_set_value)
            for model_value in config["models"]:
                model_name = str(model_value)
                for fold in range(outer_folds):
                    test_ids = assignments.loc[assignments["fold"] == fold, "row_id"]
                    test_mask = eligible["row_id"].isin(test_ids)
                    train = eligible.loc[~test_mask]
                    test = eligible.loc[test_mask]
                    if set(train[EU_VALENCE_COLUMN]) != {2, 3} or set(test[EU_VALENCE_COLUMN]) != {
                        2,
                        3,
                    }:
                        raise ValueError(
                            f"fold {fold} for {protocol} does not contain both Eu valences "
                            "in train and test"
                        )

                    for training_mode in TRAINING_MODES:
                        if training_mode == "pooled":
                            predicted, params = _fit_one(
                                train, test, feature_set, model_name, protocol, config
                            )
                            if params is not None:
                                key = (
                                    f"{protocol}|{feature_set}|{model_name}|{fold}|pooled|all"
                                )
                                parameter_records[key] = params
                        else:
                            prediction_by_index = pd.Series(index=test.index, dtype=float)
                            for valence in (2, 3):
                                valence_train = train.loc[train[EU_VALENCE_COLUMN].eq(valence)]
                                valence_test = test.loc[test[EU_VALENCE_COLUMN].eq(valence)]
                                valence_prediction, params = _fit_one(
                                    valence_train,
                                    valence_test,
                                    feature_set,
                                    model_name,
                                    protocol,
                                    config,
                                )
                                prediction_by_index.loc[valence_test.index] = valence_prediction
                                if params is not None:
                                    key = (
                                        f"{protocol}|{feature_set}|{model_name}|{fold}|"
                                        f"separate|Eu{valence}"
                                    )
                                    parameter_records[key] = params
                            predicted = prediction_by_index.loc[test.index].to_numpy(dtype=float)

                        truth = test[target].to_numpy(dtype=float)
                        metrics = regression_metrics(truth, predicted)
                        fold_rows.append(
                            {
                                "protocol": protocol,
                                "feature_set": feature_set,
                                "model": model_name,
                                "training_mode": training_mode,
                                "fold": fold,
                                "train_rows": len(train),
                                "test_rows": len(test),
                                "train_groups": (
                                    int(train[group_column].nunique())
                                    if group_column
                                    else len(train)
                                ),
                                "test_groups": (
                                    int(test[group_column].nunique()) if group_column else len(test)
                                ),
                                **metrics,
                            }
                        )
                        for valence in (2, 3):
                            mask = test[EU_VALENCE_COLUMN].to_numpy() == valence
                            valence_fold_rows.append(
                                {
                                    "protocol": protocol,
                                    "feature_set": feature_set,
                                    "model": model_name,
                                    "training_mode": training_mode,
                                    EU_VALENCE_COLUMN: valence,
                                    "fold": fold,
                                    "train_rows": int(
                                        train[EU_VALENCE_COLUMN].eq(valence).sum()
                                    ),
                                    "test_rows": int(mask.sum()),
                                    **regression_metrics(truth[mask], predicted[mask]),
                                }
                            )
                        for row_id, valence, actual, prediction in zip(
                            test["row_id"],
                            test[EU_VALENCE_COLUMN],
                            truth,
                            predicted,
                            strict=True,
                        ):
                            prediction_rows.append(
                                {
                                    "protocol": protocol,
                                    "feature_set": feature_set,
                                    "model": model_name,
                                    "training_mode": training_mode,
                                    "fold": fold,
                                    "row_id": int(row_id),
                                    EU_VALENCE_COLUMN: int(valence),
                                    "y_true": float(actual),
                                    "y_pred": float(prediction),
                                    "residual": float(prediction - actual),
                                }
                            )

    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["protocol", "feature_set", "model", "training_mode", "fold", "row_id"]
    )
    folds = pd.DataFrame(fold_rows).sort_values(
        ["protocol", "feature_set", "model", "training_mode", "fold"]
    )
    valence_folds = pd.DataFrame(valence_fold_rows).sort_values(
        ["protocol", "feature_set", "model", "training_mode", EU_VALENCE_COLUMN, "fold"]
    )
    summary = aggregate_metrics(folds).sort_values(
        ["protocol", "feature_set", "model", "training_mode"]
    )
    valence_summary = aggregate_metrics(valence_folds).sort_values(
        ["protocol", "feature_set", "model", "training_mode", EU_VALENCE_COLUMN]
    )
    predictions.to_csv(paths.predictions, index=False)
    folds.to_csv(paths.fold_metrics, index=False)
    summary.to_csv(paths.summary_metrics, index=False)
    valence_folds.to_csv(paths.valence_fold_metrics, index=False)
    valence_summary.to_csv(paths.valence_summary_metrics, index=False)
    _write_json(paths.best_params, parameter_records)
    incomplete_marker.unlink()
    return paths
