"""Run and persist reproducible Eu-emission benchmark experiments."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from ipop.features import select_features
from ipop.metrics import aggregate_metrics, regression_metrics
from ipop.models import build_dummy_pipeline, build_xgb_search
from ipop.splits import GROUP_COLUMNS, audit_split_overlap, build_outer_splits


@dataclass(frozen=True)
class RunArtifacts:
    """Filesystem locations of the seven replayable experiment artifacts."""

    splits: Path
    overlap_audit: Path
    predictions: Path
    fold_metrics: Path
    summary_metrics: Path
    best_params: Path
    run_metadata: Path


def _inner_cv(protocol: str, folds: int, seed: int) -> KFold | GroupKFold:
    """Return the protocol-matched, deterministic inner splitter."""
    if protocol == "random_row":
        return KFold(n_splits=folds, shuffle=True, random_state=seed)
    if protocol in GROUP_COLUMNS:
        return GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    raise KeyError(f"Unknown split protocol: {protocol}")


def _artifact_paths(root: Path) -> RunArtifacts:
    return RunArtifacts(
        splits=root / "splits.csv",
        overlap_audit=root / "overlap_audit.csv",
        predictions=root / "predictions.csv",
        fold_metrics=root / "fold_metrics.csv",
        summary_metrics=root / "summary_metrics.csv",
        best_params=root / "best_params.json",
        run_metadata=root / "run_metadata.json",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def run_experiment(
    frame: pd.DataFrame, config: dict[str, object], output_dir: str | Path
) -> RunArtifacts:
    """Evaluate every configured protocol, feature set, model, and outer fold.

    Outer assignments determine eligibility, so rows with missing Reference values are
    excluded from every part of the reference-disjoint experiment, including predictions.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(root)
    seed = int(config["seed"])
    outer_folds = int(config["outer_folds"])
    inner_folds = int(config["inner_folds"])
    target = str(config["target"])

    split_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    parameter_records: dict[str, dict[str, object]] = {}

    for protocol_value in config["protocols"]:
        protocol = str(protocol_value)
        assignments = build_outer_splits(frame, protocol, outer_folds, seed)
        split_frames.append(assignments)
        audit_frames.append(audit_split_overlap(frame, assignments, protocol))
        eligible = frame.loc[frame["row_id"].isin(assignments["row_id"])].copy()

        for feature_set_value in config["feature_sets"]:
            feature_set = str(feature_set_value)
            for model_value in config["models"]:
                model_name = str(model_value)
                for fold in range(outer_folds):
                    test_ids = assignments.loc[assignments["fold"] == fold, "row_id"]
                    test_mask = eligible["row_id"].isin(test_ids)
                    train = eligible.loc[~test_mask]
                    test = eligible.loc[test_mask]
                    x_train = select_features(train, feature_set)
                    x_test = select_features(test, feature_set)
                    y_train = train[target].to_numpy(dtype=float)
                    y_test = test[target].to_numpy(dtype=float)
                    group_column = GROUP_COLUMNS.get(protocol)

                    if model_name == "median":
                        estimator = build_dummy_pipeline()
                        fit_kwargs: dict[str, object] = {}
                    elif model_name == "xgboost":
                        estimator = build_xgb_search(
                            config, _inner_cv(protocol, inner_folds, seed)
                        )
                        fit_kwargs = (
                            {"groups": train[group_column].to_numpy()}
                            if group_column is not None
                            else {}
                        )
                    else:
                        raise KeyError(f"Unknown model: {model_name}")

                    estimator.fit(x_train, y_train, **fit_kwargs)
                    predicted = estimator.predict(x_test)
                    metrics = regression_metrics(y_test, predicted)
                    train_groups = (
                        int(train[group_column].nunique()) if group_column else len(train)
                    )
                    test_groups = int(test[group_column].nunique()) if group_column else len(test)
                    metric_rows.append(
                        {
                            "protocol": protocol,
                            "feature_set": feature_set,
                            "model": model_name,
                            "fold": fold,
                            "train_rows": len(train),
                            "test_rows": len(test),
                            "train_groups": train_groups,
                            "test_groups": test_groups,
                            **metrics,
                        }
                    )
                    for row_id, truth, prediction in zip(
                        test["row_id"], y_test, predicted, strict=True
                    ):
                        prediction_rows.append(
                            {
                                "protocol": protocol,
                                "feature_set": feature_set,
                                "model": model_name,
                                "fold": fold,
                                "row_id": int(row_id),
                                "y_true": float(truth),
                                "y_pred": float(prediction),
                                "residual": float(prediction - truth),
                            }
                        )
                    if hasattr(estimator, "best_params_"):
                        parameter_records[
                            f"{protocol}|{feature_set}|{model_name}|{fold}"
                        ] = dict(estimator.best_params_)

    splits = pd.concat(split_frames, ignore_index=True).sort_values(
        ["protocol", "fold", "row_id"]
    )
    audits = pd.concat(audit_frames, ignore_index=True).sort_values(
        ["protocol", "fold", "audit_column"]
    )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["protocol", "feature_set", "model", "fold", "row_id"]
    )
    folds = pd.DataFrame(metric_rows).sort_values(["protocol", "feature_set", "model", "fold"])
    summary = aggregate_metrics(folds).sort_values(["protocol", "feature_set", "model"])
    splits.to_csv(paths.splits, index=False)
    audits.to_csv(paths.overlap_audit, index=False)
    predictions.to_csv(paths.predictions, index=False)
    folds.to_csv(paths.fold_metrics, index=False)
    summary.to_csv(paths.summary_metrics, index=False)
    _write_json(paths.best_params, parameter_records)
    _write_json(
        paths.run_metadata,
        {
            "seed": seed,
            "config": config,
            "versions": {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in ("numpy", "pandas", "scikit-learn", "xgboost")
                },
            },
        },
    )
    return paths
