import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, GroupKFold

from ipop import experiment
from ipop.features import AF_COLUMNS
from ipop.models import build_dummy_pipeline, load_experiment_config


def synthetic_emission(rows: int = 50) -> pd.DataFrame:
    frame = pd.DataFrame({column: np.linspace(0, 1, rows) for column in AF_COLUMNS})
    frame["Temp. (K)"] = 298.0
    frame["Excitation source (nm)"] = np.tile([365.0, 405.0, 450.0, 465.0, 480.0], rows // 5)
    frame["Emission max. (nm)"] = 500.0 + 0.2 * frame["Excitation source (nm)"]
    frame["Formula"] = [f"F{i // 2}" for i in range(rows)]
    frame["Host"] = [f"H{i // 5}" for i in range(rows)]
    frame["Reference"] = [f"D{i // 5}" for i in range(rows)]
    frame["row_id"] = range(rows)
    return frame


def lightweight_config(
    *, protocols: list[str] | None = None, models: list[str] | None = None
) -> dict[str, object]:
    return {
        "seed": 42,
        "outer_folds": 5,
        "inner_folds": 3,
        "target": "Emission max. (nm)",
        "feature_sets": ["AF"],
        "protocols": protocols
        or ["random_row", "group_formula", "group_host", "group_reference"],
        "models": models or ["median"],
        "xgboost": {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "parameter_grid": [],
        },
    }


def test_run_persists_replayable_outer_fold_artifacts(tmp_path: Path) -> None:
    artifacts = experiment.run_experiment(
        synthetic_emission(), lightweight_config(), tmp_path / "run"
    )

    for artifact in (
        artifacts.splits,
        artifacts.overlap_audit,
        artifacts.predictions,
        artifacts.fold_metrics,
        artifacts.summary_metrics,
        artifacts.best_params,
        artifacts.run_metadata,
    ):
        assert artifact.exists()

    folds = pd.read_csv(artifacts.fold_metrics)
    summary = pd.read_csv(artifacts.summary_metrics)
    audit = pd.read_csv(artifacts.overlap_audit)
    predictions = pd.read_csv(artifacts.predictions)
    assert len(folds) == 20
    assert set(folds.columns) == {
        "protocol", "feature_set", "model", "fold", "train_rows", "test_rows",
        "train_groups", "test_groups", "mae", "rmse", "r2",
    }
    assert set(predictions.columns) == {
        "protocol", "feature_set", "model", "fold", "row_id", "y_true", "y_pred", "residual",
    }
    assert summary["fold_count"].eq(5).all()
    assert audit.loc[audit["enforced"], "overlap_count"].eq(0).all()
    metadata = json.loads(artifacts.run_metadata.read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert metadata["config"] == lightweight_config()
    assert "python" in metadata["versions"]


def test_all_protocol_preflight_artifacts_survive_first_fit_failure(
    tmp_path: Path, monkeypatch
) -> None:
    class FailingEstimator:
        def fit(self, X, y):
            raise RuntimeError("first fit failed")

    monkeypatch.setattr(experiment, "build_dummy_pipeline", FailingEstimator)
    frame = synthetic_emission()
    frame.loc[0, "Reference"] = pd.NA
    config = lightweight_config()
    artifact_names = ("splits.csv", "overlap_audit.csv", "run_metadata.json")
    for directory in (tmp_path / "first", tmp_path / "second"):
        with pytest.raises(RuntimeError, match="first fit failed"):
            experiment.run_experiment(frame, config, directory)

        assert all((directory / name).is_file() for name in artifact_names)
        splits = pd.read_csv(directory / "splits.csv")
        audits = pd.read_csv(directory / "overlap_audit.csv")
        assert len(splits) == 199
        assert set(splits["protocol"]) == set(config["protocols"])
        for protocol, assignments in splits.groupby("protocol"):
            expected_ids = set(range(1 if protocol == "group_reference" else 0, 50))
            assert set(assignments["row_id"]) == expected_ids
            assert not assignments["row_id"].duplicated().any()
            assert set(assignments["fold"]) == set(range(5))
        assert len(audits) == 60
        assert set(audits[["protocol", "fold", "audit_column"]].itertuples(
            index=False, name=None
        )) == {
            (protocol, fold, column)
            for protocol in config["protocols"]
            for fold in range(5)
            for column in ("Formula", "Host", "Reference")
        }
        assert audits["enforced"].sum() == 15
        assert audits.loc[audits["enforced"], "overlap_count"].eq(0).all()
        for table, keys in (
            (splits, ["protocol", "fold", "row_id"]),
            (audits, ["protocol", "fold", "audit_column"]),
        ):
            pd.testing.assert_frame_equal(table, table.sort_values(keys).reset_index(drop=True))
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["seed"] == 42
        assert metadata["config"] == config
        assert set(metadata["versions"]) == {
            "python", "numpy", "pandas", "scikit-learn", "xgboost"
        }
    for name in artifact_names:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_reference_protocol_excludes_missing_reference_rows_everywhere(tmp_path: Path) -> None:
    frame = synthetic_emission()
    frame.loc[0, "Reference"] = pd.NA

    artifacts = experiment.run_experiment(
        frame,
        lightweight_config(protocols=["group_reference"]),
        tmp_path / "run",
    )

    splits = pd.read_csv(artifacts.splits)
    predictions = pd.read_csv(artifacts.predictions)
    assert 0 not in set(splits["row_id"])
    assert 0 not in set(predictions["row_id"])


def test_fixed_seed_repeats_identical_splits_and_predictions(tmp_path: Path) -> None:
    config = lightweight_config(protocols=["random_row"])

    first = experiment.run_experiment(synthetic_emission(), config, tmp_path / "first")
    second = experiment.run_experiment(synthetic_emission(), config, tmp_path / "second")

    assert first.splits.read_bytes() == second.splits.read_bytes()
    assert first.predictions.read_bytes() == second.predictions.read_bytes()
    assert first.fold_metrics.read_bytes() == second.fold_metrics.read_bytes()


class RecordingGroupCV:
    def __init__(self) -> None:
        self.received_groups: np.ndarray | None = None

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 3

    def split(self, X: object, y: object = None, groups: object = None):
        self.received_groups = np.asarray(groups)
        splitter = GroupKFold(n_splits=3, shuffle=True, random_state=42)
        for train, test in splitter.split(X, y, groups):
            train_groups = set(self.received_groups[train])
            test_groups = set(self.received_groups[test])
            assert train_groups.isdisjoint(test_groups)
            yield train, test


def test_group_xgboost_passes_outer_training_groups_to_real_inner_grid_search(
    tmp_path: Path, monkeypatch
) -> None:
    inner_cvs: list[RecordingGroupCV] = []

    def make_inner_cv(protocol: str, folds: int, seed: int) -> RecordingGroupCV:
        assert (protocol, folds, seed) == ("group_formula", 3, 42)
        cv = RecordingGroupCV()
        inner_cvs.append(cv)
        return cv

    def build_fast_search(config: dict[str, object], inner_cv: RecordingGroupCV) -> GridSearchCV:
        return GridSearchCV(
            build_dummy_pipeline(),
            param_grid=[{}],
            cv=inner_cv,
            scoring="neg_mean_absolute_error",
            error_score="raise",
        )

    monkeypatch.setattr(experiment, "_inner_cv", make_inner_cv)
    monkeypatch.setattr(experiment, "build_xgb_search", build_fast_search)
    artifacts = experiment.run_experiment(
        synthetic_emission(),
        lightweight_config(protocols=["group_formula"], models=["xgboost"]),
        tmp_path / "run",
    )

    assert len(pd.read_csv(artifacts.fold_metrics)) == 5
    assert len(inner_cvs) == 5
    assert all(cv.received_groups is not None for cv in inner_cvs)
    assert all(len(set(cv.received_groups)) == 20 for cv in inner_cvs)


def test_formal_config_produces_all_160_protocol_feature_model_fold_rows(
    tmp_path: Path, monkeypatch
) -> None:
    def build_fast_search(config: dict[str, object], inner: object) -> GridSearchCV:
        return GridSearchCV(
            build_dummy_pipeline(),
            param_grid=[{}],
            cv=inner,
            scoring="neg_mean_absolute_error",
            error_score="raise",
        )

    monkeypatch.setattr(experiment, "build_xgb_search", build_fast_search)

    artifacts = experiment.run_experiment(
        synthetic_emission(),
        load_experiment_config(Path("configs/emission_xgb.yaml")),
        tmp_path / "run",
    )

    folds = pd.read_csv(artifacts.fold_metrics)
    summary = pd.read_csv(artifacts.summary_metrics)
    assert len(folds) == 160
    assert len(summary) == 32
    assert folds.groupby(["protocol", "feature_set", "model"])["fold"].nunique().eq(5).all()
