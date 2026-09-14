from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.model_selection import KFold

from ipop.models import build_dummy_pipeline, build_xgb_search, load_experiment_config


def test_config_has_declared_protocols_and_four_bounded_candidates() -> None:
    config = load_experiment_config(Path("configs/emission_xgb.yaml"))

    assert config["outer_folds"] == 5
    assert config["protocols"] == [
        "random_row",
        "group_formula",
        "group_host",
        "group_reference",
    ]
    assert len(config["xgboost"]["parameter_grid"]) == 4


@pytest.mark.parametrize(
    ("case", "mutation"),
    [
        ("wrong seed", lambda config: config.__setitem__("seed", 7)),
        ("wrong outer folds", lambda config: config.__setitem__("outer_folds", 2)),
        ("wrong inner folds", lambda config: config.__setitem__("inner_folds", 2)),
        ("incomplete protocols", lambda config: config.__setitem__("protocols", ["random_row"])),
        ("incomplete feature sets", lambda config: config.__setitem__("feature_sets", ["AF"])),
        ("missing models", lambda config: config.__setitem__("models", [])),
        (
            "wrong XGBoost objective",
            lambda config: config["xgboost"].__setitem__("objective", "reg:absoluteerror"),
        ),
        (
            "wrong XGBoost tree method",
            lambda config: config["xgboost"].__setitem__("tree_method", "approx"),
        ),
        ("wrong XGBoost jobs", lambda config: config["xgboost"].__setitem__("n_jobs", 9)),
        (
            "non-list candidate grid",
            lambda config: config["xgboost"].__setitem__("parameter_grid", "abcd"),
        ),
        (
            "non-mapping candidate",
            lambda config: config["xgboost"].__setitem__("parameter_grid", ["invalid"] * 4),
        ),
        (
            "unapproved candidate value",
            lambda config: config["xgboost"]["parameter_grid"][0].__setitem__("n_estimators", 301),
        ),
        (
            "wrong candidate count",
            lambda config: config["xgboost"].__setitem__(
                "parameter_grid", config["xgboost"]["parameter_grid"][:3]
            ),
        ),
    ],
)
def test_config_rejects_unapproved_experiment_contract(
    tmp_path: Path, case: str, mutation: object
) -> None:
    config = yaml.safe_load(Path("configs/emission_xgb.yaml").read_text(encoding="utf-8"))
    mutation(config)
    path = tmp_path / f"{case}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        load_experiment_config(path)


def test_dummy_pipeline_handles_missing_feature_values() -> None:
    model = build_dummy_pipeline()
    features = np.array([[1.0], [np.nan], [3.0]])
    target = np.array([500.0, 600.0, 700.0])

    model.fit(features, target)

    assert model.predict(np.array([[np.nan]])).tolist() == [600.0]


def test_xgb_search_uses_inner_cv_and_returns_a_fitted_estimator() -> None:
    config = load_experiment_config(Path("configs/emission_xgb.yaml"))
    search = build_xgb_search(config, KFold(n_splits=3, shuffle=True, random_state=42))
    features = np.arange(60, dtype=float).reshape(20, 3)
    target = np.linspace(500.0, 650.0, 20)

    search.fit(features, target)

    assert hasattr(search, "best_estimator_")
