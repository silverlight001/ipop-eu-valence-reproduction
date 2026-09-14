from pathlib import Path

import numpy as np
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
