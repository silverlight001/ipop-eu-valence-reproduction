import numpy as np
import pandas as pd

from ipop.metrics import aggregate_metrics, regression_metrics


def test_regression_metrics_match_known_values() -> None:
    result = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))

    assert result["mae"] == 1 / 3
    assert np.isclose(result["rmse"], np.sqrt(1 / 3))
    assert result["r2"] == 0.5


def test_aggregate_metrics_keeps_fold_values_and_sample_standard_deviation() -> None:
    folds = pd.DataFrame({"mae": [1.0, 3.0], "rmse": [2.0, 4.0], "r2": [0.5, 0.7]})

    result = aggregate_metrics(folds)

    assert result.loc[0, "mae_mean"] == 2.0
    assert np.isclose(result.loc[0, "r2_std"], np.std([0.5, 0.7], ddof=1))
