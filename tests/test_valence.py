import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ipop.features import AF_COLUMNS
from ipop.valence import run_valence_comparison


def synthetic_valence_emission(rows: int = 60) -> pd.DataFrame:
    frame = pd.DataFrame({column: np.linspace(0, 1, rows) for column in AF_COLUMNS})
    frame["Temp. (K)"] = 298.0
    frame["Excitation source (nm)"] = 400.0
    frame["Eu valence"] = np.tile([2, 3], rows // 2)
    frame["Emission max. (nm)"] = np.where(frame["Eu valence"].eq(2), 450.0, 620.0)
    frame["Formula"] = [f"F{i}" for i in range(rows)]
    frame["Host"] = [f"H{i // 2}" for i in range(rows)]
    frame["Reference"] = [f"D{i // 2}" for i in range(rows)]
    frame["row_id"] = range(rows)
    return frame


def lightweight_config() -> dict[str, object]:
    return {
        "seed": 42,
        "outer_folds": 5,
        "inner_folds": 3,
        "target": "Emission max. (nm)",
        "feature_sets": ["AF"],
        "protocols": ["random_row"],
        "models": ["median"],
        "xgboost": {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "parameter_grid": [],
        },
    }


def test_valence_comparison_uses_same_rows_and_folds_for_both_modes(tmp_path: Path) -> None:
    artifacts = run_valence_comparison(
        synthetic_valence_emission(), lightweight_config(), tmp_path / "run"
    )

    predictions = pd.read_csv(artifacts.predictions)
    folds = pd.read_csv(artifacts.fold_metrics)
    summary = pd.read_csv(artifacts.summary_metrics)
    per_valence = pd.read_csv(artifacts.valence_summary_metrics)

    assert set(predictions["training_mode"]) == {"pooled", "separate"}
    assert set(predictions["Eu valence"]) == {2, 3}
    for mode, rows in predictions.groupby("training_mode"):
        assert len(rows) == 60, mode
        assert rows["row_id"].is_unique
        assert set(rows["row_id"]) == set(range(60))
    assignment = predictions.pivot(index="row_id", columns="training_mode", values="fold")
    assert assignment["pooled"].equals(assignment["separate"])
    assert len(folds) == 10
    assert len(summary) == 2
    assert len(per_valence) == 4
    pooled_mae = summary.loc[summary["training_mode"].eq("pooled"), "mae_mean"].iloc[0]
    separate_mae = summary.loc[summary["training_mode"].eq("separate"), "mae_mean"].iloc[0]
    assert separate_mae == 0.0
    assert separate_mae < pooled_mae

    metadata = json.loads(artifacts.run_metadata.read_text(encoding="utf-8"))
    assert metadata["valence_counts"] == {"2": 30, "3": 30}
    assert metadata["training_modes"] == ["pooled", "separate"]
    assert not (artifacts.run_metadata.parent / ".incomplete").exists()


@pytest.mark.parametrize(
    "values, message",
    [
        ([2] * 59 + [pd.NA], "missing"),
        ([2] * 30 + [4] * 30, "exactly Eu valences 2 and 3"),
        ([2] * 60, "exactly Eu valences 2 and 3"),
    ],
)
def test_valence_comparison_rejects_invalid_labels(
    tmp_path: Path, values: list[object], message: str
) -> None:
    frame = synthetic_valence_emission()
    frame["Eu valence"] = values

    with pytest.raises(ValueError, match=message):
        run_valence_comparison(frame, lightweight_config(), tmp_path / "run")


def test_invalid_rerun_marks_prior_valence_bundle_incomplete(tmp_path: Path) -> None:
    run = tmp_path / "run"
    artifacts = run_valence_comparison(synthetic_valence_emission(), lightweight_config(), run)
    assert artifacts.predictions.is_file()

    invalid = synthetic_valence_emission()
    invalid.loc[0, "Eu valence"] = pd.NA
    with pytest.raises(ValueError, match="missing"):
        run_valence_comparison(invalid, lightweight_config(), run)

    assert (run / ".incomplete").is_file()
    assert not artifacts.predictions.exists()
    assert not artifacts.fold_metrics.exists()
    assert not artifacts.summary_metrics.exists()
