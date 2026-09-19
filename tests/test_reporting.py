from pathlib import Path

import pandas as pd
import pytest

from ipop.reporting import build_report


def _write_report_inputs(run: Path) -> None:
    pd.DataFrame(
        {
            "protocol": ["random_row", "group_host"],
            "feature_set": ["AF+T+ES", "AF+T+ES"],
            "model": ["xgboost", "xgboost"],
            "fold_count": [5, 5],
            "mae_mean": [15.0, 30.0],
            "mae_std": [1.0, 2.0],
            "rmse_mean": [31.0, 45.0],
            "rmse_std": [1.0, 3.0],
            "r2_mean": [0.75, 0.40],
            "r2_std": [0.02, 0.08],
        }
    ).to_csv(run / "summary_metrics.csv", index=False)
    fold_rows = []
    for protocol in ("random_row", "group_host"):
        for fold in range(5):
            fold_rows.append(
                {
                    "protocol": protocol,
                    "feature_set": "AF+T+ES",
                    "model": "xgboost",
                    "fold": fold,
                    "train_rows": 100,
                    "test_rows": 25,
                    "train_groups": 100,
                    "test_groups": 25,
                    "mae": 15.0,
                    "rmse": 31.0,
                    "r2": 0.75,
                }
            )
    pd.DataFrame(fold_rows).to_csv(run / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "protocol": ["random_row", "group_host"],
            "feature_set": ["AF+T+ES"] * 2,
            "model": ["xgboost"] * 2,
            "fold": [0, 0],
            "row_id": [1, 2],
            "y_true": [600.0, 620.0],
            "y_pred": [598.0, 600.0],
            "residual": [-2.0, -20.0],
        }
    ).to_csv(run / "predictions.csv", index=False)
    pd.DataFrame(
        {
            "protocol": ["random_row", "group_host"],
            "fold": [0, 0],
            "audit_column": ["Host", "Host"],
            "enforced": [False, True],
            "overlap_count": [1, 0],
        }
    ).to_csv(run / "overlap_audit.csv", index=False)


def _dataset_summary() -> dict[str, int]:
    return {
        "records": 3952,
        "unique_hosts": 2238,
        "unique_references": 553,
        "target_observations": 16023,
    }


def _join_summary() -> dict[str, int]:
    return {"emission_rows": 1665, "ambiguous_reference_rows": 6}


def test_report_contains_required_figures_and_reproducibility_disclosure(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_report_inputs(run)

    artifacts = build_report(run, dataset_summary=_dataset_summary(), join_summary=_join_summary())

    assert artifacts.findings.exists()
    text = artifacts.findings.read_text(encoding="utf-8")
    assert "方法级复现" in text
    assert "未公开完整超参数" in text
    assert "group_host" in text
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts.figures)


def test_report_explains_missing_required_protocol(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_report_inputs(run)
    summary = pd.read_csv(run / "summary_metrics.csv")
    summary.loc[summary["protocol"] == "group_host", "protocol"] = "group_formula"
    summary.to_csv(run / "summary_metrics.csv", index=False)
    folds = pd.read_csv(run / "fold_metrics.csv")
    folds.loc[folds["protocol"] == "group_host", "protocol"] = "group_formula"
    folds.to_csv(run / "fold_metrics.csv", index=False)

    with pytest.raises(ValueError, match="group_host"):
        build_report(run, dataset_summary=_dataset_summary(), join_summary=_join_summary())


def test_report_requires_fold_metrics_artifact(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_report_inputs(run)
    (run / "fold_metrics.csv").unlink()

    with pytest.raises(ValueError, match="fold_metrics.csv"):
        build_report(run, dataset_summary=_dataset_summary(), join_summary=_join_summary())


def test_report_rejects_fold_metrics_missing_required_column(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_report_inputs(run)
    folds = pd.read_csv(run / "fold_metrics.csv").drop(columns="fold")
    folds.to_csv(run / "fold_metrics.csv", index=False)

    with pytest.raises(ValueError, match="fold_metrics.csv.*fold"):
        build_report(run, dataset_summary=_dataset_summary(), join_summary=_join_summary())


def test_report_rejects_fold_metrics_inconsistent_with_summary(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_report_inputs(run)
    folds = pd.read_csv(run / "fold_metrics.csv")
    folds = folds.loc[~((folds["protocol"] == "random_row") & (folds["fold"] == 4))]
    folds.to_csv(run / "fold_metrics.csv", index=False)

    with pytest.raises(ValueError, match="fold_metrics.csv.*random_row"):
        build_report(run, dataset_summary=_dataset_summary(), join_summary=_join_summary())


@pytest.mark.parametrize(
    ("r2_mean", "mae_mean", "has_difference"),
    [
        (0.861, 14.611, True),
        (0.860, 14.611, False),
        (0.760, 21.612, True),
        (0.760, 21.611, False),
    ],
)
def test_report_marks_reproduction_difference_only_beyond_strict_thresholds(
    tmp_path: Path, r2_mean: float, mae_mean: float, has_difference: bool
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_report_inputs(run)
    summary = pd.read_csv(run / "summary_metrics.csv")
    random_row = summary["protocol"] == "random_row"
    summary.loc[random_row, "r2_mean"] = r2_mean
    summary.loc[random_row, "mae_mean"] = mae_mean
    summary.to_csv(run / "summary_metrics.csv", index=False)

    findings = build_report(
        run, dataset_summary=_dataset_summary(), join_summary=_join_summary()
    ).findings.read_text(encoding="utf-8")

    assert ("## 复现差异" in findings) is has_difference
