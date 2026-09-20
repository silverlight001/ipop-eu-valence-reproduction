import re
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


def _write_four_protocol_inputs(run: Path) -> None:
    """Write complete five-fold groups with distinct, consistent metric aggregates."""
    fold_rows, prediction_rows, audit_rows = [], [], []
    for protocol, r2, mae, rmse, column in (
        ("random_row", 0.8, 15, 30, None),
        ("group_formula", 0.7, 20, 35, "Formula"),
        ("group_host", 0.6, 25, 40, "Host"),
        ("group_reference", 0.4, 30, 50, "Reference"),
    ):
        for fold, offset in enumerate((-2, -1, 0, 1, 2)):
            keys = {"protocol": protocol, "feature_set": "AF+T+ES", "model": "xgboost"}
            fold_rows.append({
                **keys, "fold": fold, "train_rows": 80, "test_rows": 20,
                "train_groups": 16, "test_groups": 4,
                "r2": r2 + offset * 0.01, "mae": mae + offset, "rmse": rmse + offset * 2,
            })
            for row in range(20):
                truth = 500.0 + 10 * row
                residual = (mae + offset) * (-1 if row % 2 else 1)
                prediction_rows.append({
                    **keys, "fold": fold, "row_id": fold * 20 + row,
                    "y_true": truth, "y_pred": truth + residual, "residual": residual,
                })
            for audit_column in ("Formula", "Host", "Reference"):
                audit_rows.append({
                    "protocol": protocol, "fold": fold, "audit_column": audit_column,
                    "enforced": audit_column == column,
                    "overlap_count": 0 if audit_column == column else 2,
                })
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(run / "fold_metrics.csv", index=False)
    summary = folds.groupby(["protocol", "feature_set", "model"]).agg(
        fold_count=("fold", "count"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
    ).reset_index()
    summary.to_csv(run / "summary_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run / "predictions.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(run / "overlap_audit.csv", index=False)


def test_findings_compare_all_protocol_metrics_with_sample_sd_and_paper_reference(
    tmp_path: Path,
) -> None:
    _write_four_protocol_inputs(tmp_path)
    text = build_report(tmp_path, _dataset_summary(), _join_summary()).findings.read_text(
        encoding="utf-8"
    )
    assert "AF+T+ES" in text and "XGBoost" in text
    assert "样本标准差" in text and "ddof=1" in text
    assert "R²" in text and "MAE (nm)" in text and "RMSE (nm)" in text
    for expected_row in (
        "| random_row | 0.800 ± 0.016 | 15.000 ± 1.581 | 30.000 ± 3.162 |",
        "| group_formula | 0.700 ± 0.016 | 20.000 ± 1.581 | 35.000 ± 3.162 |",
        "| group_host | 0.600 ± 0.016 | 25.000 ± 1.581 | 40.000 ± 3.162 |",
        "| group_reference | 0.400 ± 0.016 | 30.000 ± 1.581 | 50.000 ± 3.162 |",
    ):
        assert expected_row in text
    assert "论文参考" in text
    for reference in ("0.760 ± 0.022", "14.611 ± 1.438", "30.672 ± 1.469"):
        assert reference in text


@pytest.mark.parametrize(
    "meaning",
    [
        r"重复配方.*基质家族.*random_row.*相似",
        r"合成系列.*相关.*group_reference",
        r"发表偏倚.*负结果.*代表",
        r"Eu\(II\).*Eu\(III\).*价态",
        r"组成特征.*局域配位.*晶场.*不能",
        r"插值.*新基质发现.*实验",
        r"6 条.*仅.*group_reference.*排除.*random_row.*group_formula.*group_host.*保留",
        r"方法级复现.*不通过更换种子筛选更高分",
    ],
    ids=["families", "synthesis", "publication-bias", "eu-valence", "coordination",
         "discovery", "reference-eligibility", "no-cherry-picking"],
)
def test_findings_explain_scientific_limits_and_evaluation_scope(
    tmp_path: Path, meaning: str,
) -> None:
    _write_four_protocol_inputs(tmp_path)
    text = build_report(tmp_path, _dataset_summary(), _join_summary()).findings.read_text(
        encoding="utf-8"
    )
    assert any(re.search(meaning, paragraph) for paragraph in text.split("\n\n"))


def test_report_contains_required_figures_and_reproducibility_disclosure(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_report_inputs(run)

    artifacts = build_report(run, dataset_summary=_dataset_summary(), join_summary=_join_summary())

    assert artifacts.findings.exists()
    text = artifacts.findings.read_text(encoding="utf-8")
    assert text.startswith("# IPOP Eu 发射波长预测的复现与扩展研究\n")
    assert "泛化审计" not in text
    assert "方法级复现" in text
    assert "未公开完整超参数" in text
    assert "group_host" in text
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts.figures)


def test_report_rejects_incomplete_experiment_bundle(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_report_inputs(run)
    (run / ".incomplete").write_text("incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        build_report(run, dataset_summary=_dataset_summary(), join_summary=_join_summary())


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
