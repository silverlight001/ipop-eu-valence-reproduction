import json
from pathlib import Path

import pandas as pd
import pytest

from ipop.valence_reporting import build_valence_report

PROTOCOLS = ("random_row", "group_formula", "group_host", "group_reference")


def write_report_inputs(root: Path, *, mismatched: bool = False) -> None:
    fold_rows: list[dict[str, object]] = []
    valence_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        for fold in range(5):
            for mode in ("pooled", "separate"):
                fold_rows.append({
                    "protocol": protocol,
                    "feature_set": "AF+T+ES",
                    "model": "xgboost",
                    "training_mode": mode,
                    "fold": fold,
                    "train_rows": 100,
                    "test_rows": 25 if not (mismatched and mode == "separate") else 24,
                    "train_groups": 100,
                    "test_groups": 25,
                    "mae": 20.0 if mode == "pooled" else 15.0,
                    "rmse": 30.0 if mode == "pooled" else 24.0,
                    "r2": 0.70 if mode == "pooled" else 0.80,
                })
                for valence in (2, 3):
                    valence_rows.append({
                        "protocol": protocol,
                        "feature_set": "AF+T+ES",
                        "model": "xgboost",
                        "training_mode": mode,
                        "Eu valence": valence,
                        "fold": fold,
                        "train_rows": 50,
                        "test_rows": 12,
                        "mae": (18.0 if valence == 2 else 22.0)
                        if mode == "pooled"
                        else (12.0 if valence == 2 else 17.0),
                        "rmse": 30.0 if mode == "pooled" else 24.0,
                        "r2": 0.70 if mode == "pooled" else 0.80,
                    })
                for row_id in range(fold * 5, fold * 5 + 5):
                    prediction_rows.append({
                        "protocol": protocol,
                        "feature_set": "AF+T+ES",
                        "model": "xgboost",
                        "training_mode": mode,
                        "fold": fold,
                        "row_id": row_id,
                        "Eu valence": 2 if row_id % 2 == 0 else 3,
                        "y_true": 500.0,
                        "y_pred": 500.0,
                        "residual": 0.0,
                    })
    pd.DataFrame(fold_rows).to_csv(root / "fold_metrics.csv", index=False)
    pd.DataFrame(valence_rows).to_csv(root / "valence_fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(root / "predictions.csv", index=False)
    (root / "run_metadata.json").write_text(
        json.dumps({"valence_counts": {"2": 626, "3": 1039}}), encoding="utf-8"
    )


def test_build_valence_report_writes_paired_comparison_and_figures(tmp_path: Path) -> None:
    write_report_inputs(tmp_path)

    artifacts = build_valence_report(tmp_path)

    final = pd.read_csv(artifacts.final_results)
    by_valence = pd.read_csv(artifacts.final_results_by_valence)
    assert list(final["protocol"]) == list(PROTOCOLS)
    assert final["feature_set"].eq("AF+T+ES").all()
    assert final["model"].eq("xgboost").all()
    assert final["r2_change"].eq(0.1).all()
    assert final["mae_change_nm"].eq(-5.0).all()
    assert final["mae_improvement_nm"].eq(5.0).all()
    assert set(by_valence["Eu valence"]) == {2, 3}
    assert artifacts.findings.is_file()
    text = artifacts.findings.read_text(encoding="utf-8")
    assert "626" in text and "1039" in text
    assert "四种协议的 MAE 均改善" in text
    assert "主结果固定为 `AF+T+ES / XGBoost`" in text
    assert "不是样本级预测不确定性" in text
    assert all(path.is_file() for path in artifacts.figures)


def test_build_valence_report_rejects_unpaired_fold_populations(tmp_path: Path) -> None:
    write_report_inputs(tmp_path, mismatched=True)

    with pytest.raises(ValueError, match="identical train/test populations"):
        build_valence_report(tmp_path)


def test_build_valence_report_rejects_unpaired_prediction_rows(tmp_path: Path) -> None:
    write_report_inputs(tmp_path)
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    mask = predictions["training_mode"].eq("separate")
    predictions.loc[mask.idxmax(), "row_id"] = 999
    predictions.to_csv(tmp_path / "predictions.csv", index=False)

    with pytest.raises(ValueError, match="identical row ids and folds"):
        build_valence_report(tmp_path)
