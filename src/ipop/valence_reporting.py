"""Reporting for the paired Eu valence comparison experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ipop.experiment import INCOMPLETE_MARKER_NAME

PRIMARY_FEATURE_SET = "AF+T+ES"
PRIMARY_MODEL = "xgboost"
PROTOCOLS = ("random_row", "group_formula", "group_host", "group_reference")
TRAINING_MODES = ("pooled", "separate")


@dataclass(frozen=True)
class ValenceReportArtifacts:
    findings: Path
    final_results: Path
    final_results_by_valence: Path
    figures: tuple[Path, ...]


def _paired_folds(frame: pd.DataFrame, extra_keys: tuple[str, ...] = ()) -> pd.DataFrame:
    selected = frame.loc[
        frame["feature_set"].eq(PRIMARY_FEATURE_SET) & frame["model"].eq(PRIMARY_MODEL)
    ].copy()
    keys = ["protocol", *extra_keys, "fold"]
    duplicate = selected.duplicated([*keys, "training_mode"])
    if duplicate.any():
        raise ValueError("duplicate pooled/separate fold rows in valence comparison")
    observed_modes = set(selected["training_mode"])
    if observed_modes != set(TRAINING_MODES):
        raise ValueError(f"expected pooled and separate training modes, got {observed_modes}")
    paired = selected.pivot(index=keys, columns="training_mode")
    for column in ("train_rows", "test_rows"):
        if not paired[(column, "pooled")].equals(paired[(column, "separate")]):
            raise ValueError("pooled and separate modes must use identical train/test populations")
    return paired


def _comparison_table(
    frame: pd.DataFrame, extra_keys: tuple[str, ...] = ()
) -> pd.DataFrame:
    paired = _paired_folds(frame, extra_keys).reset_index()
    rows: list[dict[str, object]] = []
    grouping = ["protocol", *extra_keys]
    grouper: object = grouping[0] if len(grouping) == 1 else grouping
    for key_values, group in paired.groupby(grouper, sort=False):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        row = dict(zip(grouping, key_values, strict=True))
        row["feature_set"] = PRIMARY_FEATURE_SET
        row["model"] = PRIMARY_MODEL
        row["fold_count"] = len(group)
        for metric in ("r2", "mae", "rmse"):
            pooled = group[(metric, "pooled")]
            separate = group[(metric, "separate")]
            unit = "_nm" if metric in {"mae", "rmse"} else ""
            row[f"pooled_{metric}_mean{unit}"] = float(pooled.mean())
            row[f"pooled_{metric}_std{unit}"] = float(pooled.std(ddof=1))
            row[f"separate_{metric}_mean{unit}"] = float(separate.mean())
            row[f"separate_{metric}_std{unit}"] = float(separate.std(ddof=1))
            change = separate - pooled
            row[f"{metric}_change{unit}"] = float(change.mean())
            row[f"{metric}_change_std{unit}"] = float(change.std(ddof=1))
            if metric in {"mae", "rmse"}:
                row[f"{metric}_improvement{unit}"] = float(-change.mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    protocol_order = {protocol: index for index, protocol in enumerate(PROTOCOLS)}
    result["_order"] = result["protocol"].map(protocol_order)
    result = result.sort_values(["_order", *extra_keys]).drop(columns="_order")
    return result.round(12).reset_index(drop=True)


def _validate_paired_predictions(predictions: pd.DataFrame) -> None:
    selected = predictions.loc[
        predictions["feature_set"].eq(PRIMARY_FEATURE_SET)
        & predictions["model"].eq(PRIMARY_MODEL)
    ].copy()
    keys = ["protocol", "feature_set", "model", "row_id"]
    if selected.duplicated([*keys, "training_mode"]).any():
        raise ValueError("duplicate pooled/separate prediction rows in valence comparison")
    paired = selected.pivot(
        index=keys,
        columns="training_mode",
        values=["fold", "Eu valence", "y_true"],
    )
    if set(paired.columns.get_level_values("training_mode")) != set(TRAINING_MODES):
        raise ValueError("pooled and separate modes must use identical row ids and folds")
    for column in ("fold", "Eu valence", "y_true"):
        pooled = paired[(column, "pooled")]
        separate = paired[(column, "separate")]
        if pooled.isna().any() or separate.isna().any() or not pooled.equals(separate):
            raise ValueError("pooled and separate modes must use identical row ids and folds")


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def build_valence_report(run_dir: str | Path) -> ValenceReportArtifacts:
    """Validate paired folds and write tables, figures, and Chinese findings."""
    root = Path(run_dir)
    if (root / INCOMPLETE_MARKER_NAME).is_file():
        raise ValueError("Valence experiment bundle is incomplete")
    folds = pd.read_csv(root / "fold_metrics.csv")
    valence_folds = pd.read_csv(root / "valence_fold_metrics.csv")
    predictions = pd.read_csv(root / "predictions.csv")
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    _validate_paired_predictions(predictions)
    final = _comparison_table(folds)
    by_valence = _comparison_table(valence_folds, ("Eu valence",))
    if set(final["protocol"]) != set(PROTOCOLS):
        missing = sorted(set(PROTOCOLS) - set(final["protocol"]))
        raise ValueError(f"valence report is missing protocols: {missing}")

    final_path = root / "final_results.csv"
    by_valence_path = root / "final_results_by_valence.csv"
    final.to_csv(final_path, index=False)
    by_valence.to_csv(by_valence_path, index=False)

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    overall_figure = figures_dir / "valence_overall_comparison.png"
    valence_figure = figures_dir / "valence_specific_mae.png"
    plot_rows: list[dict[str, object]] = []
    for row in final.itertuples(index=False):
        plot_rows.extend(
            [
                {
                    "protocol": row.protocol,
                    "training_mode": "Pooled",
                    "r2": row.pooled_r2_mean,
                },
                {
                    "protocol": row.protocol,
                    "training_mode": "Separate",
                    "r2": row.separate_r2_mean,
                },
            ]
        )
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(8.2, 5.4))
    sns.barplot(
        pd.DataFrame(plot_rows),
        x="protocol",
        y="r2",
        hue="training_mode",
        hue_order=["Pooled", "Separate"],
        order=list(PROTOCOLS),
    )
    plt.xlabel("Evaluation protocol")
    plt.ylabel("Mean outer-fold R²")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Training mode")
    _save(overall_figure)

    valence_plot_rows: list[dict[str, object]] = []
    for row in by_valence.itertuples(index=False):
        valence_plot_rows.extend(
            [
                {
                    "protocol": row.protocol,
                    "Eu valence": "Eu(II)" if getattr(row, "_1", row[1]) == 2 else "Eu(III)",
                    "training_mode": "Pooled",
                    "mae": row.pooled_mae_mean_nm,
                },
                {
                    "protocol": row.protocol,
                    "Eu valence": "Eu(II)" if getattr(row, "_1", row[1]) == 2 else "Eu(III)",
                    "training_mode": "Separate",
                    "mae": row.separate_mae_mean_nm,
                },
            ]
        )
    grid = sns.catplot(
        pd.DataFrame(valence_plot_rows),
        x="protocol",
        y="mae",
        hue="training_mode",
        hue_order=["Pooled", "Separate"],
        col="Eu valence",
        kind="bar",
        order=list(PROTOCOLS),
        height=5.2,
        aspect=0.9,
        sharey=True,
    )
    grid.set_axis_labels("Evaluation protocol", "Mean outer-fold MAE (nm)")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.tick_params(axis="x", rotation=20)
    if grid._legend is not None:
        grid._legend.set_title("Training mode")
        grid._legend.set_bbox_to_anchor((1.02, 0.5))
        grid._legend.set_loc("center left")
    grid.figure.tight_layout()
    grid.figure.savefig(valence_figure, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(grid.figure)

    counts = metadata["valence_counts"]
    mae_improved_count = int(final["mae_improvement_nm"].gt(0).sum())
    r2_improved_count = int(final["r2_change"].gt(0).sum())
    if mae_improved_count == len(PROTOCOLS):
        mae_summary = "四种协议的 MAE 均改善。"
    else:
        mae_summary = f"四种协议中有 {mae_improved_count} 种的 MAE 改善。"
    r2_declines = final.loc[final["r2_change"].lt(0), "protocol"].tolist()
    r2_summary = f"四种协议中有 {r2_improved_count} 种的 R²提高。"
    if r2_declines:
        r2_summary += "R²下降的协议为：" + "、".join(r2_declines) + "。"
    lines = [
        "# Eu²⁺/Eu³⁺区分训练对比",
        "",
        f"本实验包含 Eu²⁺ {counts['2']} 条、Eu³⁺ {counts['3']} 条。",
        "混合训练与分价态训练使用完全相同的样本、外层折和特征集合；分价态方案在每个外层训练折内分别拟合 Eu²⁺与 Eu³⁺模型，再合并对应测试预测。",
        "",
        "主结果固定为 `AF+T+ES / XGBoost`。",
        f"{mae_summary}{r2_summary}",
        "表中 ΔR² = 分价态 − 混合；MAE 改善 = 混合 − 分价态，正值表示分价态更好。",
        "",
        "下表中的 ± 为五个外层折之间的样本标准差，用于描述评估稳定性，不是样本级预测不确定性。",
        "",
        "| 协议 | 混合 R² | 分价态 R² | ΔR² | 混合 MAE | 分价态 MAE | MAE 改善 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in final.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.pooled_r2_mean:.3f} ± {row.pooled_r2_std:.3f} | "
            f"{row.separate_r2_mean:.3f} ± {row.separate_r2_std:.3f} | "
            f"{row.r2_change:+.3f} | {row.pooled_mae_mean_nm:.3f} ± "
            f"{row.pooled_mae_std_nm:.3f} | {row.separate_mae_mean_nm:.3f} ± "
            f"{row.separate_mae_std_nm:.3f} | {row.mae_improvement_nm:+.3f} nm |"
        )
    best_valence_row = by_valence.loc[by_valence["mae_improvement_nm"].idxmax()]
    worsened_rows = by_valence.loc[by_valence["mae_improvement_nm"].lt(0)]
    best_valence_label = {2: "II", 3: "III"}[int(best_valence_row["Eu valence"])]
    lines.extend(
        [
            "",
            "## 分价态观察",
            "",
            (
                f"最大的分层 MAE 改善出现在 Eu({best_valence_label}) / "
                f"{best_valence_row['protocol']}："
                f"{best_valence_row['mae_improvement_nm']:.3f} nm。"
            ),
        ]
    )
    for row in worsened_rows.itertuples(index=False):
        valence = getattr(row, "_1", row[1])
        valence_label = {2: "II", 3: "III"}[int(valence)]
        lines.append(
            f"Eu({valence_label}) / {row.protocol} 的 MAE 反而增加 "
            f"{-row.mae_improvement_nm:.3f} nm，说明总体改善并非在两个价态、所有外推场景中都成立。"
        )
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "本对比检验的是价态专属建模策略。`separate` 使用两个独立调参的模型，因此收益同时包含显式价态分组和模型容量增加，不能解释为单一价态特征的纯因果效应。后续可增加 pooled + valence indicator 对照来分离这两个因素。",
            "模型仍只使用组成统计和测量条件，尚未加入晶体结构、占位、完整光谱、寿命多指数模型或样本级不确定性量化。",
        ]
    )
    findings = root / "findings_zh.md"
    findings.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ValenceReportArtifacts(
        findings=findings,
        final_results=final_path,
        final_results_by_valence=by_valence_path,
        figures=(overall_figure, valence_figure),
    )
