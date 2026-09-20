"""Build deterministic figures and Chinese findings for an IPOP experiment run."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ipop.experiment import INCOMPLETE_MARKER_NAME


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths to the public reporting artifacts for one experiment run."""

    findings: Path
    figures: tuple[Path, ...]


FIGURE_NAMES = (
    "split_performance.png",
    "feature_ablation.png",
    "parity_random_row.png",
    "residuals_by_protocol.png",
)

FOLD_METRIC_COLUMNS = {
    "protocol",
    "feature_set",
    "model",
    "fold",
    "train_rows",
    "test_rows",
    "train_groups",
    "test_groups",
    "mae",
    "rmse",
    "r2",
}
FOLD_KEY_COLUMNS = ("protocol", "feature_set", "model")


def _require_rows(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    """Return a non-empty result selection or raise an actionable validation error."""
    if frame.empty:
        raise ValueError(f"Report input has no rows for required selection: {description}")
    return frame


def _selected_xgboost_summary(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary.loc[
        (summary["model"] == "xgboost") & (summary["feature_set"] == "AF+T+ES")
    ].sort_values("protocol")
    _require_rows(selected, "xgboost with AF+T+ES")
    required_protocols = {"random_row", "group_host"}
    missing = sorted(required_protocols - set(selected["protocol"]))
    if missing:
        raise ValueError(
            "Report requires xgboost AF+T+ES summary rows for protocols: "
            + ", ".join(missing)
        )
    return selected


def _read_and_validate_fold_metrics(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Read the fold-level artifact and verify its aggregate contract."""
    path = root / "fold_metrics.csv"
    if not path.is_file():
        raise ValueError("Report input is missing required artifact: fold_metrics.csv")
    try:
        folds = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError("Report input cannot read required artifact: fold_metrics.csv") from error

    missing_columns = sorted(FOLD_METRIC_COLUMNS - set(folds.columns))
    if missing_columns:
        raise ValueError(
            "fold_metrics.csv is missing required columns: " + ", ".join(missing_columns)
        )

    summary_columns = [*FOLD_KEY_COLUMNS, "fold_count"]
    expected = summary.loc[:, summary_columns]
    if expected.duplicated(FOLD_KEY_COLUMNS).any():
        raise ValueError("summary_metrics.csv has duplicate protocol/feature_set/model combinations")
    expected_keys = set(expected.loc[:, FOLD_KEY_COLUMNS].itertuples(index=False, name=None))
    observed_keys = set(folds.loc[:, FOLD_KEY_COLUMNS].itertuples(index=False, name=None))
    if observed_keys != expected_keys:
        raise ValueError(
            "fold_metrics.csv protocol/feature_set/model combinations disagree with summary_metrics.csv"
        )

    for row in expected.itertuples(index=False):
        key = (row.protocol, row.feature_set, row.model)
        count = row.fold_count
        if pd.isna(count) or int(count) != count or count < 1:
            raise ValueError(f"summary_metrics.csv has invalid fold_count for {key}")
        expected_folds = set(range(int(count)))
        mask = (folds["protocol"] == row.protocol) & (folds["feature_set"] == row.feature_set)
        mask &= folds["model"] == row.model
        observed_folds = set(folds.loc[mask, "fold"])
        if observed_folds != expected_folds or int(mask.sum()) != int(count):
            raise ValueError(
                "fold_metrics.csv folds for "
                f"{row.protocol}/{row.feature_set}/{row.model} disagree with "
                f"summary fold_count={int(count)}"
            )
    return folds


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def build_report(
    run_dir: str | Path,
    dataset_summary: dict[str, int],
    join_summary: dict[str, int],
) -> ReportArtifacts:
    """Create the four scientific figures and method-level Chinese findings report."""
    root = Path(run_dir)
    if (root / INCOMPLETE_MARKER_NAME).is_file():
        raise ValueError(
            "Experiment bundle is incomplete; rerun the experiment successfully before reporting"
        )
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "summary_metrics.csv")
    _read_and_validate_fold_metrics(root, summary)
    predictions = pd.read_csv(root / "predictions.csv")
    audit = pd.read_csv(root / "overlap_audit.csv")

    sns.set_theme(style="whitegrid", context="talk")
    selected = _selected_xgboost_summary(summary)

    plt.figure()
    ax = sns.barplot(selected, x="protocol", y="r2_mean", color="#3B82F6")
    ax.errorbar(
        range(len(selected)),
        selected["r2_mean"],
        yerr=selected["r2_std"],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set(xlabel="Evaluation protocol", ylabel="Mean outer-fold R²")
    plt.xticks(rotation=20, ha="right")
    _save_figure(figures_dir / FIGURE_NAMES[0])

    ablation = summary.loc[
        (summary["model"] == "xgboost") & (summary["protocol"] == "random_row")
    ].sort_values("feature_set")
    _require_rows(ablation, "xgboost random_row feature ablation")
    plt.figure()
    ax = sns.barplot(ablation, x="feature_set", y="mae_mean", color="#10B981")
    ax.set(xlabel="Feature set", ylabel="Mean outer-fold MAE (nm)")
    _save_figure(figures_dir / FIGURE_NAMES[1])

    parity = predictions.loc[
        (predictions["model"] == "xgboost")
        & (predictions["feature_set"] == "AF+T+ES")
        & (predictions["protocol"] == "random_row")
    ].sort_values(["fold", "row_id"])
    _require_rows(parity, "xgboost AF+T+ES random_row parity predictions")
    plt.figure()
    sns.scatterplot(parity, x="y_true", y="y_pred", s=25, alpha=0.65)
    low = min(parity["y_true"].min(), parity["y_pred"].min())
    high = max(parity["y_true"].max(), parity["y_pred"].max())
    plt.plot([low, high], [low, high], "--", color="black")
    plt.xlabel("Measured emission maximum (nm)")
    plt.ylabel("Predicted emission maximum (nm)")
    _save_figure(figures_dir / FIGURE_NAMES[2])

    residuals = predictions.loc[
        (predictions["model"] == "xgboost") & (predictions["feature_set"] == "AF+T+ES")
    ].sort_values(["protocol", "fold", "row_id"])
    _require_rows(residuals, "xgboost AF+T+ES residual predictions")
    plt.figure(figsize=(6.4, 7.2))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="vert: bool will be deprecated")
        sns.boxplot(residuals, x="protocol", y="residual", color="#F59E0B")
    plt.axhline(0, linestyle="--", color="black")
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("Evaluation protocol")
    plt.ylabel("Prediction minus measurement (nm)")
    _save_figure(figures_dir / FIGURE_NAMES[3])

    random_result = selected.loc[selected["protocol"] == "random_row"].iloc[0]
    differs = (
        abs(float(random_result["r2_mean"]) - 0.760) > 0.10
        or abs(float(random_result["mae_mean"]) - 14.611) > 7.0
    )
    difference_text = ""
    if differs:
        difference_text = (
            "\n## 复现差异\n\n"
            "随机行划分结果超出预先声明的方向性复现窗口。可能原因包括原论文未公开的"
            "精确划分、随机种子、超参数搜索和软件版本；本项目不通过更换种子筛选更高分。\n"
        )
    disclosure = (
        "本文结果属于方法级复现：原论文未公开完整超参数、随机种子和精确数据划分，"
        "因此不主张逐小数位重现。本项目固定 seed 42，不通过更换种子筛选更高分。"
    )
    enforced_audit = audit.loc[audit["enforced"]]
    _require_rows(enforced_audit, "enforced split-overlap audit")
    comparison = (
        "## AF+T+ES XGBoost 四协议比较\n\n"
        "下表为外层折均值 ± 折间样本标准差（ddof=1），不是均值的置信区间。\n\n"
        "| 协议 | R² | MAE (nm) | RMSE (nm) |\n"
        "| --- | --- | --- | --- |\n"
    )
    for protocol in ("random_row", "group_formula", "group_host", "group_reference"):
        for row in selected.loc[selected["protocol"] == protocol].itertuples(index=False):
            comparison += (
                f"| {protocol} | {row.r2_mean:.3f} ± {row.r2_std:.3f} | "
                f"{row.mae_mean:.3f} ± {row.mae_std:.3f} | "
                f"{row.rmse_mean:.3f} ± {row.rmse_std:.3f} |\n"
            )
    comparison += (
        "| 论文参考（随机行） | 0.760 ± 0.022 | 14.611 ± 1.438 | 30.672 ± 1.469 |\n\n"
    )
    findings = root / "findings_zh.md"
    findings.write_text(
        "# IPOP Eu 发射波长复现与泛化审计\n\n"
        f"{disclosure}\n\n"
        f"- 有效主数据：{dataset_summary['records']} 条；host：{dataset_summary['unique_hosts']}；"
        f"文献：{dataset_summary['unique_references']}；性质观测：{dataset_summary['target_observations']}。\n"
        f"- Eu 发射任务：{join_summary['emission_rows']} 条；DOI 归属有歧义："
        f"{join_summary['ambiguous_reference_rows']} 条。\n\n"
        f"这 {join_summary['ambiguous_reference_rows']} 条 DOI 归属有歧义的记录仅在 "
        "`group_reference` 评估中排除（训练与测试均排除），在 `random_row`、"
        "`group_formula` 和 `group_host` 中保留。因此文献分组与其他协议的样本范围略有不同。\n\n"
        f"{comparison}"
        "`group_formula`、`group_host` 和 `group_reference` 均要求分组键零重叠；"
        "三种分组分别检验未见配方、基质字符串和来源文献，不能视为严格嵌套的难度等级。\n\n"
        "强制分组键最大重叠数："
        f"{int(enforced_audit['overlap_count'].max())}。\n\n"
        "## 科学解释与限制\n\n"
        "重复配方和相近基质家族可能跨越 `random_row` 的训练与测试集，使模型受益于"
        "相似材料的信息；配方分组阻断完全相同配方，基质分组进一步检验未见 host。"
        "但不同 host 字符串仍可能属于同一化学家族，因此零字符串重叠并不保证化学独立。\n\n"
        "同一论文中的合成系列常共享实验流程、仪器及系统变化的掺杂条件，观测之间存在"
        "相关性；`group_reference` 把整篇文献留出，有助于检验跨文献迁移，"
        "但性能差异也可能来自测量条件和材料分布的变化，不能全部归因于泄漏。\n\n"
        "文献来源存在发表偏倚：表现较好的材料更容易被报道，失败实验和负结果可能缺失。"
        "因此当前样本不能代表完整的材料搜索空间，按文献分组也不能消除这种选择偏倚。\n\n"
        "Eu 标签混合 Eu(II) 与 Eu(III)；二者价态对应不同的发光机制与光谱特征。"
        "缺少显式价态标签时，模型可能把不同机制合并学习，误差不能仅解释为组成效应。\n\n"
        "组成特征概括元素统计，却没有显式描述 Eu 的局域配位、占据位点及晶场环境，"
        "不能区分相同组成的不同结构或局域环境；加入温度与激发波长也无法完整补足这些信息。\n\n"
        "随机行高分主要支持已见分布附近的插值能力，不能直接证明新基质发现能力。"
        "分组评估更接近特定外推场景，但仍需独立数据、结构信息和前瞻性实验验证；"
        "不能仅据本轮模型将任何候选材料宣称为具有实验前景。\n"
        f"{difference_text}",
        encoding="utf-8",
    )
    return ReportArtifacts(
        findings=findings,
        figures=tuple(figures_dir / name for name in FIGURE_NAMES),
    )
