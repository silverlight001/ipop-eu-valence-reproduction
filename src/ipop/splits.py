"""Leakage-safe outer evaluation splits and overlap audits."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

SplitProtocol = Literal["random_row", "group_formula", "group_host", "group_reference"]
GROUP_COLUMNS = {
    "group_formula": "Formula",
    "group_host": "Host",
    "group_reference": "Reference",
}
AUDIT_COLUMNS = ("Formula", "Host", "Reference")


class SplitLeakageError(ValueError):
    """Raised when an outer split violates its leakage guarantees."""


def _validate_protocol(protocol: str) -> None:
    if protocol != "random_row" and protocol not in GROUP_COLUMNS:
        raise KeyError(f"Unknown split protocol: {protocol}")


def build_outer_splits(
    frame: pd.DataFrame,
    protocol: SplitProtocol,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each eligible row to exactly one deterministic outer test fold."""
    if n_splits != 5 or seed != 42:
        raise ValueError("fixed outer split contract requires n_splits=5 and seed=42")
    _validate_protocol(protocol)
    if "row_id" not in frame:
        raise KeyError("frame must contain row_id")
    if frame["row_id"].duplicated().any():
        raise SplitLeakageError("frame contains duplicate row_id values")

    group_column = GROUP_COLUMNS.get(protocol)
    eligible = frame.dropna(subset=[group_column]).copy() if group_column else frame.copy()
    if group_column:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(eligible, groups=eligible[group_column])
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(eligible)

    records: list[dict[str, object]] = []
    for fold, (_, test_positions) in enumerate(iterator):
        for position in test_positions:
            records.append(
                {
                    "row_id": eligible.iloc[position]["row_id"],
                    "fold": fold,
                    "protocol": protocol,
                }
            )
    return pd.DataFrame.from_records(records, columns=["row_id", "fold", "protocol"]).sort_values(
        "row_id"
    ).reset_index(drop=True)


def audit_split_overlap(
    frame: pd.DataFrame, assignments: pd.DataFrame, protocol: SplitProtocol
) -> pd.DataFrame:
    """Report Formula/Host/Reference overlap for every outer fold."""
    _validate_protocol(protocol)
    required = {"row_id", "fold"}
    missing = required.difference(assignments.columns)
    if missing:
        raise KeyError(f"assignments missing columns: {sorted(missing)}")
    if assignments["row_id"].duplicated().any():
        raise SplitLeakageError("A row is assigned to more than one outer test fold")
    if frame["row_id"].duplicated().any():
        raise SplitLeakageError("frame contains duplicate row_id values")
    expected_folds = set(range(5))
    actual_folds = set(assignments["fold"])
    if actual_folds != expected_folds or assignments["fold"].value_counts().min() <= 0:
        raise SplitLeakageError("assignments must contain exactly five non-empty folds")

    eligible = frame.dropna(subset=[GROUP_COLUMNS[protocol]] if protocol != "random_row" else []).copy()
    eligible_ids = set(eligible["row_id"])
    assigned_ids = set(assignments["row_id"])
    if assigned_ids != eligible_ids:
        raise SplitLeakageError("Assignments do not cover exactly the eligible rows")

    indexed = eligible.set_index("row_id")
    audits: list[dict[str, object]] = []
    for fold in sorted(assignments["fold"].unique()):
        test_ids = assignments.loc[assignments["fold"] == fold, "row_id"]
        train_ids = assignments.loc[assignments["fold"] != fold, "row_id"]
        for audit_column in AUDIT_COLUMNS:
            overlap = set(indexed.loc[test_ids, audit_column].dropna()) & set(
                indexed.loc[train_ids, audit_column].dropna()
            )
            audits.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    "audit_column": audit_column,
                    "enforced": GROUP_COLUMNS.get(protocol) == audit_column,
                    "overlap_count": len(overlap),
                }
            )
    result = pd.DataFrame.from_records(audits)
    if result.loc[result["enforced"], "overlap_count"].ne(0).any():
        raise SplitLeakageError(f"{protocol} contains group overlap")
    return result
