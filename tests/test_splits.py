import pandas as pd
import pytest

from ipop.splits import SplitLeakageError, audit_split_overlap, build_outer_splits


def grouped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(20),
            "Formula": [f"F{i // 2}" for i in range(20)],
            "Host": [f"H{i // 4}" for i in range(20)],
            "Reference": [f"D{i // 4}" for i in range(20)],
            "Emission max. (nm)": range(500, 520),
        }
    )


@pytest.mark.parametrize(
    ("protocol", "column"),
    [
        ("group_formula", "Formula"),
        ("group_host", "Host"),
        ("group_reference", "Reference"),
    ],
)
def test_grouped_outer_splits_have_zero_group_overlap(
    protocol: str, column: str
) -> None:
    frame = grouped_frame()
    assignments = build_outer_splits(frame, protocol, n_splits=5, seed=42)
    audit = audit_split_overlap(frame, assignments, protocol)

    assert assignments["fold"].nunique() == 5
    enforced = audit.loc[audit["enforced"]]
    assert enforced["audit_column"].tolist() == [column] * 5
    assert enforced["overlap_count"].eq(0).all()
    for fold in range(5):
        test_rows = assignments.loc[assignments["fold"] == fold, "row_id"]
        train_rows = assignments.loc[assignments["fold"] != fold, "row_id"]
        assert set(frame.loc[test_rows, column]).isdisjoint(
            frame.loc[train_rows, column]
        )


def test_reference_split_excludes_rows_without_unambiguous_reference() -> None:
    frame = grouped_frame()
    frame.loc[0, "Reference"] = pd.NA
    assignments = build_outer_splits(frame, "group_reference", n_splits=5, seed=42)
    assert 0 not in set(assignments["row_id"])


def test_overlap_audit_rejects_deliberately_corrupted_assignment() -> None:
    frame = grouped_frame()
    assignments = build_outer_splits(frame, "group_formula", n_splits=5, seed=42)
    duplicated = assignments.iloc[[0]].assign(
        fold=(assignments.iloc[0]["fold"] + 1) % 5
    )
    corrupted = pd.concat([assignments, duplicated], ignore_index=True)
    with pytest.raises(SplitLeakageError):
        audit_split_overlap(frame, corrupted, "group_formula")
