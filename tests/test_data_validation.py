from pathlib import Path

import pandas as pd
import pytest

from ipop.data import (
    DatasetValidationError,
    load_master_csv,
    summarize_master,
    validate_master,
)

TARGETS = {
    "Emission max. (nm)": [500.0, None],
    "CIE x coordinate": [0.4, None],
    "CIE y coordinate": [0.3, None],
    "Int. quantum efficiency (%)": [None, 50.0],
    "Ext. quantum efficiency (%)": [None, None],
    "Thermal quenching temp. (K)": [None, 450.0],
    "1st Excitation max. (nm)": [350.0, None],
    "2nd Excitation max. (nm)": [None, None],
    "3rd Excitation max. (nm)": [None, None],
    "Decay time (ns)": [1000.0, None],
}


def test_load_master_removes_only_fully_empty_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        " Tag ": [1.0, 2.0, None],
        "Host": ["A", None, None],
        "Reference": ["doi:1", "doi:2", None],
    })
    path = tmp_path / "master.csv"
    frame.to_csv(path, index=False)

    loaded = load_master_csv(path)

    assert list(loaded.columns) == ["Tag", "Host", "Reference"]
    assert len(loaded) == 2
    assert pd.isna(loaded.loc[1, "Host"])


def test_summarize_master_counts_target_observations_once() -> None:
    frame = pd.DataFrame({
        "Tag": [1, 2],
        "Host": ["A", "B"],
        "Reference": ["doi:1", "doi:2"],
        **TARGETS,
    })

    summary = summarize_master(frame)

    assert summary.records == 2
    assert summary.unique_hosts == 2
    assert summary.unique_references == 2
    assert summary.target_observations == 7


def test_summarize_master_lists_missing_required_columns() -> None:
    with pytest.raises(DatasetValidationError, match="Decay time"):
        summarize_master(pd.DataFrame({"Tag": [1], "Host": ["A"], "Reference": ["d"]}))


def test_validate_master_enforces_published_invariants() -> None:
    frame = pd.DataFrame({
        "Tag": range(3952),
        "Host": [f"host-{i % 2238}" for i in range(3952)],
        "Reference": [f"doi:{i % 553}" for i in range(3952)],
        **{column: [1.0] * count + [None] * (3952 - count)
           for column, count in zip(
               (
                   "Emission max. (nm)", "CIE x coordinate", "CIE y coordinate",
                   "Int. quantum efficiency (%)", "Ext. quantum efficiency (%)",
                   "Thermal quenching temp. (K)", "1st Excitation max. (nm)",
                   "2nd Excitation max. (nm)", "3rd Excitation max. (nm)", "Decay time (ns)",
               ),
               (1603, 1603, 1603, 1602, 1602, 1602, 1602, 1602, 1602, 1602),
           )},
    })

    summary = validate_master(frame)

    assert summary.as_dict() == {
        "records": 3952,
        "unique_hosts": 2238,
        "unique_references": 553,
        "target_observations": 16023,
    }


def test_validate_master_reports_invariant_differences() -> None:
    frame = pd.DataFrame({
        "Tag": [1],
        "Host": ["A"],
        "Reference": ["d"],
        **{column: [values[0]] for column, values in TARGETS.items()},
    })

    with pytest.raises(DatasetValidationError, match="Published invariants differ"):
        validate_master(frame)
