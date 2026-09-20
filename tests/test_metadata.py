import pandas as pd
import pytest

from ipop.metadata import attach_emission_groups

KEYS = {
    "Formula": ["AEu", "BEu", "CEu"],
    "Temp. (K)": [298.0, 298.0, 300.0],
    "Excitation source (nm)": [400.0, 410.0, 420.0],
    "Emission max. (nm)": [500.0, 510.0, 520.0],
}


def test_metadata_join_keeps_agreed_duplicates_and_flags_ambiguous_reference() -> None:
    emission = pd.DataFrame(KEYS)
    master = pd.DataFrame({
        "Inorganic phosphor": ["AEu", "AEu", "BEu", "BEu", "CEu"],
        "Temp. (K)": [298.0, 298.0, 298.0, 298.0, 300.0],
        "Excitation source (nm)": [400.0, 400.0, 410.0, 410.0, 420.0],
        "Emission max. (nm)": [500.0, 500.0, 510.0, 510.0, 520.0],
        "Host": ["A", "A", "B", "B", "C"],
        "Reference": ["doi:1", "doi:1", "doi:2", "doi:3", "doi:4"],
        "1st dopant": ["Eu", "Eu", "Bi", "Bi", "Eu"],
        "1st dopant valency": [2, 2, 3, 3, 3],
        "2nd dopant": [pd.NA, pd.NA, "Eu", "Eu", pd.NA],
        "2nd dopant valency": [pd.NA, pd.NA, 3, 3, pd.NA],
    })

    result = attach_emission_groups(emission, master)

    assert result.prepared["Host"].tolist() == ["A", "B", "C"]
    assert result.prepared["Eu valence"].tolist() == [2, 3, 3]
    assert result.prepared.loc[0, "Reference"] == "doi:1"
    assert pd.isna(result.prepared.loc[1, "Reference"])
    assert result.audit.query("status == 'ambiguous_reference'")["row_id"].tolist() == [1]


def _single_emission() -> pd.DataFrame:
    return pd.DataFrame({key: [values[0]] for key, values in KEYS.items()})


def _master_with_hosts(hosts: list[object]) -> pd.DataFrame:
    return pd.DataFrame({
        "Inorganic phosphor": ["AEu"] * len(hosts),
        "Temp. (K)": [298.0] * len(hosts),
        "Excitation source (nm)": [400.0] * len(hosts),
        "Emission max. (nm)": [500.0] * len(hosts),
        "Host": hosts,
        "Reference": ["doi:1"] * len(hosts),
        "1st dopant": ["Eu"] * len(hosts),
        "1st dopant valency": [2] * len(hosts),
        "2nd dopant": [pd.NA] * len(hosts),
        "2nd dopant valency": [pd.NA] * len(hosts),
    })


def test_metadata_join_rejects_all_missing_host_values() -> None:
    with pytest.raises(ValueError, match="missing Host metadata"):
        attach_emission_groups(_single_emission(), _master_with_hosts([pd.NA]))


def test_metadata_join_rejects_partially_missing_host_values() -> None:
    with pytest.raises(ValueError, match="missing Host metadata"):
        attach_emission_groups(_single_emission(), _master_with_hosts(["A", pd.NA]))


def test_metadata_join_rejects_conflicting_host_values() -> None:
    with pytest.raises(ValueError, match="conflicting Host metadata"):
        attach_emission_groups(_single_emission(), _master_with_hosts(["A", "B"]))


def test_metadata_join_rejects_conflicting_eu_valence_values() -> None:
    master = _master_with_hosts(["A", "A"])
    master["1st dopant valency"] = [2, 3]

    with pytest.raises(ValueError, match="conflicting Eu valence metadata"):
        attach_emission_groups(_single_emission(), master)


@pytest.mark.parametrize("value", [pd.NA, 1, 4, "unknown"])
def test_metadata_join_rejects_missing_or_unsupported_eu_valence(value: object) -> None:
    master = _master_with_hosts(["A"])
    master["1st dopant valency"] = master["1st dopant valency"].astype(object)
    master.loc[0, "1st dopant valency"] = value

    with pytest.raises(ValueError, match="missing or unsupported Eu valence metadata"):
        attach_emission_groups(_single_emission(), master)
