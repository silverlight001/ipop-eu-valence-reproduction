import pandas as pd

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
    })

    result = attach_emission_groups(emission, master)

    assert result.prepared["Host"].tolist() == ["A", "B", "C"]
    assert result.prepared.loc[0, "Reference"] == "doi:1"
    assert pd.isna(result.prepared.loc[1, "Reference"])
    assert result.audit.query("status == 'ambiguous_reference'")["row_id"].tolist() == [1]
