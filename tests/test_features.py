import pandas as pd
import pytest

from ipop.features import AF_COLUMNS, FeatureSchemaError, select_features, validate_numeric_columns


def emission_frame() -> pd.DataFrame:
    data = {column: [float(index)] for index, column in enumerate(AF_COLUMNS)}
    data.update({
        "Temp. (K)": [298.0],
        "Excitation source (nm)": [450.0],
        "Emission max. (nm)": [610.0],
        "Formula": ["HostEu0.01"],
    })
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    ("name", "expected_width"),
    [("AF", 52), ("AF+T", 53), ("AF+ES", 53), ("AF+T+ES", 54)],
)
def test_feature_sets_have_exact_width_without_identifiers_or_target(
    name: str, expected_width: int
) -> None:
    selected = select_features(emission_frame(), name)
    assert selected.shape == (1, expected_width)
    assert "Formula" not in selected
    assert "Emission max. (nm)" not in selected


def test_missing_atomic_feature_is_rejected() -> None:
    with pytest.raises(FeatureSchemaError, match=AF_COLUMNS[0]):
        select_features(emission_frame().drop(columns=AF_COLUMNS[0]), "AF")


def test_non_numeric_feature_identifies_column_and_row() -> None:
    frame = emission_frame()
    frame[AF_COLUMNS[0]] = frame[AF_COLUMNS[0]].astype("object")
    frame.loc[0, AF_COLUMNS[0]] = "not-a-number"
    with pytest.raises(FeatureSchemaError, match=f"{AF_COLUMNS[0]}.*row 0"):
        validate_numeric_columns(frame)
