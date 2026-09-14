from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PROPERTIES = (
    "Atomic Ratio", "Atomic Number", "Atomic Weight", "Atomic Radius",
    "EN pauling", "Valence Electron", "Ionization Energy",
)
ARW_PROPERTIES = (
    "Atomic Number", "Atomic Weight", "Atomic Radius", "EN pauling",
    "Valence Electron", "Ionization Energy",
)
REDUCTIONS = ("sum", "max", "min", "diff")
AF_COLUMNS = tuple(
    [f"{name}_{reduction}" for name in RAW_PROPERTIES for reduction in REDUCTIONS]
    + [f"ARW_{name}_{reduction}" for name in ARW_PROPERTIES for reduction in REDUCTIONS]
)
FEATURE_SETS = {
    "AF": AF_COLUMNS,
    "AF+T": AF_COLUMNS + ("Temp. (K)",),
    "AF+ES": AF_COLUMNS + ("Excitation source (nm)",),
    "AF+T+ES": AF_COLUMNS + ("Temp. (K)", "Excitation source (nm)"),
}
TARGET_COLUMN = "Emission max. (nm)"


class FeatureSchemaError(ValueError):
    pass


def load_emission_features(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path).dropna(how="all").reset_index(drop=True)
    required = {*AF_COLUMNS, "Temp. (K)", "Excitation source (nm)", TARGET_COLUMN, "Formula"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FeatureSchemaError(f"Missing emission columns: {missing}")
    if len(frame) != 1665:
        raise FeatureSchemaError(f"Expected 1665 Eu emission rows, got {len(frame)}")
    if len(AF_COLUMNS) != 52:
        raise AssertionError(f"Expected 52 atomic features, got {len(AF_COLUMNS)}")
    return validate_numeric_columns(frame)


def validate_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = (*AF_COLUMNS, "Temp. (K)", "Excitation source (nm)", TARGET_COLUMN)
    for column in numeric_columns:
        converted = pd.to_numeric(result[column], errors="coerce")
        invalid_rows = result.index[result[column].notna() & converted.isna()].tolist()
        if invalid_rows:
            rendered = ", ".join(f"row {row}" for row in invalid_rows[:20])
            raise FeatureSchemaError(f"{column} contains non-numeric values at {rendered}")
        result[column] = converted
    if result[TARGET_COLUMN].isna().any():
        rows = result.index[result[TARGET_COLUMN].isna()].tolist()
        raise FeatureSchemaError(f"{TARGET_COLUMN} is missing at rows {rows[:20]}")
    return result


def select_features(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set: {name}")
    columns = FEATURE_SETS[name]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FeatureSchemaError(f"Missing feature columns: {missing}")
    return frame.loc[:, columns].copy()
