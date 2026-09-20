from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

JOIN_KEYS = ("Formula", "Temp. (K)", "Excitation source (nm)", "Emission max. (nm)")
EU_VALENCE_COLUMN = "Eu valence"


@dataclass(frozen=True)
class MetadataJoinResult:
    prepared: pd.DataFrame
    audit: pd.DataFrame


def _agreed_value(values: pd.Series) -> object:
    unique = values.dropna().unique()
    return unique[0] if len(unique) == 1 else pd.NA


def _resolved_host(values: pd.Series) -> object:
    if values.isna().any():
        return pd.NA
    unique = values.unique()
    return unique[0] if len(unique) == 1 else pd.NA


def _with_eu_valence(master: pd.DataFrame) -> pd.DataFrame:
    required = {
        "1st dopant",
        "1st dopant valency",
        "2nd dopant",
        "2nd dopant valency",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"missing Eu valence source columns: {missing}")

    result = master.copy()
    first_is_eu = (
        result["1st dopant"].astype("string").str.strip().str.casefold().eq("eu").fillna(False)
    )
    second_is_eu = (
        result["2nd dopant"].astype("string").str.strip().str.casefold().eq("eu").fillna(False)
    )
    first_valence = pd.to_numeric(result["1st dopant valency"], errors="coerce")
    second_valence = pd.to_numeric(result["2nd dopant valency"], errors="coerce")
    eu_valence = pd.Series(float("nan"), index=result.index, dtype=float)
    eu_valence.loc[first_is_eu] = first_valence.loc[first_is_eu]
    second_only = second_is_eu & ~first_is_eu
    eu_valence.loc[second_only] = second_valence.loc[second_only]
    result[EU_VALENCE_COLUMN] = eu_valence
    conflicting_positions = first_is_eu & second_is_eu & first_valence.ne(second_valence)
    result["_eu_valence_invalid"] = (
        ~result[EU_VALENCE_COLUMN].isin([2, 3]) | conflicting_positions.fillna(False)
    )
    return result


def attach_emission_groups(emission: pd.DataFrame, master: pd.DataFrame) -> MetadataJoinResult:
    left = emission.reset_index(drop=True).reset_index(names="row_id")
    right = _with_eu_valence(master).rename(columns={"Inorganic phosphor": "Formula"})
    matches = left[list(JOIN_KEYS) + ["row_id"]].merge(
        right[
            list(JOIN_KEYS)
            + ["Host", "Reference", EU_VALENCE_COLUMN, "_eu_valence_invalid"]
        ],
        on=list(JOIN_KEYS),
        how="left",
    )
    resolved = matches.groupby("row_id", sort=True).agg(
        Host=("Host", _resolved_host),
        Reference=("Reference", _agreed_value),
        **{EU_VALENCE_COLUMN: (EU_VALENCE_COLUMN, _agreed_value)},
        match_count=("Host", "size"),
        host_missing=("Host", lambda x: x.isna().any()),
        host_values=("Host", lambda x: x.dropna().nunique()),
        reference_values=("Reference", lambda x: x.dropna().nunique()),
        eu_valence_missing=(EU_VALENCE_COLUMN, lambda x: x.isna().any()),
        eu_valence_values=(EU_VALENCE_COLUMN, lambda x: x.dropna().nunique()),
        eu_valence_invalid=("_eu_valence_invalid", lambda x: x.fillna(True).any()),
    )
    missing_host_rows = resolved.index[resolved["host_missing"]].tolist()
    if missing_host_rows:
        raise ValueError(f"missing Host metadata for rows: {missing_host_rows[:20]}")
    conflicting_host_rows = resolved.index[resolved["host_values"] > 1].tolist()
    if conflicting_host_rows:
        raise ValueError(f"conflicting Host metadata for rows: {conflicting_host_rows[:20]}")

    invalid_valence_rows = resolved.index[
        resolved["eu_valence_missing"] | resolved["eu_valence_invalid"]
    ].tolist()
    if invalid_valence_rows:
        raise ValueError(
            "missing or unsupported Eu valence metadata for rows: "
            f"{invalid_valence_rows[:20]}"
        )
    conflicting_valence_rows = resolved.index[resolved["eu_valence_values"] > 1].tolist()
    if conflicting_valence_rows:
        raise ValueError(
            f"conflicting Eu valence metadata for rows: {conflicting_valence_rows[:20]}"
        )

    prepared = left.join(resolved[["Host", "Reference", EU_VALENCE_COLUMN]], on="row_id")
    prepared[EU_VALENCE_COLUMN] = prepared[EU_VALENCE_COLUMN].astype(int)
    audit = resolved.reset_index()
    audit["status"] = "resolved"
    audit.loc[audit["reference_values"] > 1, "status"] = "ambiguous_reference"
    return MetadataJoinResult(prepared=prepared, audit=audit)
