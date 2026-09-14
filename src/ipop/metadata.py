from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

JOIN_KEYS = ("Formula", "Temp. (K)", "Excitation source (nm)", "Emission max. (nm)")


@dataclass(frozen=True)
class MetadataJoinResult:
    prepared: pd.DataFrame
    audit: pd.DataFrame


def _agreed_value(values: pd.Series) -> object:
    unique = values.dropna().unique()
    return unique[0] if len(unique) == 1 else pd.NA


def attach_emission_groups(emission: pd.DataFrame, master: pd.DataFrame) -> MetadataJoinResult:
    left = emission.reset_index(drop=True).reset_index(names="row_id")
    right = master.rename(columns={"Inorganic phosphor": "Formula"})
    matches = left[list(JOIN_KEYS) + ["row_id"]].merge(
        right[list(JOIN_KEYS) + ["Host", "Reference"]], on=list(JOIN_KEYS), how="left"
    )
    resolved = matches.groupby("row_id", sort=True).agg(
        Host=("Host", _agreed_value),
        Reference=("Reference", _agreed_value),
        match_count=("Host", "size"),
        host_values=("Host", lambda x: x.dropna().nunique()),
        reference_values=("Reference", lambda x: x.dropna().nunique()),
    )
    if resolved["Host"].isna().any():
        rows = resolved.index[resolved["Host"].isna()].tolist()
        raise ValueError(f"Missing or ambiguous host metadata for rows: {rows[:20]}")

    prepared = left.join(resolved[["Host", "Reference"]], on="row_id")
    audit = resolved.reset_index()
    audit["status"] = "resolved"
    audit.loc[audit["reference_values"] > 1, "status"] = "ambiguous_reference"
    return MetadataJoinResult(prepared=prepared, audit=audit)
