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


def _resolved_host(values: pd.Series) -> object:
    if values.isna().any():
        return pd.NA
    unique = values.unique()
    return unique[0] if len(unique) == 1 else pd.NA


def attach_emission_groups(emission: pd.DataFrame, master: pd.DataFrame) -> MetadataJoinResult:
    left = emission.reset_index(drop=True).reset_index(names="row_id")
    right = master.rename(columns={"Inorganic phosphor": "Formula"})
    matches = left[list(JOIN_KEYS) + ["row_id"]].merge(
        right[list(JOIN_KEYS) + ["Host", "Reference"]], on=list(JOIN_KEYS), how="left"
    )
    resolved = matches.groupby("row_id", sort=True).agg(
        Host=("Host", _resolved_host),
        Reference=("Reference", _agreed_value),
        match_count=("Host", "size"),
        host_missing=("Host", lambda x: x.isna().any()),
        host_values=("Host", lambda x: x.dropna().nunique()),
        reference_values=("Reference", lambda x: x.dropna().nunique()),
    )
    missing_host_rows = resolved.index[resolved["host_missing"]].tolist()
    if missing_host_rows:
        raise ValueError(f"missing Host metadata for rows: {missing_host_rows[:20]}")
    conflicting_host_rows = resolved.index[resolved["host_values"] > 1].tolist()
    if conflicting_host_rows:
        raise ValueError(f"conflicting Host metadata for rows: {conflicting_host_rows[:20]}")

    prepared = left.join(resolved[["Host", "Reference"]], on="row_id")
    audit = resolved.reset_index()
    audit["status"] = "resolved"
    audit.loc[audit["reference_values"] > 1, "status"] = "ambiguous_reference"
    return MetadataJoinResult(prepared=prepared, audit=audit)
