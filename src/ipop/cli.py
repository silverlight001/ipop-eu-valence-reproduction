"""Command-line workflow for the pinned IPOP Eu-emission benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ipop.data import (
    DatasetSummary,
    build_provenance_record,
    download_artifacts,
    load_manifest,
    load_master_csv,
    validate_master,
)
from ipop.experiment import run_experiment
from ipop.features import load_emission_features
from ipop.metadata import attach_emission_groups
from ipop.models import load_experiment_config
from ipop.reporting import build_report
from ipop.valence import run_valence_comparison
from ipop.valence_reporting import build_valence_report

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/ipop_v3.json"
RAW = ROOT / "data/raw"
INTERIM = ROOT / "data/interim"
OUTPUT = ROOT / "outputs/emission-xgb"
VALENCE_OUTPUT = ROOT / "outputs/emission-valence-comparison"
MASTER_NAME = "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
EMISSION_NAME = "phosphor_20230908_Eu_only_EmP_AF.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the stable public workflow interface."""
    parser = argparse.ArgumentParser(prog="ipop")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "download",
        "validate",
        "prepare-emission",
        "run",
        "report",
        "all",
        "run-valence",
        "report-valence",
        "all-valence",
    ):
        subparsers.add_parser(command)
    return parser


def write_validation_report(summary: DatasetSummary, path: str | Path) -> None:
    """Persist the immutable dataset validation summary as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")


def command_download() -> None:
    """Fetch pinned artifacts and record their provenance."""
    manifest = load_manifest(MANIFEST)
    resolved_urls = _previous_retrieved_urls()
    reused_artifacts: set[str] = set()
    paths = download_artifacts(
        manifest, RAW, resolved_urls=resolved_urls, reused_artifacts=reused_artifacts
    )
    provenance = build_provenance_record(
        manifest, paths, datetime.now(UTC).isoformat(), resolved_urls, reused_artifacts
    )
    INTERIM.mkdir(parents=True, exist_ok=True)
    (INTERIM / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def _previous_retrieved_urls() -> dict[str, str]:
    """Reuse recorded HTTP origins without claiming cached data came from Figshare."""
    provenance_path = INTERIM / "source_provenance.json"
    if not provenance_path.is_file():
        return {}
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {
        str(artifact["name"]): str(artifact["retrieved_url"])
        for artifact in payload.get("artifacts", [])
        if isinstance(artifact, dict)
        and isinstance(artifact.get("name"), str)
        and isinstance(artifact.get("retrieved_url"), str)
        and artifact["retrieved_url"].startswith(("https://", "http://"))
    }


def command_validate() -> DatasetSummary:
    """Validate the pinned master table and write its machine-readable audit."""
    summary = validate_master(load_master_csv(RAW / MASTER_NAME))
    write_validation_report(summary, INTERIM / "validation.json")
    return summary


def command_prepare() -> dict[str, int]:
    """Attach host/reference metadata to Eu-emission feature rows."""
    emission = load_emission_features(RAW / EMISSION_NAME)
    master = load_master_csv(RAW / MASTER_NAME)
    result = attach_emission_groups(emission, master)
    ambiguous_reference_rows = int(result.audit["status"].eq("ambiguous_reference").sum())
    if ambiguous_reference_rows != 6:
        raise ValueError(
            "Expected exactly 6 ambiguous source DOI rows, "
            f"got {ambiguous_reference_rows}"
        )
    valence_counts = result.prepared["Eu valence"].value_counts().to_dict()
    if valence_counts != {3: 1039, 2: 626}:
        raise ValueError(
            "Expected Eu valence counts {2: 626, 3: 1039}, "
            f"got {dict(sorted(valence_counts.items()))}"
        )
    INTERIM.mkdir(parents=True, exist_ok=True)
    result.prepared.to_csv(INTERIM / "emission_prepared.csv", index=False)
    result.audit.to_csv(INTERIM / "metadata_join_audit.csv", index=False)
    return {
        "emission_rows": len(result.prepared),
        "ambiguous_reference_rows": ambiguous_reference_rows,
        "eu2_rows": int(valence_counts[2]),
        "eu3_rows": int(valence_counts[3]),
    }


def command_run() -> None:
    """Execute the fixed nested-CV benchmark."""
    frame = pd.read_csv(INTERIM / "emission_prepared.csv")
    config = load_experiment_config(ROOT / "configs/emission_xgb.yaml")
    config["data_provenance"] = json.loads(
        (INTERIM / "source_provenance.json").read_text(encoding="utf-8")
    )
    run_experiment(frame, config, OUTPUT)


def command_report() -> None:
    """Render Chinese findings and four visual validation figures."""
    dataset_summary = json.loads((INTERIM / "validation.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(INTERIM / "metadata_join_audit.csv")
    build_report(
        OUTPUT,
        dataset_summary,
        {
            "emission_rows": len(pd.read_csv(INTERIM / "emission_prepared.csv")),
            "ambiguous_reference_rows": int(audit["status"].eq("ambiguous_reference").sum()),
        },
    )


def command_run_valence() -> None:
    """Rerun pooled and valence-separated models on identical folds."""
    frame = pd.read_csv(INTERIM / "emission_prepared.csv")
    config = load_experiment_config(ROOT / "configs/emission_xgb.yaml")
    config["data_provenance"] = json.loads(
        (INTERIM / "source_provenance.json").read_text(encoding="utf-8")
    )
    run_valence_comparison(frame, config, VALENCE_OUTPUT)


def command_report_valence() -> None:
    """Build paired Eu-valence comparison tables, figures, and findings."""
    build_valence_report(VALENCE_OUTPUT)


def main(argv: list[str] | None = None) -> int:
    """Dispatch one public workflow command."""
    command = build_parser().parse_args(argv).command
    actions = {
        "download": command_download,
        "validate": command_validate,
        "prepare-emission": command_prepare,
        "run": command_run,
        "report": command_report,
        "run-valence": command_run_valence,
        "report-valence": command_report_valence,
    }
    if command == "all":
        command_download()
        command_validate()
        command_prepare()
        command_run()
        command_report()
    elif command == "all-valence":
        command_download()
        command_validate()
        command_prepare()
        command_run_valence()
        command_report_valence()
    else:
        actions[command]()
    return 0
