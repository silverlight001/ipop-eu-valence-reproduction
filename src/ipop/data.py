from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


class ChecksumMismatch(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    url: str
    size: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    record_id: int
    version: int
    doi: str
    license: str
    artifacts: tuple[ArtifactSpec, ...]


MASTER_TARGET_COLUMNS = (
    "Emission max. (nm)",
    "CIE x coordinate",
    "CIE y coordinate",
    "Int. quantum efficiency (%)",
    "Ext. quantum efficiency (%)",
    "Thermal quenching temp. (K)",
    "1st Excitation max. (nm)",
    "2nd Excitation max. (nm)",
    "3rd Excitation max. (nm)",
    "Decay time (ns)",
)


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSummary:
    records: int
    unique_hosts: int
    unique_references: int
    target_observations: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def load_master_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.dropna(how="all").reset_index(drop=True)


def summarize_master(frame: pd.DataFrame) -> DatasetSummary:
    required = {"Tag", "Host", "Reference", *MASTER_TARGET_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")
    return DatasetSummary(
        records=len(frame),
        unique_hosts=int(frame["Host"].nunique(dropna=True)),
        unique_references=int(frame["Reference"].nunique(dropna=True)),
        target_observations=int(frame[list(MASTER_TARGET_COLUMNS)].notna().sum().sum()),
    )


def validate_master(frame: pd.DataFrame) -> DatasetSummary:
    summary = summarize_master(frame)
    expected = DatasetSummary(3952, 2238, 553, 16023)
    if summary != expected:
        raise DatasetValidationError(
            f"Published invariants differ: expected={expected.as_dict()}, actual={summary.as_dict()}"
        )
    return summary


def load_manifest(path: str | Path) -> DatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetManifest(
        record_id=int(payload["record_id"]),
        version=int(payload["version"]),
        doi=str(payload["doi"]),
        license=str(payload["license"]),
        artifacts=tuple(ArtifactSpec(**item) for item in payload["artifacts"]),
    )


def compute_digest(path: str | Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: str | Path, artifact: ArtifactSpec) -> None:
    source = Path(path)
    if source.stat().st_size != artifact.size:
        raise ChecksumMismatch(
            f"{artifact.name}: expected {artifact.size} bytes, got {source.stat().st_size}"
        )
    for algorithm, expected in (("md5", artifact.md5), ("sha256", artifact.sha256)):
        actual = compute_digest(source, algorithm)
        if actual.lower() != expected.lower():
            raise ChecksumMismatch(
                f"{artifact.name} {algorithm}: expected {expected}, got {actual}"
            )


def build_provenance_record(
    manifest: DatasetManifest, paths: list[Path], retrieved_at: str
) -> dict[str, object]:
    by_name = {artifact.name: artifact for artifact in manifest.artifacts}
    return {
        "record_id": manifest.record_id,
        "version": manifest.version,
        "doi": manifest.doi,
        "license": manifest.license,
        "retrieved_at": retrieved_at,
        "artifacts": [
            {
                "name": path.name,
                "url": by_name[path.name].url,
                "size": path.stat().st_size,
                "md5": compute_digest(path, "md5"),
                "sha256": compute_digest(path, "sha256"),
            }
            for path in paths
        ],
    }


def _url_fetcher(url: str, destination: Path) -> None:
    urllib.request.urlretrieve(url, destination)


def download_artifacts(
    manifest: DatasetManifest,
    destination: str | Path,
    names: set[str] | None = None,
    fetcher: Callable[[str, Path], None] | None = None,
) -> list[Path]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or _url_fetcher
    selected = [a for a in manifest.artifacts if names is None or a.name in names]
    if names is not None and names != {a.name for a in selected}:
        missing = sorted(names - {a.name for a in selected})
        raise KeyError(f"Artifacts absent from manifest: {missing}")

    completed: list[Path] = []
    for artifact in selected:
        final_path = root / artifact.name
        partial_path = final_path.with_name(final_path.name + ".part")
        if final_path.exists():
            verify_artifact(final_path, artifact)
            completed.append(final_path)
            continue
        try:
            fetch(artifact.url, partial_path)
            verify_artifact(partial_path, artifact)
            os.replace(partial_path, final_path)
            completed.append(final_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
    return completed
