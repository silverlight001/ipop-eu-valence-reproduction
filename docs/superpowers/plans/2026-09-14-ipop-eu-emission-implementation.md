# IPOP Eu Emission Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local benchmark that approximates the paper's Eu emission-wavelength XGBoost result and quantifies the performance change under formula-, host-, and publication-disjoint evaluation.

**Architecture:** A small `src`-layout Python package separates pinned data acquisition, schema validation, metadata recovery, leakage-safe splits, model evaluation, and report generation. Experiments are configuration-driven and persist every split, prediction, metric, provenance record, and environment detail needed to audit a result.

**Tech Stack:** Python 3.11+, pandas, NumPy, scikit-learn, XGBoost, PyYAML, Matplotlib, seaborn, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-14-ipop-eu-emission-design.md`

## Global Constraints

- Treat Figshare record `24771186`, version `1`, as the canonical source and record the `CC BY 4.0` license.
- Preserve raw scientific values; only rows missing every column may be removed from the master CSV.
- Validate exactly 3,952 populated records, 2,238 unique hosts, 553 unique references, 16,023 target observations, 1,665 Eu emission rows, and 52 atomic features.
- Never include `Formula`, `Host`, `Reference`, row identifiers, or the emission target in the model feature matrix.
- Fit imputation and hyperparameter selection only on each outer training fold.
- Save outer-fold assignments before fitting models.
- Use five outer folds and seed `42`; fail instead of silently changing the fold count.
- A grouped split must have zero overlap for its declared grouping key.
- Do not tune against the paper's reported test metrics or suppress unfavorable grouped-split results.
- Keep downloaded data, interim tables, trained models, and generated outputs out of Git.
- Commands must work in PowerShell from `F:\AI4S` and in a standard POSIX shell from the repository root.

## File Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, dependencies, CLI entry point, pytest and Ruff settings |
| `.gitignore` | Exclude environments, caches, downloads, interim data, and run outputs |
| `data/manifests/ipop_v3.json` | Pinned DOI, version, license, file URLs, sizes, and MD5 checksums |
| `configs/emission_xgb.yaml` | Feature sets, split protocols, seed, folds, and bounded XGBoost search |
| `src/ipop/data.py` | Manifest parsing, atomic download, checksum verification, master-table loading and validation |
| `src/ipop/features.py` | Exact atomic-feature schema and target-safe feature selection |
| `src/ipop/metadata.py` | Recover host and DOI groups for Eu emission rows and report ambiguous matches |
| `src/ipop/splits.py` | Build saved row/group outer folds and audit overlap |
| `src/ipop/models.py` | Median baseline, XGBoost pipeline, and inner-fold hyperparameter search |
| `src/ipop/metrics.py` | Fold-level regression metrics and aggregation |
| `src/ipop/experiment.py` | Orchestrate feature sets, split protocols, models, and persisted artifacts |
| `src/ipop/reporting.py` | Generate CSV summaries, scientific figures, and the Chinese findings report |
| `src/ipop/cli.py`, `src/ipop/__main__.py` | `download`, `validate`, `prepare-emission`, `run`, `report`, and `all` commands |
| `tests/` | Unit and integration tests with small synthetic fixtures |
| `README.md` | Chinese narrative, installation, commands, outputs, limitations, and citation |

---

### Task 1: Project Foundation and Pinned Data Acquisition

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `data/manifests/ipop_v3.json`
- Create: `src/ipop/__init__.py`
- Create: `src/ipop/data.py`
- Create: `tests/test_data_download.py`

**Interfaces:**
- Produces: `ArtifactSpec`, `DatasetManifest`, `load_manifest(path)`, `compute_digest(path, algorithm)`, `verify_artifact(path, spec)`, `download_artifacts(manifest, destination, names=None, fetcher=None)`, and `build_provenance_record(manifest, paths, retrieved_at)`.
- Consumes: no application code; only Python standard-library I/O and the pinned JSON manifest.

- [ ] **Step 1: Add package configuration and ignored paths**

Create `pyproject.toml` with these dependency bounds and tool settings:

```toml
[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[project]
name = "ipop-reproduction"
version = "0.1.0"
description = "Reproducible IPOP Eu emission benchmark and leakage audit"
requires-python = ">=3.11"
dependencies = [
  "matplotlib>=3.9,<4",
  "numpy>=1.26,<3",
  "pandas>=2.2,<3",
  "pyyaml>=6,<7",
  "scikit-learn>=1.6,<2",
  "seaborn>=0.13,<1",
  "xgboost>=2.1,<4",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9", "ruff>=0.9,<1"]

[project.scripts]
ipop = "ipop.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/ipop"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
data/raw/
data/interim/
outputs/
*.joblib
```

Create `src/ipop/__init__.py` with only `__version__ = "0.1.0"`.

Install the editable development environment before running the first test:

```bash
python -m pip install -e ".[dev]"
```

Expected: installation completes and `python -c "import ipop"` exits with code 0.

- [ ] **Step 2: Pin the two artifacts required by this implementation cycle**

Create `data/manifests/ipop_v3.json`:

```json
{
  "record_id": 24771186,
  "version": 1,
  "doi": "10.6084/m9.figshare.24771186.v1",
  "license": "CC BY 4.0",
  "artifacts": [
    {
      "name": "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv",
      "url": "https://ndownloader.figshare.com/files/43535559",
      "size": 987261,
      "md5": "8ec75efc133963ce9961626bc483a418",
      "sha256": "9ebbee222e7b21faac0919761fbc8ee76c304c8ae3c8c89f0ab384a3d53b2924"
    },
    {
      "name": "phosphor_20230908_Eu_only_EmP_AF.csv",
      "url": "https://ndownloader.figshare.com/files/43535562",
      "size": 570738,
      "md5": "8aa14a486b06e20572a9707225f64e28",
      "sha256": "c573eee9a3893a8501be83f24807eac10f633d32d439c048c0cdc17fd8233afd"
    }
  ]
}
```

- [ ] **Step 3: Write failing acquisition tests**

Create `tests/test_data_download.py`:

```python
import hashlib
import json
from pathlib import Path

import pytest

from ipop.data import ChecksumMismatch, build_provenance_record, download_artifacts, load_manifest


def test_load_manifest_exposes_pinned_provenance(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "record_id": 24771186,
        "version": 1,
        "doi": "10.6084/m9.figshare.24771186.v1",
        "license": "CC BY 4.0",
        "artifacts": [{"name": "a.csv", "url": "https://example/a", "size": 3,
                       "md5": hashlib.md5(b"abc").hexdigest(),
                       "sha256": hashlib.sha256(b"abc").hexdigest()}],
    }), encoding="utf-8")

    manifest = load_manifest(path)

    assert manifest.record_id == 24771186
    assert manifest.license == "CC BY 4.0"
    assert manifest.artifacts[0].name == "a.csv"


def test_download_is_atomic_and_checksum_verified(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    digest = hashlib.md5(b"abc").hexdigest()
    manifest_path.write_text(json.dumps({
        "record_id": 1, "version": 1, "doi": "d", "license": "l",
        "artifacts": [{"name": "a.csv", "url": "memory://a", "size": 3, "md5": digest,
                       "sha256": hashlib.sha256(b"abc").hexdigest()}],
    }), encoding="utf-8")

    def fetcher(url: str, destination: Path) -> None:
        destination.write_bytes(b"abc")

    paths = download_artifacts(load_manifest(manifest_path), tmp_path / "raw", fetcher=fetcher)

    assert paths == [tmp_path / "raw" / "a.csv"]
    assert paths[0].read_bytes() == b"abc"
    assert not (tmp_path / "raw" / "a.csv.part").exists()


def test_download_rejects_checksum_mismatch_and_removes_partial_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "record_id": 1, "version": 1, "doi": "d", "license": "l",
        "artifacts": [{"name": "a.csv", "url": "memory://a", "size": 3,
                       "md5": hashlib.md5(b"abc").hexdigest(),
                       "sha256": hashlib.sha256(b"abc").hexdigest()}],
    }), encoding="utf-8")

    def corrupt_fetcher(url: str, destination: Path) -> None:
        destination.write_bytes(b"bad")

    with pytest.raises(ChecksumMismatch, match="a.csv"):
        download_artifacts(load_manifest(manifest_path), tmp_path / "raw", fetcher=corrupt_fetcher)

    assert not (tmp_path / "raw" / "a.csv").exists()
    assert not (tmp_path / "raw" / "a.csv.part").exists()


def test_provenance_records_both_hashes_and_retrieval_time(tmp_path: Path) -> None:
    payload = b"abc"
    artifact = tmp_path / "a.csv"
    artifact.write_bytes(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "record_id": 1, "version": 1, "doi": "d", "license": "CC BY 4.0",
        "artifacts": [{"name": "a.csv", "url": "memory://a", "size": 3,
                       "md5": hashlib.md5(payload).hexdigest(),
                       "sha256": hashlib.sha256(payload).hexdigest()}],
    }), encoding="utf-8")
    provenance = build_provenance_record(
        load_manifest(manifest_path), [artifact], "2026-09-14T00:00:00Z"
    )
    assert provenance["retrieved_at"] == "2026-09-14T00:00:00Z"
    assert provenance["artifacts"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 4: Run the tests and verify the expected import failure**

Run: `python -m pytest tests/test_data_download.py -v`

Expected: FAIL during collection because `ipop.data` does not exist.

- [ ] **Step 5: Implement manifest parsing and atomic downloads**

Create `src/ipop/data.py` with immutable dataclasses and this behavior:

```python
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
```

- [ ] **Step 6: Run tests and static checks**

Run: `python -m pytest tests/test_data_download.py -v`

Expected: 4 passed.

Run: `python -m ruff check src/ipop/data.py tests/test_data_download.py`

Expected: exit 0 with no diagnostics.

- [ ] **Step 7: Commit the acquisition foundation**

```bash
git add pyproject.toml .gitignore data/manifests/ipop_v3.json src/ipop tests/test_data_download.py
git commit -m "feat: pin and verify IPOP source data"
```

---

### Task 2: Master Dataset Cleaning and Published-Invariant Validation

**Files:**
- Modify: `src/ipop/data.py`
- Create: `tests/test_data_validation.py`

**Interfaces:**
- Consumes: `DatasetManifest` and downloaded master CSV from Task 1.
- Produces: `DatasetSummary`, `load_master_csv(path) -> DataFrame`, `summarize_master(df) -> DatasetSummary`, and `validate_master(df) -> DatasetSummary`.

- [ ] **Step 1: Write failing tests for empty-row cleaning and observation counts**

Create `tests/test_data_validation.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from ipop.data import DatasetValidationError, load_master_csv, summarize_master


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
        "Tag": [1.0, 2.0, None],
        "Host": ["A", None, None],
        "Reference": ["doi:1", "doi:2", None],
    })
    path = tmp_path / "master.csv"
    frame.to_csv(path, index=False)

    loaded = load_master_csv(path)

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
```

- [ ] **Step 2: Run tests and verify they fail because validation APIs are missing**

Run: `python -m pytest tests/test_data_validation.py -v`

Expected: FAIL during collection for missing `DatasetValidationError` or `DatasetSummary`.

- [ ] **Step 3: Implement cleaning, summary, and strict published checks**

Append to `src/ipop/data.py`:

```python
from dataclasses import asdict

import pandas as pd


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
```

- [ ] **Step 4: Run the new and existing tests**

Run: `python -m pytest tests/test_data_download.py tests/test_data_validation.py -v`

Expected: 7 passed.

- [ ] **Step 5: Commit master validation**

```bash
git add src/ipop/data.py tests/test_data_validation.py
git commit -m "feat: validate published IPOP dataset invariants"
```

---

### Task 3: Eu Emission Feature Safety and Metadata Recovery

**Files:**
- Create: `src/ipop/features.py`
- Create: `src/ipop/metadata.py`
- Create: `tests/test_features.py`
- Create: `tests/test_metadata.py`

**Interfaces:**
- Consumes: cleaned master table from `load_master_csv` and the authors' Eu emission CSV.
- Produces: `AF_COLUMNS`, `FEATURE_SETS`, `load_emission_features(path)`, `validate_numeric_columns(frame)`, `select_features(frame, name)`, `MetadataJoinResult`, and `attach_emission_groups(emission, master)`.

- [ ] **Step 1: Write failing feature-safety tests**

Create `tests/test_features.py`:

```python
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
    frame.loc[0, AF_COLUMNS[0]] = "not-a-number"
    with pytest.raises(FeatureSchemaError, match=f"{AF_COLUMNS[0]}.*row 0"):
        validate_numeric_columns(frame)
```

- [ ] **Step 2: Run feature tests and verify the module is missing**

Run: `python -m pytest tests/test_features.py -v`

Expected: FAIL during collection because `ipop.features` does not exist.

- [ ] **Step 3: Implement the exact 52-column schema and feature groups**

Create `src/ipop/features.py`. Generate the fixed column tuple from these names, preserving order:

```python
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
```

- [ ] **Step 4: Run feature tests**

Run: `python -m pytest tests/test_features.py -v`

Expected: 6 passed.

- [ ] **Step 5: Write failing metadata-join tests**

Create `tests/test_metadata.py`:

```python
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
```

- [ ] **Step 6: Run metadata tests and verify the module is missing**

Run: `python -m pytest tests/test_metadata.py -v`

Expected: FAIL during collection because `ipop.metadata` does not exist.

- [ ] **Step 7: Implement deterministic group recovery**

Create `src/ipop/metadata.py` with these public types and resolution rule:

```python
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
```

- [ ] **Step 8: Run metadata tests and all Task 3 tests**

Run: `python -m pytest tests/test_features.py tests/test_metadata.py -v`

Expected: 7 passed.

- [ ] **Step 9: Commit safe feature and group metadata preparation**

```bash
git add src/ipop/features.py src/ipop/metadata.py tests/test_features.py tests/test_metadata.py
git commit -m "feat: prepare target-safe Eu emission data"
```

---

### Task 4: Leakage-Safe Outer Splits and Audits

**Files:**
- Create: `src/ipop/splits.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Consumes: the prepared DataFrame containing `row_id`, `Formula`, `Host`, and nullable `Reference`.
- Produces: `SplitProtocol`, `build_outer_splits(frame, protocol, n_splits=5, seed=42) -> DataFrame`, and `audit_split_overlap(frame, assignments, protocol) -> DataFrame`.

- [ ] **Step 1: Write failing split tests**

Create `tests/test_splits.py`:

```python
import pandas as pd
import pytest

from ipop.splits import SplitLeakageError, audit_split_overlap, build_outer_splits


def grouped_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": range(20),
        "Formula": [f"F{i // 2}" for i in range(20)],
        "Host": [f"H{i // 4}" for i in range(20)],
        "Reference": [f"D{i // 4}" for i in range(20)],
        "Emission max. (nm)": range(500, 520),
    })


@pytest.mark.parametrize(
    ("protocol", "column"),
    [("group_formula", "Formula"), ("group_host", "Host"), ("group_reference", "Reference")],
)
def test_grouped_outer_splits_have_zero_group_overlap(protocol: str, column: str) -> None:
    frame = grouped_frame()
    assignments = build_outer_splits(frame, protocol, n_splits=5, seed=42)
    audit = audit_split_overlap(frame, assignments, protocol)

    assert assignments["fold"].nunique() == 5
    enforced = audit.loc[audit["enforced"]]
    assert enforced["audit_column"].tolist() == [column] * 5
    assert enforced["overlap_count"].eq(0).all()
    for fold in range(5):
        test_rows = assignments.loc[assignments["fold"] == fold, "row_id"]
        train_rows = assignments.loc[assignments["fold"] != fold, "row_id"]
        assert set(frame.loc[test_rows, column]).isdisjoint(frame.loc[train_rows, column])


def test_reference_split_excludes_rows_without_unambiguous_reference() -> None:
    frame = grouped_frame()
    frame.loc[0, "Reference"] = pd.NA
    assignments = build_outer_splits(frame, "group_reference", n_splits=5, seed=42)
    assert 0 not in set(assignments["row_id"])


def test_overlap_audit_rejects_deliberately_corrupted_assignment() -> None:
    frame = grouped_frame()
    assignments = build_outer_splits(frame, "group_formula", n_splits=5, seed=42)
    duplicated = assignments.iloc[[0]].assign(fold=(assignments.iloc[0]["fold"] + 1) % 5)
    corrupted = pd.concat([assignments, duplicated], ignore_index=True)
    with pytest.raises(SplitLeakageError):
        audit_split_overlap(frame, corrupted, "group_formula")
```

- [ ] **Step 2: Run split tests and verify the module is missing**

Run: `python -m pytest tests/test_splits.py -v`

Expected: FAIL during collection because `ipop.splits` does not exist.

- [ ] **Step 3: Implement row and group split generation**

Create `src/ipop/splits.py`. Use one row per outer-test assignment; training rows for fold `k` are all eligible rows assigned to folds other than `k`.

```python
from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


SplitProtocol = Literal["random_row", "group_formula", "group_host", "group_reference"]
GROUP_COLUMNS = {
    "group_formula": "Formula",
    "group_host": "Host",
    "group_reference": "Reference",
}


class SplitLeakageError(ValueError):
    pass


def build_outer_splits(
    frame: pd.DataFrame, protocol: SplitProtocol, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    eligible = frame.copy()
    group_column = GROUP_COLUMNS.get(protocol)
    if group_column is not None:
        eligible = eligible.dropna(subset=[group_column])
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(eligible, groups=eligible[group_column])
    elif protocol == "random_row":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(eligible)
    else:
        raise KeyError(f"Unknown split protocol: {protocol}")

    records: list[dict[str, object]] = []
    for fold, (_, test_positions) in enumerate(iterator):
        for position in test_positions:
            records.append({"row_id": int(eligible.iloc[position]["row_id"]), "fold": fold,
                            "protocol": protocol})
    return pd.DataFrame.from_records(records).sort_values("row_id").reset_index(drop=True)


def audit_split_overlap(
    frame: pd.DataFrame, assignments: pd.DataFrame, protocol: SplitProtocol
) -> pd.DataFrame:
    if assignments["row_id"].duplicated().any():
        raise SplitLeakageError("A row is assigned to more than one outer test fold")
    eligible = frame[frame["row_id"].isin(assignments["row_id"])].set_index("row_id")
    audits: list[dict[str, object]] = []
    for fold in sorted(assignments["fold"].unique()):
        test_ids = assignments.loc[assignments["fold"] == fold, "row_id"]
        train_ids = assignments.loc[assignments["fold"] != fold, "row_id"]
        for audit_column in ("Formula", "Host", "Reference"):
            overlap = set(eligible.loc[test_ids, audit_column].dropna()) & set(
                eligible.loc[train_ids, audit_column].dropna()
            )
            enforced = GROUP_COLUMNS.get(protocol) == audit_column
            audits.append({"protocol": protocol, "fold": int(fold),
                           "audit_column": audit_column, "enforced": enforced,
                           "overlap_count": len(overlap)})
    result = pd.DataFrame(audits)
    if result.loc[result["enforced"], "overlap_count"].ne(0).any():
        raise SplitLeakageError(f"{protocol} contains group overlap")
    return result
```

- [ ] **Step 4: Run split tests and the suite**

Run: `python -m pytest tests/test_splits.py -v`

Expected: 5 passed.

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit split protocols**

```bash
git add src/ipop/splits.py tests/test_splits.py
git commit -m "feat: add leakage-safe evaluation splits"
```

---

### Task 5: Regression Metrics, Model Pipelines, and Experiment Configuration

**Files:**
- Create: `configs/emission_xgb.yaml`
- Create: `src/ipop/metrics.py`
- Create: `src/ipop/models.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: training features and targets plus optional inner-split groups.
- Produces: `regression_metrics(y_true, y_pred)`, `aggregate_metrics(folds)`, `build_dummy_pipeline()`, `build_xgb_search(config, inner_cv)`, and `load_experiment_config(path)`.

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd

from ipop.metrics import aggregate_metrics, regression_metrics


def test_regression_metrics_match_known_values() -> None:
    result = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert result["mae"] == 1 / 3
    assert np.isclose(result["rmse"], np.sqrt(1 / 3))
    assert result["r2"] == 0.5


def test_aggregate_metrics_keeps_fold_values_and_sample_standard_deviation() -> None:
    folds = pd.DataFrame({"mae": [1.0, 3.0], "rmse": [2.0, 4.0], "r2": [0.5, 0.7]})
    result = aggregate_metrics(folds)
    assert result.loc[0, "mae_mean"] == 2.0
    assert np.isclose(result.loc[0, "r2_std"], np.std([0.5, 0.7], ddof=1))
```

- [ ] **Step 2: Run metric tests and verify the module is missing**

Run: `python -m pytest tests/test_metrics.py -v`

Expected: FAIL during collection because `ipop.metrics` does not exist.

- [ ] **Step 3: Implement fold metrics and aggregation**

Create `src/ipop/metrics.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def aggregate_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    grouping = [column for column in ("protocol", "feature_set", "model") if column in folds]
    grouped = folds.groupby(grouping, dropna=False) if grouping else [((), folds)]
    rows: list[dict[str, object]] = []
    for keys, frame in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(grouping, keys, strict=True))
        row["fold_count"] = len(frame)
        for metric in ("mae", "rmse", "r2"):
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run metric tests**

Run: `python -m pytest tests/test_metrics.py -v`

Expected: 2 passed.

- [ ] **Step 5: Create the declared experiment configuration**

Create `configs/emission_xgb.yaml`:

```yaml
seed: 42
outer_folds: 5
inner_folds: 3
target: "Emission max. (nm)"
feature_sets: ["AF", "AF+T", "AF+ES", "AF+T+ES"]
protocols: ["random_row", "group_formula", "group_host", "group_reference"]
models: ["median", "xgboost"]
xgboost:
  objective: "reg:squarederror"
  tree_method: "hist"
  n_jobs: 1
  parameter_grid:
    - {n_estimators: 300, max_depth: 3, learning_rate: 0.05, subsample: 0.8, colsample_bytree: 0.8, reg_lambda: 1.0}
    - {n_estimators: 600, max_depth: 3, learning_rate: 0.03, subsample: 0.8, colsample_bytree: 0.8, reg_lambda: 1.0}
    - {n_estimators: 300, max_depth: 5, learning_rate: 0.05, subsample: 0.8, colsample_bytree: 0.8, reg_lambda: 1.0}
    - {n_estimators: 600, max_depth: 5, learning_rate: 0.03, subsample: 0.8, colsample_bytree: 0.8, reg_lambda: 5.0}
```

- [ ] **Step 6: Write failing model tests**

Create `tests/test_models.py`:

```python
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from ipop.models import build_dummy_pipeline, build_xgb_search, load_experiment_config


def test_config_has_declared_protocols_and_four_bounded_candidates() -> None:
    config = load_experiment_config(Path("configs/emission_xgb.yaml"))
    assert config["outer_folds"] == 5
    assert config["protocols"] == [
        "random_row", "group_formula", "group_host", "group_reference"
    ]
    assert len(config["xgboost"]["parameter_grid"]) == 4


def test_dummy_pipeline_handles_missing_feature_values() -> None:
    model = build_dummy_pipeline()
    x = np.array([[1.0], [np.nan], [3.0]])
    y = np.array([500.0, 600.0, 700.0])
    model.fit(x, y)
    assert model.predict(np.array([[np.nan]])).tolist() == [600.0]


def test_xgb_search_uses_inner_cv_and_returns_a_fitted_estimator() -> None:
    config = load_experiment_config(Path("configs/emission_xgb.yaml"))
    search = build_xgb_search(config, KFold(n_splits=3, shuffle=True, random_state=42))
    x = np.arange(60, dtype=float).reshape(20, 3)
    y = np.linspace(500.0, 650.0, 20)
    search.fit(x, y)
    assert hasattr(search, "best_estimator_")
```

- [ ] **Step 7: Run model tests and verify the module is missing**

Run: `python -m pytest tests/test_models.py -v`

Expected: FAIL during collection because `ipop.models` does not exist.

- [ ] **Step 8: Implement model factories and configuration validation**

Create `src/ipop/models.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def load_experiment_config(path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"seed", "outer_folds", "inner_folds", "target", "feature_sets",
                "protocols", "models", "xgboost"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing experiment configuration keys: {missing}")
    if len(config["xgboost"]["parameter_grid"]) != 4:
        raise ValueError("The approved benchmark requires four bounded XGBoost candidates")
    return config


def build_dummy_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", DummyRegressor(strategy="median")),
    ])


def build_xgb_search(config: dict[str, object], inner_cv: object) -> GridSearchCV:
    settings = config["xgboost"]
    fixed = {key: settings[key] for key in ("objective", "tree_method", "n_jobs")}
    fixed["random_state"] = int(config["seed"])
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", XGBRegressor(**fixed)),
    ])
    candidates = [
        {f"regressor__{key}": [value] for key, value in candidate.items()}
        for candidate in settings["parameter_grid"]
    ]
    return GridSearchCV(
        pipeline,
        param_grid=candidates,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
```

- [ ] **Step 9: Run model tests and the full suite**

Run: `python -m pytest tests/test_metrics.py tests/test_models.py -v`

Expected: 5 passed.

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 10: Commit models, metrics, and configuration**

```bash
git add configs/emission_xgb.yaml src/ipop/models.py src/ipop/metrics.py tests/test_models.py tests/test_metrics.py
git commit -m "feat: configure reproducible emission models"
```

---

### Task 6: End-to-End Experiment Orchestration and Artifact Persistence

**Files:**
- Create: `src/ipop/experiment.py`
- Create: `tests/test_experiment.py`

**Interfaces:**
- Consumes: prepared emission table, configuration dictionary, feature selectors, outer assignments, model factories, and metric functions.
- Produces: `run_experiment(frame, config, output_dir) -> RunArtifacts` and files `splits.csv`, `overlap_audit.csv`, `predictions.csv`, `fold_metrics.csv`, `summary_metrics.csv`, `best_params.json`, and `run_metadata.json`.

- [ ] **Step 1: Write a failing integration test with a lightweight model set**

Create `tests/test_experiment.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipop.experiment import run_experiment
from ipop.features import AF_COLUMNS


def synthetic_emission() -> pd.DataFrame:
    rows = 50
    frame = pd.DataFrame({column: np.linspace(0, 1, rows) for column in AF_COLUMNS})
    frame["Temp. (K)"] = 298.0
    frame["Excitation source (nm)"] = np.tile([365.0, 405.0, 450.0, 465.0, 480.0], 10)
    frame["Emission max. (nm)"] = 500.0 + 0.2 * frame["Excitation source (nm)"]
    frame["Formula"] = [f"F{i // 2}" for i in range(rows)]
    frame["Host"] = [f"H{i // 5}" for i in range(rows)]
    frame["Reference"] = [f"D{i // 5}" for i in range(rows)]
    frame["row_id"] = range(rows)
    return frame


def test_run_persists_replayable_outer_fold_artifacts(tmp_path: Path) -> None:
    config = {
        "seed": 42, "outer_folds": 5, "inner_folds": 3,
        "target": "Emission max. (nm)",
        "feature_sets": ["AF"],
        "protocols": ["random_row", "group_formula", "group_host", "group_reference"],
        "models": ["median"],
        "xgboost": {"objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1,
                     "parameter_grid": []},
    }

    artifacts = run_experiment(synthetic_emission(), config, tmp_path / "run")

    assert artifacts.fold_metrics.exists()
    folds = pd.read_csv(artifacts.fold_metrics)
    summary = pd.read_csv(artifacts.summary_metrics)
    audit = pd.read_csv(artifacts.overlap_audit)
    assert len(folds) == 20
    assert summary["fold_count"].eq(5).all()
    assert audit.loc[audit["enforced"], "overlap_count"].eq(0).all()
    metadata = json.loads(artifacts.run_metadata.read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert "python" in metadata["versions"]
```

- [ ] **Step 2: Run the integration test and verify the module is missing**

Run: `python -m pytest tests/test_experiment.py -v`

Expected: FAIL during collection because `ipop.experiment` does not exist.

- [ ] **Step 3: Implement experiment orchestration**

Create `src/ipop/experiment.py` with an immutable artifact-path result and an explicit loop over protocol, feature set, model, and fold:

```python
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from ipop.features import select_features
from ipop.metrics import aggregate_metrics, regression_metrics
from ipop.models import build_dummy_pipeline, build_xgb_search
from ipop.splits import GROUP_COLUMNS, audit_split_overlap, build_outer_splits


@dataclass(frozen=True)
class RunArtifacts:
    splits: Path
    overlap_audit: Path
    predictions: Path
    fold_metrics: Path
    summary_metrics: Path
    best_params: Path
    run_metadata: Path


def _inner_cv(protocol: str, folds: int, seed: int) -> object:
    if protocol == "random_row":
        return KFold(n_splits=folds, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=folds, shuffle=True, random_state=seed)


def run_experiment(frame: pd.DataFrame, config: dict[str, object], output_dir: str | Path) -> RunArtifacts:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    split_frames = []
    audit_frames = []
    prediction_rows = []
    metric_rows = []
    parameter_records: dict[str, dict[str, object]] = {}

    for protocol in config["protocols"]:
        assignments = build_outer_splits(
            frame, protocol, int(config["outer_folds"]), int(config["seed"])
        )
        split_frames.append(assignments)
        audit_frames.append(audit_split_overlap(frame, assignments, protocol))
        eligible = frame[frame["row_id"].isin(assignments["row_id"])].copy()

        for feature_set in config["feature_sets"]:
            for model_name in config["models"]:
                for fold in range(int(config["outer_folds"])):
                    test_ids = set(assignments.loc[assignments["fold"] == fold, "row_id"])
                    test_mask = eligible["row_id"].isin(test_ids)
                    train = eligible.loc[~test_mask]
                    test = eligible.loc[test_mask]
                    x_train = select_features(train, feature_set)
                    x_test = select_features(test, feature_set)
                    y_train = train[config["target"]].to_numpy(dtype=float)
                    y_test = test[config["target"]].to_numpy(dtype=float)

                    if model_name == "median":
                        estimator = build_dummy_pipeline()
                        fit_kwargs = {}
                    elif model_name == "xgboost":
                        inner = _inner_cv(protocol, int(config["inner_folds"]), int(config["seed"]))
                        estimator = build_xgb_search(config, inner)
                        group_column = GROUP_COLUMNS.get(protocol)
                        fit_kwargs = ({"groups": train[group_column].to_numpy()}
                                      if group_column is not None else {})
                    else:
                        raise KeyError(f"Unknown model: {model_name}")

                    estimator.fit(x_train, y_train, **fit_kwargs)
                    predicted = estimator.predict(x_test)
                    metrics = regression_metrics(y_test, predicted)
                    group_column = GROUP_COLUMNS.get(protocol)
                    train_groups = (int(train[group_column].nunique())
                                    if group_column is not None else len(train))
                    test_groups = (int(test[group_column].nunique())
                                   if group_column is not None else len(test))
                    metric_rows.append({"protocol": protocol, "feature_set": feature_set,
                                        "model": model_name, "fold": fold,
                                        "train_rows": len(train), "test_rows": len(test),
                                        "train_groups": train_groups, "test_groups": test_groups,
                                        **metrics})
                    for row_id, truth, prediction in zip(test["row_id"], y_test, predicted, strict=True):
                        prediction_rows.append({"protocol": protocol, "feature_set": feature_set,
                                                "model": model_name, "fold": fold,
                                                "row_id": int(row_id), "y_true": float(truth),
                                                "y_pred": float(prediction),
                                                "residual": float(prediction - truth)})
                    if hasattr(estimator, "best_params_"):
                        parameter_records[f"{protocol}|{feature_set}|{model_name}|{fold}"] = (
                            estimator.best_params_
                        )

    paths = RunArtifacts(
        splits=root / "splits.csv",
        overlap_audit=root / "overlap_audit.csv",
        predictions=root / "predictions.csv",
        fold_metrics=root / "fold_metrics.csv",
        summary_metrics=root / "summary_metrics.csv",
        best_params=root / "best_params.json",
        run_metadata=root / "run_metadata.json",
    )
    pd.concat(split_frames, ignore_index=True).to_csv(paths.splits, index=False)
    pd.concat(audit_frames, ignore_index=True).to_csv(paths.overlap_audit, index=False)
    pd.DataFrame(prediction_rows).to_csv(paths.predictions, index=False)
    folds = pd.DataFrame(metric_rows)
    folds.to_csv(paths.fold_metrics, index=False)
    aggregate_metrics(folds).to_csv(paths.summary_metrics, index=False)
    paths.best_params.write_text(json.dumps(parameter_records, indent=2), encoding="utf-8")
    metadata = {
        "seed": int(config["seed"]),
        "config": config,
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in ("numpy", "pandas", "scikit-learn", "xgboost")
        }},
    }
    paths.run_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths
```

- [ ] **Step 4: Run the experiment integration test**

Run: `python -m pytest tests/test_experiment.py -v`

Expected: 1 passed and seven artifact files created inside pytest's temporary directory.

- [ ] **Step 5: Add a deterministic-replay assertion**

Append this test to `tests/test_experiment.py` before changing production code:

```python
def test_fixed_seed_repeats_identical_splits_and_predictions(tmp_path: Path) -> None:
    config = {
        "seed": 42, "outer_folds": 5, "inner_folds": 3,
        "target": "Emission max. (nm)", "feature_sets": ["AF"],
        "protocols": ["random_row"], "models": ["median"],
        "xgboost": {"objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1,
                     "parameter_grid": []},
    }
    first = run_experiment(synthetic_emission(), config, tmp_path / "first")
    second = run_experiment(synthetic_emission(), config, tmp_path / "second")
    assert first.splits.read_bytes() == second.splits.read_bytes()
    assert first.predictions.read_bytes() == second.predictions.read_bytes()
```

- [ ] **Step 6: Run the new test and confirm it passes without nondeterministic changes**

Run: `python -m pytest tests/test_experiment.py -v`

Expected: 2 passed. If it fails, set deterministic XGBoost options and stable row sorting in production code, then rerun until both artifacts match.

- [ ] **Step 7: Run the complete suite and commit orchestration**

Run: `python -m pytest -q`

Expected: all tests pass.

```bash
git add src/ipop/experiment.py tests/test_experiment.py
git commit -m "feat: persist auditable emission experiments"
```

---

### Task 7: Scientific Figures and Chinese Findings Report

**Files:**
- Create: `src/ipop/reporting.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Consumes: `summary_metrics.csv`, `fold_metrics.csv`, `predictions.csv`, `overlap_audit.csv`, and dataset/join audit dictionaries.
- Produces: `build_report(run_dir, dataset_summary, join_summary) -> ReportArtifacts`, four PNG figures, and `findings_zh.md`.

- [ ] **Step 1: Write failing reporting tests**

Create `tests/test_reporting.py`:

```python
from pathlib import Path

import pandas as pd

from ipop.reporting import build_report


def test_report_contains_required_figures_and_reproducibility_disclosure(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({
        "protocol": ["random_row", "group_host"],
        "feature_set": ["AF+T+ES", "AF+T+ES"],
        "model": ["xgboost", "xgboost"], "fold_count": [5, 5],
        "mae_mean": [15.0, 30.0], "mae_std": [1.0, 2.0],
        "rmse_mean": [31.0, 45.0], "rmse_std": [1.0, 3.0],
        "r2_mean": [0.75, 0.40], "r2_std": [0.02, 0.08],
    }).to_csv(run / "summary_metrics.csv", index=False)
    pd.DataFrame({
        "protocol": ["random_row", "group_host"], "feature_set": ["AF+T+ES"] * 2,
        "model": ["xgboost"] * 2, "fold": [0, 0], "row_id": [1, 2],
        "y_true": [600.0, 620.0], "y_pred": [598.0, 600.0], "residual": [-2.0, -20.0],
    }).to_csv(run / "predictions.csv", index=False)
    pd.DataFrame({
        "protocol": ["random_row", "group_host"], "fold": [0, 0],
        "audit_column": ["Host", "Host"], "enforced": [False, True],
        "overlap_count": [1, 0],
    }).to_csv(run / "overlap_audit.csv", index=False)

    artifacts = build_report(
        run,
        dataset_summary={"records": 3952, "unique_hosts": 2238,
                         "unique_references": 553, "target_observations": 16023},
        join_summary={"emission_rows": 1665, "ambiguous_reference_rows": 6},
    )

    assert artifacts.findings.exists()
    text = artifacts.findings.read_text(encoding="utf-8")
    assert "方法级复现" in text
    assert "未公开完整超参数" in text
    assert "group_host" in text
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts.figures)
```

- [ ] **Step 2: Run the report test and verify the module is missing**

Run: `python -m pytest tests/test_reporting.py -v`

Expected: FAIL during collection because `ipop.reporting` does not exist.

- [ ] **Step 3: Implement deterministic tables, plots, and narrative**

Create `src/ipop/reporting.py` with Matplotlib's noninteractive `Agg` backend. The public result type and output names are fixed:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class ReportArtifacts:
    findings: Path
    figures: tuple[Path, ...]


FIGURE_NAMES = (
    "split_performance.png",
    "feature_ablation.png",
    "parity_random_row.png",
    "residuals_by_protocol.png",
)


def build_report(
    run_dir: str | Path,
    dataset_summary: dict[str, int],
    join_summary: dict[str, int],
) -> ReportArtifacts:
    root = Path(run_dir)
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)
    summary = pd.read_csv(root / "summary_metrics.csv")
    predictions = pd.read_csv(root / "predictions.csv")
    audit = pd.read_csv(root / "overlap_audit.csv")

    sns.set_theme(style="whitegrid", context="talk")
    selected = summary.query("model == 'xgboost' and feature_set == 'AF+T+ES'")
    ax = sns.barplot(selected, x="protocol", y="r2_mean", color="#3B82F6")
    ax.errorbar(range(len(selected)), selected["r2_mean"], yerr=selected["r2_std"], fmt="none",
                color="black", capsize=4)
    ax.set(xlabel="Evaluation protocol", ylabel="Mean outer-fold R²")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / FIGURE_NAMES[0], dpi=180)
    plt.close()

    ablation = summary.query("model == 'xgboost' and protocol == 'random_row'")
    sns.barplot(ablation, x="feature_set", y="mae_mean", color="#10B981")
    plt.xlabel("Feature set")
    plt.ylabel("Mean outer-fold MAE (nm)")
    plt.tight_layout()
    plt.savefig(figures_dir / FIGURE_NAMES[1], dpi=180)
    plt.close()

    parity = predictions.query("model == 'xgboost' and feature_set == 'AF+T+ES' and protocol == 'random_row'")
    sns.scatterplot(parity, x="y_true", y="y_pred", s=25, alpha=0.65)
    low = min(parity["y_true"].min(), parity["y_pred"].min())
    high = max(parity["y_true"].max(), parity["y_pred"].max())
    plt.plot([low, high], [low, high], "--", color="black")
    plt.xlabel("Measured emission maximum (nm)")
    plt.ylabel("Predicted emission maximum (nm)")
    plt.tight_layout()
    plt.savefig(figures_dir / FIGURE_NAMES[2], dpi=180)
    plt.close()

    residuals = predictions.query("model == 'xgboost' and feature_set == 'AF+T+ES'")
    sns.boxplot(residuals, x="protocol", y="residual", color="#F59E0B")
    plt.axhline(0, linestyle="--", color="black")
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("Evaluation protocol")
    plt.ylabel("Prediction minus measurement (nm)")
    plt.tight_layout()
    plt.savefig(figures_dir / FIGURE_NAMES[3], dpi=180)
    plt.close()

    random_result = selected.loc[selected["protocol"] == "random_row"].iloc[0]
    host_result = selected.loc[selected["protocol"] == "group_host"].iloc[0]
    differs = (
        abs(float(random_result["r2_mean"]) - 0.760) > 0.10
        or abs(float(random_result["mae_mean"]) - 14.611) > 7.0
    )
    difference_text = ""
    if differs:
        difference_text = (
            "\n## 复现差异\n\n"
            "随机行划分结果超出预先声明的方向性复现窗口。可能原因包括原论文未公开的"
            "精确划分、随机种子、超参数搜索和软件版本；本项目不通过更换种子筛选更高分。\n"
        )
    disclosure = (
        "本文结果属于方法级复现：原论文未公开完整超参数、随机种子和精确数据划分，"
        "因此不主张逐小数位重现。"
    )
    findings = root / "findings_zh.md"
    findings.write_text(
        "# IPOP Eu 发射波长复现与泛化审计\n\n"
        f"{disclosure}\n\n"
        f"- 有效主数据：{dataset_summary['records']} 条；host：{dataset_summary['unique_hosts']}；"
        f"文献：{dataset_summary['unique_references']}；性质观测：{dataset_summary['target_observations']}。\n"
        f"- Eu 发射任务：{join_summary['emission_rows']} 条；DOI 归属有歧义："
        f"{join_summary['ambiguous_reference_rows']} 条。\n"
        f"- random_row：R²={random_result['r2_mean']:.3f}，MAE={random_result['mae_mean']:.2f} nm。\n"
        f"- group_host：R²={host_result['r2_mean']:.3f}，MAE={host_result['mae_mean']:.2f} nm。\n\n"
        "`group_formula`、`group_host` 和 `group_reference` 均要求分组键零重叠；"
        "这些结果衡量的是比随机行划分更接近新材料发现的外推能力。\n\n"
        f"强制分组键最大重叠数："
        f"{int(audit.loc[audit['enforced'], 'overlap_count'].max())}。\n"
        f"{difference_text}",
        encoding="utf-8",
    )
    return ReportArtifacts(findings=findings,
                           figures=tuple(figures_dir / name for name in FIGURE_NAMES))
```

- [ ] **Step 4: Run the report test and inspect generated images**

Run: `python -m pytest tests/test_reporting.py -v`

Expected: 1 passed.

Open the four temporary PNGs produced by the test or reproduce the fixture under `outputs/report-smoke`. Verify labels are legible, axes are not clipped, and each chart states its metric and protocol.

- [ ] **Step 5: Run the complete suite and commit reporting**

Run: `python -m pytest -q`

Expected: all tests pass.

```bash
git add src/ipop/reporting.py tests/test_reporting.py
git commit -m "feat: report IPOP generalization audit"
```

---

### Task 8: CLI, Chinese README, and Full-Data Reproduction Run

**Files:**
- Create: `src/ipop/cli.py`
- Create: `src/ipop/__main__.py`
- Create: `tests/test_cli.py`
- Create: `README.md`
- Create at runtime: `data/raw/*`, `data/interim/emission_prepared.csv`, `data/interim/metadata_join_audit.csv`, `outputs/emission-xgb/*`

**Interfaces:**
- Consumes: every public interface from Tasks 1-7.
- Produces: `python -m ipop {download,validate,prepare-emission,run,report,all}`, the public README workflow, and the verified full-data result directory.

- [ ] **Step 1: Write failing CLI parser and validation-output tests**

Create `tests/test_cli.py`:

```python
import json
from pathlib import Path

import pandas as pd

from ipop.cli import build_parser, write_validation_report
from ipop.data import DatasetSummary


def test_parser_exposes_six_commands() -> None:
    parser = build_parser()
    for command in ("download", "validate", "prepare-emission", "run", "report", "all"):
        args = parser.parse_args([command])
        assert args.command == command


def test_validation_report_is_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    write_validation_report(DatasetSummary(3952, 2238, 553, 16023), path)
    assert json.loads(path.read_text(encoding="utf-8"))["target_observations"] == 16023
```

- [ ] **Step 2: Run CLI tests and verify the module is missing**

Run: `python -m pytest tests/test_cli.py -v`

Expected: FAIL during collection because `ipop.cli` does not exist.

- [ ] **Step 3: Implement CLI commands with explicit default paths**

Create `src/ipop/cli.py` using `argparse`. The parser must define these defaults:

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/ipop_v3.json"
RAW = ROOT / "data/raw"
INTERIM = ROOT / "data/interim"
OUTPUT = ROOT / "outputs/emission-xgb"
MASTER_NAME = "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
EMISSION_NAME = "phosphor_20230908_Eu_only_EmP_AF.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ipop")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "validate", "prepare-emission", "run", "report", "all"):
        subparsers.add_parser(command)
    return parser


def write_validation_report(summary: DatasetSummary, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")


def command_download() -> None:
    manifest = load_manifest(MANIFEST)
    paths = download_artifacts(manifest, RAW)
    provenance = build_provenance_record(
        manifest, paths, datetime.now(timezone.utc).isoformat()
    )
    INTERIM.mkdir(parents=True, exist_ok=True)
    (INTERIM / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def command_validate() -> DatasetSummary:
    summary = validate_master(load_master_csv(RAW / MASTER_NAME))
    write_validation_report(summary, INTERIM / "validation.json")
    return summary


def command_prepare() -> dict[str, int]:
    emission = load_emission_features(RAW / EMISSION_NAME)
    master = load_master_csv(RAW / MASTER_NAME)
    result = attach_emission_groups(emission, master)
    INTERIM.mkdir(parents=True, exist_ok=True)
    result.prepared.to_csv(INTERIM / "emission_prepared.csv", index=False)
    result.audit.to_csv(INTERIM / "metadata_join_audit.csv", index=False)
    return {
        "emission_rows": len(result.prepared),
        "ambiguous_reference_rows": int(result.audit["status"].eq("ambiguous_reference").sum()),
    }


def command_run() -> None:
    frame = pd.read_csv(INTERIM / "emission_prepared.csv")
    config = load_experiment_config(ROOT / "configs/emission_xgb.yaml")
    config["data_provenance"] = json.loads(
        (INTERIM / "source_provenance.json").read_text(encoding="utf-8")
    )
    run_experiment(frame, config, OUTPUT)


def command_report() -> None:
    dataset_summary = json.loads((INTERIM / "validation.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(INTERIM / "metadata_join_audit.csv")
    build_report(OUTPUT, dataset_summary, {
        "emission_rows": len(pd.read_csv(INTERIM / "emission_prepared.csv")),
        "ambiguous_reference_rows": int(audit["status"].eq("ambiguous_reference").sum()),
    })


def main(argv: list[str] | None = None) -> int:
    command = build_parser().parse_args(argv).command
    actions = {
        "download": command_download,
        "validate": command_validate,
        "prepare-emission": command_prepare,
        "run": command_run,
        "report": command_report,
    }
    if command == "all":
        command_download()
        command_validate()
        command_prepare()
        command_run()
        command_report()
    else:
        actions[command]()
    return 0
```

Create `src/ipop/__main__.py`:

```python
from ipop.cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests and command help**

Run: `python -m pytest tests/test_cli.py -v`

Expected: 2 passed.

Run: `python -m ipop --help`

Expected: help lists `download`, `validate`, `prepare-emission`, `run`, `report`, and `all`.

- [ ] **Step 5: Write the Chinese README**

Create `README.md` with these exact sections:

```markdown
# IPOP Eu 发射波长复现与泛化审计

本项目复现 Jang 等人在 Scientific Reports 发表的 IPOP 数据集 Eu 激活荧光粉发射波长基线，并比较随机行划分与配方、基质、文献分组划分。重点不是追逐单一最高 R²，而是判断模型面对未见材料体系时的真实泛化能力。

## 数据与许可
## 为什么不能声称逐数值复现
## 环境安装
## 一键运行
## 分步运行
## 输出文件
## 评估协议
## 结果解读
## 已知限制
## 后续研究路线
## 引用
```

Under “环境安装”, document `python -m venv .venv`, platform-appropriate activation, and `python -m pip install -e ".[dev]"`. Under “一键运行”, document `python -m ipop all`. Under “引用”, include both DOI `10.1038/s41598-024-58351-w` and Figshare DOI `10.6084/m9.figshare.24771186.v1`, and state that the dataset is CC BY 4.0.

- [ ] **Step 6: Run unit, integration, lint, and packaging checks**

Run: `python -m pytest -q`

Expected: all tests pass.

Run: `python -m ruff check .`

Expected: exit 0 with no diagnostics.

Run: `python -m pip install -e ".[dev]"`

Expected: editable installation completes successfully.

- [ ] **Step 7: Execute the pinned full-data workflow**

Run: `python -m ipop download`

Expected: two files appear in `data/raw`; both pass the pinned MD5, SHA-256, and size checks. `data/interim/source_provenance.json` records the DOI, version, license, retrieval time, source URLs, and both hashes.

Run: `python -m ipop validate`

Expected `data/interim/validation.json`:

```json
{
  "records": 3952,
  "unique_hosts": 2238,
  "unique_references": 553,
  "target_observations": 16023
}
```

Run: `python -m ipop prepare-emission`

Expected: 1,665 prepared rows; 52 atomic features; 0 ambiguous hosts; 6 rows with ambiguous source DOI recorded in `metadata_join_audit.csv`.

Run: `python -m ipop run`

Expected:

- `splits.csv` contains five outer test folds for every protocol.
- `overlap_audit.csv` reports zero overlap for formula, host, and reference protocols.
- `fold_metrics.csv` contains `4 protocols x 4 feature sets x 2 models x 5 folds = 160` rows.
- `summary_metrics.csv` contains 32 rows.
- `predictions.csv`, `best_params.json`, and `run_metadata.json` are nonempty.

Run: `python -m ipop report`

Expected: `findings_zh.md` and four legible PNG figures under `outputs/emission-xgb`.

- [ ] **Step 8: Inspect scientific results without metric cherry-picking**

Open `outputs/emission-xgb/summary_metrics.csv` and check the `random_row`, `AF+T+ES`, `xgboost` row against the paper's reference `R² 0.760 +/- 0.022`, `MAE 14.611 +/- 1.438 nm`, and `RMSE 30.672 +/- 1.469 nm`.

If mean random-row R² differs by more than `0.10` or mean MAE differs by more than `7 nm`, add a “复现差异” subsection to `outputs/emission-xgb/findings_zh.md` naming the observed differences and the likely causes: unpublished original split, seed, hyperparameters, and platform behavior. Do not rerun different seeds to choose a more favorable score.

Open each generated PNG and verify that labels are legible, error bars are visible, legends do not cover data, and the grouped-protocol comparison is not truncated.

- [ ] **Step 9: Run a fresh final verification**

Run: `python -m pytest -q`

Expected: all tests pass with zero failures.

Run: `python -m ruff check .`

Expected: exit 0 with no diagnostics.

Run: `git diff --check`

Expected: exit 0 with no whitespace errors.

Run: `git status --short`

Expected: only intended source, configuration, tests, documentation, and tracked manifest changes are listed; raw/interim/output artifacts remain ignored.

- [ ] **Step 10: Commit the runnable first-cycle demo**

```bash
git add README.md src/ipop/cli.py src/ipop/__main__.py tests/test_cli.py
git commit -m "feat: deliver runnable IPOP emission audit"
```

Record the final test count and the four principal `AF+T+ES` XGBoost results in the implementation handoff. Do not commit downloaded data or generated model artifacts.
