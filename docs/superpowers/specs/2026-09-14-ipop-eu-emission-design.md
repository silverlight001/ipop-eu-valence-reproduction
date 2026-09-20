# IPOP Eu Emission Prediction: Reproduction and Extended Study

Date: 2026-09-14

Status: Approved in chat for specification

## 1. Purpose

Build a local, reproducible AI4Science demo around the 2024 Scientific Reports article, *Optical property dataset of inorganic phosphor* (DOI: 10.1038/s41598-024-58351-w). The first implementation cycle will reproduce the Eu-activated phosphor emission-wavelength baseline at the method level and then test how performance changes under scientifically stricter data splits.

The demo must distinguish two questions:

1. Can an XGBoost model reproduce the paper's reported performance range under a conventional random row split?
2. How well does the model generalize to unseen formulas, unseen hosts, and unseen source publications?

This distinction is the project's central contribution. It turns a notebook-level reproduction into a reproducible study of split-dependent generalization and data-leakage risks in literature-derived materials datasets.

## 2. Scope

### Included in this implementation cycle

- Download and pin IPOP version 3.0 from Figshare.
- Validate provenance, checksums, license metadata, and the published dataset counts.
- Clean the raw master CSV without silently changing scientific values.
- Load the authors' Eu-only emission-peak feature table.
- Reproduce an XGBoost emission-wavelength baseline using the published 52 atomic features and measurement conditions.
- Reproduce the paper's feature ablation: atomic features only, atomic features plus temperature, atomic features plus excitation source, and all three feature groups.
- Compare four evaluation protocols: random rows, grouped by phosphor formula, grouped by host, and grouped by source DOI.
- Save split assignments, predictions, metrics, figures, environment metadata, and model configuration.
- Provide a documented command-line workflow and automated tests.

### Explicitly deferred

- The other eight Eu targets: excitation peaks, decay time, CIE coordinates, IQE, EQE, and T50.
- Models covering Ce, Mn, Tb, Dy, or all activators.
- Crystal-structure models using MP-ID or ICSD structures.
- Multi-task learning, uncertainty-aware screening, inverse design, and experimental active learning.
- A web interface.

Each deferred item is a separate research increment after the first demo is validated. This prevents sparse-property tasks and structure acquisition from obscuring the core reproduction.

## 3. Source Data and Reproducibility Boundary

The canonical source is Figshare record 24771186, version 1, published under CC BY 4.0. The GitHub repository is a convenient mirror, not the provenance authority.

The raw master CSV contains 3,952 populated records followed by 437 completely empty rows when read using standard CSV tooling. The loader will remove rows only when every column is missing. It will not impute or drop partially populated scientific records.

Dataset validation must reproduce these published invariants:

- 3,952 populated phosphor records.
- 2,238 unique host strings.
- 553 unique source DOI strings.
- 16,023 non-null target observations, defined as the sum of emission maximum, three excitation maxima, decay time, CIE x, CIE y, internal quantum efficiency, external quantum efficiency, and T50 observations.
- 1,665 rows in the authors' Eu-only emission-peak feature table.
- 52 composition-derived atomic features before measurement conditions.

The paper and its supplementary information do not disclose a complete train/test split, random seed, XGBoost hyperparameters, or hyperparameter-search procedure. Therefore, this project promises a transparent method-level reproduction, not bit-for-bit recovery of the authors' unpublished run. All deviations and assumptions will be recorded in the generated report.

## 4. Scientific Evaluation Design

### 4.1 Target and features

The first target is maximum photoluminescence emission wavelength in nanometres.

Feature groups are:

- `AF`: the 52 atomic features supplied by the authors.
- `T`: measurement temperature in kelvin.
- `ES`: excitation-source wavelength in nanometres.

The human-readable `Formula` column and all target columns are identifiers or outcomes and must never enter the feature matrix.

### 4.2 Models

The primary model is `XGBRegressor`, matching the paper. A median `DummyRegressor` is required as a sanity baseline. Hyperparameters will be declared in version-controlled configuration and tuned only inside training folds. Test folds must never influence feature selection, imputation, or tuning.

The first implementation will favor a small, documented search space over a large optimization campaign. The purpose is reproducibility and comparison across split protocols, not leaderboard maximization.

### 4.3 Split protocols

All protocols use five outer folds where the group count permits it. Split assignments are saved with the outputs.

1. `random_row`: shuffled five-fold splitting over observations. This is the closest transparent approximation to the paper's apparent row-level evaluation.
2. `group_formula`: no exact phosphor formula can appear in both training and test folds.
3. `group_host`: no host string can appear in both training and test folds.
4. `group_reference`: no source DOI can appear in both training and test folds.

The group metadata for the Eu feature table will be recovered from the master table by an exact join on formula, temperature, excitation wavelength, and emission target. If more than one master record matches, the loader may retain the row only when all matching records agree on the required group label. Ambiguous rows must be reported rather than assigned arbitrarily.

### 4.4 Metrics and uncertainty

For every feature combination, model, and split protocol, report:

- MAE in nanometres.
- RMSE in nanometres.
- coefficient of determination, R-squared.
- fold mean, standard deviation, and individual fold values.
- number of train/test rows and groups.
- train/test overlap audit for formula, host, and DOI.

The paper's `AF+T+ES` result (`MAE 14.611 +/- 1.438 nm`, `RMSE 30.672 +/- 1.469 nm`, `R-squared 0.760 +/- 0.022`) is a reference point, not a value to fit by trial and error. A random-row result within 0.10 R-squared and 7 nm MAE of the reference is considered directionally reproduced. Results outside that window are not hidden; they trigger a documented discrepancy analysis.

### 4.5 Interpretation

The main report will compare the apparent random-row performance with increasingly strict generalization settings. It will explicitly discuss:

- repeated formulas and host families;
- correlated synthesis series within a paper;
- literature-source bias;
- mixed Eu(II)/Eu(III) labels;
- the limits of composition-only features for local coordination and crystal-field effects;
- the difference between interpolation and novel-host discovery.

No candidate will be described as experimentally promising based solely on this first-cycle model.

## 5. Architecture

The project will be a small Python package with configuration-driven experiments.

```text
F:\AI4S
|-- README.md
|-- pyproject.toml
|-- configs/
|   `-- emission_xgb.yaml
|-- data/
|   |-- raw/              # downloaded, ignored by git
|   |-- interim/          # validated joins and group labels, ignored by git
|   `-- manifests/        # provenance and checksums, tracked
|-- src/ipop/
|   |-- data.py           # download, checksums, loading, schema validation
|   |-- features.py       # target-safe feature-group selection
|   |-- metadata.py       # exact master-table joins and ambiguity reporting
|   |-- splits.py         # row and group-aware folds plus overlap audits
|   |-- models.py         # training pipeline and bounded parameter search
|   |-- metrics.py        # fold metrics and aggregation
|   |-- experiment.py     # orchestration and artifact persistence
|   |-- reporting.py      # tables and static scientific figures
|   `-- cli.py            # command-line entry points
|-- tests/
|   |-- fixtures/
|   |-- test_data.py
|   |-- test_features.py
|   |-- test_metadata.py
|   |-- test_splits.py
|   |-- test_experiment.py
|   `-- test_reporting.py
`-- outputs/              # generated results, ignored except small examples
```

Module boundaries are intentionally narrow. Data acquisition does not train models; split logic does not know about XGBoost; report generation consumes saved tables rather than hidden in-memory state.

## 6. Data Flow

1. `download` reads the pinned Figshare metadata, downloads named files, and records URLs, sizes, SHA-256 checksums, retrieval time, DOI, version, and license.
2. `validate` drops only fully empty rows, normalizes column headers, checks the published invariants, and emits a machine-readable validation report.
3. `prepare-emission` validates the Eu feature schema and joins group labels from the master table. It emits a row-level join audit and fails if required metadata cannot be recovered within the declared ambiguity policy.
4. `run` materializes outer splits first. For each fold, all imputation and hyperparameter selection operate on training data only. It saves configuration, fold assignments, predictions, and metrics.
5. `report` reads saved artifacts and produces comparison tables, parity plots, residual plots, feature-ablation plots, and a concise Markdown findings report.

Repeated runs with the same data checksum, configuration, package versions, and seed must reproduce the same split assignments and numerical results within deterministic-library tolerances.

## 7. Failure Handling

- Network or download failures leave no file marked as complete.
- A checksum mismatch stops the run and identifies the affected file.
- Missing or renamed required columns produce a schema error listing the differences.
- Published count mismatches fail validation instead of being silently accepted.
- Non-numeric target or feature values identify their source rows.
- Insufficient groups for five folds produce a clear configuration error.
- Ambiguous metadata joins are written to an audit table and excluded only under the documented policy.
- Any nonzero formula, host, or DOI overlap in its corresponding grouped protocol fails the split audit.
- Model or reporting failures preserve completed upstream artifacts for diagnosis.

## 8. Testing Strategy

Implementation will follow test-driven development.

### Unit tests

- Remove fully empty CSV rows while retaining partially populated rows.
- Recompute the 16,023-observation definition correctly.
- Reject schema drift and checksum mismatches.
- Select exactly 52 AF columns and prevent identifiers or targets from entering `X`.
- Resolve unambiguous group metadata and surface ambiguous joins.
- Prove group disjointness for formula, host, and DOI splits.
- Detect deliberately introduced overlap.
- Compute MAE, RMSE, and R-squared from known arrays.

### Integration tests

- Run the complete experiment on a tiny synthetic fixture.
- Confirm that preprocessing is fit inside each training fold.
- Confirm deterministic split and prediction artifacts for a fixed seed.
- Build all expected report files from fixture results.

### Full-data verification

- Run the public-data invariant checks.
- Run the median and XGBoost emission benchmarks.
- Verify that all four protocols emit five folds and pass their overlap audits.
- Regenerate the report from saved artifacts in a clean process.

## 9. Deliverables and Acceptance Criteria

The first cycle is complete when:

1. A clean checkout can install dependencies and retrieve the pinned public data using documented commands.
2. Validation reproduces all declared IPOP counts and records source checksums.
3. The Eu emission benchmark runs end to end for four feature combinations and four split protocols.
4. Every grouped split has zero overlap for its grouping key.
5. Metrics, fold assignments, predictions, configuration, and environment metadata are saved and sufficient to audit a run.
6. The report compares results with the paper without cherry-picking and explains material differences.
7. Unit and integration tests pass, and a fresh full-data smoke run succeeds.
8. The README provides a short Chinese project narrative suitable for presenting the demo, while commands and artifact schemas remain technically precise.

## 10. Planned Follow-on Research

After the first cycle is accepted, follow-on specifications will be written in this order:

1. Extend the same leakage-safe benchmark to all Eu optical-property tasks, with honest handling of sparse EQE and T50 data.
2. Add activator identity and valence-aware models for Ce, Mn, Tb, Dy, and other dopants.
3. Compare hand-engineered atomic features with modern composition encoders and host-structure representations.
4. Add calibrated uncertainty and applicability-domain checks.
5. Define multi-objective candidate screening and an experiment-facing active-learning loop.

This ordering ensures that later discovery claims rest on a verified evaluation foundation.
