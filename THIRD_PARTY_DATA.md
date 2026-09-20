# Third-party data attribution

This repository redistributes two files from the **Optical property dataset of inorganic phosphor (IPOP dataset ver. 3.0)** for transparent, reproducible analysis.

## Original dataset

- Creator: Seunghun Jang
- Title: *Optical property dataset of inorganic phosphor (IPOP dataset ver. 3.0, 20231208)*
- Repository: figshare
- Version: 1
- DOI: <https://doi.org/10.6084/m9.figshare.24771186.v1>
- Official mirror: <https://github.com/KRICT-DATA/IPOP-dataset-ver-3.0>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

Related data paper:

> Jang, S., Na, G. S., Choi, Y. & Chang, H. Optical property dataset of inorganic phosphor. *Scientific Reports* **14**, 7639 (2024). <https://doi.org/10.1038/s41598-024-58351-w>

## Redistributed files

| Path | SHA-256 |
| --- | --- |
| `data/raw/Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv` | `9ebbee222e7b21faac0919761fbc8ee76c304c8ae3c8c89f0ab384a3d53b2924` |
| `data/raw/phosphor_20230908_Eu_only_EmP_AF.csv` | `c573eee9a3893a8501be83f24807eac10f633d32d439c048c0cdc17fd8233afd` |

The raw files are redistributed without modification. Their origin, retrieved URL, byte size, MD5, SHA-256, and retrieval status are recorded in `data/interim/source_provenance.json` and `data/manifests/ipop_v3.json`.

## Derived files

Files under `data/interim/` and `outputs/` are computational derivatives created by this repository from the attributed IPOP data. They do not replace the original dataset citation. Any use of the derived files should also cite the original data DOI and paper.

## Attribution requirements

CC BY 4.0 permits sharing and adaptation provided appropriate credit is given, the license is linked, and changes are indicated. This repository adds matching metadata, Eu valence labels copied from the original master table, cross-validation assignments, predictions, metrics, figures, and discussion. It does not claim ownership of the original IPOP records.
