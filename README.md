# Bank Customer Segmentation (PCA + K-Means)

[![CI Pipeline](https://github.com/subki72/bank-customer-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/subki72/bank-customer-segmentation/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end banking customer segmentation platform powered by **Principal Component Analysis (PCA)**, **K-Means Clustering**, business persona profiling, CLI batch execution, and a production-hardened **FastAPI** service.

---

## Architecture Overview

The repository uses a **Dual-Mode Non-Breaking Architecture**:
1. **Interactive Research (`Clustering_Model.ipynb`)**: Preserved and patched in-place so exploratory analysis runs cleanly without data-loss or scaling bugs.
2. **Production Service & Package (`src/customer_segmentation/` + `main.py`)**: Modular, type-annotated scikit-learn pipeline with FastAPI endpoints (`/health`, `/analyze`, `/segment`), Pydantic v2 validation, structured logging, and multi-stage non-root Docker packaging.

```text
clustering_data.csv / JSON Payload
        │
        ▼
[preprocessing.py] ──▶ Schema Validation ─▶ Group/Global Median Age Imputation
        │                                ─▶ Two-Sided Z-Score Filter (|z| < 3)
        │                                ─▶ Ordinal Fallbacks + ColumnTransformer
        ▼
[clustering.py]    ──▶ sklearn Pipeline: Preprocessor -> PCA(3) -> KMeans(k)
        │
        ▼
[profiling.py]     ──▶ Segment Persona Ranking + Cross-Sell Product Recommendation
        │
        ▼
[api.py / main.py] ──▶ FastAPI (/health, /analyze, /segment) & CLI (--health, --demo, --serve)
```

---

## Stage 1 Bugs Identified & Remediated

| Defect | Legacy Behavior | Remediation |
|---|---|---|
| **Demographic Erasure** | `data.dropna()` dropped all 4,045 `Janda/Duda` rows (`Usia` is 100% `NaN` in that segment). | Group-aware (`Status_Perkawinan`) + global median `Usia` imputation preserves 100% of `Janda/Duda` records. |
| **One-Sided Outlier Filter** | `(z_scores < 3).all(axis=1)` missed negative `-3 sigma` outliers and used `%pip install`. | Replaced with two-sided `(np.abs(z_scores) < 3).all(axis=1)` with zero-variance protection. |
| **Double Standardization** | `StandardScaler` was applied to `X_num` and re-applied to `data_combined` + one-hot columns. | Single-pass `ColumnTransformer` standardizes numeric/ordinal features once and preserves binary one-hot indicators. |

---

## Project Structure

```text
.
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                  # Ruff, Flake8, Pytest (>=75% cov), Semgrep, Docker, Trivy
│   │   └── release.yml             # Tag-triggered container build & GitHub Release
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
├── docs/
│   ├── adr/001-pca-kmeans-segmentation.md
│   └── runbook.md
├── src/customer_segmentation/
│   ├── __init__.py
│   ├── config.py                   # Schema constants & env-driven SegmentationConfig
│   ├── preprocessing.py            # Imputation, two-sided Z-score, single-pass ColumnTransformer
│   ├── clustering.py               # PCA + K-Means Pipeline, metrics, synthetic generator
│   ├── profiling.py                # Cluster persona ranking & cross-sell product selection
│   └── api.py                      # FastAPI service (/health, /analyze, /segment)
├── tests/                          # Unit, integration, API, and CLI tests (97%+ coverage)
├── main.py                         # CLI entrypoint (--health, --demo, --serve)
├── Clustering_Model.ipynb          # Patched EDA & modeling notebook
├── clustering_data.csv             # 28,129 banking customer records
├── Dockerfile                      # Multi-stage python:3.11-slim non-root container
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Local Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
make install
```

### 2. CLI Execution (100% Offline)

```bash
# Service health probe
python main.py --health

# Run segmentation benchmark on local dataset (5,000-row sample)
python main.py --demo --sample-limit 5000

# Run on full 28,129-row dataset
python main.py --demo --sample-limit 0
```

### 3. FastAPI Server & Docker

```bash
# Local Uvicorn dev server
make serve

# Containerized execution
docker compose up --build
```

---

## API Contract

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | `GET` | Liveness and model readiness status. |
| `/analyze` | `POST` | Segment submitted customer batch or local `clustering_data.csv` and return metrics + personas. |
| `/segment` | `POST` | Assign cluster, persona tier, and cross-sell product recommendation for a single customer. |

Example `POST /segment` payload:

```json
{
  "GCIF": "G55501",
  "Area": "Jakarta",
  "Jalur_Pembukaan": "Aplikasi Digital",
  "Vintage": "> 4 Tahun",
  "Usia": null,
  "Jenis_Kelamin": "Perempuan",
  "Status_Perkawinan": "Janda/Duda",
  "Jumlah_Anak": 2,
  "Pendidikan": "Sarjana",
  "Produk_Tabungan": 1,
  "Produk_Deposito": 1,
  "Produk_Kartu_Kredit": 0,
  "Produk_Kredit_Rumah": 0,
  "Produk_Kredit_Kendaraan": 0,
  "Produk_Kredit_Dana_Tunai": 0,
  "Pendapatan_Tahunan": 520,
  "Total_Relationship_Balance": 410000000
}
```

---

## Quality & Testing

```bash
make lint   # Ruff check, Ruff format check, Flake8
make test   # Pytest suite with >=75% coverage gate
```

See [docs/runbook.md](docs/runbook.md) for operational troubleshooting and [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
