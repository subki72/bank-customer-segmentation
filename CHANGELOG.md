# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-27

### Added
- Modular Python package under `src/customer_segmentation/` (`config.py`, `preprocessing.py`, `clustering.py`, `profiling.py`, `api.py`).
- FastAPI inference and analytics service exposing `/health`, `/analyze`, and `/segment` with Pydantic v2 schema validation, CORS middleware, and centralized error handling.
- CLI runner `main.py` supporting `--health`, `--demo`, `--n-clusters`, `--sample-limit`, and `--serve`.
- Automated unit and integration test suite under `tests/` (`test_preprocessing.py`, `test_clustering.py`, `test_profiling.py`, `test_api.py`, `test_cli.py`) reaching 97%+ coverage.
- Multi-stage non-root `Dockerfile`, `docker-compose.yml`, `Makefile`, `.env.example`, `.gitignore`, `.pre-commit-config.yaml`, Dependabot, and GitHub Actions CI/Release workflows.
- Governance documentation (`CONTRIBUTING.md`, `SECURITY.md`, `docs/runbook.md`, `docs/adr/001-pca-kmeans-segmentation.md`).

### Fixed
- **Demographic Erasure (`Clustering_Model.ipynb` & `preprocessing.py`)**: Replaced blind `data.dropna()` with group-aware (`Status_Perkawinan`) + global median `Usia` imputation so the 4,045 `Janda/Duda` customer records with 100% missing ages are preserved instead of silently dropped.
- **One-Sided Z-Score Filter (`Clustering_Model.ipynb` & `preprocessing.py`)**: Changed `(z_scores < 3).all(axis=1)` to `(np.abs(z_scores) < 3).all(axis=1)` and removed `%pip install scipy` notebook magic.
- **Double Standardization (`Clustering_Model.ipynb` & `preprocessing.py`)**: Eliminated duplicate `StandardScaler` calls on already-scaled numeric features and switched to single-pass `ColumnTransformer` (`StandardScaler` on numeric/ordinal features + `OneHotEncoder(handle_unknown="ignore")` on nominal categoricals).
