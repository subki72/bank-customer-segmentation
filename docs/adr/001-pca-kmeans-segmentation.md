# ADR 001: Single-Pass PCA + K-Means Pipeline and Demographic Preservation

- **Status**: Accepted
- **Date**: 2026-11-27

## Context

The legacy exploration notebook (`Clustering_Model.ipynb`) exhibited three production defects:
1. Blind `data.dropna()` dropped all 4,045 `Janda/Duda` customer rows because `Usia` was 100% missing for that segment (`14.38%` of total customers).
2. A one-sided Z-score filter `(z_scores < 3).all(axis=1)` failed to remove negative `-3 sigma` anomalies and relied on `%pip install scipy` inline magic.
3. `StandardScaler` was applied twice: once on numeric variables (`X_num`) and again across the concatenated `data_combined` frame including one-hot indicators.

## Decision

1. **Group-Aware Age Imputation**: Impute missing `Usia` via `Status_Perkawinan` group median with global median fallback (`43.0` on full CSV) and named constant fallback (`40.0`) so `Janda/Duda` customers remain represented.
2. **Two-Sided Z-Score Filter**: Enforce `np.abs(z_scores) < threshold` with zero-variance protection.
3. **Single-Pass ColumnTransformer + Unified Pipeline**: Package `ColumnTransformer` (`StandardScaler` on numeric/ordinal features once + `OneHotEncoder(handle_unknown="ignore")` on nominal categoricals), `PCA(n_components=3, random_state=42)`, and `KMeans(n_clusters=3, random_state=42, n_init=10)` inside `sklearn.pipeline.Pipeline`.
4. **Dual-Mode Architecture**: Keep `Clustering_Model.ipynb` patched in-place for interactive EDA while exposing modular Python APIs (`src/customer_segmentation/`), CLI (`main.py`), and a FastAPI service (`api.py`).

## Consequences

- **Positive**: Eliminates train/inference skew, preserves 100% of `Janda/Duda` customers, prevents double-scaling distortions, and enables deterministic unit/API testing.
- **Trade-off**: Preserving `Janda/Duda` with median-imputed ages alongside unscaled one-hot indicators yields a realistic Silhouette score (`~0.295` on the 27,698 cleaned CSV rows) rather than the artificially inflated `~0.39` produced when excluding 4,045 widowed customers and double-scaling sparse indicators.
