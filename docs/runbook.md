# Operational Runbook — Bank Customer Segmentation

## 1. Service Overview

- **Package**: `src/customer_segmentation/`
- **CLI Entrypoint**: `main.py` (`--health`, `--demo`, `--serve`)
- **API Framework**: FastAPI + Uvicorn on port `8000`
- **Primary Dataset**: `clustering_data.csv` (28,129 Indonesian retail banking records)

## 2. Health Verification & Smoke Checks

### CLI Health Probe
```bash
python main.py --health
```
Expected output: JSON with `"status": "ok"` and `"dataset_available": true`.

### HTTP Health Probe
```bash
curl -fsS http://localhost:8000/health
```
Expected output: `{"status":"ok","service":"bank-customer-segmentation","version":"1.0.0","model_ready":...}`.

### Offline Benchmark Run
```bash
python main.py --demo --sample-limit 5000
```

## 3. Configuration & Environment Variables

Copy `.env.example` to `.env` before overriding defaults in non-dev environments:
- `DATA_PATH`: Path to customer dataset CSV (default `clustering_data.csv`).
- `DEFAULT_N_CLUSTERS`: Default K-Means cluster count `k >= 2` (default `3`).
- `PCA_N_COMPONENTS`: Target PCA dimensions (default `3`).
- `ZSCORE_THRESHOLD`: Two-sided outlier magnitude threshold `|z| < threshold` (default `3.0`).
- `SILHOUETTE_SAMPLE_SIZE`: Cap used during Silhouette scoring to bound runtime latency and memory (`5000`).

## 4. Incident & Troubleshooting Playbook

| Symptom | Likely Cause | Resolution |
|---|---|---|
| `POST /analyze` returns `404` | `DATA_PATH` CSV missing inside container/host | Mount or copy `clustering_data.csv`, or pass inline `customers` JSON list. |
| `POST /analyze` returns `422` | Fewer valid rows than `n_clusters` or negative financials | Verify payload conforms to `CustomerRecord` schema bounds. |
| `Janda/Duda` cluster count drops to `0` | Regression to blind `dropna()` in custom preprocessing | Verify `impute_missing_ages()` is called before `dropna()`. |
| High latency on full 28k Silhouette | `SILHOUETTE_SAMPLE_SIZE` set too high | Keep `SILHOUETTE_SAMPLE_SIZE <= 5000` for O(n) bounded evaluation. |

## 5. Rollback Procedure

1. Identify last healthy release tag (`git tag -l`).
2. Re-deploy previous container image tag or checkout target tag (`git checkout v1.0.0`).
3. Re-verify `GET /health` and `python main.py --demo --sample-limit 500`.
