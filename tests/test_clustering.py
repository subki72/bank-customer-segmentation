"""Unit tests for PCA + K-Means segmentation pipeline, persistence, and data loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.customer_segmentation.clustering import (
    SegmentationPipeline,
    generate_synthetic_customers,
    load_customer_data,
)
from src.customer_segmentation.config import SegmentationConfig


def test_segmentation_pipeline_fit_predict_produces_valid_clusters() -> None:
    """Verify end-to-end PCA + K-Means clustering produces valid metrics and preserves Janda/Duda."""
    df = generate_synthetic_customers(n_samples=120, random_state=42, include_widowed_nan_age=True)
    model = SegmentationPipeline(SegmentationConfig(n_clusters=3, pca_n_components=3, random_state=42))

    result = model.fit_predict(df)

    assert len(result.labels) == result.report.rows_after_cleaning
    assert len(result.cluster_counts) == 3
    assert sum(result.cluster_counts.values()) == result.report.rows_after_cleaning
    assert result.report.preserved_widowed_count > 0
    assert -1.0 <= result.silhouette_score <= 1.0
    assert result.silhouette_score > 0.1
    assert result.davies_bouldin_score > 0.0
    assert result.calinski_harabasz_score > 0.0
    assert len(result.explained_variance_ratio) == 3
    assert sum(result.explained_variance_ratio) <= 1.0 + 1e-6


def test_segmentation_pipeline_predict_requires_fit_and_preserves_inference_rows() -> None:
    """Verify unfitted guard and single-row prediction without outlier filtering."""
    df = generate_synthetic_customers(n_samples=60, random_state=9)
    model = SegmentationPipeline()

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(df.iloc[:2])

    model.fit_predict(df)

    # Single extreme row with NaN age should still receive a valid cluster label
    probe = df.iloc[[0]].copy()
    probe.loc[probe.index[0], "Usia"] = np.nan
    probe.loc[probe.index[0], "Pendapatan_Tahunan"] = 9999.0
    preds = model.predict(probe)

    assert preds.shape == (1,)
    assert int(preds[0]) in {0, 1, 2}


def test_segmentation_pipeline_rejects_invalid_configs_and_tiny_inputs() -> None:
    """Verify constructor and fit_predict validation guards."""
    with pytest.raises(ValueError, match="n_clusters"):
        SegmentationPipeline(SegmentationConfig(n_clusters=1))

    with pytest.raises(ValueError, match="pca_n_components"):
        SegmentationPipeline(SegmentationConfig(pca_n_components=0))

    tiny_df = generate_synthetic_customers(n_samples=3, random_state=1).iloc[:2]
    model = SegmentationPipeline(SegmentationConfig(n_clusters=3))
    with pytest.raises(ValueError, match="at least 3 rows"):
        model.fit_predict(tiny_df)


def test_segmentation_pipeline_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Verify serialized pipeline produces identical cluster predictions after reload."""
    df = generate_synthetic_customers(n_samples=90, random_state=21)
    model = SegmentationPipeline(SegmentationConfig(n_clusters=3, random_state=21))
    model.fit_predict(df)

    artifact = tmp_path / "models" / "segmentation.joblib"
    saved_path = model.save(artifact)
    assert saved_path.exists()

    reloaded = SegmentationPipeline.load(saved_path)
    original_preds = model.predict(df.iloc[:10])
    reloaded_preds = reloaded.predict(df.iloc[:10])

    assert np.array_equal(original_preds, reloaded_preds)

    with pytest.raises(RuntimeError, match="unfitted"):
        SegmentationPipeline().save(tmp_path / "unfitted.joblib")

    with pytest.raises(FileNotFoundError, match="not found"):
        SegmentationPipeline.load(tmp_path / "missing.joblib")


def test_load_customer_data_sampling_and_missing_file(tmp_path: Path) -> None:
    """Verify offline CSV loader handles deterministic sampling and missing files."""
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_customer_data(missing)

    csv_path = tmp_path / "sample.csv"
    generate_synthetic_customers(n_samples=50, random_state=5).to_csv(csv_path, index=False)

    sampled = load_customer_data(csv_path, sample_limit=15, random_state=5)
    assert len(sampled) == 15

    full = load_customer_data(csv_path, sample_limit=None)
    assert len(full) == 50
