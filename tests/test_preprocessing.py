"""Unit tests for preprocessing, imputation, outlier filtering, and single-pass scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.customer_segmentation.clustering import generate_synthetic_customers
from src.customer_segmentation.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_ORDINAL_FEATURES,
    SegmentationConfig,
)
from src.customer_segmentation.preprocessing import (
    build_preprocessor,
    encode_ordinal_features,
    impute_missing_ages,
    preprocess_dataframe,
    remove_outliers_zscore,
    validate_input_dataframe,
)


def test_validate_input_dataframe_rejects_empty_and_missing_columns() -> None:
    """Verify upfront guard clauses reject empty or incomplete customer tables."""
    with pytest.raises(ValueError, match="empty"):
        validate_input_dataframe(pd.DataFrame())

    df = generate_synthetic_customers(n_samples=12, random_state=7).drop(columns=["Pendapatan_Tahunan"])
    with pytest.raises(ValueError, match="Pendapatan_Tahunan"):
        validate_input_dataframe(df)


def test_impute_missing_ages_preserves_janda_duda_segment() -> None:
    """Verify Janda/Duda customers with 100% null ages are imputed, not dropped."""
    # Arrange
    df = generate_synthetic_customers(n_samples=40, random_state=42, include_widowed_nan_age=True)
    widowed_count = int((df["Status_Perkawinan"] == "Janda/Duda").sum())
    assert widowed_count > 0
    assert df.loc[df["Status_Perkawinan"] == "Janda/Duda", "Usia"].isna().all()

    # Act
    cleaned, imputed_count = impute_missing_ages(df)

    # Assert
    assert imputed_count == widowed_count
    assert len(cleaned) == len(df)
    assert int((cleaned["Status_Perkawinan"] == "Janda/Duda").sum()) == widowed_count
    assert not cleaned["Usia"].isna().any()


def test_impute_missing_ages_handles_all_nan_dataset() -> None:
    """Verify fallback age applies cleanly when every record lacks age."""
    df = generate_synthetic_customers(n_samples=9, random_state=11)
    df["Usia"] = np.nan

    cleaned, imputed_count = impute_missing_ages(df)

    assert imputed_count == 9
    assert (cleaned["Usia"] == 40.0).all()


def test_remove_outliers_zscore_drops_both_negative_and_positive_extremes() -> None:
    """Verify two-sided Z-score removes -4 sigma outliers as well as +4 sigma outliers."""
    # Arrange: 30 tight baseline rows around balanced midpoint + 1 low (-4 sigma) + 1 high (+4 sigma)
    n_base = 30
    base_df = pd.DataFrame(
        {
            "Usia": [40.0] * n_base + [5.0, 95.0],
            "Pendapatan_Tahunan": [400.0] * n_base + [-200.0, 2000.0],
            "Total_Relationship_Balance": [200_000_000.0] * n_base + [-500_000_000.0, 900_000_000.0],
        }
    )

    # Act
    filtered, removed = remove_outliers_zscore(
        base_df,
        columns=("Usia", "Pendapatan_Tahunan", "Total_Relationship_Balance"),
        threshold=3.0,
    )

    # Assert: BOTH the -4 sigma negative outlier and +4 sigma positive outlier are removed
    assert removed == 2
    assert len(filtered) == n_base
    assert filtered["Pendapatan_Tahunan"].min() >= 0.0
    assert filtered["Total_Relationship_Balance"].min() >= 0.0


def test_remove_outliers_zscore_guards_and_zero_variance() -> None:
    """Verify threshold validation and constant-column stability."""
    constant_df = pd.DataFrame(
        {
            "Usia": [35.0, 35.0, 35.0, 35.0],
            "Pendapatan_Tahunan": [300.0, 300.0, 300.0, 300.0],
            "Total_Relationship_Balance": [100_000.0, 100_000.0, 100_000.0, 100_000.0],
        }
    )
    filtered, removed = remove_outliers_zscore(constant_df, threshold=3.0)
    assert removed == 0
    assert len(filtered) == 4

    with pytest.raises(ValueError, match="positive"):
        remove_outliers_zscore(constant_df, threshold=0.0)


def test_encode_ordinal_features_handles_unseen_categories() -> None:
    """Verify safe fallback mapping on unexpected Vintage and Pendidikan labels."""
    df = generate_synthetic_customers(n_samples=6, random_state=3)
    df.loc[0, "Vintage"] = "10+ Tahun Baru"
    df.loc[1, "Pendidikan"] = "Postdoctoral"

    encoded = encode_ordinal_features(df)

    assert encoded.loc[0, "Vintage"] == 2.5
    assert encoded.loc[1, "Pendidikan_Map"] == 3.0
    assert not encoded["Vintage"].isna().any()
    assert not encoded["Pendidikan_Map"].isna().any()


def test_single_pass_preprocessor_avoids_double_standardization() -> None:
    """Verify numeric/ordinal features scale once while one-hot columns stay binary."""
    raw_df = generate_synthetic_customers(n_samples=60, random_state=42)
    prepared_df, report = preprocess_dataframe(raw_df, config=SegmentationConfig())

    assert report.preserved_widowed_count > 0
    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(prepared_df)

    num_dim = len(NUMERIC_ORDINAL_FEATURES)
    numeric_block = matrix[:, :num_dim]
    onehot_block = matrix[:, num_dim:]

    # Numeric block is standardized (zero mean, unit variance for non-constant features)
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(numeric_block.std(axis=0), 1.0, atol=1e-7)
    # One-hot block stays strictly {0.0, 1.0} (not distorted by a second StandardScaler)
    assert set(np.unique(onehot_block)).issubset({0.0, 1.0})
    assert onehot_block.shape[1] >= len(CATEGORICAL_COLUMNS)
