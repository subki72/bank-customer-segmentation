"""Deterministic data cleaning, imputation, outlier filtering, and encoding."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.customer_segmentation.config import (
    CATEGORICAL_COLUMNS,
    DEFAULT_EDUCATION_FALLBACK,
    DEFAULT_FALLBACK_AGE,
    DEFAULT_VINTAGE_FALLBACK,
    DROP_COLUMNS,
    EDUCATION_MAPPING,
    FINANCIAL_COLUMNS,
    MAPPED_EDUCATION_COLUMN,
    NUMERIC_ORDINAL_FEATURES,
    ORDINAL_EDUCATION_COLUMN,
    PRODUCT_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    VINTAGE_MAPPING,
    ZSCORE_COLUMNS,
    SegmentationConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingReport:
    """Audit summary of rows retained, imputed, and filtered during preprocessing."""

    rows_input: int
    rows_after_cleaning: int
    ages_imputed: int
    duplicates_removed: int
    outliers_removed: int
    preserved_widowed_count: int


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate that input data is non-empty and contains required banking columns.

    Args:
        df: Raw customer DataFrame.

    Raises:
        ValueError: If DataFrame is empty or missing required columns.
    """
    if df is None or df.empty:
        raise ValueError("Input customer DataFrame cannot be empty.")

    missing_cols = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required customer columns: {', '.join(missing_cols)}")


def impute_missing_ages(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Impute missing ``Usia`` values via group-median then global-median fallback.

    Preserves demographic segments such as ``Janda/Duda`` where age may be 100% null.

    Args:
        df: Validated customer DataFrame.

    Returns:
        Tuple of ``(cleaned_df, imputed_age_count)``.
    """
    cleaned = df.copy()
    cleaned["Usia"] = pd.to_numeric(cleaned["Usia"], errors="coerce")
    missing_mask = cleaned["Usia"].isna()
    imputed_count = int(missing_mask.sum())

    if imputed_count > 0:
        group_medians = cleaned.groupby("Status_Perkawinan")["Usia"].transform("median")
        global_median = cleaned["Usia"].median()
        fallback_age = float(global_median) if pd.notna(global_median) else DEFAULT_FALLBACK_AGE
        cleaned["Usia"] = cleaned["Usia"].fillna(group_medians).fillna(fallback_age)
        logger.info("Imputed %d missing age values (fallback median=%.1f)", imputed_count, fallback_age)

    # Ensure remaining mandatory fields are safe before dropping unrecoverable rows
    cleaned["Jumlah_Anak"] = pd.to_numeric(cleaned["Jumlah_Anak"], errors="coerce").fillna(0)
    for col in CATEGORICAL_COLUMNS:
        cleaned[col] = cleaned[col].fillna("Unknown").astype(str)
    cleaned[ORDINAL_EDUCATION_COLUMN] = cleaned[ORDINAL_EDUCATION_COLUMN].fillna("SMA").astype(str)

    mandatory_numeric = [*PRODUCT_COLUMNS, *FINANCIAL_COLUMNS]
    for col in mandatory_numeric:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=mandatory_numeric).reset_index(drop=True)
    return cleaned, imputed_count


def remove_outliers_zscore(
    df: pd.DataFrame,
    columns: Sequence[str] = ZSCORE_COLUMNS,
    threshold: float = 3.0,
) -> tuple[pd.DataFrame, int]:
    """Remove extreme lower and upper outliers using a two-sided Z-score filter.

    Args:
        df: Cleaned customer DataFrame.
        columns: Numeric columns evaluated for Z-score bounds.
        threshold: Positive Z-score magnitude cutoff.

    Returns:
        Tuple of ``(filtered_df, removed_row_count)``.

    Raises:
        ValueError: If ``threshold`` is not positive or ``columns`` are absent.
    """
    if threshold <= 0:
        raise ValueError(f"Z-score threshold must be positive, got {threshold}")

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for outlier filtering: {', '.join(missing)}")

    if len(df) < 3:
        return df.copy().reset_index(drop=True), 0

    values = df[list(columns)].to_numpy(dtype=float)
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=0)

    # Avoid NaN propagation on constant zero-variance columns (e.g., identical synthetic ages)
    safe_stds = np.where(stds == 0.0, 1.0, stds)
    z_scores = np.where(stds == 0.0, 0.0, (values - means) / safe_stds)

    inlier_mask = (np.abs(z_scores) < threshold).all(axis=1)
    filtered = df.loc[inlier_mask].reset_index(drop=True)
    removed = int(len(df) - len(filtered))
    if removed > 0:
        logger.info("Removed %d extreme two-sided outliers (|z| >= %.2f)", removed, threshold)
    return filtered, removed


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal banking attributes (``Vintage``, ``Pendidikan``) with safe fallbacks.

    Args:
        df: Customer DataFrame with ``Vintage`` and ``Pendidikan`` columns.

    Returns:
        Copy of DataFrame with numeric ``Vintage`` and ``Pendidikan_Map`` columns.
    """
    encoded = df.copy()

    if pd.api.types.is_numeric_dtype(encoded["Vintage"]):
        encoded["Vintage"] = encoded["Vintage"].astype(float).fillna(DEFAULT_VINTAGE_FALLBACK)
    else:
        mapped_vintage = encoded["Vintage"].astype(str).str.strip().map(VINTAGE_MAPPING)
        encoded["Vintage"] = mapped_vintage.fillna(DEFAULT_VINTAGE_FALLBACK).astype(float)

    mapped_edu = encoded[ORDINAL_EDUCATION_COLUMN].astype(str).str.strip().map(EDUCATION_MAPPING)
    encoded[MAPPED_EDUCATION_COLUMN] = mapped_edu.fillna(DEFAULT_EDUCATION_FALLBACK).astype(float)
    return encoded


def build_preprocessor() -> ColumnTransformer:
    """Create a single-pass scikit-learn ColumnTransformer without double standardization.

    Returns:
        Configured ``ColumnTransformer`` applying ``StandardScaler`` once to numeric/ordinal
        features and ``OneHotEncoder`` to nominal demographic attributes.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_ORDINAL_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_dataframe(
    df: pd.DataFrame,
    config: SegmentationConfig | None = None,
    remove_outliers: bool = True,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Run full deterministic preprocessing pipeline on raw customer records.

    Args:
        df: Raw customer DataFrame matching ``REQUIRED_INPUT_COLUMNS``.
        config: Optional ``SegmentationConfig`` overriding default thresholds.
        remove_outliers: Set ``False`` during single-record inference to preserve all inputs.

    Returns:
        Tuple of ``(prepared_df, PreprocessingReport)``.
    """
    cfg = config or SegmentationConfig()
    validate_input_dataframe(df)

    rows_input = len(df)
    working, ages_imputed = impute_missing_ages(df)

    cols_to_drop = [col for col in DROP_COLUMNS if col in working.columns]
    if cols_to_drop:
        working = working.drop(columns=cols_to_drop)

    before_dedup = len(working)
    if remove_outliers:
        working = working.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_dedup - len(working)

    outliers_removed = 0
    if remove_outliers:
        working, outliers_removed = remove_outliers_zscore(
            working,
            columns=ZSCORE_COLUMNS,
            threshold=cfg.zscore_threshold,
        )

    working = encode_ordinal_features(working)
    preserved_widowed = int((working["Status_Perkawinan"] == "Janda/Duda").sum())

    report = PreprocessingReport(
        rows_input=rows_input,
        rows_after_cleaning=len(working),
        ages_imputed=ages_imputed,
        duplicates_removed=duplicates_removed,
        outliers_removed=outliers_removed,
        preserved_widowed_count=preserved_widowed,
    )
    return working, report
