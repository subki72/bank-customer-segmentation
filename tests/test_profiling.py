"""Unit tests for cluster business persona profiling and cross-sell recommendations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.customer_segmentation.clustering import SegmentationPipeline, generate_synthetic_customers
from src.customer_segmentation.config import PRODUCT_COLUMNS, SegmentationConfig
from src.customer_segmentation.profiling import generate_cluster_profiles


def test_generate_cluster_profiles_assigns_ordered_personas_and_recommendations() -> None:
    """Verify segment profiling computes valid financials, shares, and cross-sell products."""
    df = generate_synthetic_customers(n_samples=150, random_state=42)
    model = SegmentationPipeline(SegmentationConfig(n_clusters=3, random_state=42))
    result = model.fit_predict(df)

    profiles = generate_cluster_profiles(result.clustered_df)

    assert len(profiles) == 3
    assert sum(p.customer_count for p in profiles) == len(result.clustered_df)
    assert pytest.approx(sum(p.share_pct for p in profiles), abs=0.1) == 100.0

    # Verify persona labels align monotonically with relationship balance ranking
    by_balance = sorted(profiles, key=lambda p: p.mean_relationship_balance)
    assert [p.persona_name for p in by_balance] == [
        "Emerging Mass Savers",
        "Mid-Tier Family Growth",
        "Affluent Wealth Builders",
    ]

    for profile in profiles:
        assert profile.recommended_product in PRODUCT_COLUMNS
        assert len(profile.business_action) > 10
        assert profile.dominant_area != "Unknown"


def test_generate_cluster_profiles_supports_custom_k_and_validates_inputs() -> None:
    """Verify generic tier labels for k != 3 and explicit ValueError on invalid inputs."""
    df = generate_synthetic_customers(n_samples=80, random_state=8)
    model = SegmentationPipeline(SegmentationConfig(n_clusters=2, random_state=8))
    result = model.fit_predict(df)

    profiles = generate_cluster_profiles(result.clustered_df)
    assert len(profiles) == 2
    assert {p.persona_name for p in profiles} == {"Value Tier 1", "Value Tier 2"}

    with pytest.raises(ValueError, match="empty"):
        generate_cluster_profiles(pd.DataFrame())

    with pytest.raises(ValueError, match="Cluster_PCA"):
        generate_cluster_profiles(df)
