"""Business persona profiling and cross-sell recommendations per customer cluster."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Final

import pandas as pd

from src.customer_segmentation.config import PRODUCT_COLUMNS

logger = logging.getLogger(__name__)

PRODUCT_RECOMMENDATION_ACTIONS: Final[dict[str, str]] = {
    "Produk_Deposito": "Offer tiered high-yield term deposit and wealth retention bundle.",
    "Produk_Kartu_Kredit": "Cross-sell lifestyle cashback credit card with installment perks.",
    "Produk_Kredit_Rumah": "Target mortgage (KPR) pre-approval and family asset financing.",
    "Produk_Kredit_Kendaraan": "Promote low-rate auto loan (KKB) upgrade campaign.",
    "Produk_Kredit_Dana_Tunai": "Provide flexible personal cash facility for liquidity needs.",
    "Produk_Tabungan": "Onboard digital high-interest primary savings account.",
}

PERSONA_TIER_LABELS: Final[tuple[str, ...]] = (
    "Emerging Mass Savers",
    "Mid-Tier Family Growth",
    "Affluent Wealth Builders",
)


@dataclass(frozen=True)
class ClusterProfile:
    """Actionable business summary for a single customer segment."""

    cluster_id: int
    customer_count: int
    share_pct: float
    persona_name: str
    mean_age: float
    mean_annual_income: float
    mean_relationship_balance: float
    mean_product_count: float
    dominant_area: str
    dominant_marital_status: str
    dominant_education: str
    recommended_product: str
    business_action: str

    def to_dict(self) -> dict[str, object]:
        """Serialize cluster profile into JSON-friendly dictionary.

        Returns:
            Dictionary representation of ``ClusterProfile``.
        """
        return asdict(self)


def _safe_mode(series: pd.Series, fallback: str = "Unknown") -> str:
    """Return the most frequent non-null string in ``series`` safely."""
    modes = series.dropna().mode()
    if modes.empty:
        return fallback
    return str(modes.iloc[0])


def _select_cross_sell_product(cluster_df: pd.DataFrame) -> tuple[str, str]:
    """Identify the under-penetrated banking product with highest cross-sell potential."""
    available_products = [col for col in PRODUCT_COLUMNS if col in cluster_df.columns]
    if not available_products:
        return "Produk_Deposito", PRODUCT_RECOMMENDATION_ACTIONS["Produk_Deposito"]

    adoption_rates = {col: float(cluster_df[col].mean()) for col in available_products}
    recommended = min(adoption_rates, key=lambda col: adoption_rates[col])
    action = PRODUCT_RECOMMENDATION_ACTIONS.get(
        recommended,
        "Offer tailored cross-sell advisory package.",
    )
    return recommended, action


def _compute_persona_map(profiles: list[ClusterProfile]) -> dict[int, str]:
    """Rank clusters by relationship balance and income so personas remain consistent."""
    ranked = sorted(
        profiles,
        key=lambda item: (item.mean_relationship_balance, item.mean_annual_income),
    )
    persona_map: dict[int, str] = {}
    if len(ranked) == 3:
        for idx, item in enumerate(ranked):
            persona_map[item.cluster_id] = PERSONA_TIER_LABELS[idx]
    else:
        for idx, item in enumerate(ranked, start=1):
            persona_map[item.cluster_id] = f"Value Tier {idx}"
    return persona_map


def _summarize_cluster(cid: int, group: pd.DataFrame, total_rows: int) -> ClusterProfile:
    """Compute raw demographic, financial, and cross-sell attributes for one cluster."""
    available_products = [col for col in PRODUCT_COLUMNS if col in group.columns]
    mean_prod = float(group[available_products].sum(axis=1).mean()) if available_products else 0.0
    rec_product, rec_action = _select_cross_sell_product(group)

    return ClusterProfile(
        cluster_id=cid,
        customer_count=len(group),
        share_pct=round((len(group) / total_rows) * 100.0, 2),
        persona_name="",
        mean_age=round(float(group["Usia"].mean()), 2),
        mean_annual_income=round(float(group["Pendapatan_Tahunan"].mean()), 2),
        mean_relationship_balance=round(float(group["Total_Relationship_Balance"].mean()), 2),
        mean_product_count=round(mean_prod, 2),
        dominant_area=_safe_mode(group["Area"]),
        dominant_marital_status=_safe_mode(group["Status_Perkawinan"]),
        dominant_education=_safe_mode(group["Pendidikan"]),
        recommended_product=rec_product,
        business_action=rec_action,
    )


def generate_cluster_profiles(
    df_clustered: pd.DataFrame,
    cluster_col: str = "Cluster_PCA",
) -> list[ClusterProfile]:
    """Build demographic, financial, and cross-sell profiles for each discovered cluster.

    Args:
        df_clustered: Preprocessed DataFrame containing ``cluster_col``.
        cluster_col: Column storing integer cluster IDs.

    Returns:
        Sorted list of ``ClusterProfile`` instances ordered by ``cluster_id``.

    Raises:
        ValueError: If DataFrame is empty or missing ``cluster_col``.
    """
    if df_clustered is None or df_clustered.empty:
        raise ValueError("Clustered DataFrame cannot be empty.")
    if cluster_col not in df_clustered.columns:
        raise ValueError(f"Missing cluster assignment column: {cluster_col}")

    total_rows = len(df_clustered)
    raw_profiles = [
        _summarize_cluster(int(cid), df_clustered[df_clustered[cluster_col] == int(cid)], total_rows)
        for cid in sorted(df_clustered[cluster_col].unique())
    ]

    persona_map = _compute_persona_map(raw_profiles)
    profiles = [replace(item, persona_name=persona_map[item.cluster_id]) for item in raw_profiles]
    logger.info("Generated %d business segment profiles across %d customers", len(profiles), total_rows)
    return profiles
