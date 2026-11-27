"""PCA + K-Means customer segmentation pipeline, evaluation metrics, and data utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline

from src.customer_segmentation.config import (
    EDUCATION_MAPPING,
    VINTAGE_MAPPING,
    SegmentationConfig,
)
from src.customer_segmentation.preprocessing import (
    PreprocessingReport,
    build_preprocessor,
    preprocess_dataframe,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationResult:
    """Container for fitted cluster assignments and internal validation metrics."""

    labels: np.ndarray
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    explained_variance_ratio: list[float]
    cluster_counts: dict[int, int]
    report: PreprocessingReport
    clustered_df: pd.DataFrame


class SegmentationPipeline:
    """Unified single-pass Preprocessing -> PCA -> K-Means segmentation model."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.config = config or SegmentationConfig()
        if self.config.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.config.n_clusters}")
        if self.config.pca_n_components < 1:
            raise ValueError(f"pca_n_components must be >= 1, got {self.config.pca_n_components}")

        self.pipeline: Pipeline | None = None

    def _build_sklearn_pipeline(self, n_components: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("pca", PCA(n_components=n_components, random_state=self.config.random_state)),
                (
                    "kmeans",
                    KMeans(
                        n_clusters=self.config.n_clusters,
                        random_state=self.config.random_state,
                        n_init=10,
                    ),
                ),
            ]
        )

    def _evaluate_metrics(
        self,
        pca_features: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute internal clustering metrics with bounded Silhouette sampling."""
        if len(np.unique(labels)) < 2:
            return 0.0, 0.0, 0.0
        sample_cap = min(len(pca_features), self.config.silhouette_sample_size)
        sil = float(
            silhouette_score(
                pca_features,
                labels,
                sample_size=sample_cap if len(pca_features) > sample_cap else None,
                random_state=self.config.random_state,
            )
        )
        db_score = float(davies_bouldin_score(pca_features, labels))
        ch_score = float(calinski_harabasz_score(pca_features, labels))
        return sil, db_score, ch_score

    def fit_predict(self, df: pd.DataFrame) -> SegmentationResult:
        """Preprocess raw customer data, fit PCA + K-Means, and evaluate cluster quality.

        Args:
            df: Raw customer DataFrame.

        Returns:
            ``SegmentationResult`` with cluster labels, metrics, and cleaned DataFrame.

        Raises:
            ValueError: If cleaned sample count is smaller than ``n_clusters``.
        """
        prepared_df, report = preprocess_dataframe(df, config=self.config, remove_outliers=True)
        if len(prepared_df) < self.config.n_clusters:
            raise ValueError(f"Need at least {self.config.n_clusters} rows after cleaning, got {len(prepared_df)}.")

        transformed = build_preprocessor().fit_transform(prepared_df)
        effective_components = min(self.config.pca_n_components, transformed.shape[0], transformed.shape[1])

        self.pipeline = self._build_sklearn_pipeline(n_components=effective_components)
        labels = self.pipeline.fit_predict(prepared_df)
        pca_features = self.pipeline[:-1].transform(prepared_df)
        sil, db_score, ch_score = self._evaluate_metrics(pca_features, labels)

        pca_step: PCA = self.pipeline.named_steps["pca"]
        explained = [round(float(val), 4) for val in pca_step.explained_variance_ratio_]
        counts_series = pd.Series(labels).value_counts().sort_index()
        cluster_counts = {int(k): int(v) for k, v in counts_series.items()}

        clustered_df = prepared_df.copy()
        clustered_df["Cluster_PCA"] = labels
        logger.info(
            "audit_log: Fitted PCA(%d)+KMeans(%d) on %d rows (silhouette=%.4f, preserved_widowed=%d)",
            effective_components,
            self.config.n_clusters,
            len(clustered_df),
            sil,
            report.preserved_widowed_count,
        )
        return SegmentationResult(
            labels=labels,
            silhouette_score=round(sil, 4),
            davies_bouldin_score=round(db_score, 4),
            calinski_harabasz_score=round(ch_score, 4),
            explained_variance_ratio=explained,
            cluster_counts=cluster_counts,
            report=report,
            clustered_df=clustered_df,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Assign cluster labels to new customer rows without outlier row dropping.

        Args:
            df: Customer DataFrame matching expected input schema.

        Returns:
            1D numpy array of predicted cluster integers.

        Raises:
            RuntimeError: If pipeline has not been fitted yet.
        """
        if self.pipeline is None:
            raise RuntimeError("SegmentationPipeline must be fitted before calling predict().")

        prepared_df, _ = preprocess_dataframe(df, config=self.config, remove_outliers=False)
        return self.pipeline.predict(prepared_df)

    def save(self, artifact_path: Path | str) -> Path:
        """Persist fitted pipeline to disk via joblib.

        Args:
            artifact_path: Target ``.joblib`` file path.

        Returns:
            Resolved ``Path`` of the saved model.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted SegmentationPipeline.")
        target = Path(artifact_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "config": self.config}, target)
        logger.info("Saved fitted segmentation pipeline to %s", target)
        return target

    @classmethod
    def load(cls, artifact_path: Path | str) -> SegmentationPipeline:
        """Load a previously serialized ``SegmentationPipeline``.

        Args:
            artifact_path: Existing ``.joblib`` file path.

        Returns:
            Loaded ``SegmentationPipeline`` ready for inference.
        """
        source = Path(artifact_path)
        if not source.exists():
            raise FileNotFoundError(f"Model artifact not found: {source}")
        payload = joblib.load(source)
        instance = cls(config=payload["config"])
        instance.pipeline = payload["pipeline"]
        return instance


def load_customer_data(
    csv_path: Path | str,
    sample_limit: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load local CSV customer dataset cleanly with optional deterministic sampling.

    Args:
        csv_path: Path to local CSV file (e.g. ``clustering_data.csv``).
        sample_limit: Optional row cap; ``None`` or ``<= 0`` loads all rows.
        random_state: Reproducible seed when sampling.

    Returns:
        Loaded customer DataFrame.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    if sample_limit is not None and sample_limit > 0 and len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=random_state).reset_index(drop=True)
    logger.info("Loaded %d customer records from %s", len(df), path)
    return df


def _build_tier_financials(
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate distinct behavioral tiers so synthetic clustering separates reliably."""
    tier = np.arange(n_samples) % 3
    ages = np.where(
        tier == 0,
        rng.integers(22, 34, n_samples),
        np.where(tier == 1, rng.integers(34, 50, n_samples), rng.integers(50, 68, n_samples)),
    ).astype(float)
    incomes = np.where(
        tier == 0,
        rng.integers(120, 260, n_samples),
        np.where(tier == 1, rng.integers(280, 480, n_samples), rng.integers(520, 820, n_samples)),
    )
    balances = np.where(
        tier == 0,
        rng.integers(5_000_000, 90_000_000, n_samples),
        np.where(
            tier == 1,
            rng.integers(120_000_000, 320_000_000, n_samples),
            rng.integers(380_000_000, 780_000_000, n_samples),
        ),
    )
    return tier, ages, incomes, balances


def generate_synthetic_customers(
    n_samples: int = 120,
    random_state: int = 42,
    include_widowed_nan_age: bool = True,
) -> pd.DataFrame:
    """Generate deterministic offline Indonesian banking customer fixtures for testing.

    Args:
        n_samples: Number of synthetic customers to generate (minimum 3).
        random_state: Deterministic numpy RNG seed.
        include_widowed_nan_age: Whether ``Janda/Duda`` records include ``NaN`` in ``Usia``.

    Returns:
        Synthetic DataFrame conforming to ``clustering_data.csv`` schema.
    """
    if n_samples < 3:
        raise ValueError(f"n_samples must be >= 3, got {n_samples}")

    rng = np.random.default_rng(random_state)
    tier, ages, incomes, balances = _build_tier_financials(n_samples, rng)
    marital_statuses = np.array(["Menikah", "Belum Menikah", "Cerai", "Janda/Duda"])
    marital_col = marital_statuses[np.arange(n_samples) % len(marital_statuses)]
    if include_widowed_nan_age:
        ages = np.where(marital_col == "Janda/Duda", np.nan, ages)

    tabungan = rng.integers(0, 2, n_samples)
    deposito = np.where(tier == 2, 1, rng.integers(0, 2, n_samples))
    kartu_kredit = np.where(tier >= 1, 1, 0)
    kredit_rumah = np.where(tier == 1, 1, 0)
    kredit_kendaraan = rng.integers(0, 2, n_samples)
    kredit_dana = rng.integers(0, 2, n_samples)
    total_produk = tabungan + deposito + kartu_kredit + kredit_rumah + kredit_kendaraan + kredit_dana
    channels = ["Cabang", "Telemarketing", "Aplikasi Digital", "Internet Banking"]

    return pd.DataFrame(
        {
            "GCIF": [f"G{10000 + idx}" for idx in range(n_samples)],
            "Area": rng.choice(["Jakarta", "Bogor", "Bandung", "Surabaya", "Jogja", "Solo"], n_samples),
            "Jalur_Pembukaan": rng.choice(channels, n_samples),
            "Vintage": rng.choice(list(VINTAGE_MAPPING.keys()), n_samples),
            "Usia": ages,
            "Jenis_Kelamin": rng.choice(["Laki-laki", "Perempuan"], n_samples),
            "Status_Perkawinan": marital_col,
            "Jumlah_Anak": np.where(tier == 0, 0, rng.integers(0, 4, n_samples)),
            "Pendidikan": rng.choice(list(EDUCATION_MAPPING.keys()), n_samples),
            "Produk_Tabungan": tabungan,
            "Produk_Deposito": deposito,
            "Produk_Kartu_Kredit": kartu_kredit,
            "Produk_Kredit_Rumah": kredit_rumah,
            "Produk_Kredit_Kendaraan": kredit_kendaraan,
            "Produk_Kredit_Dana_Tunai": kredit_dana,
            "Total_Kepemilikan_Produk": total_produk,
            "Pendapatan_Tahunan": incomes,
            "Total_Relationship_Balance": balances,
        }
    )
