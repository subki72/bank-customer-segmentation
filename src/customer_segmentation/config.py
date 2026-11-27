"""Centralized configuration and schema constants for customer segmentation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

DROP_COLUMNS: Final[tuple[str, ...]] = ("GCIF", "Jalur_Pembukaan")
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = ("Area", "Jenis_Kelamin", "Status_Perkawinan")
PRODUCT_COLUMNS: Final[tuple[str, ...]] = (
    "Produk_Tabungan",
    "Produk_Deposito",
    "Produk_Kartu_Kredit",
    "Produk_Kredit_Rumah",
    "Produk_Kredit_Kendaraan",
    "Produk_Kredit_Dana_Tunai",
)
FINANCIAL_COLUMNS: Final[tuple[str, ...]] = ("Pendapatan_Tahunan", "Total_Relationship_Balance")
ZSCORE_COLUMNS: Final[tuple[str, ...]] = ("Usia", "Pendapatan_Tahunan", "Total_Relationship_Balance")
NUMERIC_BASE_COLUMNS: Final[tuple[str, ...]] = ("Vintage", "Usia", "Jumlah_Anak")
ORDINAL_EDUCATION_COLUMN: Final[str] = "Pendidikan"
MAPPED_EDUCATION_COLUMN: Final[str] = "Pendidikan_Map"

NUMERIC_ORDINAL_FEATURES: Final[tuple[str, ...]] = (
    *NUMERIC_BASE_COLUMNS,
    *PRODUCT_COLUMNS,
    *FINANCIAL_COLUMNS,
    MAPPED_EDUCATION_COLUMN,
)

REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    *CATEGORICAL_COLUMNS,
    *NUMERIC_BASE_COLUMNS,
    ORDINAL_EDUCATION_COLUMN,
    *PRODUCT_COLUMNS,
    *FINANCIAL_COLUMNS,
)

VINTAGE_MAPPING: Final[dict[str, float]] = {
    "< 1 Tahun": 0.5,
    "2 - 3 Tahun": 2.5,
    "> 4 Tahun": 5.0,
}

EDUCATION_MAPPING: Final[dict[str, int]] = {
    "Tidak Sekolah Formal": 0,
    "SD": 1,
    "SMP": 2,
    "SMA": 3,
    "Sarjana": 4,
    "Magister": 5,
    "Doktor": 6,
}

DEFAULT_FALLBACK_AGE: Final[float] = 40.0
DEFAULT_VINTAGE_FALLBACK: Final[float] = 2.5
DEFAULT_EDUCATION_FALLBACK: Final[int] = 3


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure structured root logging for CLI and API execution.

    Args:
        level: Standard Python logging level name (e.g. ``INFO``, ``DEBUG``).
    """
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Configured structured logging at level %s", logging.getLevelName(resolved_level))


@dataclass(frozen=True)
class SegmentationConfig:
    """Runtime parameters loaded from environment variables with safe defaults."""

    data_path: str = "clustering_data.csv"
    n_clusters: int = 3
    pca_n_components: int = 3
    zscore_threshold: float = 3.0
    random_state: int = 42
    silhouette_sample_size: int = 5000
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:8000")
    request_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> SegmentationConfig:
        """Instantiate configuration from environment variables.

        Returns:
            Populated ``SegmentationConfig`` instance.
        """
        raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
        origins = tuple(item.strip() for item in raw_origins.split(",") if item.strip())
        return cls(
            data_path=os.getenv("DATA_PATH", "clustering_data.csv"),
            n_clusters=max(2, int(os.getenv("DEFAULT_N_CLUSTERS", "3"))),
            pca_n_components=max(1, int(os.getenv("PCA_N_COMPONENTS", "3"))),
            zscore_threshold=float(os.getenv("ZSCORE_THRESHOLD", "3.0")),
            random_state=int(os.getenv("RANDOM_STATE", "42")),
            silhouette_sample_size=max(100, int(os.getenv("SILHOUETTE_SAMPLE_SIZE", "5000"))),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            cors_origins=origins or ("http://localhost:3000",),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30.0")),
        )
