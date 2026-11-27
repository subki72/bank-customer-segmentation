"""Bank customer segmentation package built with PCA + K-Means."""

from typing import Final

from src.customer_segmentation.clustering import (
    SegmentationPipeline,
    SegmentationResult,
    generate_synthetic_customers,
    load_customer_data,
)
from src.customer_segmentation.config import SegmentationConfig, configure_logging
from src.customer_segmentation.preprocessing import PreprocessingReport, preprocess_dataframe
from src.customer_segmentation.profiling import ClusterProfile, generate_cluster_profiles

__version__: Final[str] = "1.0.0"

__all__ = [
    "ClusterProfile",
    "PreprocessingReport",
    "SegmentationConfig",
    "SegmentationPipeline",
    "SegmentationResult",
    "__version__",
    "configure_logging",
    "generate_cluster_profiles",
    "generate_synthetic_customers",
    "load_customer_data",
    "preprocess_dataframe",
]
