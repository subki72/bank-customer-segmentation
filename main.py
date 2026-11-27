"""CLI entrypoint for Bank Customer Segmentation pipeline and FastAPI service."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import uvicorn

from src.customer_segmentation.clustering import (
    SegmentationPipeline,
    generate_synthetic_customers,
    load_customer_data,
)
from src.customer_segmentation.config import SegmentationConfig, configure_logging
from src.customer_segmentation.profiling import generate_cluster_profiles

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Bank Customer Segmentation CLI & FastAPI Service Runner",
    )
    parser.add_argument("--health", action="store_true", help="Output service health check JSON and exit.")
    parser.add_argument("--demo", action="store_true", help="Run offline segmentation benchmark and print summary.")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI Uvicorn server.")
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of K-Means customer clusters (default: 3).")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5000,
        help="Optional sample cap for --demo (0 loads full dataset; default: 5000).",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to local customer CSV dataset.")
    parser.add_argument("--host", type=str, default=None, help="Bind host for --serve mode.")
    parser.add_argument("--port", type=int, default=None, help="Bind port for --serve mode.")
    return parser


def run_health(config: SegmentationConfig) -> dict[str, object]:
    """Build health payload and verify local environment readiness."""
    dataset_exists = Path(config.data_path).exists()
    payload: dict[str, object] = {
        "status": "ok",
        "service": "bank-customer-segmentation",
        "version": "1.0.0",
        "dataset_available": dataset_exists,
        "data_path": config.data_path,
    }
    logger.info("Health check passed (dataset_available=%s)", dataset_exists)
    return payload


def run_demo(config: SegmentationConfig, sample_limit: int) -> dict[str, object]:
    """Execute offline segmentation pipeline and return metrics + business personas."""
    data_path = Path(config.data_path)
    if data_path.exists():
        effective_limit = sample_limit if sample_limit > 0 else None
        df = load_customer_data(data_path, sample_limit=effective_limit, random_state=config.random_state)
        source_mode = "csv"
    else:
        logger.warning("Dataset %s not found; falling back to synthetic fixture.", data_path)
        df = generate_synthetic_customers(n_samples=240, random_state=config.random_state)
        source_mode = "synthetic"

    model = SegmentationPipeline(config=config)
    result = model.fit_predict(df)
    profiles = generate_cluster_profiles(result.clustered_df)

    summary: dict[str, object] = {
        "source": source_mode,
        "rows_input": result.report.rows_input,
        "rows_clustered": result.report.rows_after_cleaning,
        "ages_imputed": result.report.ages_imputed,
        "outliers_removed": result.report.outliers_removed,
        "preserved_widowed_count": result.report.preserved_widowed_count,
        "silhouette_score": result.silhouette_score,
        "davies_bouldin_score": result.davies_bouldin_score,
        "calinski_harabasz_score": result.calinski_harabasz_score,
        "explained_variance_ratio": result.explained_variance_ratio,
        "profiles": [profile.to_dict() for profile in profiles],
    }
    logger.info(
        "Demo finished: rows=%d silhouette=%.4f clusters=%d",
        result.report.rows_after_cleaning,
        result.silhouette_score,
        len(profiles),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint handling --health, --demo, and --serve modes."""
    base_cfg = SegmentationConfig.from_env()
    configure_logging(base_cfg.log_level)

    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = SegmentationConfig(
        data_path=args.data_path or base_cfg.data_path,
        n_clusters=args.n_clusters,
        pca_n_components=base_cfg.pca_n_components,
        zscore_threshold=base_cfg.zscore_threshold,
        random_state=base_cfg.random_state,
        silhouette_sample_size=base_cfg.silhouette_sample_size,
        log_level=base_cfg.log_level,
        api_host=args.host or base_cfg.api_host,
        api_port=args.port or base_cfg.api_port,
        cors_origins=base_cfg.cors_origins,
        request_timeout_seconds=base_cfg.request_timeout_seconds,
    )

    if args.serve:
        logger.info("Starting FastAPI server on %s:%d", cfg.api_host, cfg.api_port)
        uvicorn.run(
            "src.customer_segmentation.api:app",
            host=cfg.api_host,
            port=cfg.api_port,
            reload=False,
        )
        return 0

    if args.demo:
        summary = run_demo(cfg, sample_limit=args.sample_limit)
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        return 0

    # Default or explicit --health executes health check cleanly offline
    health = run_health(cfg)
    sys.stdout.write(json.dumps(health, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
