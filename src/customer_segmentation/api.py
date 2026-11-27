"""FastAPI service exposing customer segmentation analysis, inference, and health probes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.customer_segmentation.clustering import (
    SegmentationPipeline,
    generate_synthetic_customers,
    load_customer_data,
)
from src.customer_segmentation.config import SegmentationConfig, configure_logging
from src.customer_segmentation.profiling import ClusterProfile, generate_cluster_profiles

logger = logging.getLogger(__name__)

_CONFIG = SegmentationConfig.from_env()
configure_logging(_CONFIG.log_level)


class CustomerRecord(BaseModel):
    """Validated input schema representing one bank customer profile."""

    model_config = ConfigDict(extra="ignore")

    GCIF: str = Field(default="G00001", min_length=1, max_length=64)
    Area: str = Field(default="Jakarta", min_length=1, max_length=64)
    Jalur_Pembukaan: str = Field(default="Cabang", min_length=1, max_length=64)
    Vintage: str | float = Field(default="2 - 3 Tahun")
    Usia: float | None = Field(default=None, ge=17.0, le=120.0)
    Jenis_Kelamin: str = Field(default="Laki-laki", min_length=1, max_length=32)
    Status_Perkawinan: str = Field(default="Menikah", min_length=1, max_length=32)
    Jumlah_Anak: int = Field(default=0, ge=0, le=20)
    Pendidikan: str = Field(default="Sarjana", min_length=1, max_length=64)
    Produk_Tabungan: Literal[0, 1] = 1
    Produk_Deposito: Literal[0, 1] = 0
    Produk_Kartu_Kredit: Literal[0, 1] = 0
    Produk_Kredit_Rumah: Literal[0, 1] = 0
    Produk_Kredit_Kendaraan: Literal[0, 1] = 0
    Produk_Kredit_Dana_Tunai: Literal[0, 1] = 0
    Pendapatan_Tahunan: float = Field(ge=0.0)
    Total_Relationship_Balance: float = Field(ge=0.0)


class AnalyzeRequest(BaseModel):
    """Request payload for batch customer segmentation."""

    n_clusters: int = Field(default=3, ge=2, le=10)
    sample_limit: int = Field(default=2000, ge=10, le=50000)
    customers: list[CustomerRecord] | None = None


class ClusterProfileSchema(BaseModel):
    """Serialized cluster persona returned by ``/analyze``."""

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


class AnalyzeResponse(BaseModel):
    """Response schema for ``POST /analyze``."""

    n_clusters: int
    rows_input: int
    rows_clustered: int
    ages_imputed: int
    outliers_removed: int
    preserved_widowed_count: int
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    explained_variance_ratio: list[float]
    profiles: list[ClusterProfileSchema]


class SegmentResponse(BaseModel):
    """Response schema for single-customer ``POST /segment``."""

    gcif: str
    cluster_id: int
    persona_name: str
    recommended_product: str
    business_action: str


class HealthResponse(BaseModel):
    """Liveness and model readiness response."""

    status: str
    service: str
    version: str
    model_ready: bool


class _ModelRegistry:
    """Thread-safe in-memory cache holding the active fitted segmentation pipeline."""

    def __init__(self) -> None:
        self.pipeline: SegmentationPipeline | None = None
        self.profiles: dict[int, ClusterProfile] = {}

    def get_or_initialize(self, config: SegmentationConfig) -> tuple[SegmentationPipeline, dict[int, ClusterProfile]]:
        if self.pipeline is not None and self.profiles:
            return self.pipeline, self.profiles

        data_file = Path(config.data_path)
        if data_file.exists():
            df = load_customer_data(data_file, sample_limit=2000, random_state=config.random_state)
        else:
            df = generate_synthetic_customers(n_samples=180, random_state=config.random_state)

        model = SegmentationPipeline(config=config)
        result = model.fit_predict(df)
        profile_list = generate_cluster_profiles(result.clustered_df)
        self.pipeline = model
        self.profiles = {p.cluster_id: p for p in profile_list}
        return self.pipeline, self.profiles


_REGISTRY = _ModelRegistry()

app = FastAPI(
    title="Bank Customer Segmentation API",
    description="PCA + K-Means customer segmentation and business persona recommendation service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_CONFIG.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    """Convert domain validation errors into explicit HTTP 422 responses."""
    logger.warning("Validation error during request: %s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled server exceptions safely without leaking tracebacks."""
    logger.exception("Unhandled error in API service: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Return service liveness and pipeline readiness status."""
    return HealthResponse(
        status="ok",
        service="bank-customer-segmentation",
        version="1.0.0",
        model_ready=_REGISTRY.pipeline is not None,
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Segmentation"])
def analyze_customers(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Run full preprocessing, PCA, K-Means clustering, and persona profiling."""
    cfg = SegmentationConfig(
        data_path=_CONFIG.data_path,
        n_clusters=payload.n_clusters,
        pca_n_components=_CONFIG.pca_n_components,
        zscore_threshold=_CONFIG.zscore_threshold,
        random_state=_CONFIG.random_state,
        silhouette_sample_size=_CONFIG.silhouette_sample_size,
        log_level=_CONFIG.log_level,
        cors_origins=_CONFIG.cors_origins,
        request_timeout_seconds=_CONFIG.request_timeout_seconds,
    )

    if payload.customers:
        raw_df = pd.DataFrame([record.model_dump() for record in payload.customers])
    else:
        data_file = Path(cfg.data_path)
        if not data_file.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found on server.")
        raw_df = load_customer_data(data_file, sample_limit=payload.sample_limit, random_state=cfg.random_state)

    model = SegmentationPipeline(config=cfg)
    result = model.fit_predict(raw_df)
    profiles = generate_cluster_profiles(result.clustered_df)

    _REGISTRY.pipeline = model
    _REGISTRY.profiles = {profile.cluster_id: profile for profile in profiles}

    return AnalyzeResponse(
        n_clusters=cfg.n_clusters,
        rows_input=result.report.rows_input,
        rows_clustered=result.report.rows_after_cleaning,
        ages_imputed=result.report.ages_imputed,
        outliers_removed=result.report.outliers_removed,
        preserved_widowed_count=result.report.preserved_widowed_count,
        silhouette_score=result.silhouette_score,
        davies_bouldin_score=result.davies_bouldin_score,
        calinski_harabasz_score=result.calinski_harabasz_score,
        explained_variance_ratio=result.explained_variance_ratio,
        profiles=[ClusterProfileSchema(**profile.to_dict()) for profile in profiles],
    )


@app.post("/segment", response_model=SegmentResponse, tags=["Segmentation"])
def segment_single_customer(customer: CustomerRecord) -> SegmentResponse:
    """Predict cluster assignment and recommended product for a single customer."""
    model, profiles = _REGISTRY.get_or_initialize(_CONFIG)
    customer_df = pd.DataFrame([customer.model_dump()])
    cluster_id = int(model.predict(customer_df)[0])
    profile = profiles.get(cluster_id)
    if profile is None:
        raise HTTPException(status_code=500, detail="Cluster profile lookup failed.")

    return SegmentResponse(
        gcif=customer.GCIF,
        cluster_id=cluster_id,
        persona_name=profile.persona_name,
        recommended_product=profile.recommended_product,
        business_action=profile.business_action,
    )
