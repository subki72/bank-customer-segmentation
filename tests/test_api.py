"""Integration tests for FastAPI endpoints, schema validation, and error handling."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fastapi.testclient import TestClient

from src.customer_segmentation import api
from src.customer_segmentation.clustering import generate_synthetic_customers


def _synthetic_records(n_samples: int = 45) -> list[dict[str, Any]]:
    df = generate_synthetic_customers(n_samples=n_samples, random_state=42, include_widowed_nan_age=True)
    # Replace NaN with None for valid JSON payload serialization
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def test_health_endpoint_returns_ok() -> None:
    """Verify GET /health returns service liveness metadata."""
    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "bank-customer-segmentation"
    assert body["version"] == "1.0.0"
    assert isinstance(body["model_ready"], bool)


def test_analyze_endpoint_segments_batch_and_preserves_widowed_customers() -> None:
    """Verify POST /analyze preserves Janda/Duda null-age customers and removes negative outliers."""
    client = TestClient(api.app)
    records = _synthetic_records(n_samples=48)
    # Inject a negative financial outlier (-4 sigma equivalent)
    records.append(
        {
            **records[0],
            "GCIF": "G99999",
            "Usia": 40.0,
            "Pendapatan_Tahunan": 0.0,
            "Total_Relationship_Balance": 0.0,
        }
    )

    response = client.post(
        "/analyze",
        json={"n_clusters": 3, "customers": records},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["n_clusters"] == 3
    assert body["rows_input"] == 49
    assert body["ages_imputed"] >= 1
    assert body["preserved_widowed_count"] >= 1
    assert len(body["profiles"]) == 3
    assert len(body["explained_variance_ratio"]) == 3


def test_segment_single_customer_and_validation_errors() -> None:
    """Verify POST /segment assigns valid cluster and rejects negative income with HTTP 422."""
    client = TestClient(api.app)
    # Warm up model via batch analyze so no file lookup is needed
    client.post("/analyze", json={"n_clusters": 3, "customers": _synthetic_records(45)})

    valid_customer = {
        "GCIF": "G55501",
        "Area": "Jakarta",
        "Jalur_Pembukaan": "Aplikasi Digital",
        "Vintage": "> 4 Tahun",
        "Usia": None,
        "Jenis_Kelamin": "Perempuan",
        "Status_Perkawinan": "Janda/Duda",
        "Jumlah_Anak": 2,
        "Pendidikan": "Sarjana",
        "Produk_Tabungan": 1,
        "Produk_Deposito": 1,
        "Produk_Kartu_Kredit": 0,
        "Produk_Kredit_Rumah": 0,
        "Produk_Kredit_Kendaraan": 0,
        "Produk_Kredit_Dana_Tunai": 0,
        "Pendapatan_Tahunan": 520.0,
        "Total_Relationship_Balance": 410_000_000.0,
    }
    resp = client.post("/segment", json=valid_customer)
    assert resp.status_code == 200
    result = resp.json()
    assert result["gcif"] == "G55501"
    assert result["cluster_id"] in {0, 1, 2}
    assert result["recommended_product"].startswith("Produk_")

    invalid_customer = {**valid_customer, "Pendapatan_Tahunan": -100.0}
    bad_resp = client.post("/segment", json=invalid_customer)
    assert bad_resp.status_code == 422


def test_analyze_missing_dataset_returns_404(monkeypatch: Any) -> None:
    """Verify explicit HTTP 404 when server CSV path does not exist."""
    client = TestClient(api.app)
    monkeypatch.setattr(api, "_CONFIG", replace(api._CONFIG, data_path="nonexistent_dataset_file.csv"))
    response = client.post("/analyze", json={"n_clusters": 3, "sample_limit": 100})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_segment_auto_initializes_registry_when_cold(monkeypatch: Any) -> None:
    """Verify POST /segment lazily initializes pipeline with synthetic fallback if cold."""
    cold_registry = api._ModelRegistry()
    monkeypatch.setattr(api, "_REGISTRY", cold_registry)
    monkeypatch.setattr(api, "_CONFIG", replace(api._CONFIG, data_path="absent_cold_start.csv"))

    client = TestClient(api.app)
    sample = _synthetic_records(n_samples=3)[0]
    response = client.post("/segment", json=sample)

    assert response.status_code == 200
    assert response.json()["cluster_id"] in {0, 1, 2}
    assert cold_registry.pipeline is not None


def test_global_exception_handler_masks_unhandled_exceptions(monkeypatch: Any) -> None:
    """Verify unexpected internal errors return generic 500 JSON without stack trace."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("sensitive internal trace details")

    monkeypatch.setattr(api._REGISTRY, "get_or_initialize", _boom)
    client = TestClient(api.app, raise_server_exceptions=False)
    sample = _synthetic_records(n_samples=3)[0]
    response = client.post("/segment", json=sample)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
