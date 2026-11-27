"""CLI entrypoint tests covering health check and offline demo execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from main import main
from src.customer_segmentation.clustering import generate_synthetic_customers


def test_cli_health_returns_zero_and_valid_json(capsys: Any) -> None:
    """Verify --health outputs structured JSON health payload."""
    exit_code = main(["--health"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["version"] == "1.0.0"


def test_cli_demo_runs_offline_with_csv_and_synthetic_fallback(
    tmp_path: Path,
    capsys: Any,
) -> None:
    """Verify --demo works seamlessly both with local CSV and synthetic fallback."""
    csv_path = tmp_path / "customers.csv"
    generate_synthetic_customers(n_samples=90, random_state=42).to_csv(csv_path, index=False)

    exit_code = main(["--demo", "--data-path", str(csv_path), "--sample-limit", "60"])
    out_csv = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert out_csv["source"] == "csv"
    assert out_csv["rows_input"] == 60
    assert len(out_csv["profiles"]) == 3
    assert out_csv["preserved_widowed_count"] > 0

    # Fallback mode when CSV path is absent
    missing_csv = tmp_path / "absent.csv"
    exit_code_fb = main(["--demo", "--data-path", str(missing_csv), "--n-clusters", "3"])
    out_fb = json.loads(capsys.readouterr().out)

    assert exit_code_fb == 0
    assert out_fb["source"] == "synthetic"
    assert len(out_fb["profiles"]) == 3
