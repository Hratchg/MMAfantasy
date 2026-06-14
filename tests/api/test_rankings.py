"""Tests for GET /api/v1/rankings/{division} endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_rankings_lightweight(client: TestClient):
    """Lightweight rankings return sorted fighters by Elo desc."""
    response = client.get("/api/v1/rankings/Lightweight")
    assert response.status_code == 200

    data = response.json()
    assert data["division"] == "Lightweight"
    assert len(data["fighters"]) >= 1

    # Verify sorted by Elo descending
    elos = [f["elo"] for f in data["fighters"]]
    assert elos == sorted(elos, reverse=True)

    # Khabib should be ranked first (1555 Elo)
    assert data["fighters"][0]["name"] == "Khabib Nurmagomedov"


def test_rankings_with_top_param(client: TestClient):
    """Top parameter limits the number of returned fighters."""
    response = client.get("/api/v1/rankings/Lightweight?top=2")
    assert response.status_code == 200

    data = response.json()
    assert len(data["fighters"]) <= 2


def test_rankings_division_not_found(client: TestClient):
    """Non-existent division returns 404."""
    response = client.get("/api/v1/rankings/Nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_rankings_top_cap(client: TestClient):
    """Top parameter exceeding 100 returns 422 validation error."""
    response = client.get("/api/v1/rankings/Lightweight?top=200")
    assert response.status_code == 422
