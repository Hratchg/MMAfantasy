"""Tests for GET /api/v1/fighters/{name} endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_fighter_single_match(client: TestClient):
    """Single match returns full rating details with divisions."""
    response = client.get("/api/v1/fighters/Khabib")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Khabib Nurmagomedov"
    assert data["id"] is not None
    assert len(data["divisions"]) >= 1

    # Check Lightweight division is present
    lw = next(d for d in data["divisions"] if d["division"] == "Lightweight")
    assert lw["elo"] is not None
    assert lw["wins"] >= 1
    assert lw["total_fights"] >= 1


def test_fighter_multiple_matches(client: TestClient):
    """Multiple matches return 200 with count and candidate list (D-13)."""
    response = client.get("/api/v1/fighters/Silva")
    assert response.status_code == 200

    data = response.json()
    assert data["count"] >= 2
    assert len(data["results"]) >= 2

    names = [r["name"] for r in data["results"]]
    assert any("Anderson" in n for n in names)
    assert any("Antonio" in n for n in names)


def test_fighter_not_found(client: TestClient):
    """Non-existent fighter returns 404 with descriptive detail."""
    response = client.get("/api/v1/fighters/nonexistent_xyz_12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_fighter_has_domain_elo(client: TestClient):
    """Khabib's Lightweight division has striking and grappling Elo."""
    response = client.get("/api/v1/fighters/Khabib")
    assert response.status_code == 200

    data = response.json()
    lw = next(d for d in data["divisions"] if d["division"] == "Lightweight")
    assert lw["striking_elo"] is not None
    assert lw["grappling_elo"] is not None
    # Khabib: striking 1515, grappling 1535
    assert lw["striking_elo"] == 1515.0
    assert lw["grappling_elo"] == 1535.0
