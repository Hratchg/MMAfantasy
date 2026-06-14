"""Tests for GET /api/v1/matchup endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_matchup_success(client_matchup: TestClient):
    """Successful matchup returns all comparison sections."""
    response = client_matchup.get("/api/v1/matchup?a=Khabib&b=Conor")
    assert response.status_code == 200

    data = response.json()
    assert data["fighter_a_name"] == "Khabib Nurmagomedov"
    assert data["fighter_b_name"] == "Conor McGregor"

    # Verify all sections are present
    assert "physical" in data
    assert "elo" in data
    assert "striking" in data
    assert "grappling" in data
    assert "style" in data

    # Verify Elo comparison has values
    assert data["elo"]["fighter_a_overall"] is not None
    assert data["elo"]["fighter_b_overall"] is not None


def test_matchup_fighter_not_found(client_matchup: TestClient):
    """Missing fighter returns 404."""
    response = client_matchup.get("/api/v1/matchup?a=Khabib&b=nonexistent_xyz_12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_matchup_same_fighter(client_matchup: TestClient):
    """Comparing a fighter to themselves returns 400."""
    response = client_matchup.get("/api/v1/matchup?a=Khabib&b=Khabib")
    assert response.status_code == 400
    assert "themselves" in response.json()["detail"].lower()
