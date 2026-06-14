"""Tests for GET /api/v1/fighters/{name}/history endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_history_success(client: TestClient):
    """Khabib's Elo history returns chronological snapshots."""
    response = client.get("/api/v1/fighters/Khabib/history")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Khabib Nurmagomedov"
    assert data["fighter_id"] is not None
    assert len(data["history"]) == 2  # Two overall snapshots for Khabib

    # Verify chronological order (asc by fight_date)
    dates = [h["fight_date"] for h in data["history"]]
    assert dates == sorted(dates)

    # First snapshot: 2018-10-06
    assert data["history"][0]["fight_date"] == "2018-10-06"
    assert data["history"][0]["elo_after_shrinkage"] == 1525.0

    # Second snapshot: 2019-09-07
    assert data["history"][1]["fight_date"] == "2019-09-07"
    assert data["history"][1]["elo_after_shrinkage"] == 1555.0


def test_history_not_found(client: TestClient):
    """Non-existent fighter returns 404."""
    response = client.get("/api/v1/fighters/nonexistent_xyz_12345/history")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
