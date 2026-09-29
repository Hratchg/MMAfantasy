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


# ── S09 regression tests ────────────────────────────────────────────────────


def test_exact_name_with_cross_source_twin_returns_rating(
    client: TestClient, cross_source_duplicates
):
    """Finding 1: a ufcstats + Kaggle twin pair resolves to the canonical row."""
    khabib = cross_source_duplicates["canonical"]["khabib"]
    response = client.get("/api/v1/fighters/Khabib Nurmagomedov")
    assert response.status_code == 200
    data = response.json()
    assert "divisions" in data, f"expected a rating response, got {data}"
    assert data["id"] == khabib.id


def test_substring_with_cross_source_twins_lists_each_person_once(
    client: TestClient, cross_source_duplicates
):
    """Finding 1: the candidate list holds canonical rows only."""
    canonical = cross_source_duplicates["canonical"]
    response = client.get("/api/v1/fighters/Silva")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert {r["id"] for r in data["results"]} == {
        canonical["anderson"].id,
        canonical["antonio"].id,
    }
    anderson = next(r for r in data["results"] if r["id"] == canonical["anderson"].id)
    assert anderson["elo"] == 1515.0
    assert anderson["division"] == "Middleweight"


def test_like_wildcards_are_literal(client: TestClient):
    """Finding 3: '%' and '_' no longer match every fighter."""
    assert client.get("/api/v1/fighters/%25").status_code == 404
    assert client.get("/api/v1/fighters/_").status_code == 404


def test_candidate_list_is_capped_and_batched(client: TestClient, session):
    """Finding 3: a broad query returns at most 25 candidates without a
    per-candidate query fan-out."""
    from sqlalchemy import event

    from ufc_prediction.models.fighter import Fighter

    session.add_all([Fighter(name=f"Broad Match {i:02d}", source="ufcstats") for i in range(40)])
    session.flush()

    statements: list[str] = []

    def _count(conn, cursor, statement, *args):
        statements.append(statement)

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", _count)
    try:
        response = client.get("/api/v1/fighters/Broad")
    finally:
        event.remove(engine, "before_cursor_execute", _count)

    assert response.status_code == 200
    assert response.json()["count"] == 25
    # Previously 1 search + 4 queries per candidate (161 for 40 candidates).
    assert len(statements) <= 5, statements
