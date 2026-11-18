"""Unit tests for the pure helpers in scripts/refresh_corpus.py."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from scripts.refresh_corpus import (
    SEED_TABLES,
    RefreshReport,
    libpq_url,
    render_summary,
    rewrite_goldens,
    rewrite_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS = {t: 1_000_000 + i for i, t in enumerate(SEED_TABLES)}


def test_libpq_url_strips_driver() -> None:
    assert (
        libpq_url("postgresql+psycopg://ufc:ufc@localhost:5432/ufc_prediction")
        == "postgresql://ufc:ufc@localhost:5432/ufc_prediction"
    )


def test_rewrite_goldens_updates_every_table_in_the_real_file() -> None:
    source = (REPO_ROOT / "tests/integration/test_db_seed.py").read_text(encoding="utf-8")

    out = rewrite_goldens(source, COUNTS)

    for table, n in COUNTS.items():
        assert f'    "{table}": {n:_},' in out
    assert out.count("\n") == source.count("\n")


def test_rewrite_goldens_rejects_unknown_table() -> None:
    with pytest.raises(ValueError, match="no_such_table"):
        rewrite_goldens('    "fights": 1,\n', {"no_such_table": 1})


def test_rewrite_provenance_updates_counts_size_sha_and_date() -> None:
    source = (REPO_ROOT / "data/seed/PROVENANCE.md").read_text(encoding="utf-8")

    out = rewrite_provenance(
        source, COUNTS, 12_345_678, "ab" * 32, date(2026, 11, 23), date(2026, 11, 21)
    )

    for table, n in COUNTS.items():
        assert re.search(rf"^ {table}\s+\|\s+{n}$", out, flags=re.MULTILINE)
    assert "| Compressed dump size | 12,345,678 bytes (11.77 MB) |" in out
    assert f"| SHA256 | `{'ab' * 32}` |" in out
    assert out.startswith("# UFC Corpus v3.0 Seed")
    assert "**Generated:** 2026-11-23" in out
    assert "corpus current to 2026-11-21" in out


def test_render_summary_without_new_events() -> None:
    assert "No new UFCStats events" in render_summary(RefreshReport(since=date(2026, 11, 14)))


def test_render_summary_lists_events_deltas_and_gaps() -> None:
    report = RefreshReport(
        since=date(2026, 11, 14),
        new_events=[(date(2026, 11, 21), "UFC Fight Night: A vs. B", 12, 11)],
        counts_before={"fights": 17_256, "venues": 176},
        counts_after={"fights": 17_268, "venues": 176},
        unseeded_debutants=["New Guy"],
        unmatched_venues=["Somewhere, Nowhere"],
        warnings=["`ufc scrape odds` exited 1"],
        dump_sha256="cd" * 32,
        dump_bytes=2**20,
    )

    out = render_summary(report)

    assert "| 2026-11-21 | UFC Fight Night: A vs. B | 12 | 11 |" in out
    assert "| `fights` | 17,256 | 17,268 |" in out
    assert "| `venues` |" not in out
    assert "New Guy" in out
    assert "Somewhere, Nowhere" in out
    assert "`ufc scrape odds` exited 1" in out
    assert "/retrain-gate" in out
