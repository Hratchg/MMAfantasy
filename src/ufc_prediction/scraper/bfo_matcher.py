"""Fuzzy name matching for BFO fighter names against DB fighter records.

Pattern reused from scraper/sherdog.py::match_fighter_name (lines 344-393).
Diverges from Sherdog in two ways:
  1. Default threshold 80 (not 85) — per RESEARCH.md Pitfall 6: smaller
     candidate pool per fight (2 fighters per BFO row vs ~20 in a Sherdog
     search page) warrants looser matching.
  2. Adds normalize_name() to pre-process both sides (Saint/St, Junior/Jr)
     before fuzz.ratio scoring. Addresses the 'Saint-Preux' / 'St-Preux'
     regression called out in Pitfall 6.

Anti-pattern avoided: uses ``fuzz.ratio`` rather than the partial-ratio
variant, since substring matching is too aggressive for fighter names
('Jon Jones' ~= 'Jon Jones Jr.') per RESEARCH.md anti-patterns.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

_SAINT_RE = re.compile(r"\bsaint\b", re.IGNORECASE)
_JUNIOR_RE = re.compile(r"\bjunior\b", re.IGNORECASE)
_HYPHEN_RE = re.compile(r"[-]+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation/hyphens, collapse Saint/St and Junior/Jr.

    Addresses Pitfall 6: ``fuzz.ratio("Saint-Preux", "St-Preux") ~= 78``,
    below the default 80 threshold. Post-normalize both become
    ``st preux`` and score 100.

    The transformation chain:
      1. Lowercase
      2. 'saint' -> 'st' (word boundary)
      3. 'junior' -> 'jr' (word boundary)
      4. Hyphens -> spaces
      5. Strip remaining non-word/non-space punctuation (Unicode-aware)
      6. Collapse whitespace runs to a single space, trim edges
    """
    s = name.lower()
    s = _SAINT_RE.sub("st", s)
    s = _JUNIOR_RE.sub("jr", s)
    s = _HYPHEN_RE.sub(" ", s)
    s = _PUNCT_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def match_bfo_name(
    bfo_name: str,
    candidates: list[tuple[int, str]],
    threshold: int = 80,
) -> tuple[int, str] | None:
    """Return (fighter_id, db_name) of best fuzzy match >= threshold, else None.

    Tries direct match AND reversed-name variant ("Last, First" <-> "First Last").
    Normalizes both sides before scoring (see ``normalize_name``).

    Args:
        bfo_name: Fighter name from a BestFightOdds row.
        candidates: List of ``(fighter_id, db_name)`` tuples from our DB.
        threshold: Minimum ``fuzz.ratio`` score (0-100). Default 80 per
            RESEARCH.md Pitfall 6 — BFO candidate pools are small and the
            normalization is more aggressive than Sherdog's.

    Returns:
        The best ``(fighter_id, db_name)`` tuple if its score meets or
        exceeds ``threshold``. Otherwise ``None``.
    """
    if not candidates:
        return None

    best_score = 0.0
    best_match: tuple[int, str] | None = None

    norm_bfo = normalize_name(bfo_name)

    for fid, db_name in candidates:
        norm_db = normalize_name(db_name)
        score = fuzz.ratio(norm_bfo, norm_db)

        # Reversed candidate name fallback: DB stores "Last, First"
        if "," in db_name:
            parts = [p.strip() for p in db_name.split(",", 1)]
            rev_db = normalize_name(f"{parts[1]} {parts[0]}")
            score = max(score, fuzz.ratio(norm_bfo, rev_db))

        # Reversed BFO name fallback: BFO gives "Last, First"
        if "," in bfo_name:
            parts = [p.strip() for p in bfo_name.split(",", 1)]
            rev_bfo = normalize_name(f"{parts[1]} {parts[0]}")
            score = max(score, fuzz.ratio(rev_bfo, norm_db))

        if score > best_score:
            best_score = score
            best_match = (fid, db_name)

    if best_score >= threshold and best_match is not None:
        return best_match
    return None
