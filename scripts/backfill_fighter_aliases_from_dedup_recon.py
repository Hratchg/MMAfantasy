"""Plan 28-04 Task 2 — Backfill fighter_aliases from the 360-row no-pair-match
set surfaced by ``scripts/recon_dedup.py`` Section 3.

Discovery pass (default):
    PYTHONPATH=src python scripts/backfill_fighter_aliases_from_dedup_recon.py

  Reads the no-pair-match query, runs the six-tier matcher, writes
  ``.planning/phases/28-referee-venue-ingestion-pipeline/28-04-PROPOSED-ALIASES.csv``
  for operator review, exits with HALT notice. NO DB WRITES.

Apply pass (after operator review of CSV):
    PYTHONPATH=src python scripts/backfill_fighter_aliases_from_dedup_recon.py --apply

  Reads the reviewed CSV; for each row with ``confidence != "skip"`` AND
  ``match_tier != "ambiguous-same-card-lastname"``, INSERTs into
  ``fighter_aliases (alias_name, source, fighter_id)`` with ON CONFLICT
  (alias_name, source) DO NOTHING. Prompts ``confirm? [y/N]`` on stdin
  BEFORE ``session.commit()``.

Six-tier matcher (Tier 1-3 per Plan 28-04 ``<interfaces>``; Tier 4-6
per Path C extension closing the CHECKPOINT 1 52-row no-match gap):

  Tier 1 (high): exact last-name match + ≥1 first-name token overlap
                 between kaggle name and ufcstats candidate name (after
                 lowercase + whitespace + diacritic normalization).
  Tier 2 (med):  exact last-name match with NO first-token overlap
                 (e.g., short-form first names like "Tim" → "Timothy",
                 or initial-form like "A. Silva" → "Anderson Silva").
                 Requires same-card uniqueness (only one ufcstats fighter
                 with that last name on the event date) — otherwise drops
                 to tier-3 or ambiguous-same-card-lastname.
  Tier 3 (med):  hardcoded ``NICKNAME_REGISTRY`` lookup for nickname-style
                 names that share no surname with the canonical form
                 (Rampage → Quinton, Cro Cop → Filipovic, etc.).
  Tier 4 (med):  ``WOMENS_MARRIED_NAME_REGISTRY`` lookup — Kaggle uses
                 fight-name (often maiden / ring name); ufcstats canonical
                 reflects post-marriage / legal name. E.g., Cris Cyborg →
                 Cristiane Justino. Path C addition.
  Tier 5 (med):  ``BRAZILIAN_NICKNAME_REGISTRY`` lookup — covers Brazilian
                 fight-nicknames (Rafael Feijao → Rafael Cavalcante),
                 Brazilian "legal-name superset" variants (Glaico Franca
                 → Glaico Franca Moreira), suffix/hyphen/typo cases
                 (Waldo Cortes-Acosta → Waldo Cortes Acosta), and
                 concatenated Asian romanizations (Wuliji Buren →
                 Wulijiburen). Path C addition.
  Tier 6 (med):  algorithmic Asian name-order tolerance — surname-first
                 vs given-name-first. Match when normalized token SETs
                 are equal (permutation) AND ≥2 token overlap (avoids
                 false positives like ``John Smith`` ↮ ``Jane Smith``).
                 Path C addition.

  Same-card uniqueness gate (Tier 1/2 only): before promoting a tier-1
                 or tier-2 candidate, check the proposed ufcstats
                 fighter's last name is UNIQUE among all ufcstats
                 fighters on the event date. Tier 1 gate only fires
                 when the kaggle first-name is an INITIAL-form token
                 (length ≤ 1) — a full first-name token (e.g., "Jacare")
                 is itself the disambiguator and DOES NOT trigger the
                 gate (Path C bug fix). Tier 2 gate fires unconditionally
                 (no first-name overlap → no disambiguator available).
                 If gate fires and tier-3 fails, emit ``unresolved`` +
                 confidence=``low`` + match_tier=``ambiguous-same-card-lastname``.
                 Tier 4-6 are registry/algorithmic and bypass the gate
                 (the registry / token-set permutation is itself definitive).

  Registry global-fallback: if a Tier 4 or Tier 5 registry hit points to
                 a canonical name that is NOT in the same-card candidate
                 pool (e.g., Miranda Maverick — her first kaggle fight
                 was missing from ufcstats), the matcher resolves via a
                 global ufcstats fighter lookup with confidence=``low``
                 so the operator can spot-check.

  Residual: anything not matched by any tier returns match_tier=
                 ``"no-match"`` with confidence=``"requires-manual-review"``
                 (NOT ``"n/a"`` — distinguishes residual from same-card-
                 ambiguous which uses confidence=``"low"``). Path C update.
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from collections import defaultdict
from datetime import date as date_type
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ufc_prediction.db.session import SessionLocal


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Output CSV path (operator-reviewed before --apply)
DEFAULT_OUT_CSV = Path(
    ".planning/phases/28-referee-venue-ingestion-pipeline/28-04-PROPOSED-ALIASES.csv"
)

# Hardcoded tier-3 nickname registry. Diacritic-stripped canonical
# ufcstats names per Plan 28-04 ``<interfaces>``. Keys are kaggle-side
# nickname forms (case-sensitive — matched case-insensitively via
# ``_norm()`` at lookup time).
NICKNAME_REGISTRY: dict[str, str] = {
    "Rampage Jackson": "Quinton Jackson",
    "Mirko Cro Cop": "Mirko Filipovic",
    "Minotauro Nogueira": "Antonio Rodrigo Nogueira",
    "Lil Nog": "Rogerio Nogueira",
    "Suga Rashad": "Rashad Evans",
}


# ─────────────────────────────────────────────────────────────────────
# Tier 4 — Women's MMA married-name registry (Path C extension).
# Kaggle reflects the fighter's fight-name (often pre-marriage);
# ufcstats canonical reflects post-marriage / legal name. Seeded from
# the 52-no-match set; verified against the ufcstats DB on each
# kaggle row's event date (every value here is a confirmed
# Fighter.name with source='ufcstats').
# ─────────────────────────────────────────────────────────────────────
WOMENS_MARRIED_NAME_REGISTRY: dict[str, str] = {
    "Cris Cyborg": "Cristiane Justino",
    "Cristiane Cyborg": "Cristiane Justino",
    "Joanne Calderwood": "Joanne Wood",
    "Tecia Torres": "Tecia Pennington",
    "Michelle Waterson": "Michelle Waterson-Gomez",
    "Nina Ansaroff": "Nina Nunes",
    "Katlyn Chookagian": "Katlyn Cerminara",
    "Veronica Macedo": "Veronica Hardy",
    "Yana Kunitskaya": "Yana Santos",
    "Ariane Lipski": "Ariane da Silva",
    "Brianna Van Buren": "Brianna Fortino",
}


# ─────────────────────────────────────────────────────────────────────
# Tier 5 — Brazilian fight-nickname + suffix/typo/hyphen/concat registry
# (Path C extension). Covers four sub-patterns that all manifest as
# kaggle-side ≠ ufcstats-side surnames but DO have a confirmable
# canonical mapping verified against the ufcstats DB on the kaggle
# row's event date:
#   (a) Brazilian fight-nickname → legal name (Rafael Feijao →
#       Rafael Cavalcante; William Patolino → William Macario;
#       Patricio Freire → Patricio Pitbull).
#   (b) Brazilian / Spanish legal-name superset (kaggle drops second
#       surname): Glaico Franca → Glaico Franca Moreira; Wendell
#       Oliveira → Wendell Oliveira Marques; Alvaro Herrera → Alvaro
#       Herrera Mendoza; Humberto Brown → Humberto Brown Morrison;
#       Rodolfo Rubio → Rodolfo Rubio Perez; Vernon Ramos → Vernon
#       Ramos Ho; Montserrat Conejo → Montserrat Conejo Ruiz; Raphael
#       Pessoa Nunes → Raphael Pessoa; Omar Antonio Morales Ferrer →
#       Omar Morales.
#   (c) Suffix / hyphen / apostrophe / typo divergence: Carlo
#       Pedersoli → Carlo Pedersoli Jr.; Kai Kamaka → Kai Kamaka III;
#       Roldan Sangcha-an → Roldan Sangcha'an; Waldo Cortes-Acosta →
#       Waldo Cortes Acosta; Kai Kara France → Kai Kara-France; Ali
#       Qaisi → Ali AlQaisi; Ode Obsourne → Ode Osbourne; Youssef
#       Zalel → Youssef Zalal; Isabela De Pauda → Isabela de Padua;
#       Zhalgas Zhamagulov → Zhalgas Zhumagulov.
#   (d) Concatenated Asian romanization (no token gap): Wuliji Buren
#       → Wulijiburen; Su Mudaerji → Sumudaerji; Heili Alateng →
#       Alatengheili; Aori Qileng → Aoriqileng; Rong Zhu → Rongzhu.
#   (e) Confirmed canonical exists with same name but matcher's
#       per-date scoping missed it (registry forces resolution):
#       Jacare Souza, Uriah Hall (these would ALSO be unblocked by
#       the Tier-1 gate bug fix below; registry-seeded as defense-in-
#       depth so an apply-pass on a stale CSV still resolves them).
#   (f) Cross-org / global-fallback cases where the canonical exists
#       in the ufcstats DB but is NOT on the kaggle row's event date
#       (matcher uses the global ufcstats name→id index): Bruno
#       Rodrigues → Bruno Korea; Derrick → Derrick Krantz; Mizuki
#       Inoue → Mizuki; Leonardo Augusto Leleco → Leonardo Guimaraes;
#       Tiago Trator → Tiago dos Santos e Silva; Miranda Maverick →
#       Miranda Maverick.
# ─────────────────────────────────────────────────────────────────────
BRAZILIAN_NICKNAME_REGISTRY: dict[str, str] = {
    # (a) Brazilian fight-nickname → legal name
    "Rafael Feijao": "Rafael Cavalcante",
    "William Patolino": "William Macario",
    "Patricio Freire": "Patricio Pitbull",
    # (b) Legal-name superset (kaggle drops second surname)
    "Glaico Franca": "Glaico Franca Moreira",
    "Wendell Oliveira": "Wendell Oliveira Marques",
    "Alvaro Herrera": "Alvaro Herrera Mendoza",
    "Humberto Brown": "Humberto Brown Morrison",
    "Rodolfo Rubio": "Rodolfo Rubio Perez",
    "Vernon Ramos": "Vernon Ramos Ho",
    "Montserrat Conejo": "Montserrat Conejo Ruiz",
    "Raphael Pessoa Nunes": "Raphael Pessoa",
    "Omar Antonio Morales Ferrer": "Omar Morales",
    # (c) Suffix / hyphen / apostrophe / typo divergence
    "Carlo Pedersoli": "Carlo Pedersoli Jr.",
    "Kai Kamaka": "Kai Kamaka III",
    "Roldan Sangcha-an": "Roldan Sangcha'an",
    "Waldo Cortes-Acosta": "Waldo Cortes Acosta",
    "Kai Kara France": "Kai Kara-France",
    "Ali Qaisi": "Ali AlQaisi",
    "Ode Obsourne": "Ode Osbourne",
    "Youssef Zalel": "Youssef Zalal",
    "Isabela De Pauda": "Isabela de Padua",
    "Zhalgas Zhamagulov": "Zhalgas Zhumagulov",
    # (d) Concatenated Asian romanization (no token gap)
    "Wuliji Buren": "Wulijiburen",
    "Su Mudaerji": "Sumudaerji",
    "Heili Alateng": "Alatengheili",
    "Aori Qileng": "Aoriqileng",
    "Rong Zhu": "Rongzhu",
    # (e) Defense-in-depth seeding (also unblocked by Tier-1 gate fix)
    "Jacare Souza": "Jacare Souza",
    "Uriah Hall": "Uriah Hall",
    # (f) Global-fallback canonical (not on event-date roster)
    "Bruno Rodrigues": "Bruno Korea",
    "Derrick": "Derrick Krantz",
    "Mizuki Inoue": "Mizuki",
    "Leonardo Augusto Leleco": "Leonardo Guimaraes",
    "Tiago Trator": "Tiago dos Santos e Silva",
    "Miranda Maverick": "Miranda Maverick",
}


# ─────────────────────────────────────────────────────────────────────
# Tier 6 — Asian name-order tolerance (Path C extension; algorithmic,
# no registry). This module-level constant is exposed for test
# import-symmetry with the other registries; the algorithm itself
# lives in ``_tier6_token_set_permutation_match()``. The dict is
# empty by design — Tier 6 is rule-based, not lookup-based.
# ─────────────────────────────────────────────────────────────────────
ASIAN_NAME_REGISTRY: dict[str, str] = {}

# CSV header (operator-reviewable)
CSV_HEADER = [
    "kaggle_fighter_id",
    "kaggle_fighter_name",
    "proposed_ufcstats_fighter_id",
    "proposed_ufcstats_fighter_name",
    "match_tier",
    "evidence",
    "confidence",
]


# ─────────────────────────────────────────────────────────────────────
# Name normalization helpers
# ─────────────────────────────────────────────────────────────────────


def _strip_diacritics(s: str) -> str:
    """NFKD decomposition + drop combining marks (handles "Filipović" →
    "Filipovic"). Matches the diacritic-stripped form used in
    NICKNAME_REGISTRY values.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm(s: str) -> str:
    """Canonical name normalization: strip diacritics, lowercase, collapse
    whitespace, strip leading/trailing whitespace, strip trailing dots
    (handles initial forms like "A. Silva")."""
    s = _strip_diacritics(s).lower().strip()
    s = " ".join(s.split())  # collapse internal whitespace
    return s


def _tokens(name: str) -> list[str]:
    """Split normalized name into tokens. Initial-form tokens (single
    letter + optional dot) are returned as just the letter (e.g.,
    "a. silva" → ["a", "silva"])."""
    norm = _norm(name)
    out: list[str] = []
    for raw in norm.split():
        tok = raw.rstrip(".")
        if tok:
            out.append(tok)
    return out


def _last_name(name: str) -> str:
    toks = _tokens(name)
    return toks[-1] if toks else ""


def _first_tokens(name: str) -> set[str]:
    """All tokens EXCEPT the last (the surname)."""
    toks = _tokens(name)
    return set(toks[:-1])


# ─────────────────────────────────────────────────────────────────────
# Three-tier matcher (pure-Python; no DB)
# ─────────────────────────────────────────────────────────────────────


def _lookup_registry(kaggle_name: str, registry: dict[str, str]) -> str | None:
    """Case-insensitive normalized lookup against any registry dict.
    Returns the canonical ufcstats name string or None."""
    norm_key = _norm(kaggle_name)
    for k, v in registry.items():
        if _norm(k) == norm_key:
            return v
    return None


def _resolve_registry_canonical(
    canonical_name: str,
    ufcstats_candidates: list[tuple[str, int]],
    global_ufcstats_name_index: dict[str, int] | None,
) -> tuple[int | None, str]:
    """Resolve a registry-supplied canonical name to a fighter_id.

    Preference order:
      1. Same-card candidate pool (most defensible: opponent context lines up).
      2. Global ufcstats name→id index (defensible when the kaggle row
         lives on a date the canonical wasn't fighting — e.g., Miranda
         Maverick whose first kaggle fight was missing from ufcstats).

    Returns (matched_id_or_None, resolution_label) where resolution_label
    is one of {"same-card", "global", "missing"} so the caller can
    annotate evidence/confidence appropriately.
    """
    canonical_norm = _norm(canonical_name)
    for cname, cid in ufcstats_candidates:
        if _norm(cname) == canonical_norm:
            return cid, "same-card"
    if global_ufcstats_name_index is not None:
        gid = global_ufcstats_name_index.get(canonical_norm)
        if gid is not None:
            return gid, "global"
    return None, "missing"


def _tier6_token_set_permutation_match(
    kaggle_name: str,
    ufcstats_candidates: list[tuple[str, int]],
) -> tuple[int | None, str | None]:
    """Tier 6 — algorithmic Asian name-order tolerance.

    Rule: normalize both names; tokenize; match when the token SETS are
    EQUAL (a permutation) AND there are ≥2 tokens (so a single shared
    surname like "Smith" can't match by itself — defends against false
    positives like ``John Smith`` ↮ ``Jane Smith``).

    Returns (matched_id_or_None, matched_ufcstats_name_or_None).
    """
    k_toks = set(_tokens(kaggle_name))
    if len(k_toks) < 2:
        return None, None
    for cname, cid in ufcstats_candidates:
        c_toks = set(_tokens(cname))
        if len(c_toks) < 2:
            continue
        if k_toks == c_toks:
            return cid, cname
    return None, None


def match_kaggle_to_ufcstats(
    kaggle_name: str,
    ufcstats_candidates: list[tuple[str, int]],
    nickname_registry: dict[str, str],
    event_date: date_type,
    same_card_lastname_index: dict[date_type, dict[str, list[int]]],
    global_ufcstats_name_index: dict[str, int] | None = None,
) -> tuple[int | None, int | str | None, str, str]:
    """Match a kaggle fighter name to an ufcstats fighter_id via six
    progressive tiers + same-card uniqueness gate.

    Args:
      kaggle_name: the kaggle-side fighter name (may include diacritics,
                   nickname forms, or initial-form first names)
      ufcstats_candidates: list of (ufcstats_name, ufcstats_fighter_id)
                          candidates — typically all ufcstats fighters
                          appearing on the event_date.
      nickname_registry: dict mapping kaggle-side nickname form (case-
                         insensitive) to ufcstats canonical name (tier 3).
      event_date: the date the kaggle fight was held.
      same_card_lastname_index: dict[date, dict[normalized-lastname,
                                list[ufcstats_fighter_id]]]; used by the
                                uniqueness gate to detect ambiguous
                                same-card cases.
      global_ufcstats_name_index: optional dict[normalized-name, fighter_id]
                                spanning the entire ufcstats fighter table;
                                used by Tier 4/5 registry fallback when
                                the canonical isn't on the kaggle row's
                                event-date roster (Miranda Maverick case).

    Returns:
      (matched_id, tier, evidence, confidence) where:
        - matched_id: ufcstats fighter_id (int) or None if no match
        - tier: 1 | 2 | 3 | 4 | 5 | 6 (int) or the sentinel string
                "ambiguous-same-card-lastname" / "no-match" or None
        - evidence: human-readable reason string
        - confidence: "high" | "med" | "low" | "requires-manual-review"
    """
    kaggle_last = _last_name(kaggle_name)
    kaggle_first_toks = _first_tokens(kaggle_name)
    candidates_by_last: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for cname, cid in ufcstats_candidates:
        candidates_by_last[_last_name(cname)].append((cname, cid))

    # Same-card lastname collisions for this date (for the uniqueness gate)
    card_lastname_index = same_card_lastname_index.get(event_date, {})

    # ── Tier 1: exact last-name + first-token overlap ──────────────────
    last_matches = candidates_by_last.get(kaggle_last, [])
    tier1_hits: list[tuple[str, int]] = []
    for cname, cid in last_matches:
        c_first_toks = _first_tokens(cname)
        # First-token overlap: any token in common, OR initial-form match
        # (e.g., kaggle "A. Silva" → ufcstats "Anderson Silva": kaggle
        # first toks = {"a"}, ufcstats first toks = {"anderson"} →
        # initial-token match if "a" prefixes "anderson").
        overlap = kaggle_first_toks & c_first_toks
        initial_match = any(
            len(kt) == 1 and any(ct.startswith(kt) for ct in c_first_toks)
            for kt in kaggle_first_toks
        )
        if overlap or initial_match:
            tier1_hits.append((cname, cid))

    # Path C bug fix: the Tier-1 same-card gate over-fired whenever
    # multiple ufcstats fighters shared the kaggle surname on the card,
    # even when the kaggle first-name was a full (non-initial) token
    # that uniquely picked out a single candidate. Result: legitimate
    # tier-1 matches like ``Jacare Souza`` → ``Jacare Souza`` (id=5229,
    # 4 Souzas on card but only one Jacare) were demoted to no-match.
    #
    # Narrow fix: the gate only fires when the kaggle first-name is
    # itself ambiguous (an initial-form token, length ≤ 1). Full first
    # names ARE the disambiguator and don't need additional protection.
    # The double-Smith test (Test 7, "A. Smith") still triggers the
    # gate because kaggle_first_toks = {"a"} (initial-form).
    kaggle_has_initial_first = any(len(kt) <= 1 for kt in kaggle_first_toks)
    if len(tier1_hits) == 1:
        cname, cid = tier1_hits[0]
        same_card = card_lastname_index.get(kaggle_last, [])
        if len(same_card) > 1 and cid in same_card and kaggle_has_initial_first:
            # Kaggle first-name is an initial AND same-card collision:
            # genuine ambiguity → fall through to lower tiers / gate.
            pass
        else:
            return (
                cid,
                1,
                (f"tier1: exact last={kaggle_last!r} + first-token overlap → {cname!r} (id={cid})"),
                "high",
            )
    elif len(tier1_hits) > 1:
        # Multiple tier-1 hits → ambiguous; fall through to gate
        pass

    # ── Tier 2: exact last-name match WITHOUT first-token overlap ─────
    #   Only if the same-card uniqueness gate passes (exactly 1 ufcstats
    #   fighter with this last name on the card).
    if last_matches:
        same_card = card_lastname_index.get(kaggle_last, [])
        unique_ids_on_card = {cid for _, cid in last_matches if cid in same_card}
        if len(unique_ids_on_card) == 1:
            cname, cid = next(
                ((cn, ci) for cn, ci in last_matches if ci in unique_ids_on_card),
                (None, None),
            )
            if cid is not None:
                return (
                    cid,
                    2,
                    (
                        f"tier2: exact last={kaggle_last!r}; same-card "
                        f"uniqueness gate passed → {cname!r} (id={cid})"
                    ),
                    "med",
                )
        # If gate would have promoted a tier-2 candidate but there are >1
        # last-name collisions on the card, that's the ambiguous case —
        # fall through to tier 3 and then to the explicit ambiguous
        # sentinel below.

    # ── Tier 3: nickname registry ─────────────────────────────────────
    registry_canonical = _lookup_registry(kaggle_name, nickname_registry)
    if registry_canonical is not None:
        mid, where = _resolve_registry_canonical(
            registry_canonical,
            ufcstats_candidates,
            global_ufcstats_name_index,
        )
        if mid is not None and where == "same-card":
            return (
                mid,
                3,
                (f"tier3: nickname registry {kaggle_name!r} → {registry_canonical!r} (id={mid})"),
                "med",
            )
        if mid is not None and where == "global":
            return (
                mid,
                3,
                (
                    f"tier3: nickname registry {kaggle_name!r} → "
                    f"{registry_canonical!r} (id={mid}; global ufcstats "
                    f"lookup — canonical NOT on event-date roster)"
                ),
                "low",
            )
        return (
            None,
            3,
            (
                f"tier3: nickname registry {kaggle_name!r} → "
                f"{registry_canonical!r} (NOT in same-card candidate pool — "
                f"operator must look up canonical id manually)"
            ),
            "low",
        )

    # ── Tier 4: women's MMA married-name registry ─────────────────────
    wmma_canonical = _lookup_registry(kaggle_name, WOMENS_MARRIED_NAME_REGISTRY)
    if wmma_canonical is not None:
        mid, where = _resolve_registry_canonical(
            wmma_canonical,
            ufcstats_candidates,
            global_ufcstats_name_index,
        )
        if mid is not None and where == "same-card":
            return (
                mid,
                4,
                (
                    f"tier4: women's married-name registry lookup "
                    f"{kaggle_name!r} → {wmma_canonical!r} (id={mid})"
                ),
                "med",
            )
        if mid is not None and where == "global":
            return (
                mid,
                4,
                (
                    f"tier4: women's married-name registry {kaggle_name!r} "
                    f"→ {wmma_canonical!r} (id={mid}; global ufcstats lookup)"
                ),
                "low",
            )
        return (
            None,
            4,
            (
                f"tier4: women's married-name registry {kaggle_name!r} → "
                f"{wmma_canonical!r} (NOT in same-card pool AND not in "
                f"global ufcstats index)"
            ),
            "low",
        )

    # ── Tier 5: Brazilian fight-nickname / suffix / typo / concat ────
    brazilian_canonical = _lookup_registry(
        kaggle_name,
        BRAZILIAN_NICKNAME_REGISTRY,
    )
    if brazilian_canonical is not None:
        mid, where = _resolve_registry_canonical(
            brazilian_canonical,
            ufcstats_candidates,
            global_ufcstats_name_index,
        )
        if mid is not None and where == "same-card":
            return (
                mid,
                5,
                (
                    f"tier5: Brazilian-nickname/suffix registry "
                    f"{kaggle_name!r} → {brazilian_canonical!r} (id={mid})"
                ),
                "med",
            )
        if mid is not None and where == "global":
            return (
                mid,
                5,
                (
                    f"tier5: Brazilian-nickname/suffix registry "
                    f"{kaggle_name!r} → {brazilian_canonical!r} (id={mid}; "
                    f"global ufcstats lookup — canonical NOT on event-date roster)"
                ),
                "low",
            )
        return (
            None,
            5,
            (
                f"tier5: Brazilian-nickname/suffix registry {kaggle_name!r} "
                f"→ {brazilian_canonical!r} (NOT in same-card pool AND not "
                f"in global ufcstats index)"
            ),
            "low",
        )

    # ── Tier 6: Asian name-order token-set permutation ────────────────
    tier6_id, tier6_name = _tier6_token_set_permutation_match(
        kaggle_name,
        ufcstats_candidates,
    )
    if tier6_id is not None:
        return (
            tier6_id,
            6,
            (
                f"tier6: name-token-set permutation match "
                f"{_norm(kaggle_name)!r} ↔ {_norm(tier6_name)!r} "
                f"(id={tier6_id})"
            ),
            "med",
        )

    # ── Ambiguous-same-card-lastname sentinel ─────────────────────────
    # If we get here AND there are >1 ufcstats fighters with this last
    # name on the card AND no tier resolved, emit the explicit
    # ambiguous sentinel (confidence='low' — distinguishes from
    # genuine residual which uses 'requires-manual-review').
    if len(card_lastname_index.get(kaggle_last, [])) > 1:
        return (
            None,
            "ambiguous-same-card-lastname",
            (
                f"same-card uniqueness gate fired: {len(card_lastname_index[kaggle_last])} "
                f"ufcstats fighters surnamed {kaggle_last!r} on {event_date}; "
                f"operator must disambiguate via opponent context"
            ),
            "low",
        )

    # ── No match (residual) ──────────────────────────────────────────
    # Path C update: residual confidence is 'requires-manual-review'
    # (NOT 'n/a') so operator review can distinguish residual classes.
    # Evidence records what was tried.
    n_card_candidates = len(ufcstats_candidates)
    return (
        None,
        "no-match",
        (
            f"no-match: tried tiers 1-6 against {n_card_candidates} ufcstats "
            f"fighters on {event_date}; not in any registry; no name-token "
            f"permutation match"
        ),
        "requires-manual-review",
    )


# ─────────────────────────────────────────────────────────────────────
# Discovery — runs the recon Section 3 query + matcher + emits CSV
# ─────────────────────────────────────────────────────────────────────

# Recon Section 3 no-pair-match SQL (from: scripts/recon_dedup.py Section 3,
# lines 257-285). Inlined here per Plan 28-04 action step 3 with attribution.
_NO_PAIR_MATCH_SQL = """
WITH kf AS (
  SELECT f.id AS fid, e.source AS src, e.date AS d,
         fa.id AS faid, fa.name AS aname,
         fb.id AS fbid, fb.name AS bname,
         LOWER(TRIM(fa.name)) AS a, LOWER(TRIM(fb.name)) AS b
  FROM fights f
  JOIN events e ON f.event_id = e.id
  JOIN fighters fa ON f.fighter_a_id = fa.id
  JOIN fighters fb ON f.fighter_b_id = fb.id
  WHERE e.source LIKE 'kaggle%'
),
uf AS (
  SELECT e.date AS d,
         LOWER(TRIM(fa.name)) AS a, LOWER(TRIM(fb.name)) AS b
  FROM fights f
  JOIN events e ON f.event_id = e.id
  JOIN fighters fa ON f.fighter_a_id = fa.id
  JOIN fighters fb ON f.fighter_b_id = fb.id
  WHERE e.source = 'ufcstats'
)
SELECT kf.d, kf.faid, kf.aname, kf.fbid, kf.bname
FROM kf
LEFT JOIN uf ON uf.d = kf.d AND (
  (uf.a = kf.a AND uf.b = kf.b) OR (uf.a = kf.b AND uf.b = kf.a)
)
WHERE uf.d IS NULL
ORDER BY kf.d, kf.faid
"""


# Same-card index: for every event date that has at least one ufcstats fight,
# build the set of ufcstats fighters on the card grouped by normalized last
# name. This is the constant-time lookup the uniqueness gate uses.
_SAME_CARD_INDEX_SQL = """
SELECT e.date, f.id AS fighter_id, f.name AS fighter_name
FROM events e
JOIN fights ft ON ft.event_id = e.id
JOIN fighters f ON f.id = ft.fighter_a_id OR f.id = ft.fighter_b_id
WHERE e.source = 'ufcstats' AND f.source = 'ufcstats'
"""


def _build_same_card_lastname_index(
    session: Any,
) -> dict[date_type, dict[str, list[int]]]:
    """Build the same-card last-name index for the uniqueness gate."""
    idx: dict[date_type, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[date_type, int]] = set()
    for d, fid, fname in session.execute(text(_SAME_CARD_INDEX_SQL)).all():
        if (d, fid) in seen:
            continue
        seen.add((d, fid))
        ln = _last_name(fname)
        if ln:
            idx[d][ln].append(fid)
    return idx


# Global ufcstats name → fighter_id index. Used by Tier 4/5 registry
# fallback when the registry's canonical isn't on the kaggle row's
# event-date roster (e.g., Miranda Maverick — her first kaggle fight
# was missing from ufcstats, but her canonical fighter row exists).
_GLOBAL_UFCSTATS_NAME_SQL = """
SELECT id, name FROM fighters WHERE source = 'ufcstats'
"""


def _build_global_ufcstats_name_index(session: Any) -> dict[str, int]:
    """Build a normalized-name → fighter_id index for ALL ufcstats
    fighters in the DB. Last-wins on collision (acceptable: the
    registry tier's job is to point at a SPECIFIC canonical name, and
    if two ufcstats rows share a normalized name the operator must
    disambiguate at review)."""
    idx: dict[str, int] = {}
    for fid, fname in session.execute(text(_GLOBAL_UFCSTATS_NAME_SQL)).all():
        nname = _norm(fname)
        if nname:
            idx[nname] = fid
    return idx


def _candidates_for_date(session: Any, event_date: date_type) -> list[tuple[str, int]]:
    """Return all ufcstats fighters appearing on event_date as
    (name, fighter_id) tuples."""
    rows = session.execute(
        text("""
        SELECT DISTINCT f.name, f.id
        FROM events e
        JOIN fights ft ON ft.event_id = e.id
        JOIN fighters f ON f.id = ft.fighter_a_id OR f.id = ft.fighter_b_id
        WHERE e.source = 'ufcstats' AND f.source = 'ufcstats' AND e.date = :d
    """),
        {"d": event_date},
    ).all()
    return [(r[0], r[1]) for r in rows]


def discover(out_csv_path: Path = DEFAULT_OUT_CSV) -> dict[str, int]:
    """Discovery pass. Reads no-pair-match rows; runs matcher; emits CSV.
    No DB writes.

    Returns a dict of count summaries (handy for tests and stdout report).
    """
    with SessionLocal() as session:
        rows = session.execute(text(_NO_PAIR_MATCH_SQL)).all()
        same_card_idx = _build_same_card_lastname_index(session)
        global_ufc_idx = _build_global_ufcstats_name_index(session)

        # Per-(kaggle_fid, kaggle_name) deduplication — we want one alias
        # row per unique kaggle fighter, not one per fight.
        per_fighter: dict[tuple[int, str], dict[str, Any]] = {}
        # also remember candidate pools per date so we don't re-query
        candidates_cache: dict[date_type, list[tuple[str, int]]] = {}

        for row in rows:
            d, faid, aname, fbid, bname = row
            for fid, name in [(faid, aname), (fbid, bname)]:
                key = (fid, name)
                if key in per_fighter:
                    continue
                if d not in candidates_cache:
                    candidates_cache[d] = _candidates_for_date(session, d)
                cands = candidates_cache[d]
                matched_id, tier, evidence, confidence = match_kaggle_to_ufcstats(
                    kaggle_name=name,
                    ufcstats_candidates=cands,
                    nickname_registry=NICKNAME_REGISTRY,
                    event_date=d,
                    same_card_lastname_index=same_card_idx,
                    global_ufcstats_name_index=global_ufc_idx,
                )
                # Look up the canonical name if we matched
                proposed_name = ""
                if matched_id is not None:
                    for cn, ci in cands:
                        if ci == matched_id:
                            proposed_name = cn
                            break
                    if not proposed_name:
                        # Match returned an id NOT in same-card pool
                        # (registry global-fallback) — look up globally.
                        gn = session.execute(
                            text("SELECT name FROM fighters WHERE id = :id"), {"id": matched_id}
                        ).scalar()
                        proposed_name = gn or ""
                per_fighter[key] = {
                    "kaggle_fighter_id": fid,
                    "kaggle_fighter_name": name,
                    "proposed_ufcstats_fighter_id": (
                        matched_id if matched_id is not None else "UNRESOLVED"
                    ),
                    "proposed_ufcstats_fighter_name": proposed_name,
                    "match_tier": tier if tier is not None else "no-match",
                    "evidence": evidence,
                    "confidence": confidence,
                }
        session.rollback()  # defense-in-depth — no writes anyway

    # Sort: confidence DESC (high → med → low → requires-manual-review),
    # then match_tier ASC. 'n/a' kept for backward-compat with mocked tests.
    confidence_rank = {
        "high": 0,
        "med": 1,
        "low": 2,
        "requires-manual-review": 3,
        "n/a": 4,
    }

    def _sort_key(r: dict) -> tuple[int, str]:
        return (
            confidence_rank.get(r["confidence"], 9),
            str(r["match_tier"]),
        )

    sorted_rows = sorted(per_fighter.values(), key=_sort_key)

    # Write CSV
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in sorted_rows:
            writer.writerow(r)

    # Summaries
    tier_counts: dict[str, int] = defaultdict(int)
    conf_counts: dict[str, int] = defaultdict(int)
    for r in sorted_rows:
        tier_counts[str(r["match_tier"])] += 1
        conf_counts[r["confidence"]] += 1

    return {
        "total_rows": len(sorted_rows),
        "tier_counts": dict(tier_counts),
        "conf_counts": dict(conf_counts),
        "ambiguous_same_card": tier_counts.get("ambiguous-same-card-lastname", 0),
        "out_csv_path": str(out_csv_path),
    }


# ─────────────────────────────────────────────────────────────────────
# Apply — reads operator-reviewed CSV, inserts into fighter_aliases
# ─────────────────────────────────────────────────────────────────────


def apply_csv(csv_path: Path = DEFAULT_OUT_CSV) -> dict[str, int]:
    """Apply pass. Reads operator-reviewed CSV; inserts each row with
    ``confidence != "skip"`` AND match_tier != ambiguous into
    ``fighter_aliases``. Prompts confirm before commit.

    Returns dict of insert counts.
    """
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        return {"inserted": 0, "skipped": 0, "errors": 0}

    rows_to_apply: list[dict] = []
    skipped = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (
                r["confidence"] == "skip"
                or r["confidence"] == "requires-manual-review"
                or r["match_tier"] == "ambiguous-same-card-lastname"
                or r["proposed_ufcstats_fighter_id"] == "UNRESOLVED"
                or not r["proposed_ufcstats_fighter_id"]
            ):
                skipped += 1
                continue
            rows_to_apply.append(r)

    print(f"Loaded {len(rows_to_apply) + skipped} rows from {csv_path}")
    print(f"  to apply: {len(rows_to_apply)}")
    print(f"  skipped (confidence=skip/requires-manual-review / ambiguous / unresolved): {skipped}")

    # Determine the kaggle source label per (kaggle_fighter_id) lookup —
    # the fighter_aliases.source column must be the kaggle source
    # ('kaggle-mdabbert' or 'kaggle-rajeevw'), not 'ufcstats'.
    with SessionLocal() as session:
        # Pre-fetch sources for all kaggle fighter_ids
        kaggle_ids = [int(r["kaggle_fighter_id"]) for r in rows_to_apply]
        if kaggle_ids:
            src_rows = session.execute(
                text("SELECT id, source FROM fighters WHERE id = ANY(:ids)"), {"ids": kaggle_ids}
            ).all()
            kaggle_id_to_source = {r[0]: r[1] for r in src_rows}
        else:
            kaggle_id_to_source = {}

        # Dry-run preview
        print("\nDry-run preview (first 5):")
        for r in rows_to_apply[:5]:
            print(
                f"  alias_name={r['kaggle_fighter_name']!r} "
                f"source={kaggle_id_to_source.get(int(r['kaggle_fighter_id']), '???')!r} "
                f"→ fighter_id={r['proposed_ufcstats_fighter_id']} "
                f"({r['proposed_ufcstats_fighter_name']})"
            )

        # Interactive gate
        ans = input("\nconfirm? [y/N]: ").strip().lower()
        if ans != "y":
            print("ABORTED — no rows written.")
            session.rollback()
            return {"inserted": 0, "skipped": skipped, "aborted": 1}

        inserted = 0
        conflicts = 0
        errors = 0
        for r in rows_to_apply:
            kid = int(r["kaggle_fighter_id"])
            ufc_id = int(r["proposed_ufcstats_fighter_id"])
            alias_name = r["kaggle_fighter_name"]
            kaggle_source = kaggle_id_to_source.get(kid)
            if kaggle_source is None:
                errors += 1
                continue
            try:
                # ON CONFLICT DO NOTHING via raw SQL (SQLAlchemy ORM
                # IntegrityError-on-flush would abort the transaction).
                result = session.execute(
                    text("""
                    INSERT INTO fighter_aliases (fighter_id, alias_name, source)
                    VALUES (:fid, :alias, :src)
                    ON CONFLICT (alias_name, source) DO NOTHING
                """),
                    {"fid": ufc_id, "alias": alias_name, "src": kaggle_source},
                )
                if result.rowcount == 1:
                    inserted += 1
                else:
                    conflicts += 1
            except IntegrityError as e:
                errors += 1
                print(f"  ERROR inserting {alias_name!r}: {e}", file=sys.stderr)
                session.rollback()

        session.commit()
        print(f"\nDone. inserted={inserted} on-conflict-skipped={conflicts} errors={errors}")
        return {
            "inserted": inserted,
            "skipped": skipped,
            "conflicts": conflicts,
            "errors": errors,
        }


# ─────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover (default) or --apply operator-reviewed "
            "fighter alias mappings from the Plan 28-03 "
            "no-pair-match recon residual."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Read the operator-reviewed CSV and INSERT into fighter_aliases "
        "(with confirm prompt). Default: discovery only — no DB writes.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Path to CSV (default: {DEFAULT_OUT_CSV}).",
    )
    args = parser.parse_args()

    if args.apply:
        result = apply_csv(args.csv)
        print(f"\nApply summary: {result}")
        return 0

    # Discovery
    result = discover(args.csv)
    print(f"\nDiscovery summary:")
    print(f"  CSV written to: {result['out_csv_path']}")
    print(f"  Total rows: {result['total_rows']}")
    print(f"  By tier: {result['tier_counts']}")
    print(f"  By confidence: {result['conf_counts']}")
    print(f"  Same-card-ambiguous rows: {result['ambiguous_same_card']}")
    print(f"\nHALT — review CSV before --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
