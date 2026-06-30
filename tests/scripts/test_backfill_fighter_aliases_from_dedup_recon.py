"""Plan 28-04 Task 2a — unit tests for the matcher in
``scripts/backfill_fighter_aliases_from_dedup_recon.py``.

Covers the original seven behaviors (Tier 1-3 + same-card gate) PLUS the
Path C extension (Tiers 4-6) added to close the CHECKPOINT 1 52-row
no-match gap:

  Original (Tier 1-3):
   1. Tier 1 high-confidence exact-name match
   2. Tier 1 / Tier 2 lowercase / diacritic normalization (DOES match)
   3. Tier 3 nickname registry (Rampage Jackson → Quinton Jackson)
   4. Tier 3 nickname registry (Mirko Cro Cop → Mirko Filipovic)
   5. No-match returns (None, None, "no-match", "n/a")
   6. Discovery pass without ``--apply`` makes ZERO write calls (mocked DB)
   7. **Same-card uniqueness gate** (double-Smith fixture) — ambiguous
      matches route to ``unresolved`` with confidence=``low`` +
      match_tier=``ambiguous-same-card-lastname`` (NOT a tier-1
      false-positive)
   8. NICKNAME_REGISTRY (tier-3) documented hops present.

  Path C extension (Tier 4-6):
   9.  Tier 4 women's married-name (Cris Cyborg → Cristiane Justino).
   10. Tier 5 Brazilian fight-nickname (Rafael Feijao → Rafael Cavalcante).
   11. Tier 6 Asian name-token-set permutation (Tiequan Zhang → Zhang
       Tiequan) AND negative ("John Smith" ↮ "Jane Smith" — must NOT
       match at tier 6).
   12. Residual path: kaggle name not in any registry and no candidates
       on the date → match_tier='no-match' AND confidence=
       'requires-manual-review' (NOT 'n/a' — distinguishes residual from
       same-card-ambiguous).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from scripts.backfill_fighter_aliases_from_dedup_recon import (
    ASIAN_NAME_REGISTRY,
    BRAZILIAN_NICKNAME_REGISTRY,
    NICKNAME_REGISTRY,
    WOMENS_MARRIED_NAME_REGISTRY,
    discover,
    match_kaggle_to_ufcstats,
)


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Tier 1: exact-last + first-token overlap → "high"
# ─────────────────────────────────────────────────────────────────────


def test_tier1_exact_lastname_with_first_token_overlap():
    """``Bobby Green`` (kaggle) → ``Bobby Green`` (ufcstats) when present.

    The kaggle-mdabbert ingestion path created separate ``fighters`` rows
    for the same person — so the candidate pool DOES include an
    identically-named ufcstats fighter for many cases. Tier 1 catches
    these immediately.
    """
    candidates = [
        ("Bobby Green", 5176),  # the actual ufcstats canonical (per recon)
        ("King Green", 5177),  # red herring: same last, no first overlap
        ("Maurice Greene", 5991),
    ]
    same_card_idx = {date(2099, 1, 1): {"green": [5176, 5177]}}  # 2 Greens
    # When the same-card index has >1 last-name match, the gate forces
    # ambiguity — UNLESS the matcher has unambiguous evidence. For this
    # test we relax: a non-conflicting date (no same-card collision).
    same_card_idx_clean = {date(2099, 1, 1): {"green": [5176]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="Bobby Green",
        ufcstats_candidates=candidates,
        nickname_registry={},
        event_date=date(2099, 1, 1),
        same_card_lastname_index=same_card_idx_clean,
    )
    matched_id, tier, evidence, confidence = result
    assert matched_id == 5176
    assert tier == 1
    assert confidence == "high"
    assert "bobby" in evidence.lower() or "green" in evidence.lower()


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Tier 1 / 2: case + whitespace normalization
# ─────────────────────────────────────────────────────────────────────


def test_normalization_lowercase_and_whitespace():
    """``  Junior Dos Santos  `` (kaggle, padded) vs ``Junior dos Santos``
    (ufcstats, mixed case) → tier 1 match via case + whitespace normalization.
    """
    candidates = [("Junior dos Santos", 5050), ("Frank Mir", 2129)]
    same_card_idx = {date(2010, 11, 13): {"santos": [5050], "mir": [2129]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="  Junior Dos Santos  ",
        ufcstats_candidates=candidates,
        nickname_registry={},
        event_date=date(2010, 11, 13),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, _evidence, confidence = result
    assert matched_id == 5050
    assert tier == 1
    assert confidence == "high"


# ─────────────────────────────────────────────────────────────────────
# Test 3 — Tier 3 nickname registry: Rampage → Quinton
# ─────────────────────────────────────────────────────────────────────


def test_tier3_nickname_registry_rampage_jackson():
    """``Rampage Jackson`` (kaggle, nickname form) → ``Quinton Jackson``
    (ufcstats canonical) via NICKNAME_REGISTRY tier 3 lookup.

    Tier 1 must fail (no first-name token overlap: ``rampage`` ∉
    ``quinton``). Tier 2 must reject (last-name ``jackson`` is too common —
    multiple ufcstats Jacksons exist in the candidate pool, so the
    same-card gate makes tier 2 ambiguous). Tier 3 resolves via the
    explicit registry hop.
    """
    candidates = [
        ("Quinton Jackson", 4615),
        ("Kevin Jackson", 4256),
        ("Eugene Jackson", 4321),
        ("Jeremy Jackson", 4441),
    ]
    # 4 Jacksons on the same card → tier 1/2 gate forces ambiguity
    same_card_idx = {date(2010, 5, 29): {"jackson": [4615, 4256, 4321, 4441]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="Rampage Jackson",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2010, 5, 29),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, _evidence, confidence = result
    assert matched_id == 4615
    assert tier == 3
    assert confidence == "med"


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Tier 3 nickname registry: Mirko Cro Cop → Mirko Filipovic
# ─────────────────────────────────────────────────────────────────────


def test_tier3_nickname_registry_mirko_cro_cop():
    """``Mirko Cro Cop`` (kaggle) → ``Mirko Filipovic`` (ufcstats) via
    NICKNAME_REGISTRY. The last name ``Cop`` does not exist in ufcstats
    as a fighter surname (Tier 1 + Tier 2 both fail by construction),
    so Tier 3 is the only resolution path.
    """
    candidates = [
        ("Mirko Filipovic", 6789),
        ("Chris Cope", 4978),  # red herring with similar surname
        ("Kit Cope", 4511),
    ]
    same_card_idx = {date(2010, 6, 12): {"filipovic": [6789]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="Mirko Cro Cop",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2010, 6, 12),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, _evidence, confidence = result
    assert matched_id == 6789
    assert tier == 3
    assert confidence == "med"


# ─────────────────────────────────────────────────────────────────────
# Test 5 — no-match
# ─────────────────────────────────────────────────────────────────────


def test_no_match_for_wholly_unique_name():
    """``John WhollyUniqueName XYZ`` against unrelated candidates returns
    ``(None, "no-match", evidence, "requires-manual-review")``.

    Path C update (CHECKPOINT 1'): residual confidence flipped from
    ``"n/a"`` to ``"requires-manual-review"`` so that operator review
    can distinguish three residual classes:
      - ``ambiguous-same-card-lastname`` (confidence=low) — gate fired
      - ``no-match`` (confidence=requires-manual-review) — Tiers 1-6 all
        tried and exhausted; operator must decide
    The match_tier is still the sentinel ``"no-match"`` (per plan spec).
    """
    candidates = [
        ("Conor McGregor", 4567),
        ("Khabib Nurmagomedov", 4890),
    ]
    same_card_idx = {date(2020, 1, 1): {"mcgregor": [4567], "nurmagomedov": [4890]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="John WhollyUniqueName XYZ",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2020, 1, 1),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, evidence, confidence = result
    assert matched_id is None
    assert tier == "no-match"
    assert "no-match" in evidence.lower() or "no candidates" in evidence.lower()
    assert confidence == "requires-manual-review"


# ─────────────────────────────────────────────────────────────────────
# Test 6 — discovery without --apply: zero write calls
# ─────────────────────────────────────────────────────────────────────


def test_discover_makes_no_db_writes(tmp_path):
    """``discover()`` MUST NOT open a write transaction. We patch
    ``SessionLocal`` at the script's import site and assert no ``.add`` /
    ``.commit`` calls were made.
    """
    mock_session = MagicMock()
    mock_session.execute.return_value.all.return_value = []  # empty result
    mock_session_local = MagicMock()
    mock_session_local.return_value.__enter__.return_value = mock_session
    mock_session_local.return_value.__exit__.return_value = None

    out_csv = tmp_path / "test-aliases.csv"
    with patch(
        "scripts.backfill_fighter_aliases_from_dedup_recon.SessionLocal",
        mock_session_local,
    ):
        discover(out_csv_path=out_csv)

    # discover() reads but does not write
    assert mock_session.add.call_count == 0
    assert mock_session.commit.call_count == 0
    # CSV file should exist (header at minimum) even if empty result set
    assert out_csv.exists()


# ─────────────────────────────────────────────────────────────────────
# Test 7 — same-card uniqueness gate (double-Smith fixture)
# ─────────────────────────────────────────────────────────────────────


def test_same_card_lastname_gate_double_smith():
    """**Synthetic double-Smith fixture** (mandatory per plan).

    Two ufcstats fighters surnamed ``Smith`` on event date 2099-01-01
    (Alice Smith, fid=8001; Carol Smith, fid=8002). A kaggle row ``A. Smith``
    on the same date MUST route to ``unresolved`` with confidence=``low``
    and match_tier=``ambiguous-same-card-lastname`` — NOT a tier-1
    false-positive against either Smith.

    The matcher cannot disambiguate without opponent context (Alice's
    opponent vs Carol's opponent), so the gate fires.
    """
    candidates = [
        ("Alice Smith", 8001),
        ("Carol Smith", 8002),
        ("Bob Jones", 8003),
        ("Dave Brown", 8004),
    ]
    # Two Smiths on the same date → uniqueness gate must fire
    same_card_idx = {date(2099, 1, 1): {"smith": [8001, 8002]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="A. Smith",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2099, 1, 1),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, evidence, confidence = result
    assert matched_id is None, (
        f"matcher MUST NOT promote a candidate when two Smiths are on the "
        f"same card; got id={matched_id}"
    )
    assert confidence == "low"
    # The plan specifies the match_tier sentinel string explicitly
    assert "ambiguous" in str(tier).lower() or tier == "ambiguous-same-card-lastname"
    assert "ambiguous" in evidence.lower() or "same-card" in evidence.lower()


# ─────────────────────────────────────────────────────────────────────
# Test 8 (bonus sanity) — NICKNAME_REGISTRY contains the documented hops
# ─────────────────────────────────────────────────────────────────────


def test_nickname_registry_documented_hops_present():
    """Regression sentinel: the five documented hops in the plan's
    ``<interfaces>`` block (Rampage, Cro Cop, Minotauro, Lil Nog,
    Suga Rashad) MUST be present in NICKNAME_REGISTRY.
    """
    required = {
        "Rampage Jackson": "Quinton Jackson",
        "Mirko Cro Cop": "Mirko Filipovic",
        "Minotauro Nogueira": "Antonio Rodrigo Nogueira",
        "Lil Nog": "Rogerio Nogueira",
        "Suga Rashad": "Rashad Evans",
    }
    for k, v in required.items():
        assert k in NICKNAME_REGISTRY, f"missing nickname key: {k!r}"
        assert NICKNAME_REGISTRY[k] == v, (
            f"NICKNAME_REGISTRY[{k!r}] = {NICKNAME_REGISTRY[k]!r}; expected {v!r}"
        )


# ═════════════════════════════════════════════════════════════════════
# Path C extension — Tiers 4, 5, 6 + residual + bug-fix regression
# ═════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Test 9 — Tier 4: women's married-name registry
# ─────────────────────────────────────────────────────────────────────


def test_tier4_womens_married_name_cris_cyborg():
    """``Cris Cyborg`` (kaggle ring-name) → ``Cristiane Justino`` (ufcstats
    post-marriage canonical) via WOMENS_MARRIED_NAME_REGISTRY tier 4.

    Tier 1 fails (no last-name overlap: 'cyborg' ∉ 'justino'). Tier 2
    fails (no 'cyborg' candidates on the card). Tier 3 also fails
    (NICKNAME_REGISTRY does not include WMMA cases). Tier 4 resolves.
    """
    candidates = [
        ("Cristiane Justino", 9001),
        ("Demian Maia", 9002),
        ("Stipe Miocic", 9003),
    ]
    same_card_idx = {date(2016, 5, 14): {"justino": [9001], "maia": [9002], "miocic": [9003]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="Cris Cyborg",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2016, 5, 14),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, evidence, confidence = result
    assert matched_id == 9001
    assert tier == 4
    assert confidence == "med"
    assert "tier4" in evidence.lower() or "married" in evidence.lower()


def test_tier4_registry_documented_seeds_present():
    """Regression sentinel: the seed WMMA hops MUST be present in
    WOMENS_MARRIED_NAME_REGISTRY (case-insensitive keys; values are
    diacritic-stripped ufcstats canonical forms).
    """
    required_keys = [
        "cris cyborg",
        "joanne calderwood",
        "tecia torres",
        "michelle waterson",
        "nina ansaroff",
        "katlyn chookagian",
    ]
    norm_keys = {k.lower(): v for k, v in WOMENS_MARRIED_NAME_REGISTRY.items()}
    for k in required_keys:
        assert k in norm_keys, f"missing WMMA registry key: {k!r}"


# ─────────────────────────────────────────────────────────────────────
# Test 10 — Tier 5: Brazilian fight-nickname registry
# ─────────────────────────────────────────────────────────────────────


def test_tier5_brazilian_nickname_rafael_feijao():
    """``Rafael Feijao`` (kaggle ring-name) → ``Rafael Cavalcante``
    (ufcstats legal-name) via BRAZILIAN_NICKNAME_REGISTRY tier 5.

    Tier 1 fails (last-name 'feijao' ∉ candidates). Tier 4 fails (not
    a WMMA case). Tier 5 resolves.
    """
    candidates = [
        ("Rafael Cavalcante", 9101),
        ("Fabricio Werdum", 9102),
        ("William Macario", 9103),
    ]
    same_card_idx = {date(2013, 6, 8): {"cavalcante": [9101], "werdum": [9102], "macario": [9103]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="Rafael Feijao",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2013, 6, 8),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, evidence, confidence = result
    assert matched_id == 9101
    assert tier == 5
    assert confidence == "med"
    assert "tier5" in evidence.lower() or "brazilian" in evidence.lower()


def test_tier5_registry_documented_seeds_present():
    """Regression sentinel: the seed Brazilian-nickname hops MUST be
    present in BRAZILIAN_NICKNAME_REGISTRY."""
    required_keys = [
        "rafael feijao",
        "william patolino",
    ]
    norm_keys = {k.lower(): v for k, v in BRAZILIAN_NICKNAME_REGISTRY.items()}
    for k in required_keys:
        assert k in norm_keys, f"missing Brazilian-nickname registry key: {k!r}"


# ─────────────────────────────────────────────────────────────────────
# Test 11 — Tier 6: Asian name-token-set permutation (positive + negative)
# ─────────────────────────────────────────────────────────────────────


def test_tier6_asian_name_token_permutation_positive():
    """``Tiequan Zhang`` (kaggle given-first) → ``Zhang Tiequan`` (ufcstats
    surname-first) via Tier 6 token-set permutation match.

    Tokens {'tiequan', 'zhang'} == {'zhang', 'tiequan'} → match.
    """
    candidates = [
        ("Zhang Tiequan", 9201),
        ("Michael Bisping", 9202),
        ("BJ Penn", 9203),
    ]
    same_card_idx = {date(2011, 2, 26): {"tiequan": [9201], "bisping": [9202], "penn": [9203]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="Tiequan Zhang",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2011, 2, 26),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, evidence, confidence = result
    assert matched_id == 9201
    assert tier == 6
    assert confidence == "med"
    assert "tier6" in evidence.lower() or "permutation" in evidence.lower()


def test_tier6_asian_name_negative_does_not_match_different_first_name():
    """**Negative case (mandatory per plan):** ``John Smith`` (kaggle)
    MUST NOT match ``Jane Smith`` (ufcstats) at tier 6.

    Token sets {'john', 'smith'} vs {'jane', 'smith'} overlap by 1 token
    only ('smith'); the tier 6 rule requires that the token SET is a
    permutation (i.e., equal sets). 1 of 2 overlap ≠ permutation.

    Furthermore, the kaggle row should fall through to ambiguous-same-card
    (if multiple Smiths on card) OR no-match (if not) — either way, NOT
    tier 6.
    """
    candidates = [
        ("Jane Smith", 9301),
        ("Bob Jones", 9302),
    ]
    same_card_idx = {date(2099, 1, 1): {"smith": [9301], "jones": [9302]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="John Smith",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2099, 1, 1),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, _evidence, _confidence = result
    assert tier != 6, (
        f"Tier 6 MUST NOT fire on {'{john,smith}'} vs {'{jane,smith}'} — got tier={tier}"
    )
    # Either no match or a non-tier-6 resolution. Acceptable terminations:
    #   - tier 1/2 (if first-token gate matches — for 'smith' the full
    #     first-name 'john' is not in {'jane'} so tier 1 fails) → falls
    #     through eventually to residual.
    # Definitive assertion: result is NOT tier 6.


# ─────────────────────────────────────────────────────────────────────
# Test 12 — Residual: requires-manual-review confidence
# ─────────────────────────────────────────────────────────────────────


def test_residual_unresolved_uses_requires_manual_review_confidence():
    """Per Path C spec: anything not matched by Tiers 1-6 gets
    ``match_tier='no-match'`` AND ``confidence='requires-manual-review'``
    (NOT ``'n/a'``) — distinguishes from ``ambiguous-same-card-lastname``
    (confidence=``'low'``)."""
    candidates = []  # empty candidate pool — guaranteed no-match
    same_card_idx = {date(2025, 12, 31): {}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="Unknown Person",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2025, 12, 31),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, evidence, confidence = result
    assert matched_id is None
    assert tier == "no-match"
    assert confidence == "requires-manual-review"
    assert "no-match" in evidence.lower() or "no candidates" in evidence.lower()


# ─────────────────────────────────────────────────────────────────────
# Test 13 — Bug-fix regression: Tier-1 gate must NOT over-fire when
#           kaggle first-name is full (not an initial) and uniquely
#           identifies the candidate among same-surname collisions
# ─────────────────────────────────────────────────────────────────────


def test_tier1_gate_does_not_over_fire_on_unique_full_first_name():
    """**Bug fix regression (Rule 1 — surfaced during Path C):**

    The original Tier-1 same-card gate demoted to no-match whenever
    multiple ufcstats fighters shared the kaggle surname on the card,
    even if the kaggle first-name was a FULL (non-initial) token that
    uniquely picked out a single candidate. This caused legitimate
    tier-1 matches like ``Jacare Souza`` → ``Jacare Souza`` to fail
    (5 Souzas on the card; only one is named Jacare).

    Path C fix: the gate must only fire when the kaggle first-name is
    an INITIAL-form token (length ≤ 1) — i.e., truly ambiguous. Full
    first-name tokens are themselves the disambiguator.

    This regression test:
      - 4 Souzas on the card (Jacare, Edimilson, Wendell-with-different-
        last, etc.)
      - Kaggle row: ``Jacare Souza`` (full first name)
      - Expected: tier 1 resolves to id=5229 (the Jacare Souza)
      - Must NOT route to ambiguous-same-card-lastname.

    The original double-Smith test (Test 7) still passes because in
    that case the kaggle first-name IS an initial (``A.``).
    """
    candidates = [
        ("Jacare Souza", 5229),
        ("Edimilson Souza", 5263),
        ("Ketlen Souza", 6524),
        ("Livinha Souza", 5964),
    ]
    # 4 Souzas on the same card
    same_card_idx = {date(2013, 5, 18): {"souza": [5229, 5263, 6524, 5964]}}
    result = match_kaggle_to_ufcstats(
        kaggle_name="Jacare Souza",
        ufcstats_candidates=candidates,
        nickname_registry=NICKNAME_REGISTRY,
        event_date=date(2013, 5, 18),
        same_card_lastname_index=same_card_idx,
    )
    matched_id, tier, _evidence, confidence = result
    assert matched_id == 5229, (
        f"Tier-1 gate over-fired: expected id=5229 (Jacare Souza), got id={matched_id}, tier={tier}"
    )
    assert tier == 1
    assert confidence == "high"
