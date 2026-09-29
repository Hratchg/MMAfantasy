"""Cross-source "twin" fight matching (ufcstats fight <-> kaggle fight).

A twin is the same bout ingested from another source: same event date and the
same two fighters by canonical name (fighter ids are per-source, so matching
must be name-based — see scripts/recon_dedup.py).
"""

from __future__ import annotations

from datetime import date

from ufc_prediction.dedup.twins import FightSides, find_twins, map_fighters_by_name

D = date(2020, 5, 16)


def _f(
    fight_id: int,
    source: str,
    a: tuple[int, str],
    b: tuple[int, str],
    d: date = D,
) -> FightSides:
    return FightSides(
        fight_id=fight_id,
        source=source,
        event_date=d,
        fighter_a_id=a[0],
        fighter_a_name=a[1],
        fighter_b_id=b[0],
        fighter_b_name=b[1],
    )


UFC = _f(1, "ufcstats", (10, "Georges St-Pierre"), (11, "B.J. Penn"))


def test_twin_matches_on_date_and_canonical_names_in_either_order() -> None:
    k = _f(2, "kaggle-rajeevw", (20, "BJ Penn"), (21, "Georges Saint Pierre"))
    match = find_twins([UFC], [k])
    assert match.twins == {1: [k]}


def test_different_date_or_opponent_is_not_a_twin() -> None:
    other_day = _f(
        2, "kaggle-rajeevw", (20, "BJ Penn"), (21, "Georges St-Pierre"), date(2020, 5, 17)
    )
    other_opp = _f(3, "kaggle-rajeevw", (22, "Matt Hughes"), (21, "Georges St-Pierre"))
    match = find_twins([UFC], [other_day, other_opp])
    assert match.twins == {1: []}


def test_every_candidate_twin_is_returned_so_callers_can_refuse_ambiguity() -> None:
    k1 = _f(2, "kaggle-rajeevw", (20, "BJ Penn"), (21, "Georges St-Pierre"))
    k2 = _f(3, "kaggle-mdabbert", (30, "Georges St-Pierre"), (31, "BJ Penn"))
    match = find_twins([UFC], [k1, k2])
    assert match.twins[1] == [k1, k2]


def test_targets_sharing_a_key_are_flagged_duplicate() -> None:
    dup = _f(9, "ufcstats", (12, "BJ Penn"), (13, "Georges St Pierre"))
    k = _f(2, "kaggle-rajeevw", (20, "BJ Penn"), (21, "Georges St-Pierre"))
    match = find_twins([UFC, dup], [k])
    assert match.duplicate_targets == {1, 9}
    assert match.twins == {1: [], 9: []}


def test_unkeyable_target_same_canonical_names() -> None:
    same = _f(5, "ufcstats", (14, "Jon Jones"), (15, "jon jones"))
    match = find_twins([same], [])
    assert match.unkeyable == {5}
    assert match.twins == {5: []}


def test_map_fighters_by_name() -> None:
    k = _f(2, "kaggle-rajeevw", (20, "BJ Penn"), (21, "Georges Saint-Pierre"))
    assert map_fighters_by_name(UFC, k) == {20: 11, 21: 10}


def test_map_fighters_by_name_refuses_when_names_do_not_pair_up() -> None:
    k = _f(2, "kaggle-rajeevw", (20, "BJ Penn"), (21, "Matt Hughes"))
    assert map_fighters_by_name(UFC, k) is None
    same = _f(3, "kaggle-rajeevw", (20, "BJ Penn"), (21, "B.J. Penn"))
    assert map_fighters_by_name(UFC, same) is None
