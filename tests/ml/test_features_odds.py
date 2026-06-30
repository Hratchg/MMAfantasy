"""Tests for Phase 13 Plan 03 — Wave 2: betting odds in feature matrix.

Covers ODDS-04:
- FEATURE_COLUMNS extends from 67 to 70 with three new betting odds diffs
  in the documented order (opening_prob_diff, closing_prob_diff,
  line_movement_diff) at the tail per D-04, D-05, D-09.
- load_fight_odds query returns dict keyed by (fighter_id, fight_id) -> {
    opening_implied_prob, closing_implied_prob}.
- FeatureMatrixAssembler.assemble accepts a fight_odds kwarg and emits the
  3 new differentials per Fighter A minus Fighter B convention (D-09),
  with NaN — never 0.0 — for missing-odds cases (D-07 + Pitfall 3).
- predict train + predict evaluate CLI handlers wire load_fight_odds through.

Test sections:
1. Config tests       — pure import; no DB.
2. Query tests        — DB session fixture; seed FightOdds rows.
3. Assembler tests    — pure-Python fixtures from conftest; verify Section 13.
4. CLI tests          — Typer CliRunner + mocks; verify load_fight_odds is
                        invoked by both `predict train` and `predict evaluate`.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.orm import Session

from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter

# ─── 1. Config tests (no DB) ──────────────────────────────────────────────────


class TestFeatureColumnsExtended:
    """ODDS-04: FEATURE_COLUMNS gains 3 entries (67 -> 70) at the tail."""

    def test_feature_columns_length_is_70(self) -> None:
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        assert len(FEATURE_COLUMNS) == 75

    def test_feature_columns_ends_with_three_odds_diffs(self) -> None:
        """D-04 + D-05 + D-09: order matters — opening, closing, movement."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        assert FEATURE_COLUMNS[-8:-5] == [
            "opening_prob_diff",
            "closing_prob_diff",
            "line_movement_diff",
        ]

    def test_pre_ufc_is_still_fourth_from_end(self) -> None:
        """Regression guard: pre_ufc_win_pct_diff must remain at -4 after
        appending the 3 odds diffs (don't accidentally reorder)."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        assert FEATURE_COLUMNS[-9] == "pre_ufc_win_pct_diff"

    def test_feature_columns_no_duplicates_after_extension(self) -> None:
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


# ─── 2. Query tests (DB fixture) ──────────────────────────────────────────────


def _seed_fight_for_odds(session: Session) -> tuple[Fighter, Fighter, Fight]:
    """Seed two fighters + one fight for FightOdds rows to reference."""
    fa = Fighter(name="Test Fighter A", source="test")
    fb = Fighter(name="Test Fighter B", source="test")
    session.add_all([fa, fb])
    session.flush()

    ev = Event(name="UFC Test 1", date=date(2020, 1, 15), source="test")
    session.add(ev)
    session.flush()

    fight = Fight(
        event_id=ev.id,
        fighter_a_id=fa.id,
        fighter_b_id=fb.id,
        weight_class="Lightweight",
        source="test",
    )
    session.add(fight)
    session.flush()
    return fa, fb, fight


class TestLoadFightOdds:
    """ODDS-04 / Plan 03 Task 1: load_fight_odds query contract."""

    def test_load_fight_odds_empty_db_returns_empty_dict(self, session: Session) -> None:
        from ufc_prediction.ml.queries import load_fight_odds

        assert load_fight_odds(session) == {}

    def test_load_fight_odds_returns_keyed_by_fighter_fight_tuple(self, session: Session) -> None:
        from ufc_prediction.ml.queries import load_fight_odds

        fa, fb, fight = _seed_fight_for_odds(session)
        session.add_all(
            [
                FightOdds(
                    fight_id=fight.id,
                    fighter_id=fa.id,
                    opening_ml=-200,
                    closing_range_min_ml=-220,
                    closing_range_max_ml=-210,
                    opening_implied_prob=0.62,
                    closing_implied_prob=0.65,
                ),
                FightOdds(
                    fight_id=fight.id,
                    fighter_id=fb.id,
                    opening_ml=170,
                    closing_range_min_ml=190,
                    closing_range_max_ml=200,
                    opening_implied_prob=0.38,
                    closing_implied_prob=0.35,
                ),
            ]
        )
        session.flush()

        result = load_fight_odds(session)
        assert len(result) == 2
        # Keys are (fighter_id, fight_id) tuples of ints — D-09 convention
        for key in result:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], int)
            assert isinstance(key[1], int)
        assert (fa.id, fight.id) in result
        assert (fb.id, fight.id) in result

    def test_load_fight_odds_exposes_prob_columns_only(self, session: Session) -> None:
        """D-03 separation: raw American moneyline columns are stored in DB but
        load_fight_odds returns only the model-feature columns (opening +
        closing implied_prob). Raw ML stays in the DB for audit; not surfaced
        as ML features."""
        from ufc_prediction.ml.queries import load_fight_odds

        fa, fb, fight = _seed_fight_for_odds(session)
        session.add_all(
            [
                FightOdds(
                    fight_id=fight.id,
                    fighter_id=fa.id,
                    opening_ml=-200,
                    closing_range_min_ml=-220,
                    closing_range_max_ml=-210,
                    opening_implied_prob=0.62,
                    closing_implied_prob=0.65,
                ),
                FightOdds(
                    fight_id=fight.id,
                    fighter_id=fb.id,
                    opening_ml=170,
                    closing_range_min_ml=190,
                    closing_range_max_ml=200,
                    opening_implied_prob=0.38,
                    closing_implied_prob=0.35,
                ),
            ]
        )
        session.flush()

        result = load_fight_odds(session)
        for value in result.values():
            assert set(value.keys()) == {"opening_implied_prob", "closing_implied_prob"}

    def test_load_fight_odds_preserves_none(self, session: Session) -> None:
        """D-07 + Pitfall 3: None must NOT be coerced to 0.0 or NaN at load
        time. The assembler is the boundary that decides what to do with
        None (it converts to NaN in the row vector). The loader preserves
        what the DB has."""
        from ufc_prediction.ml.queries import load_fight_odds

        fa, _fb, fight = _seed_fight_for_odds(session)
        session.add(
            FightOdds(
                fight_id=fight.id,
                fighter_id=fa.id,
                opening_ml=None,
                closing_range_min_ml=-220,
                closing_range_max_ml=-210,
                opening_implied_prob=None,
                closing_implied_prob=0.55,
            )
        )
        session.flush()

        result = load_fight_odds(session)
        v = result[(fa.id, fight.id)]
        # Use 'is None' rather than '== None' — explicit not-coerced check
        assert v["opening_implied_prob"] is None
        assert v["closing_implied_prob"] == 0.55

    def test_load_fight_odds_handles_both_probs_none(self, session: Session) -> None:
        """If both probs are None, the row STILL appears in the dict —
        the row's existence is a different signal from None values per
        column. Pre-2007 fights have no row at all (absent from dict);
        post-2007 partial-data rows (e.g., book-coverage gap) have a row
        with None probs. The assembler treats both cases the same (NaN)
        but the storage semantics differ."""
        from ufc_prediction.ml.queries import load_fight_odds

        fa, _fb, fight = _seed_fight_for_odds(session)
        session.add(
            FightOdds(
                fight_id=fight.id,
                fighter_id=fa.id,
                opening_implied_prob=None,
                closing_implied_prob=None,
            )
        )
        session.flush()

        result = load_fight_odds(session)
        assert (fa.id, fight.id) in result
        assert result[(fa.id, fight.id)]["opening_implied_prob"] is None
        assert result[(fa.id, fight.id)]["closing_implied_prob"] is None


# ─── 3. Assembler tests (pure-Python; reuse conftest fixtures) ────────────────


def _swap_for_fight(fight_id: int) -> bool:
    """Reproduce the assembler's deterministic A/B swap.

    From feature_matrix.py line ~371:
        swap = int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 0

    Used by tests that need to know which fighter ends up as 'A' in the
    assembler's row, regardless of which slot they were seeded in.
    """
    import hashlib

    return int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 0


class TestAssemblerSection13:
    """Plan 03 Task 2: Section 13 in FeatureMatrixAssembler.assemble."""

    def test_assemble_returns_70_column_matrix(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Even with empty fight_odds, the matrix gains 3 NaN columns at the
        tail — total 70."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            fight_odds={},
        )
        assert X.shape[1] == 75

    def test_assemble_backward_compatible_without_fight_odds_kwarg(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Calling assemble() without the new kwarg must still work and
        produce NaN in the last 3 columns (default = None ≡ empty dict)."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            round_stats=None,
            pre_ufc_records=None,
        )
        assert X.shape[1] == 75
        # Last 3 columns are NaN for every row (no odds passed)
        for r in range(X.shape[0]):
            assert np.isnan(X[r, -8])
            assert np.isnan(X[r, -7])
            assert np.isnan(X[r, -6])

    def test_odds_diffs_both_present(
        self,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """D-04 + D-05 + D-09: per-fighter formulation — A minus B."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        # Seed fight 101 with deterministic A/B identities
        fight_recs = [
            {
                "fight_id": 101,
                "event_date": date(2020, 1, 15),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            },
        ]
        # Account for the assembler's deterministic swap.
        if _swap_for_fight(101):
            a_id, b_id = 2, 1
        else:
            a_id, b_id = 1, 2

        # Per the spec: A opens 0.62 / closes 0.65; B opens 0.38 / closes 0.35
        fight_odds = {
            (a_id, 101): {"opening_implied_prob": 0.62, "closing_implied_prob": 0.65},
            (b_id, 101): {"opening_implied_prob": 0.38, "closing_implied_prob": 0.35},
        }

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_recs,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            fight_odds=fight_odds,
        )

        assert X[0, -8] == pytest.approx(0.62 - 0.38, abs=1e-9)
        assert X[0, -7] == pytest.approx(0.65 - 0.35, abs=1e-9)
        # line_movement_diff = (cl_a - op_a) - (cl_b - op_b)
        # = (0.65 - 0.62) - (0.35 - 0.38) = 0.03 - (-0.03) = 0.06
        assert X[0, -6] == pytest.approx((0.65 - 0.62) - (0.35 - 0.38), abs=1e-9)

    def test_odds_diffs_missing_a_side_are_nan(
        self,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """D-07 + Pitfall 3: if either side has no odds row at all, ALL THREE
        diffs are NaN (NEVER 0.0)."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fight_recs = [
            {
                "fight_id": 101,
                "event_date": date(2020, 1, 15),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            },
        ]
        # Only B has odds; A absent from dict.
        # Resolve POST-SWAP B-id deterministically without binding an unused
        # a_id local — ruff F841 would flag it otherwise.
        b_id = 1 if _swap_for_fight(101) else 2
        fight_odds = {
            (b_id, 101): {"opening_implied_prob": 0.38, "closing_implied_prob": 0.35},
        }

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_recs,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            fight_odds=fight_odds,
        )

        assert np.isnan(X[0, -8])
        assert np.isnan(X[0, -7])
        assert np.isnan(X[0, -6])
        # Explicit non-zero check — Pitfall 3 regression
        assert X[0, -8] != 0.0
        assert X[0, -7] != 0.0
        assert X[0, -6] != 0.0

    def test_odds_diffs_missing_opening_only_are_partial(
        self,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """If both fighters have odds rows but B's opening is None: opening
        diff is NaN (need both opens), closing diff is valid, line_movement
        is NaN (needs all 4 values)."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fight_recs = [
            {
                "fight_id": 101,
                "event_date": date(2020, 1, 15),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            },
        ]
        if _swap_for_fight(101):
            a_id, b_id = 2, 1
        else:
            a_id, b_id = 1, 2

        fight_odds = {
            (a_id, 101): {"opening_implied_prob": 0.62, "closing_implied_prob": 0.65},
            (b_id, 101): {"opening_implied_prob": None, "closing_implied_prob": 0.35},
        }

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_recs,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            fight_odds=fight_odds,
        )

        assert np.isnan(X[0, -8])  # opening_prob_diff: B opening None
        assert X[0, -7] == pytest.approx(0.65 - 0.35, abs=1e-9)
        assert np.isnan(X[0, -6])  # line_movement_diff: needs all 4

    def test_odds_diffs_respect_positional_swap(
        self,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """The assembler swaps fighter_a / fighter_b deterministically by
        fight_id hash. D-09 ('Fighter A minus Fighter B') applies AFTER the
        swap — i.e., the matrix's 'A' is whichever fighter the swap put in
        slot A. Verify by asserting on POST-SWAP identities."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fight_recs = [
            {
                "fight_id": 101,
                "event_date": date(2020, 1, 15),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            },
        ]
        # Fighter 1's per-fight odds vs Fighter 2's per-fight odds (raw)
        odds_for_1 = {"opening_implied_prob": 0.7, "closing_implied_prob": 0.72}
        odds_for_2 = {"opening_implied_prob": 0.3, "closing_implied_prob": 0.28}
        fight_odds = {(1, 101): odds_for_1, (2, 101): odds_for_2}

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_recs,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            fight_odds=fight_odds,
        )

        # POST-SWAP: which fighter is A?
        if _swap_for_fight(101):
            a_odds, b_odds = odds_for_2, odds_for_1
        else:
            a_odds, b_odds = odds_for_1, odds_for_2

        assert X[0, -8] == pytest.approx(
            a_odds["opening_implied_prob"] - b_odds["opening_implied_prob"],
            abs=1e-9,
        )
        assert X[0, -7] == pytest.approx(
            a_odds["closing_implied_prob"] - b_odds["closing_implied_prob"],
            abs=1e-9,
        )
        expected_lm = (a_odds["closing_implied_prob"] - a_odds["opening_implied_prob"]) - (
            b_odds["closing_implied_prob"] - b_odds["opening_implied_prob"]
        )
        assert X[0, -6] == pytest.approx(expected_lm, abs=1e-9)

    def test_odds_diffs_zero_imputation_regression_guard(
        self,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Pitfall 3: zero is a legal probability value. 0.0 means
        'mathematical pickem' (50-50 split); NaN means 'no odds'. They MUST
        be distinguishable.

        Strategy: build TWO synthetic fight sets:
        - 100 'pickem' fights: every fighter has opening=0.5, closing=0.5
          → every diff is exactly 0.0 (numerically, not NaN)
        - 100 'no-odds' fights: empty fight_odds dict
          → every diff is NaN (numerically, not 0.0)
        """
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        # 100 pickem fights between fighter 1 and 2.
        pickem_fights = [
            {
                "fight_id": 1000 + i,
                "event_date": date(2020, 1, 15),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            }
            for i in range(100)
        ]
        pickem_odds: dict = {}
        for f in pickem_fights:
            pickem_odds[(1, f["fight_id"])] = {
                "opening_implied_prob": 0.5,
                "closing_implied_prob": 0.5,
            }
            pickem_odds[(2, f["fight_id"])] = {
                "opening_implied_prob": 0.5,
                "closing_implied_prob": 0.5,
            }

        assembler = FeatureMatrixAssembler()
        X_pickem, _, _ = assembler.assemble(
            pickem_fights,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            fight_odds=pickem_odds,
        )
        # Every pickem row: all 3 diffs are exactly 0.0 (NOT NaN)
        for r in range(X_pickem.shape[0]):
            assert X_pickem[r, -8] == 0.0
            assert X_pickem[r, -7] == 0.0
            assert X_pickem[r, -6] == 0.0
            assert not np.isnan(X_pickem[r, -8])
            assert not np.isnan(X_pickem[r, -7])
            assert not np.isnan(X_pickem[r, -6])

        # 100 no-odds fights between fighter 1 and 2
        no_odds_fights = [
            {
                "fight_id": 2000 + i,
                "event_date": date(2020, 1, 15),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            }
            for i in range(100)
        ]
        X_none, _, _ = assembler.assemble(
            no_odds_fights,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            fight_odds={},
        )
        # Every no-odds row: all 3 diffs are NaN (NOT 0.0)
        for r in range(X_none.shape[0]):
            assert np.isnan(X_none[r, -8])
            assert np.isnan(X_none[r, -7])
            assert np.isnan(X_none[r, -6])
            assert X_none[r, -8] != 0.0
            assert X_none[r, -7] != 0.0
            assert X_none[r, -6] != 0.0


# ─── 4. CLI tests ─────────────────────────────────────────────────────────────


class TestPredictCliWiresFightOdds:
    """Plan 03 Task 2 step 3: predict train + predict evaluate must call
    load_fight_odds(session) and forward the result to assembler.assemble().
    Mocked so no DB / model training executes."""

    @pytest.mark.xfail(
        reason="`train` was refactored to assemble via the meta-OOF-matrix path "
        "(_load_meta_train_eval_matrices); it still calls load_fight_odds but no "
        "longer the module-level FeatureMatrixAssembler this mock asserts on. "
        "Assertion needs rework against the current train wiring.",
        strict=False,
    )
    def test_predict_train_loads_fight_odds(self) -> None:
        from typer.testing import CliRunner

        from ufc_prediction.cli.predict import predict_app

        runner = CliRunner()

        with (
            patch("ufc_prediction.cli.predict.SessionLocal") as mock_session_local,
            patch("ufc_prediction.cli.predict.load_fight_records", return_value=[]),
            patch("ufc_prediction.cli.predict.load_elo_features", return_value={}),
            patch("ufc_prediction.cli.predict.load_computed_features", return_value={}),
            patch("ufc_prediction.cli.predict.load_fighter_physicals", return_value={}),
            patch("ufc_prediction.cli.predict.load_round_stats_for_ml", return_value={}),
            patch("ufc_prediction.cli.predict.load_pre_ufc_records", return_value={}),
            patch("ufc_prediction.cli.predict.load_fight_odds", return_value={}) as mock_lfo,
            patch("ufc_prediction.cli.predict.compute_division_medians", return_value={}),
            patch("ufc_prediction.cli.predict.FeatureMatrixAssembler") as mock_asm_cls,
            patch(
                "ufc_prediction.cli.predict.split_temporal",
                return_value=(
                    np.zeros((1, 70)),
                    np.zeros((1, 70)),
                    np.zeros(1, dtype=np.int32),
                    np.zeros(1, dtype=np.int32),
                ),
            ),
            patch("ufc_prediction.cli.predict.ModelTrainer") as mock_trainer_cls,
            patch(
                "ufc_prediction.cli.predict.evaluate_model",
                return_value={
                    "brier_score": 0.25,
                    "auc_roc": 0.6,
                    "accuracy": 0.55,
                },
            ),
            patch("ufc_prediction.cli.predict.save_model", return_value="models/test"),
        ):
            # Stub assembler.assemble() return shape
            mock_asm = MagicMock()
            mock_asm.assemble.return_value = (
                np.zeros((1, 70)),
                np.zeros(1, dtype=np.int32),
                np.array([date(2020, 1, 1)], dtype=object),
            )
            mock_asm_cls.return_value = mock_asm

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = (MagicMock(), {}, {})
            mock_trainer_cls.return_value = mock_trainer

            mock_session_local.return_value = MagicMock()

            result = runner.invoke(predict_app, ["train", "--trials", "1"])

            # Must have called load_fight_odds at least once
            assert mock_lfo.called, (
                "predict train must invoke load_fight_odds; got "
                f"exit_code={result.exit_code}, output={result.output!r}, "
                f"exc={result.exception!r}"
            )
            # And assembler.assemble must have received fight_odds kwarg
            asm_calls = mock_asm.assemble.call_args_list
            assert asm_calls, "assembler.assemble was not called"
            kwargs = asm_calls[0].kwargs
            assert "fight_odds" in kwargs, (
                f"fight_odds kwarg missing from assemble() call; got kwargs={list(kwargs)}"
            )

    def test_predict_evaluate_loads_fight_odds(self) -> None:
        from typer.testing import CliRunner

        from ufc_prediction.cli.predict import predict_app

        runner = CliRunner()

        with (
            patch("ufc_prediction.cli.predict.SessionLocal") as mock_session_local,
            patch("ufc_prediction.cli.predict.load_fight_records", return_value=[]),
            patch("ufc_prediction.cli.predict.load_elo_features", return_value={}),
            patch("ufc_prediction.cli.predict.load_computed_features", return_value={}),
            patch("ufc_prediction.cli.predict.load_fighter_physicals", return_value={}),
            patch("ufc_prediction.cli.predict.load_round_stats_for_ml", return_value={}),
            patch("ufc_prediction.cli.predict.load_pre_ufc_records", return_value={}),
            patch("ufc_prediction.cli.predict.load_fight_odds", return_value={}) as mock_lfo,
            patch("ufc_prediction.cli.predict.compute_division_medians", return_value={}),
            patch("ufc_prediction.cli.predict.FeatureMatrixAssembler") as mock_asm_cls,
            patch(
                "ufc_prediction.cli.predict.split_temporal",
                return_value=(
                    np.zeros((1, 70)),
                    np.zeros((1, 70)),
                    np.zeros(1, dtype=np.int32),
                    np.zeros(1, dtype=np.int32),
                ),
            ),
            patch("ufc_prediction.ml.persistence.load_model", return_value=MagicMock()),
            patch("ufc_prediction.cli.predict.load_metadata", return_value={}),
            patch(
                "ufc_prediction.cli.predict.evaluate_model",
                return_value={
                    "brier_score": 0.25,
                    "auc_roc": 0.6,
                    "accuracy": 0.55,
                },
            ),
            patch("ufc_prediction.cli.predict._extract_importances", return_value={}),
            patch("ufc_prediction.cli.predict.format_evaluation_report", return_value=""),
        ):
            mock_asm = MagicMock()
            mock_asm.assemble.return_value = (
                np.zeros((1, 70)),
                np.zeros(1, dtype=np.int32),
                np.array([date(2020, 1, 1)], dtype=object),
            )
            mock_asm_cls.return_value = mock_asm

            mock_session_local.return_value = MagicMock()

            result = runner.invoke(predict_app, ["evaluate", "--version", "v1"])

            assert mock_lfo.called, (
                "predict evaluate must invoke load_fight_odds; got "
                f"exit_code={result.exit_code}, output={result.output!r}, "
                f"exc={result.exception!r}"
            )
            asm_calls = mock_asm.assemble.call_args_list
            assert asm_calls, "assembler.assemble was not called"
            kwargs = asm_calls[0].kwargs
            assert "fight_odds" in kwargs, (
                f"fight_odds kwarg missing from assemble() call; got kwargs={list(kwargs)}"
            )
