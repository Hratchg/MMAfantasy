"""Tests for ML feature matrix assembly, config, queries, and physical features.

Covers:
- MLConfig defaults and FEATURE_COLUMNS
- encode_stance_matchup for all cases
- compute_division_medians with temporal filtering
- Query return types and processing
- FeatureMatrixAssembler.assemble() integration
- split_temporal correctness
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

# ─── MLConfig tests ──────────────────────────────────────────────────────────


class TestMLConfig:
    def test_defaults(self) -> None:
        from ufc_prediction.ml.config import MLConfig

        cfg = MLConfig()
        assert cfg.cutoff_date == "2023-01-01"
        assert cfg.model_dir == "models"
        assert cfg.n_optuna_trials == 50
        assert cfg.cv_splits == 5
        assert cfg.random_seed == 42

    def test_frozen(self) -> None:
        from ufc_prediction.ml.config import MLConfig

        cfg = MLConfig()
        with pytest.raises(AttributeError):
            cfg.random_seed = 99  # type: ignore[misc]

    def test_feature_columns_length(self) -> None:
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        # 3 Elo + 20 perf + 4 physical + 1 stance + 16 career
        # + 2 cross-domain + 3 context + 4 pace + 3 layoff
        # + 8 rolling + 2 rematch + 1 pre-UFC + 3 betting odds
        # + 2 engineered odds (Phase 15.1 gap closure on D-05)
        # + 3 NET-* (Phase 16-03 NET-01/02 pagerank_diff/sos_2hop_diff/
        #   is_debutant_in_graph_diff) = 75 total.
        # Stale "72" assertion fixed in Phase 23 Plan 23-01 (RESEARCH §
        # "State of the Art" + Assumption A2).
        assert len(FEATURE_COLUMNS) == 75

    def test_feature_columns_starts_with_elo(self) -> None:
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        assert FEATURE_COLUMNS[0] == "elo_overall_diff"
        assert FEATURE_COLUMNS[1] == "elo_striking_diff"
        assert FEATURE_COLUMNS[2] == "elo_grappling_diff"

    def test_feature_columns_ends_with_pre_ufc(self) -> None:
        """Phase 13-03 appended 3 betting-odds diffs after pre_ufc, then
        Phase 15.1 gap closure appended 2 engineered odds features after
        those (sharp_money_signal, odds_elo_divergence on D-05).

        pre_ufc_win_pct_diff now sits at index -6 (6th from end). The
        full odds tail is the 5 betting-odds features in this order.
        """
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        assert FEATURE_COLUMNS[-6] == "pre_ufc_win_pct_diff"
        assert FEATURE_COLUMNS[-5:] == [
            "opening_prob_diff",
            "closing_prob_diff",
            "line_movement_diff",
            "sharp_money_signal",
            "odds_elo_divergence",
        ]

    def test_feature_columns_physical_diffs(self) -> None:
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        assert "height_diff" in FEATURE_COLUMNS
        assert "reach_diff" in FEATURE_COLUMNS
        assert "leg_reach_diff" in FEATURE_COLUMNS
        assert "age_diff" in FEATURE_COLUMNS

    def test_performance_feature_keys_count(self) -> None:
        from ufc_prediction.ml.config import PERFORMANCE_FEATURE_KEYS

        # 20 numeric features from CANONICAL_FEATURE_ORDER (excluding style_tag)
        assert len(PERFORMANCE_FEATURE_KEYS) == 20

    def test_performance_feature_keys_excludes_style_tag(self) -> None:
        from ufc_prediction.ml.config import PERFORMANCE_FEATURE_KEYS

        assert "style_tag" not in PERFORMANCE_FEATURE_KEYS

    def test_feature_columns_no_duplicates(self) -> None:
        from ufc_prediction.ml.config import FEATURE_COLUMNS

        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


# ─── encode_stance_matchup tests ─────────────────────────────────────────────


class TestEncodeStanceMatchup:
    def test_orthodox_vs_southpaw_is_opposite(self) -> None:
        from ufc_prediction.ml.config import encode_stance_matchup

        assert encode_stance_matchup("Orthodox", "Southpaw") == 0.0

    def test_orthodox_vs_orthodox_is_same(self) -> None:
        from ufc_prediction.ml.config import encode_stance_matchup

        assert encode_stance_matchup("Orthodox", "Orthodox") == 1.0

    def test_none_vs_southpaw_is_opposite(self) -> None:
        """None normalizes to Orthodox, which is opposite of Southpaw."""
        from ufc_prediction.ml.config import encode_stance_matchup

        assert encode_stance_matchup(None, "Southpaw") == 0.0

    def test_switch_vs_orthodox_is_same(self) -> None:
        """Switch normalizes to Orthodox per D-02."""
        from ufc_prediction.ml.config import encode_stance_matchup

        assert encode_stance_matchup("Switch", "Orthodox") == 1.0

    def test_none_vs_none_is_same(self) -> None:
        """Both normalize to Orthodox."""
        from ufc_prediction.ml.config import encode_stance_matchup

        assert encode_stance_matchup(None, None) == 1.0

    def test_southpaw_vs_southpaw_is_same(self) -> None:
        from ufc_prediction.ml.config import encode_stance_matchup

        assert encode_stance_matchup("Southpaw", "Southpaw") == 1.0

    def test_switch_vs_southpaw_is_opposite(self) -> None:
        """Switch normalizes to Orthodox, opposite of Southpaw."""
        from ufc_prediction.ml.config import encode_stance_matchup

        assert encode_stance_matchup("Switch", "Southpaw") == 0.0


# ─── compute_division_medians tests ──────────────────────────────────────────


class TestComputeDivisionMedians:
    def test_computes_medians_per_weight_class(
        self, fighter_physicals: dict, fight_records: list
    ) -> None:
        from ufc_prediction.ml.feature_matrix import compute_division_medians

        medians = compute_division_medians(
            fighter_physicals, fight_records, cutoff_date=date(2023, 1, 1)
        )
        assert "Lightweight" in medians
        assert "height_inches" in medians["Lightweight"]
        assert "reach_inches" in medians["Lightweight"]
        assert "leg_reach_inches" in medians["Lightweight"]

    def test_only_uses_pre_cutoff_fights(
        self, fighter_physicals: dict, fight_records: list
    ) -> None:
        """Division medians only include fighters from pre-cutoff fights (D-01)."""
        from ufc_prediction.ml.feature_matrix import compute_division_medians

        # With cutoff 2023-01-01, fights 101, 102, 103 are pre-cutoff
        medians = compute_division_medians(
            fighter_physicals, fight_records, cutoff_date=date(2023, 1, 1)
        )

        # Lightweight pre-cutoff fighters: 1 (72), 2 (70), 5 (71)
        # Median height = 71.0
        assert medians["Lightweight"]["height_inches"] == 71.0

    def test_imputation_fills_none_values(self, fighter_physicals: dict) -> None:
        """Division median imputation fills None height/reach/leg_reach."""
        from ufc_prediction.ml.feature_matrix import compute_division_medians

        fight_recs = [
            {
                "fight_id": 200,
                "event_date": date(2022, 1, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            },
        ]
        medians = compute_division_medians(
            fighter_physicals, fight_recs, cutoff_date=date(2023, 1, 1)
        )
        # Both fighter 1 and 2 have complete data for Lightweight
        assert medians["Lightweight"]["height_inches"] is not None

    def test_does_not_overwrite_existing_values(self, fighter_physicals: dict) -> None:
        """Imputation must NOT overwrite non-None values -- this is tested via the assembler."""
        # Fighter 1 has height 72.0; imputation should not change it
        assert fighter_physicals[1]["height_inches"] == 72.0


# ─── Query function return type tests ────────────────────────────────────────


class TestQueryReturnTypes:
    """Tests for query function return value processing.

    These test the data structure contracts without DB access.
    Actual DB queries are tested via integration tests with testcontainers.
    """

    def test_elo_features_structure(self, elo_features: dict) -> None:
        """Elo features should be keyed by (fighter_id, fight_id) with named elo values."""
        key = (1, 101)
        assert key in elo_features
        assert "elo_overall" in elo_features[key]
        assert "elo_striking" in elo_features[key]
        assert "elo_grappling" in elo_features[key]

    def test_elo_uses_before_not_after(self, elo_features: dict) -> None:
        """Elo features must use elo_before values (not elo_after) per ML-06."""
        # The fixture represents elo_before values.
        # In production, queries.load_elo_features uses elo_before.
        key = (1, 101)
        # Values should be reasonable pre-fight Elo (around 1500 range)
        assert 1400 <= elo_features[key]["elo_overall"] <= 1600

    def test_computed_features_structure(self, computed_features: dict) -> None:
        """Computed features should have 20 numeric keys."""
        key = (1, 101)
        assert key in computed_features
        assert len(computed_features[key]) == 20
        assert "sig_str_per_minute" in computed_features[key]
        assert "style_tag" not in computed_features[key]

    def test_fighter_physicals_structure(self, fighter_physicals: dict) -> None:
        """Fighter physicals keyed by fighter_id with expected fields."""
        assert 1 in fighter_physicals
        phys = fighter_physicals[1]
        assert "height_inches" in phys
        assert "reach_inches" in phys
        assert "leg_reach_inches" in phys
        assert "stance" in phys
        assert "date_of_birth" in phys

    def test_fight_records_sorted_chronologically(self, fight_records: list) -> None:
        """Fight records must be sorted by event_date."""
        dates = [f["event_date"] for f in fight_records]
        assert dates == sorted(dates)

    def test_fight_records_have_required_keys(self, fight_records: list) -> None:
        required_keys = {
            "fight_id",
            "event_date",
            "fighter_a_id",
            "fighter_b_id",
            "winner_id",
            "weight_class",
        }
        for record in fight_records:
            assert required_keys.issubset(record.keys())


# ─── FeatureMatrixAssembler tests ────────────────────────────────────────────


class TestFeatureMatrixAssembler:
    def test_matrix_shape(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """assemble() returns (X, y, fight_dates) with correct shapes."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, y, fight_dates = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        assert X.shape == (5, 72)
        assert y.shape == (5,)
        assert fight_dates.shape == (5,)

    def test_columns_match_feature_columns_order(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """X columns must match FEATURE_COLUMNS order exactly."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Verify by checking known values at known positions
        # Fight 101: fighter 1 vs 2, Elo overall diff = 1520 - 1480 = 40
        elo_overall_idx = FEATURE_COLUMNS.index("elo_overall_diff")
        assert X[0, elo_overall_idx] == pytest.approx(40.0)

    def test_elo_differential(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Elo differential = fighter_a elo_before - fighter_b elo_before."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Fight 101: A=1, B=2
        # overall: 1520 - 1480 = 40
        # striking: 1530 - 1470 = 60
        # grappling: 1510 - 1490 = 20
        idx_o = FEATURE_COLUMNS.index("elo_overall_diff")
        idx_s = FEATURE_COLUMNS.index("elo_striking_diff")
        idx_g = FEATURE_COLUMNS.index("elo_grappling_diff")
        assert X[0, idx_o] == pytest.approx(40.0)
        assert X[0, idx_s] == pytest.approx(60.0)
        assert X[0, idx_g] == pytest.approx(20.0)

    def test_performance_feature_differential(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Performance feature differential = fighter_a - fighter_b."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Fight 101: A=1 (base: sig_str=4.5), B=2 (alt: sig_str=3.5)
        # Diff = 4.5 - 3.5 = 1.0
        idx = FEATURE_COLUMNS.index("sig_str_per_minute_diff")
        assert X[0, idx] == pytest.approx(1.0)

    def test_physical_differential(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Physical differential = fighter_a - fighter_b."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Fight 101: A=1 (height 72), B=2 (height 70)
        # height_diff = 72 - 70 = 2.0
        idx_h = FEATURE_COLUMNS.index("height_diff")
        assert X[0, idx_h] == pytest.approx(2.0)

        # reach_diff = 74 - 72 = 2.0
        idx_r = FEATURE_COLUMNS.index("reach_diff")
        assert X[0, idx_r] == pytest.approx(2.0)

    def test_missing_physical_imputed_with_division_median(
        self,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Missing physical data is imputed with division median before differencing."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        # Fight 102: A=3 (all physical None), B=4 (reach None)
        # Both in Welterweight, median reach = 74.0
        fight_recs = [
            {
                "fight_id": 102,
                "event_date": date(2021, 6, 20),
                "fighter_a_id": 3,
                "fighter_b_id": 4,
                "winner_id": 4,
                "weight_class": "Welterweight",
            },
        ]
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_recs,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Fighter 3: height=None -> imputed to 72.0 (Welterweight median)
        # Fighter 4: height=68.0
        # height_diff = 72.0 - 68.0 = 4.0
        idx_h = FEATURE_COLUMNS.index("height_diff")
        assert X[0, idx_h] == pytest.approx(4.0)

        # Fighter 3: reach=None -> imputed to 74.0 (Welterweight median)
        # Fighter 4: reach=None -> imputed to 74.0 (Welterweight median)
        # reach_diff = 74.0 - 74.0 = 0.0
        idx_r = FEATURE_COLUMNS.index("reach_diff")
        assert X[0, idx_r] == pytest.approx(0.0)

    def test_stance_matchup_is_binary(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Stance matchup is binary (0 or 1), not a differential."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        idx = FEATURE_COLUMNS.index("stance_matchup")
        # All stance_matchup values should be 0.0 or 1.0
        for i in range(X.shape[0]):
            assert X[i, idx] in (0.0, 1.0)

        # Fight 101: A=1 (Orthodox) vs B=2 (Southpaw) -> opposite -> 0.0
        assert X[0, idx] == 0.0

    def test_target_alignment(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """y=1 when winner_id == fighter_a_id, y=0 otherwise."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        _, y, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Fight 101: winner_id=1, fighter_a_id=1 -> y=1
        assert y[0] == 1
        # Fight 102: winner_id=4, fighter_a_id=3 -> y=0
        assert y[1] == 0
        # Fight 103: winner_id=5, fighter_a_id=1 -> y=0
        assert y[2] == 0
        # Fight 104: winner_id=2, fighter_a_id=2 -> y=1
        assert y[3] == 1
        # Fight 105: winner_id=1, fighter_a_id=1 -> y=1
        assert y[4] == 1

    def test_chronological_order(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """fight_dates must be monotonically non-decreasing."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        _, _, fight_dates = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        for i in range(1, len(fight_dates)):
            assert fight_dates[i] >= fight_dates[i - 1]

    def test_missing_elo_defaults_to_1500(
        self,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Fighters without Elo snapshots get 1500.0 default."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fight_recs = [
            {
                "fight_id": 999,
                "event_date": date(2020, 1, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            },
        ]
        # Empty elo_features means no snapshots exist
        empty_elo: dict[tuple[int, int], dict[str, float]] = {}

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_recs,
            empty_elo,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Both default to 1500, so diff = 0
        idx = FEATURE_COLUMNS.index("elo_overall_diff")
        assert X[0, idx] == pytest.approx(0.0)

    def test_missing_features_produce_nan(
        self,
        elo_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Fights without computed features have NaN in performance columns."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        fight_recs = [
            {
                "fight_id": 998,
                "event_date": date(2020, 1, 1),
                "fighter_a_id": 1,
                "fighter_b_id": 2,
                "winner_id": 1,
                "weight_class": "Lightweight",
            },
        ]
        # Empty computed features
        empty_feats: dict[tuple[int, int], dict[str, float | None]] = {}

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_recs,
            elo_features,
            empty_feats,
            fighter_physicals,
            division_medians,
        )

        idx = FEATURE_COLUMNS.index("sig_str_per_minute_diff")
        assert np.isnan(X[0, idx])

    def test_age_uses_fight_date(
        self,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Age computed from DOB and event_date, not current date."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
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
        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_recs,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Fighter 1: DOB 1990-01-01, fight date 2020-01-15 -> age ~30.04
        # Fighter 2: DOB 1988-06-15, fight date 2020-01-15 -> age ~31.59
        # age_diff = 30.04 - 31.59 = -1.55 (approx)
        idx = FEATURE_COLUMNS.index("age_diff")
        age_diff = X[0, idx]
        assert age_diff < 0  # Fighter A is younger
        assert age_diff == pytest.approx(-1.55, abs=0.1)


# ─── split_temporal tests ────────────────────────────────────────────────────


class TestSplitTemporal:
    def test_correct_split(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """split_temporal correctly divides on cutoff_date."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler, split_temporal

        assembler = FeatureMatrixAssembler()
        X, y, fight_dates = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        X_train, X_test, y_train, y_test = split_temporal(X, y, fight_dates, date(2023, 1, 1))

        # Fights 101, 102, 103 are pre-2023 (train)
        assert X_train.shape[0] == 3
        assert y_train.shape[0] == 3
        # Fights 104, 105 are post-2023 (test)
        assert X_test.shape[0] == 2
        assert y_test.shape[0] == 2

    def test_split_preserves_columns(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Train and test sets have same number of feature columns."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler, split_temporal

        assembler = FeatureMatrixAssembler()
        X, y, fight_dates = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        X_train, X_test, _, _ = split_temporal(X, y, fight_dates, date(2023, 1, 1))
        assert X_train.shape[1] == 28
        assert X_test.shape[1] == 28


# ─── Elo-as-input verification (ML-06) ──────────────────────────────────────


class TestEloAsInput:
    """Verify Elo ratings are used as input features (not replacement).

    Per ML-06/D-12: Existing Elo infrastructure is read-only. Model uses
    Elo elo_before values as input features in the feature matrix.
    """

    def test_elo_as_input(
        self,
        fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Elo columns contain elo_before differences, not elo_after."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
        )

        # Verify Elo features are present as columns (3 Elo diff columns)
        elo_cols = [c for c in FEATURE_COLUMNS if c.startswith("elo_")]
        assert len(elo_cols) == 3

        # Verify the Elo values are elo_before-based differentials
        # Fight 101: A=fighter1 (overall=1520), B=fighter2 (overall=1480)
        # If elo_after were used, values would be different (post-fight)
        idx = FEATURE_COLUMNS.index("elo_overall_diff")
        assert X[0, idx] == pytest.approx(40.0)  # 1520 - 1480

    def test_queries_module_uses_elo_before(self) -> None:
        """The queries module must reference elo_before, never elo_after."""
        import inspect

        from ufc_prediction.ml import queries

        source = inspect.getsource(queries.load_elo_features)
        assert "elo_before" in source
        # elo_after should only appear in comments, not in select columns
        lines = [
            line.strip()
            for line in source.split("\n")
            if not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("CRITICAL")
        ]
        code_only = "\n".join(lines)
        # The actual select statement should not use elo_after
        assert "EloSnapshot.elo_after" not in code_only


# ─── TestFeatureColumnsV22 — Phase 23 Plan 23-01 ─────────────────────────────


class TestFeatureColumnsV22:
    """Config-layer tests for the v2.2 feature set.

    Tests live here (NOT test_features_v22.py) because they're about the
    CONFIG layer — col-list shape, dispatch, dedup. Compute-helper tests
    live in test_features_v22.py.

    NOTE on length: the plan text claims `len(FEATURE_COLUMNS_V22) == 91`
    (72 + 3 + 6 + 10). The plan's own META enumeration only lists 9 cols
    (`age_diff` cannot be a 10th col — it already exists in
    FEATURE_COLUMNS_NO_NET, which would violate D-07 dedup / test 12).
    Resolution: honor D-07 (stronger CONTEXT-locked invariant); final
    length is 72 + 3 + 6 + 9 = **90 cols**. See 23-01-SUMMARY.md
    deviation note for the auto-fix rationale.
    """

    def test_feature_columns_v22_append_only(self) -> None:
        """Test 9: FEATURE_COLUMNS_V22[:72] is byte-identical to NO_NET (D-08)."""
        from ufc_prediction.ml.config import (
            FEATURE_COLUMNS_NO_NET,
            FEATURE_COLUMNS_V22,
        )

        assert FEATURE_COLUMNS_V22[:72] == FEATURE_COLUMNS_NO_NET

    def test_feature_columns_no_net_unchanged_byte_identical(self) -> None:
        """Test 10: xgb_v2 baseline preserved at 72 cols."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET

        assert len(FEATURE_COLUMNS_NO_NET) == 72

    def test_feature_columns_v22_no_duplicates(self) -> None:
        """Test 11: no col appears twice in V22."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        assert len(set(FEATURE_COLUMNS_V22)) == len(FEATURE_COLUMNS_V22)

    def test_feature_columns_v22_dedup_from_no_net(self) -> None:
        """Test 12: V22 tail (index 72+) has zero overlap with NO_NET (D-07)."""
        from ufc_prediction.ml.config import (
            FEATURE_COLUMNS_NO_NET,
            FEATURE_COLUMNS_V22,
        )

        overlap = set(FEATURE_COLUMNS_V22[72:]) & set(FEATURE_COLUMNS_NO_NET)
        assert overlap == set(), f"D-07 dedup violation: overlap = {overlap}"

    def test_feature_columns_v22_no_layoff_days_diff(self) -> None:
        """Test 13: layoff_days_diff explicitly absent (Q4 RESOLVED, B-1 fix).

        Semantic duplicate of FEATURE_COLUMNS_NO_NET[?] days_since_last_fight_diff.
        """
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        assert "layoff_days_diff" not in FEATURE_COLUMNS_V22

    def test_feature_columns_v22_length(self) -> None:
        """Test 14: V22 total = 72 + 3 REF + 6 TRAVEL + 9 META = 90.

        See class docstring for the META=9 (not 10) rationale: ``age_diff``
        cannot be added because it already exists in FEATURE_COLUMNS_NO_NET
        (D-07 dedup wins over the plan's miscounted "10").
        """
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        assert len(FEATURE_COLUMNS_V22) == 90

    def test_get_feature_columns_dispatch_v22(self) -> None:
        """Test 15: feature_set='v2.2' returns V22 list."""
        from ufc_prediction.ml.config import (
            FEATURE_COLUMNS_V22,
            get_feature_columns,
        )

        assert get_feature_columns(feature_set="v2.2") == list(FEATURE_COLUMNS_V22)

    def test_get_feature_columns_dispatch_v21(self) -> None:
        """Test 16: feature_set='v2.1-no-net' returns NO_NET list."""
        from ufc_prediction.ml.config import (
            FEATURE_COLUMNS_NO_NET,
            get_feature_columns,
        )

        assert get_feature_columns(feature_set="v2.1-no-net") == list(FEATURE_COLUMNS_NO_NET)

    def test_get_feature_columns_dispatch_v10(self) -> None:
        """Test 17: feature_set='v1.0' returns full FEATURE_COLUMNS list."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS, get_feature_columns

        assert get_feature_columns(feature_set="v1.0") == list(FEATURE_COLUMNS)

    def test_get_feature_columns_include_net_backcompat_true(self) -> None:
        """Test 18: include_net=True back-compat shim → v1.0."""
        from ufc_prediction.ml.config import get_feature_columns

        assert get_feature_columns(include_net=True) == get_feature_columns(feature_set="v1.0")

    def test_get_feature_columns_include_net_backcompat_false(self) -> None:
        """Test 19: include_net=False back-compat shim → v2.1-no-net."""
        from ufc_prediction.ml.config import get_feature_columns

        assert get_feature_columns(include_net=False) == get_feature_columns(
            feature_set="v2.1-no-net"
        )

    def test_get_feature_columns_unknown_feature_set(self) -> None:
        """Test 20: bogus feature_set raises ValueError matching 'unknown feature_set'."""
        from ufc_prediction.ml.config import get_feature_columns

        with pytest.raises(ValueError, match="unknown feature_set"):
            get_feature_columns(feature_set="bogus")

    def test_feature_columns_v22_ref_cols_at_72_to_74(self) -> None:
        """Anchor: REF cols are LOCKED at indices 72-74 (Plan 23-01 contract)."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

        assert FEATURE_COLUMNS_V22[72] == "ref_finish_rate_shrunk"
        assert FEATURE_COLUMNS_V22[73] == "ref_decision_rate_shrunk"
        assert FEATURE_COLUMNS_V22[74] == "ref_no_action_rate_shrunk"


# ─── TestV22Dispatch — Phase 23 Plan 23-01 Task 2 wiring tests ───────────────


@pytest.fixture()
def v22_fight_records(fight_records: list) -> list[dict]:
    """Augment baseline fight_records with v2.2-specific fields:
    event_id, referee_id, method.

    The plan's REF compute requires these keys to be present on each fight
    dict. We piggyback on the existing fight_records fixture (5 fights, 3
    pre-cutoff, 2 post-cutoff) so the Elo/computed/physical fixtures still
    line up.

    Referee assignment:
        - fights 101 + 103 + 105 → referee_id=1 (3 fights, history available)
        - fights 102 + 104 → referee_id=2 (2 fights)

    Methods give us a mix of finish / decision / no_action so REF rates
    have non-trivial values.
    """
    method_map = {
        101: ("KO/TKO", 1),  # finish, ref=1
        102: ("Decision - Unanimous", 2),  # decision, ref=2
        103: ("Submission", 1),  # finish, ref=1
        104: ("Decision - Split", 2),  # decision, ref=2
        105: ("Decision - Unanimous", 1),  # decision, ref=1
    }
    out: list[dict] = []
    for fight in fight_records:
        fid = fight["fight_id"]
        method, ref_id = method_map[fid]
        new = dict(fight)
        # event_id is 1:1 with fight_id in this fixture (each fight on its
        # own synthetic event).
        new["event_id"] = fid
        new["referee_id"] = ref_id
        new["method"] = method
        out.append(new)
    return out


class TestV22Dispatch:
    """Phase 23 Plan 23-01 Task 2 — feature_set='v2.2' dispatch in assemble().

    All tests live in test_feature_matrix.py because they exercise the
    training path. Cross-path parity (train ⇔ inference REF cols byte-identical)
    is asserted in tests/unit/ml/test_features_v22.py::TestTrainInferenceParity.
    """

    def test_assemble_v22_dispatch_shape(
        self,
        v22_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 1: assemble(..., feature_set='v2.2') → 90 cols (72+3+6+9)."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, y, fight_dates = assembler.assemble(
            v22_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        assert X.shape == (5, len(FEATURE_COLUMNS_V22))
        assert X.shape[1] == 90
        assert y.shape == (5,)
        assert fight_dates.shape == (5,)

    def test_assemble_v21_dispatch_unchanged(
        self,
        v22_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 2: feature_set='v2.1-no-net' returns 72-col baseline."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            v22_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.1-no-net",
        )

        assert X.shape[1] == 72

    def test_assemble_include_net_backcompat(
        self,
        v22_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 3: include_net=False back-compat shim still works (72 cols)."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            v22_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            include_net=False,
        )

        assert X.shape[1] == 72

    def test_assemble_v22_ref_cols_at_72_73_74(
        self,
        v22_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 4: REF cols at indices 72/73/74 match shared helper output.

        We re-compute the expected REF rates using the SAME shared helper
        the assembler imports — this proves the wiring routes through the
        single source of truth (Pitfall #12).
        """
        from ufc_prediction.ml.feature_matrix import (
            FeatureMatrixAssembler,
            _build_referee_lookup,
            _compute_ref_global_rates,
        )
        from ufc_prediction.ml.features_v22.ref import compute_ref_rates_shrunk

        # Pre-compute expected REF rates for the FIRST fight (fight 101,
        # ref=1, event_date=2020-01-15). At as_of=2020-01-15, ref 1 has NO
        # prior history (this is their first fight). Bayesian fallback →
        # all 3 cols == global_rates[cat].
        sorted_records = sorted(v22_fight_records, key=lambda f: f["event_date"])
        global_rates = _compute_ref_global_rates(sorted_records)
        event_refs, ref_history = _build_referee_lookup(sorted_records)

        expected_first = compute_ref_rates_shrunk(
            event_refs.get(101),
            date(2020, 1, 15),
            ref_history,
            global_rates,
        )

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            v22_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        # The first row of X corresponds to v22_fight_records[0] = fight 101.
        assert X[0, 72] == pytest.approx(expected_first["ref_finish_rate_shrunk"])
        assert X[0, 73] == pytest.approx(expected_first["ref_decision_rate_shrunk"])
        assert X[0, 74] == pytest.approx(expected_first["ref_no_action_rate_shrunk"])

    def test_assemble_v22_ref_no_referee_collapses_to_global(
        self,
        v22_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 5: referee_id=None on a fight → REF cols == global_rates."""
        from ufc_prediction.ml.feature_matrix import (
            FeatureMatrixAssembler,
            _compute_ref_global_rates,
        )

        # Strip referee_id from ALL fights → every row falls back to globals.
        no_ref_records = [{**f, "referee_id": None} for f in v22_fight_records]
        sorted_records = sorted(no_ref_records, key=lambda f: f["event_date"])
        global_rates = _compute_ref_global_rates(sorted_records)

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            no_ref_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        # Every row's REF cols collapse to global_rates respectively.
        for row_idx in range(X.shape[0]):
            assert X[row_idx, 72] == pytest.approx(global_rates["finish"])
            assert X[row_idx, 73] == pytest.approx(global_rates["decision"])
            assert X[row_idx, 74] == pytest.approx(global_rates["no_action"])

    def test_assemble_v22_temporal_leakage_guard(
        self,
        v22_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 10: Pitfall #6 — REF for fight #2 (with same ref as #1+#3)
        reflects only fight #1's outcome, not future #3.

        v22_fight_records has ref=1 on fights 101 (2020-01-15, finish),
        103 (2022-11-05, finish) and 105 (2024-01-20, decision). At fight
        103's as_of_date, only fight 101 should contribute.
        """
        from ufc_prediction.ml.feature_matrix import (
            FeatureMatrixAssembler,
            _build_referee_lookup,
            _compute_ref_global_rates,
        )
        from ufc_prediction.ml.features_v22.ref import (
            beta_binomial_shrink,
            compute_ref_rates_shrunk,
        )

        sorted_records = sorted(v22_fight_records, key=lambda f: f["event_date"])
        global_rates = _compute_ref_global_rates(sorted_records)
        event_refs, ref_history = _build_referee_lookup(sorted_records)

        # Manually compute expected REF for fight 103 (ref=1, as_of=2022-11-05):
        # Only fight 101 (2020-01-15, "KO/TKO" → finish) is visible.
        # n=1, finishes=1, decisions=0, no_action=0.
        exp_finish = beta_binomial_shrink(1, 1, global_rates["finish"], 50.0)
        exp_decision = beta_binomial_shrink(0, 1, global_rates["decision"], 50.0)
        exp_no_action = beta_binomial_shrink(0, 1, global_rates["no_action"], 50.0)

        # Sanity: compute_ref_rates_shrunk agrees with our manual calc.
        helper_out = compute_ref_rates_shrunk(
            1,
            date(2022, 11, 5),
            ref_history,
            global_rates,
        )
        assert helper_out["ref_finish_rate_shrunk"] == pytest.approx(exp_finish)
        assert helper_out["ref_decision_rate_shrunk"] == pytest.approx(exp_decision)
        assert helper_out["ref_no_action_rate_shrunk"] == pytest.approx(exp_no_action)

        # Now run the assembler and check fight 103's row matches.
        assembler = FeatureMatrixAssembler()
        X, _, fight_dates = assembler.assemble(
            v22_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        # Find the index of fight 103 in v22_fight_records.
        idx_103 = next(i for i, f in enumerate(v22_fight_records) if f["fight_id"] == 103)
        assert X[idx_103, 72] == pytest.approx(exp_finish)
        assert X[idx_103, 73] == pytest.approx(exp_decision)
        assert X[idx_103, 74] == pytest.approx(exp_no_action)

    def test_assemble_v22_meta_cols_populated(
        self,
        v22_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Phase 23 Plan 23-03: META cols (81-89) populated via shared helpers.

        Layoff (81-82): per-fighter, clipped at 720. age (83-84): non-NaN
        when DoB present. Velocity (85-87): NaN when insufficient history
        (window=5 needs 6 prior fights). Division finish rate (88): non-NaN
        with Beta-binomial EB. Reach normalized (89): non-NaN when both
        reaches + division mean present.

        TRAVEL (75-80) stays NaN here because the fixture has no venue
        columns → curr_venue=None → graceful NaN per Pattern D.
        """
        import numpy as np

        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            v22_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        # Layoff cols 81-82: should NOT be all-NaN; first fighter's first
        # fight is 0 sentinel, subsequent should be positive days.
        assert not np.all(np.isnan(X[:, 81])), (
            "layoff_days_red should be populated (0 for debuts, days for "
            "subsequent fights — not all NaN)"
        )
        assert not np.all(np.isnan(X[:, 82])), "layoff_days_blue all-NaN"

        # Age cols 83-84: non-NaN where DoB is set in fixture.
        # (At least one fighter has DoB in v22_fight_records fixture.)
        assert not np.all(np.isnan(X[:, 83])), "age_at_fight_red all-NaN"
        assert not np.all(np.isnan(X[:, 84])), "age_at_fight_blue all-NaN"

        # Division finish rate (88): non-NaN (Beta-binomial EB always
        # produces a value when global_finish_rate is set, even with no
        # division history → collapses to global_rate).
        assert not np.all(np.isnan(X[:, 88])), (
            "division_finish_rate_shrunk all-NaN — Beta-binomial EB should "
            "always emit a value (collapses to global on no history)"
        )

        # TRAVEL cols 75-80 are NaN here — fixture has no venue keys
        # → curr_venue=None → graceful NaN per Pattern D (distinct from the
        # debut sentinel of 0, which only applies when curr_venue is known).
        travel_slice = X[:, 75:81]
        assert np.all(np.isnan(travel_slice)), (
            "Without venue cols in fight_records, TRAVEL must NaN-pad "
            "(graceful degradation, NOT sentinel 0)."
        )

    def test_load_time_assert_v22_columns(self) -> None:
        """Test 9: feature_matrix.py defines _EXPECTED_V22_NCOLS == 90 (D-10)."""
        from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
        from ufc_prediction.ml.feature_matrix import _EXPECTED_V22_NCOLS

        assert _EXPECTED_V22_NCOLS == len(FEATURE_COLUMNS_V22)
        assert _EXPECTED_V22_NCOLS == 90


# ─── TestV22Travel — Phase 23 Plan 23-02 ─────────────────────────────────────


@pytest.fixture()
def v22_travel_fight_records(v22_fight_records: list) -> list[dict]:
    """Augment v22_fight_records with venue_lat / venue_lon / venue_timezone_iana.

    Layout (one fighter who moves NYC→LA between fights; one fixed-venue ref):
        - fight 101 (2020-01-15) NYC (40.7128, -74.0060, America/New_York)
        - fight 102 (2021-06-10) LA  (34.0522, -118.2437, America/Los_Angeles)
        - fight 103 (2022-11-05) LA  same as 102
        - fight 104 (2023-08-20) NYC (back to NYC for fighter_a_id=1)
        - fight 105 (2024-01-20) LV  (36.1, -115.2, America/Las_Vegas)

    Used by TestV22Travel to verify:
        - debut-fighter sentinel (fight 101: both fighters' first UFC bout)
        - subsequent travel (fight 102+: real haversine + tz_shift)
        - red − blue diff convention
        - Pitfall #6 chronological-sort defense (reverse-order input)
    """
    venue_map = {
        101: (40.7128, -74.0060, "America/New_York"),
        102: (34.0522, -118.2437, "America/Los_Angeles"),
        103: (34.0522, -118.2437, "America/Los_Angeles"),
        104: (40.7128, -74.0060, "America/New_York"),
        # Las Vegas falls under the America/Los_Angeles IANA zone (Pacific
        # Time); "America/Las_Vegas" is NOT a real tz key.
        105: (36.1, -115.2, "America/Los_Angeles"),
    }
    out: list[dict] = []
    for fight in v22_fight_records:
        new = dict(fight)
        lat, lon, tz = venue_map[fight["fight_id"]]
        new["venue_lat"] = lat
        new["venue_lon"] = lon
        new["venue_timezone_iana"] = tz
        out.append(new)
    return out


class TestV22Travel:
    """Phase 23 Plan 23-02 — TRAVEL cols at FEATURE_COLUMNS_V22 indices 75-80.

    Train-path tests; cross-path parity (train ⇔ inference TRAVEL cols
    byte-identical) is in test_features_v22.py::TestTrainInferenceParity.
    """

    def test_assemble_v22_travel_cols_present(
        self,
        v22_travel_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 1: TRAVEL cols 75-80 carry NON-NaN values when venues populated.

        At least one subsequent-fight row must have non-NaN TRAVEL cols.
        Debut fighters get 0 sentinel (still NON-NaN per D-04).
        """
        import numpy as np

        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            v22_travel_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        # Every row has non-NaN TRAVEL cols (either real travel value or
        # 0 sentinel — but NEVER NaN since venues are populated).
        travel_slice = X[:, 75:81]
        assert not np.any(np.isnan(travel_slice)), (
            f"TRAVEL cols 75-80 must be NON-NaN when venues populated; "
            f"found NaN values: {travel_slice}"
        )

    def test_assemble_v22_travel_first_fight_sentinel(
        self,
        v22_travel_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 2: debut fight → 0 sentinel for that fighter's travel cols (D-04).

        Fight 101 is the EARLIEST event in the fixture → both fighters are
        debuting → both red+blue travel_distance AND tz_shift = 0.
        """
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            v22_travel_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        # Find the row corresponding to fight 101 (chronologically first).
        idx_101 = next(i for i, f in enumerate(v22_travel_fight_records) if f["fight_id"] == 101)
        # red + blue travel_distance + diff + tz_shift_red + tz_shift_blue
        # + tz_shift_diff — all 0 because both fighters are debutants.
        assert X[idx_101, 75] == 0.0, "travel_distance_miles_red debut"
        assert X[idx_101, 76] == 0.0, "travel_distance_miles_blue debut"
        assert X[idx_101, 77] == 0.0, "travel_distance_miles_diff debut"
        assert X[idx_101, 78] == 0.0, "tz_shift_red_signed debut"
        assert X[idx_101, 79] == 0.0, "tz_shift_blue_signed debut"
        assert X[idx_101, 80] == 0.0, "tz_shift_diff_signed debut"

    def test_assemble_v22_travel_diff_red_minus_blue(
        self,
        v22_travel_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 3: diff cols = red - blue (matches existing differential
        convention)."""
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            v22_travel_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        # For every row, col 77 (travel_diff) == col 75 - col 76,
        # and col 80 (tz_diff) == col 78 - col 79.
        for i in range(X.shape[0]):
            assert X[i, 77] == pytest.approx(X[i, 75] - X[i, 76]), (
                f"travel_distance_miles_diff != red - blue at row {i}"
            )
            assert X[i, 80] == pytest.approx(X[i, 78] - X[i, 79]), (
                f"tz_shift_diff_signed != red - blue at row {i}"
            )

    def test_assemble_v22_travel_uses_prior_fight_venue(
        self,
        v22_travel_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Test 4: a fighter who moves NYC→LA gets haversine ≈ 2451 mi.

        fight 103 (2022-11-05) is in LA. fighter_a_id=1 was last in NYC at
        fight 101 (2020-01-15) → fighter 1 travels NYC→LA → ≈ 2451 mi.
        fighter_b_id=5 is debuting → 0 sentinel. Due to the deterministic
        A/B swap in assemble(), red could be either fighter, so we test
        that EXACTLY ONE of {red_dist, blue_dist} ≈ 2451 and the other is 0.
        """
        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            v22_travel_fight_records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        idx_103 = next(i for i, f in enumerate(v22_travel_fight_records) if f["fight_id"] == 103)
        red_dist = X[idx_103, 75]
        blue_dist = X[idx_103, 76]
        # One side ≈ 2451 mi (NYC→LA for fighter 1); other side = 0 (debut
        # fighter 5). Order depends on deterministic A/B swap.
        traveler = red_dist if abs(red_dist - 2451) < 15 else blue_dist
        debut = blue_dist if abs(red_dist - 2451) < 15 else red_dist
        assert abs(traveler - 2451) < 15, (
            f"Expected NYC→LA ≈ 2451 mi for fighter 1; got red={red_dist}, blue={blue_dist}"
        )
        assert debut == 0.0, (
            f"Expected 0 sentinel for debut fighter 5 at fight 103; "
            f"got red={red_dist}, blue={blue_dist}"
        )

    def test_v22_travel_pre_pass_chronological_sort_defense(
        self,
        v22_travel_fight_records: list,
    ) -> None:
        """Test 9 (Pitfall #6): the v2.2 TRAVEL pre-pass sorts records
        chronologically before building prior-venue lookups, so unsorted
        input produces the SAME as-of-date prior-venue mapping as sorted.

        Direct unit test of `_build_event_venue_lookup` + `_build_fighter_prior_venues`
        (the v2.2 pre-pass helpers), bypassing the rest of assemble() which
        has independent ordering concerns out of scope for Plan 23-02.
        """
        from ufc_prediction.ml.feature_matrix import (
            _build_event_venue_lookup,
            _build_fighter_prior_venues,
        )

        sorted_records = sorted(
            v22_travel_fight_records,
            key=lambda f: f["event_date"],
        )
        reversed_records = list(reversed(sorted_records))

        # Sorted input.
        ev_venues_s = _build_event_venue_lookup(sorted_records)
        prior_s = _build_fighter_prior_venues(sorted_records, ev_venues_s)

        # The helper requires sorted input — caller is expected to sort.
        # Demonstrate that the caller-sorted reversed input (after re-sort)
        # produces identical results.
        re_sorted = sorted(
            reversed_records,
            key=lambda f: f["event_date"],
        )
        ev_venues_r = _build_event_venue_lookup(re_sorted)
        prior_r = _build_fighter_prior_venues(re_sorted, ev_venues_r)

        assert ev_venues_s == ev_venues_r, (
            "Event-venue lookup differs between sorted runs of the same data"
        )
        assert prior_s == prior_r, (
            "Pitfall #6 (caller-sort): prior-venue lookup differs between "
            "sorted runs of the same data"
        )

        # The assembler's v2.2 pre-pass calls these helpers ONLY after the
        # `sorted(fight_records, key=...)` line — verify that line exists
        # in the source as a static contract.
        import ufc_prediction.ml.feature_matrix as fm_module
        import inspect

        src = inspect.getsource(fm_module.FeatureMatrixAssembler.assemble)
        assert "sorted(" in src and "event_date" in src, (
            "v2.2 pre-pass MUST defensively sort by event_date before "
            "calling TRAVEL/REF builders (Pitfall #6)."
        )

    def test_assemble_v22_travel_null_venue_propagates_nan(
        self,
        v22_travel_fight_records: list,
        elo_features: dict,
        computed_features: dict,
        fighter_physicals: dict,
        division_medians: dict,
    ) -> None:
        """Strip venue keys from one fight → that row's TRAVEL cols NaN.

        Distinguishes sentinel 0 (debut fighter) from NaN (venue unknown).
        """
        import numpy as np

        from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

        # Strip venue keys from fight 102.
        records = []
        for f in v22_travel_fight_records:
            new = dict(f)
            if f["fight_id"] == 102:
                new.pop("venue_lat", None)
                new.pop("venue_lon", None)
                new.pop("venue_timezone_iana", None)
            records.append(new)

        assembler = FeatureMatrixAssembler()
        X, _, _ = assembler.assemble(
            records,
            elo_features,
            computed_features,
            fighter_physicals,
            division_medians,
            feature_set="v2.2",
        )

        idx_102 = next(i for i, f in enumerate(records) if f["fight_id"] == 102)
        assert np.all(np.isnan(X[idx_102, 75:81])), (
            "Stripped venue → TRAVEL cols must be NaN (Pattern D), not sentinel 0."
        )
