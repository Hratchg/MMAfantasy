"""Tests for CLI predict sub-app.

Covers predict_app registration, command existence, and callback
for 'ufc predict A vs B' syntax per D-10.
"""

from __future__ import annotations

import typer


class TestPredictAppRegistration:
    """Tests for predict_app as a Typer sub-app."""

    def test_predict_app_is_typer_instance(self):
        """predict_app is a Typer instance with name 'predict'."""
        from ufc_prediction.cli.predict import predict_app

        assert isinstance(predict_app, typer.Typer)

    def test_predict_app_has_name(self):
        """predict_app has the expected name."""
        from ufc_prediction.cli.predict import predict_app

        assert predict_app.info.name == "predict"

    def test_train_command_exists(self):
        """predict_app has a 'train' command."""
        from ufc_prediction.cli.predict import predict_app

        command_names = [
            cmd.name or cmd.callback.__name__ for cmd in predict_app.registered_commands
        ]
        assert "train" in command_names

    def test_evaluate_command_exists(self):
        """predict_app has an 'evaluate' command."""
        from ufc_prediction.cli.predict import predict_app

        command_names = [
            cmd.name or cmd.callback.__name__ for cmd in predict_app.registered_commands
        ]
        assert "evaluate" in command_names

    def test_callback_exists(self):
        """predict_app exposes 'A vs B' prediction via the 'matchup' subcommand
        (which replaced the earlier top-level predict-callback shorthand)."""
        from ufc_prediction.cli.predict import predict_app

        command_names = [
            cmd.name or cmd.callback.__name__ for cmd in predict_app.registered_commands
        ]
        assert "matchup" in command_names

    def test_main_includes_predict(self):
        """Main app has predict_app registered."""
        from ufc_prediction.cli.main import app

        # Check that predict sub-app is registered
        sub_app_names = []
        for group in app.registered_groups:
            if group.typer_instance and group.typer_instance.info.name:
                sub_app_names.append(group.typer_instance.info.name)
        assert "predict" in sub_app_names


class TestPredictMatchupBoutContext:
    """Operator-approved D1: `ufc predict matchup` forwards --weight-class,
    --rounds and --title/--no-title to ModelPredictor.predict on both
    order-invariant runs. Omitted flags forward nothing, so the feature builder
    keeps the stored-row / fallback behaviour."""

    _RESULT = {
        "fighter_a": "Fighter A",
        "fighter_b": "Fighter B",
        "win_probability": 0.6,
        "model_probability_a": 0.6,
        "base_prob": 0.6,
        "meta_prob": None,
    }

    def _invoke(self, *extra):
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from ufc_prediction.cli.predict import predict_app

        mod = "ufc_prediction.cli.predict"
        # autospec: the mocked predict() enforces the real ModelPredictor.predict
        # signature, so a misnamed kwarg raises instead of passing silently.
        with (
            patch(f"{mod}.ModelPredictor", autospec=True) as mock_predictor_cls,
            patch(f"{mod}.SessionLocal", return_value=MagicMock()),
            patch(f"{mod}._display_prediction"),
        ):
            predict = mock_predictor_cls.return_value.predict
            predict.return_value = dict(self._RESULT)
            result = CliRunner().invoke(
                predict_app, ["matchup", "Fighter A", "vs", "Fighter B", *extra]
            )
        return result, predict

    def test_flags_forwarded_to_both_runs(self):
        result, predict = self._invoke("--weight-class", "Welterweight", "--rounds", "5", "--title")
        assert result.exit_code == 0, (result.output, result.exception)
        assert predict.call_count == 2
        for call in predict.call_args_list:
            assert call.kwargs == {
                "weight_class": "Welterweight",
                "num_rounds": 5,
                "is_title_fight": True,
            }

    def test_no_title_forwards_false(self):
        result, predict = self._invoke("--rounds", "3", "--no-title")
        assert result.exit_code == 0, (result.output, result.exception)
        for call in predict.call_args_list:
            assert call.kwargs == {"num_rounds": 3, "is_title_fight": False}

    def test_omitted_flags_forward_nothing(self):
        result, predict = self._invoke()
        assert result.exit_code == 0, (result.output, result.exception)
        assert predict.call_count == 2
        for call in predict.call_args_list:
            assert call.kwargs == {}

    def test_rejects_rounds_other_than_3_or_5(self):
        result, predict = self._invoke("--rounds", "4")
        assert result.exit_code == 2, result.output
        predict.assert_not_called()

    def test_rejects_unknown_weight_class(self):
        result, predict = self._invoke("--weight-class", "welterweight")
        assert result.exit_code == 2, result.output
        predict.assert_not_called()


class TestPredictTrainPassesFeatureColumns:
    """S02 finding 1: `predict train` must hand its --feature-set column list to
    ModelTrainer.train so importances are keyed at the trained width (72 for the
    default v2.1-no-net), not the 75-col FEATURE_COLUMNS default."""

    def test_default_feature_set_columns_reach_trainer(self, tmp_path):
        from datetime import date
        from unittest.mock import MagicMock, patch

        import numpy as np
        from typer.testing import CliRunner

        from ufc_prediction.cli.predict import predict_app
        from ufc_prediction.ml.config import get_feature_columns

        cols = get_feature_columns(feature_set="v2.1-no-net")
        width = len(cols)
        mod = "ufc_prediction.cli.predict"
        with (
            patch(f"{mod}.SessionLocal", return_value=MagicMock()),
            patch(f"{mod}.load_fight_records", return_value=[]),
            patch(f"{mod}.load_elo_features", return_value={}),
            patch(f"{mod}.load_computed_features", return_value={}),
            patch(f"{mod}.load_fighter_physicals", return_value={}),
            patch(f"{mod}.load_round_stats_for_ml", return_value={}),
            patch(f"{mod}.load_pre_ufc_records", return_value={}),
            patch(f"{mod}.load_fight_odds", return_value={}),
            patch(f"{mod}.compute_division_medians", return_value={}),
            patch(f"{mod}.FeatureMatrixAssembler") as mock_asm_cls,
            patch(
                f"{mod}.split_temporal",
                return_value=(
                    np.zeros((2, width)),
                    np.zeros((2, width)),
                    np.zeros(2, dtype=np.int32),
                    np.zeros(2, dtype=np.int32),
                ),
            ),
            patch(f"{mod}.ModelTrainer") as mock_trainer_cls,
            patch(
                f"{mod}.evaluate_model",
                return_value={"brier_score": 0.2, "auc_roc": 0.7, "accuracy": 0.7},
            ),
            patch(f"{mod}.save_model", return_value=tmp_path / "xgb_s02.joblib"),
            patch(f"{mod}._display_training_summary"),
        ):
            mock_asm_cls.return_value.assemble.return_value = (
                np.zeros((2, width)),
                np.zeros(2, dtype=np.int32),
                np.array([date(2020, 1, 1)] * 2, dtype=object),
            )
            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.train.return_value = (MagicMock(), {}, {})

            result = CliRunner().invoke(
                predict_app,
                [
                    "train",
                    "--trials",
                    "1",
                    "--version",
                    "s02",
                    "--model-dir",
                    str(tmp_path),
                    "--force",
                ],
            )

        assert result.exit_code == 0, (result.output, result.exception)
        mock_trainer.train.assert_called_once()
        assert mock_trainer.train.call_args.kwargs.get("feature_columns") == cols
