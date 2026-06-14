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
            cmd.name or cmd.callback.__name__
            for cmd in predict_app.registered_commands
        ]
        assert "train" in command_names

    def test_evaluate_command_exists(self):
        """predict_app has an 'evaluate' command."""
        from ufc_prediction.cli.predict import predict_app

        command_names = [
            cmd.name or cmd.callback.__name__
            for cmd in predict_app.registered_commands
        ]
        assert "evaluate" in command_names

    def test_callback_exists(self):
        """predict_app has a callback for 'predict A vs B' syntax."""
        from ufc_prediction.cli.predict import predict_app

        assert predict_app.registered_callback is not None

    def test_main_includes_predict(self):
        """Main app has predict_app registered."""
        from ufc_prediction.cli.main import app

        # Check that predict sub-app is registered
        sub_app_names = []
        for group in app.registered_groups:
            if group.typer_instance and group.typer_instance.info.name:
                sub_app_names.append(group.typer_instance.info.name)
        assert "predict" in sub_app_names
