"""Tests for configure_logging() + init_sentry() (Plan 35-02, API-V24-06)."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from ufc_prediction.api.middleware.request_id import (
    partner_label_ctxvar,
    request_id_ctxvar,
)
from ufc_prediction.obs import logging as obs_logging
from ufc_prediction.obs import sentry as obs_sentry


@pytest.fixture(autouse=True)
def reset_root_logger():
    """Strip handlers before/after each test to keep tests isolated."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for h in saved_handlers:
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


@pytest.fixture(autouse=True)
def reset_contextvars():
    """Snapshot contextvars and reset on teardown."""
    rid_token = request_id_ctxvar.set(request_id_ctxvar.get())
    pl_token = partner_label_ctxvar.set(partner_label_ctxvar.get())
    yield
    request_id_ctxvar.reset(rid_token)
    partner_label_ctxvar.reset(pl_token)


def _captured_json_line(capsys) -> dict:
    captured = capsys.readouterr().err.strip().splitlines()
    assert captured, "expected at least one log line on stderr"
    return json.loads(captured[-1])


def test_configure_logging_emits_json_with_required_fields(capsys):
    obs_logging.configure_logging()
    logging.getLogger("ufc_prediction").info("hello")
    record = _captured_json_line(capsys)
    for key in ("timestamp", "level", "logger", "message", "request_id"):
        assert key in record, f"missing key {key!r} in {record!r}"
    assert record["message"] == "hello"
    assert record["logger"] == "ufc_prediction"
    assert record["level"] == "INFO"


def test_configure_logging_injects_request_id_from_contextvar(capsys):
    obs_logging.configure_logging()
    token = request_id_ctxvar.set("abc-123-")
    try:
        logging.getLogger("ufc_prediction").info("scoped")
    finally:
        request_id_ctxvar.reset(token)
    record = _captured_json_line(capsys)
    assert record["request_id"] == "abc-123-"


def test_configure_logging_request_id_is_dash_outside_scope(capsys):
    obs_logging.configure_logging()
    logging.getLogger("ufc_prediction").info("outside")
    record = _captured_json_line(capsys)
    assert record["request_id"] == "-"


def test_init_sentry_no_op_without_dsn(monkeypatch):
    monkeypatch.setattr(obs_sentry.settings, "sentry_dsn", None)
    mock_init = MagicMock()
    monkeypatch.setattr(obs_sentry.sentry_sdk, "init", mock_init)
    obs_sentry.init_sentry()
    mock_init.assert_not_called()


def test_init_sentry_wires_integrations_when_dsn_set(monkeypatch):
    monkeypatch.setattr(
        obs_sentry.settings,
        "sentry_dsn",
        "https://abc@sentry.example.com/1",
    )
    monkeypatch.setattr(obs_sentry.settings, "sentry_traces_sample_rate", 0.25)
    mock_init = MagicMock()
    monkeypatch.setattr(obs_sentry.sentry_sdk, "init", mock_init)
    obs_sentry.init_sentry()
    assert mock_init.called
    kwargs = mock_init.call_args.kwargs
    assert kwargs["dsn"] == "https://abc@sentry.example.com/1"
    assert kwargs["traces_sample_rate"] == 0.25
    integrations = kwargs["integrations"]
    int_types = {type(i).__name__ for i in integrations}
    assert "FastApiIntegration" in int_types
    assert "LoggingIntegration" in int_types


def test_configure_logging_honors_log_level(monkeypatch, capsys):
    monkeypatch.setattr(obs_logging.settings, "log_level", "DEBUG")
    obs_logging.configure_logging()
    assert logging.getLogger().level == logging.DEBUG

    monkeypatch.setattr(obs_logging.settings, "log_level", "INFO")
    obs_logging.configure_logging()
    assert logging.getLogger().level == logging.INFO
