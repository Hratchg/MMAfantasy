"""Structured JSON logging configuration (API-V24-06).

`configure_logging()` installs a `python-json-logger` formatter on the
root logger and enriches every record with the current request_id and
partner_label contextvars. Phase 35-02 builds the formatter; Plan 35-06
calls `configure_logging()` from `create_app()`.

Log record schema (per CONTEXT specifics):
    {
      "timestamp": "...",
      "level": "INFO",
      "logger": "ufc_prediction.api.routers.predict",
      "message": "...",
      "request_id": "<uuid4 or '-'>",
      "partner_label": "<partner or null>",
      ...                                # path / method / status_code / duration_ms
                                         # injected by access-log middleware (Plan 35-06)
    }
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger

from ufc_prediction.api.middleware.request_id import (
    partner_label_ctxvar,
    request_id_ctxvar,
)
from ufc_prediction.config import settings


class ContextEnrichingJsonFormatter(jsonlogger.JsonFormatter):  # type: ignore[misc,name-defined]
    """JsonFormatter that injects request-scoped contextvars into every record."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["request_id"] = request_id_ctxvar.get()
        log_record["partner_label"] = partner_label_ctxvar.get()


def configure_logging() -> None:
    """Install JSON formatter on the root logger.

    Idempotent — clears existing handlers before adding the JSON one so
    repeated calls (e.g. across test fixtures) don't stack handlers.
    Honors `settings.log_level`.
    """
    formatter = ContextEnrichingJsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Drop existing handlers to keep idempotent behavior under repeat init.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level)
