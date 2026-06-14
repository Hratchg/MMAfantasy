"""Sentry error tracking init (API-V24-06).

`init_sentry()` is a no-op when `settings.sentry_dsn` is None (default in
dev/test). When set, wires both FastApiIntegration (request capture) and
LoggingIntegration (ERROR-and-above log records become Sentry events).
"""

from __future__ import annotations

import logging

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from ufc_prediction.config import settings


def init_sentry() -> None:
    """Initialize sentry-sdk if a DSN is configured; otherwise no-op."""
    if not settings.sentry_dsn:
        return
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        integrations=[
            FastApiIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
    )
