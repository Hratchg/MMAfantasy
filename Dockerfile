# syntax=docker/dockerfile:1.7

# ============================================================
# Builder stage — Astral uv official image (Python 3.13)
# ============================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# uv-managed environment
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.13

WORKDIR /app

# Install deps WITHOUT the project itself (cache layer keyed on lockfile)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

# Now copy the project source and install it into the venv
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY models/ ./models/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

# ============================================================
# Runtime stage — minimal python:3.13-slim (no build toolchain)
# ============================================================
FROM python:3.13-slim AS runtime

# OCI labels (REQ DEPLOY-V24-01 — version 2.3.0 sourced from 36-01 Task 3)
LABEL org.opencontainers.image.title="UFC Fight Prediction" \
      org.opencontainers.image.description="Production API for UFC fight prediction (Elo + ML)" \
      org.opencontainers.image.version="2.3.0" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/ufc-fight-prediction/ufc-fight-prediction"

# Runtime deps: curl for HEALTHCHECK; tini for PID 1 signal handling (optional but standard)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (uid 10001 per locked decision)
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app

# Copy venv from builder (deps + project installed there)
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy source + migrations + alembic.ini + models (runtime needs these at /app)
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/migrations /app/migrations
COPY --from=builder --chown=app:app /app/alembic.ini /app/alembic.ini
COPY --from=builder --chown=app:app /app/models /app/models

# Prepend venv binaries to PATH so `uvicorn`, `alembic`, `python` come from the venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "ufc_prediction.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
