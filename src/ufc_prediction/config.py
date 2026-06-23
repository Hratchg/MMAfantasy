from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _split_csv(v: str | list[str]) -> list[str]:
    """Parse a comma-separated string into a list of stripped non-empty values.

    Pass-through if input is already a list (so direct in-process construction
    like `Settings(api_keys=["k1", "k2"])` still works for tests). Used as
    a pre-validator for env vars that pydantic-settings would otherwise try
    to JSON-decode and reject for bare comma-separated input.
    """
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://ufc:ufc@localhost:5433/ufc_prediction"
    echo_sql: bool = False

    # Runtime environment selector. Drives env-gated /docs + /redoc behavior
    # (API-V24-05) and other per-env defaults. Bound to UFC_ENV env var (Phase
    # 36 README documents the UFC_-prefixed name); bare ENV accepted too for
    # backward compat with v2.3 deployments.
    env: Literal["dev", "staging", "prod"] = Field(
        default="dev",
        validation_alias=AliasChoices("UFC_ENV", "ENV"),
    )

    # API-V24-01 storage decision: plaintext keys in env, host secret store encrypted
    # at rest. Format: "partner_label:secret_token" entries, comma-separated.
    # Default empty so missing env var blocks all API access by default.
    # NoDecode disables JSON decoding so the field_validator below handles
    # the CSV split (pydantic-settings would otherwise reject bare CSV input).
    api_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("UFC_API_KEYS", "API_KEYS"),
    )

    # CORS allowlist (API-V24-03). Default empty so missing env var denies
    # preflight by default — operator MUST set UFC_CORS_ORIGINS in prod.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("UFC_CORS_ORIGINS", "CORS_ORIGINS"),
    )

    # Structured logging level (API-V24-06).
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("UFC_LOG_LEVEL", "LOG_LEVEL"),
    )

    # Sentry error tracking (API-V24-06). None disables Sentry init.
    sentry_dsn: str | None = None

    # Sentry trace sample rate (0.0 = no tracing, 1.0 = trace all). Default
    # 0.1 follows Sentry recommendation for production cost balance.
    sentry_traces_sample_rate: float = 0.1

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_api_keys(cls, v: str | list[str]) -> list[str]:
        return _split_csv(v)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: str | list[str]) -> list[str]:
        return _split_csv(v)


settings = Settings()
