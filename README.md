# UFC Fight Prediction

A Python service that estimates win probabilities for UFC matchups. Give it two fighter names, get back a calibrated probability for each and an explainable breakdown (Elo ratings, top contributing features).

**Current release: [v3.0](CHANGELOG.md)** (2026-06-10) — handoff package. About **75% accuracy** on recent fights, in line with the closing betting line (which itself hits ~70–75% on this sport).

> Want to use this? **→ [`docs/INSTALL.md`](docs/INSTALL.md)** has a tested one-page walkthrough from `git clone` to your first prediction.
> Want to contribute? **→ [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** + [`docs/GLOSSARY.md`](docs/GLOSSARY.md) + [`CONTRIBUTING.md`](CONTRIBUTING.md).
> Taking ownership of the project? **→ [`TECHNICAL_HANDOFF.md`](TECHNICAL_HANDOFF.md)**.
> Hit a problem? **→ [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)**.

---

## What it is, in plain English

A predictor with three layers stacked on top of each other:

1. **Elo ratings.** Per-fighter, per-skill (overall / striking / grappling), updated chronologically after every fight.
2. **An XGBoost model** trained on 72 features per matchup (Elo differences, reach / height / age gaps, recent finish rate, days since last fight, betting odds where available, etc.).
3. **A logistic regression on top** that mixes the XGBoost output with closing betting odds to produce the final probability.

The combined model is called **META-V22** and lives at `models/meta/meta_v2.joblib`. It's the canonical one and is selected by default.

The accuracy ceiling on UFC fights is real: single punches flip 8-second outcomes, and the closing betting market itself only hits ~70–75%. This model gets to the same neighbourhood, but its real value is the *explainability* — "fighter A wins 56% because of a +22 striking-Elo edge and a +0.04 closing-odds delta", not just a bare probability.

## Quickstart

Five-minute version (full walkthrough in [`docs/INSTALL.md`](docs/INSTALL.md)):

```bash
# 1. Clone + install Python deps (needs Python 3.13)
git clone <repo-url>
cd ufc-fight-prediction
uv sync

# 2. Install pg_restore (needed by `ufc db seed`)
brew install libpq
echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 3. Start Postgres + load the corpus dump (Postgres 17+ required; see docs/INSTALL.md)
docker compose up -d db   # canonical: host port 5433, db ufc_prediction, postgres:18
export DATABASE_URL='postgresql+psycopg://ufc:ufc@localhost:5433/ufc_prediction'
uv run ufc db seed

# 4. Predict a matchup (the `vs` literal is required)
uv run ufc predict matchup "Israel Adesanya" vs "Sean Strickland"
```

You'll see a Rich-formatted table with win probability, Elo ratings, and the top contributing features.

## Documentation by audience

| You are… | Read |
|---|---|
| A new user wanting to run this locally | [`docs/INSTALL.md`](docs/INSTALL.md) |
| A partner integrating the API | [`docs/QUICKSTART-PARTNER.md`](docs/QUICKSTART-PARTNER.md) + [`docs/PARTNER-CONTRACT.md`](docs/PARTNER-CONTRACT.md) + JSON schemas in [`src/ufc_prediction/contracts/`](src/ufc_prediction/contracts/) |
| A new contributor to the codebase | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) → [`docs/GLOSSARY.md`](docs/GLOSSARY.md) → [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Taking ownership of the project | [`TECHNICAL_HANDOFF.md`](TECHNICAL_HANDOFF.md) — the practical handoff doc |
| Wondering what's broken | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) — per-scraper status + v3.0 regressions |
| Curious about the model methodology | [`METHODOLOGY_CLIENT.md`](METHODOLOGY_CLIENT.md) |
| Curious about the release history | [`CHANGELOG.md`](CHANGELOG.md) |

## Repository layout

```
src/ufc_prediction/
├── api/          FastAPI HTTP layer (v1/, routers/, app.py)
├── cli/          Typer-based `ufc` CLI commands
├── ml/           ML primitives (predictor, evaluator, gate verifier)
├── elo/          Multi-domain Elo computation (overall/striking/grappling)
├── features/     Feature engineering (FeatureMatrixAssembler)
├── contracts/    Partner-facing JSON schemas + OpenAPI specs
├── scraper/      Data ingestion (UFCStats, Sherdog, BFO)
├── matchup/      Fighter-pair comparison helpers
└── db/           SQLAlchemy 2.x ORM models

migrations/       Alembic migrations
scripts/          One-off CLIs + PDF generators
docs/             User + developer + partner docs
data/seed/        Portable corpus dump (ufc_corpus_v30.dump)
models/           Production model artifacts (joblib + JSON metadata)
tests/            pytest 8.x — unit, integration, regression, contracts
.planning/        Internal planning artifacts (GSD workflow; not required reading)
```

## Tech stack

- **Python 3.13+** (`pyproject.toml` pin: `>=3.13.0,!=3.13.1`)
- **Package manager:** [uv](https://github.com/astral-sh/uv) — required (not pip / poetry)
- **Web:** FastAPI 0.128+, uvicorn, Pydantic 2.13+
- **ML:** XGBoost 3.2+, scikit-learn 1.6+, scipy 1.17+, numpy 2.0+
- **Data:** SQLAlchemy 2.0+, Alembic, PostgreSQL (psycopg 3.3+)
- **CLI:** Typer 0.24+, rich
- **Testing:** pytest 8.x

## Developer setup

Pre-commit hooks are installed on first clone:

```bash
# 1. Install uv (one-time, system-wide)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync project dependencies
uv sync --frozen

# 3. Install pre-commit + hook environments
uv tool install pre-commit
pre-commit install

# 4. (Optional) run all hooks against the full tree once
pre-commit run --all-files
```

The hook chain runs on every `git commit`: `ruff` lint+format, `mypy --strict` against the admit-list in `pyproject.toml`, file hygiene checks, and the **audit-01-protected-files** guard that blocks edits to byte-identity-locked files (see [`CONTRIBUTING.md` § AUDIT-01 protected files](CONTRIBUTING.md#audit-01-protected-files-do-not-casually-edit)).

| Situation | What to do |
| --- | --- |
| Hook surfaces a real problem | Fix the underlying issue. Do NOT bypass. |
| Touching an AUDIT-01 protected file with operator approval | `AUDIT01_OVERRIDE=1 git commit ...` AND cite the override in the PR description per [`CONTRIBUTING.md` checklist](CONTRIBUTING.md). |
| Local hook environment broken (e.g. ruff version skew mid-update) | `git commit --no-verify ...`. CI re-runs the full hook chain, so a pre-commit failure on `master` is impossible. |

CI re-runs everything in [`.github/workflows/ci.yml`](.github/workflows/ci.yml). PRs cannot merge into `master` until both `lint-and-test` and `pre-commit` pass.

## Deploy (Docker)

This project ships with a production-ready `Dockerfile` that runs the FastAPI
app under uvicorn. It runs on any container host (a VM with Docker, ECS, Cloud
Run, Render, Railway, Kubernetes, etc.) — point it at a PostgreSQL database via
`DATABASE_URL` and set the API key(s).

### Prerequisites

- A container host that can run the `Dockerfile`.
- A reachable PostgreSQL database (managed Postgres — RDS / Cloud SQL / Supabase / Neon — or your own). Load the corpus into it once with `ufc db seed` (see [`docs/INSTALL.md`](docs/INSTALL.md)).

### Configuration (environment variables)

| Env Var | Required | Default | Example |
|---|---|---|---|
| `DATABASE_URL` | yes | — | `postgresql+psycopg://user:pass@host/db` |
| `UFC_API_KEYS` | yes | — | `partner-alpha:rawsecret1,partner-beta:rawsecret2` |
| `UFC_CORS_ORIGINS` | no | `[]` | `https://partner.example.com,https://app.partner.com` |
| `SENTRY_DSN` | no | unset | `https://abc123@o0.ingest.sentry.io/0` |
| `SENTRY_TRACES_SAMPLE_RATE` | no | `0.1` | `0.5` |
| `UFC_LOG_LEVEL` | no | `INFO` | `DEBUG` |
| `UFC_ENV` | no | `dev` | `prod` (gates `/docs` + `/redoc` to 404) |

### Build, run + verify

```bash
docker build -t ufc-fp .

# The container serves on port 8080 (override with -e PORT=...).
docker run -d -p 8080:8080 \
  -e DATABASE_URL='postgresql+psycopg://user:pass@host/db' \
  -e UFC_API_KEYS='partner-alpha:rawsecret1' \
  -e UFC_ENV='prod' \
  ufc-fp

# Liveness (no auth)
curl http://localhost:8080/health
# {"status":"ok","version":"2.3.0"}

# Predict (requires X-API-Key)
curl -X POST http://localhost:8080/api/v1/predict \
  -H "X-API-Key: partner-alpha:rawsecret1" \
  -H "Content-Type: application/json" \
  -d '{"fighter_a": "Conor McGregor", "fighter_b": "Khabib Nurmagomedov"}'
```

## Data sources & attribution

The model is built on three external data sources. Each is acknowledged below with its current TOS / use posture per the operator's good-faith reading.

- **UFCStats** — <https://ufcstats.com>. Public per-fight statistics. Pure-fantasy framing. Cache lives under `data/ufcstats_event_detail_cache/` (runtime-only).
- **Sherdog** — <https://www.sherdog.com>. Public fight database used for fighter-career profile reconstruction. Pure-fantasy framing. Cache under `data/sherdog_html_cache/` (runtime-only).
- **BestFightOdds** — <https://www.bestfightodds.com>. Closing-odds aggregator; commercial use assumed contractually OK per the operator. Refreshed weekly via [`.github/workflows/refresh-odds.yml`](.github/workflows/refresh-odds.yml). Pure-fantasy framing; this is not a sportsbook.

## Disclaimer

> **Entertainment / informational use only.** This software is provided "as is"; partners are responsible for assessing fitness for purpose and complying with applicable laws regarding fantasy contests and games of skill.
>
> UFC Fight Prediction is a fantasy-MMA decision-support tool intended to help fantasy players reason about matchups using historical performance data. For informational and entertainment purposes; not financial or wagering advice; statistical estimates may diverge from actual outcomes. Predictions are NOT predictions of future events, NOT guarantees of any particular outcome, and NOT a recommendation to place, modify, or refrain from any wager, fantasy pick, or other transaction. Model uncertainty is real and material: closing-odds-derived features can be missing or stale at predict time, sample sizes for debutants and short-tenured fighters are small, and the underlying corpus is incomplete in well-documented ways. Partners integrating this API into a downstream product are solely responsible for assessing fitness for purpose, complying with local laws regarding fantasy contests, sports wagering, and games of skill, and clearly framing model outputs to their end users as statistical estimates rather than outcome predictions or guaranteed results.

The full 200-word disclaimer is at [`src/ufc_prediction/api/disclaimer.py`](src/ufc_prediction/api/disclaimer.py) and is surfaced in every `/api/v1/predict` response.

## License

MIT License. See [`LICENSE`](LICENSE) for the full text.
