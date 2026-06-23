# Install & Use — UFC Fight Prediction

> This page takes a technical Python user from `git clone` to a first prediction in
> roughly five minutes of reading. No prior UFC ML background is assumed. By the end
> you will have a local Postgres corpus, three working CLI commands
> (`ufc predict matchup`, `ufc fighter lookup`, `ufc fighter rankings`), and an
> optional FastAPI server you can hit with `curl`.

## Prerequisites

- Python `>=3.13.0,!=3.13.1` — pinned in `pyproject.toml`.
- [`uv`](https://github.com/astral-sh/uv) — canonical install / venv manager for this repo. `uv` is the only supported toolchain entry point; `pip install -e .` is unsupported.
- Docker (recommended) **or** a local Postgres 17+ install. (Postgres **17 or newer** — `ufc db seed` restores via a modern host `pg_restore`, which emits a `SET transaction_timeout` that Postgres 16 servers reject.)
- `pg_restore` on your `PATH` — `ufc db seed` shells out to it. On macOS: `brew install libpq` then `export PATH="/opt/homebrew/opt/libpq/bin:$PATH"`. On Linux it ships with `postgresql-client`. (`psql` from the same package is handy for the connectivity check below but not strictly required.)
- Linux or macOS. Windows is out of scope for v3.0.

## 1. Clone the repo

```bash
git clone https://github.com/<your-org>/ufc-fight-prediction.git
cd ufc-fight-prediction
```

**What you should see:** a new `ufc-fight-prediction/` directory containing
`README.md`, `pyproject.toml`, `src/ufc_prediction/`, `docs/`, and `models/`.

## 2. Install Python deps

```bash
uv sync
```

**What you should see:** `uv` resolves the lockfile, creates `.venv/`, and ends
with a line like `Installed N packages in …s`. The `ufc` CLI entry point becomes
available via `uv run ufc …`. If `uv` itself is missing, install it first with
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

## 3. Start a local Postgres

Primary path — Docker Compose (recommended). This is the canonical local DB: it
matches the `database_url` default in `config.py`, `.env.example`, and the
shipped corpus provenance (host port **5433**, database **`ufc_prediction`**).

```bash
docker compose up -d db
```

Raw `docker run` alternative (same credentials, host port `5433`, Postgres 18):

```bash
docker run -d --name ufc-pg \
  -e POSTGRES_USER=ufc \
  -e POSTGRES_PASSWORD=ufc \
  -e POSTGRES_DB=ufc_prediction \
  -p 5433:5432 \
  postgres:18-alpine
```

> **Postgres version:** use **17 or newer**. `ufc db seed` restores the corpus
> with the host `pg_restore`; a modern client (17/18) emits `SET
> transaction_timeout`, which a Postgres 16 server rejects and the restore
> fails. The bundled `docker-compose.yml` pins `postgres:18` for this reason.

Bare-metal fallback: install Postgres 17+ (`brew install postgresql@17` on macOS, distro package on Linux), then create role `ufc` and database `ufc_prediction` matching the credentials above (and either run it on port `5433` or adjust `DATABASE_URL`).

Create a project-local `.env` file at the repo root (the `postgresql+psycopg://`
scheme is required — SQLAlchemy rejects a bare `postgres://` URL):

```env
DATABASE_URL=postgresql+psycopg://ufc:ufc@localhost:5433/ufc_prediction
```

**What you should see:** `docker ps` lists the Postgres container as `Up`. `psql "$DATABASE_URL" -c '\dt'` connects and returns an empty table list (that is expected pre-seed).

## 4. Load the corpus seed

Primary — load the shipped corpus snapshot:

```bash
uv run ufc db seed
```

`ufc db seed` pops a portable corpus snapshot (fights, fighters, Elo history,
odds, computed features) into your local Postgres in one shot. The snapshot
ships at `data/seed/ufc_corpus_v30.dump` (pg_dump custom-format, compressed);
restore takes ~30-60s on a clean container. After restore, the command runs a
`ModelPredictor` sanity check pinned to the canonical META-V22 stack (xgb_v2 +
meta_v2) to catch the predictor-default-version regression early — see
`KNOWN_ISSUES.md` (Phase 89 deliverable) for that story.

Use `uv run ufc db status` after restore to verify per-table row counts and
the alembic head pointer landed cleanly.

Secondary — empty-schema bootstrap (no corpus data):

```bash
uv run alembic upgrade head
```

This creates every table from the Alembic migration chain but leaves them
empty. You can then ingest data manually via the scrapers under
`src/ufc_prediction/data/` if you do not want to wait for the corpus dump.

**What you should see:** `ufc db seed` completes with row counts per table
(12 rows, ending with `alembic_version=1`) followed by the green
`✓ ModelPredictor v2 instantiated` line. `alembic upgrade head` reports a sequence of
`Running upgrade … -> …` lines and exits 0.

## 5. Predict

Three CLI commands cover the day-to-day operator surface. Run them in order.

### Matchup prediction

```bash
uv run ufc predict matchup "Israel Adesanya" vs "Sean Strickland"
```

```
╭────────────────────────────── Fight Prediction ──────────────────────────────╮
│ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓                           │
│ ┃            ┃ Israel Adesanya ┃ Sean Strickland ┃                           │
│ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩                           │
│ │ ML Model   │      49.3%      │      50.7%      │                           │
│ │ Elo Rating │      1532       │      1554       │                           │
│ │ Elo Prob   │      46.9%      │      53.1%      │                           │
│ └────────────┴─────────────────┴─────────────────┘                           │
╰──────────────────────────────────────────────────────────────────────────────╯
        Top Contributing Features
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Feature                  ┃ Importance ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ closing_prob_diff        │     0.0511 │
│ elo_overall_diff         │     0.0287 │
│ odds_elo_divergence      │     0.0273 │
│ opening_prob_diff        │     0.0221 │
│ age_diff                 │     0.0214 │
│ is_debut_diff            │     0.0190 │
│ win_streak_diff          │     0.0189 │
│ opp_adj_ctrl_time_diff   │     0.0159 │
│ loss_streak_diff         │     0.0159 │
│ is_short_turnaround_diff │     0.0154 │
└──────────────────────────┴────────────┘
```

> **What you should see:** the table above, with `ML Model` giving Adesanya
> 49.3% and Strickland 50.7%. (The prediction is order-invariant — you get the
> same numbers whichever fighter you list first.) Numbers will drift over time
> as the corpus refreshes; what matters is that the command exits 0 and prints
> both blocks.

The `ML Model` row is the META-V22 stacked output — a calibrated win
probability that combines the gradient-boosted base model (`xgb_v2`) with the
Elo signal. The `Elo Rating` row is each fighter's current middleweight Elo,
and `Elo Prob` is the head-to-head probability implied by the Elo gap alone.
When ML and Elo disagree, the ML number is the authoritative pick — Elo is
shown for context.

### Fighter lookup

```bash
uv run ufc fighter lookup "Israel Adesanya"
```

```
Israel Adesanya

          Light Heavyweight
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value             ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Elo Rating    │ 1561              │
│ Striking Elo  │ 1551              │
│ Grappling Elo │ 1493              │
│ Division      │ Light Heavyweight │
│ Record        │ 0W - 1L           │
│ Total Fights  │ 1                 │
│ Last Fight    │ 2021-03-06        │
└───────────────┴───────────────────┘

          Middleweight
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Metric        ┃ Value        ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Elo Rating    │ 1532         │
│ Striking Elo  │ 1521         │
│ Grappling Elo │ 1525         │
│ Division      │ Middleweight │
│ Record        │ 13W - 5L     │
│ Total Fights  │ 18           │
│ Last Fight    │ 2026-03-28   │
└───────────────┴──────────────┘
```

> **What you should see:** one card per division the fighter has competed in.
> Adesanya shows both Middleweight (his canonical division) and a one-fight
> Light Heavyweight stint from 2021.

Each card reports the fighter's per-division Elo rating decomposed into overall
/ striking / grappling, plus their UFC record and most recent bout. Per
CLAUDE.md, fighters with fewer than ~8 UFC fights have higher Elo variance
(small-sample noise is structural, not a bug) — treat the Light Heavyweight
card here as illustrative rather than authoritative.

### Division rankings

```bash
uv run ufc fighter rankings middleweight
```

```
              Middleweight Rankings (Top 15)
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Rank ┃ Fighter            ┃  Elo ┃ Fights ┃ Last Active ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│    1 │ Mike Swick         │ 1656 │      6 │  2007-04-07 │
│    2 │ Brendan Allen      │ 1626 │     16 │  2025-10-18 │
│    3 │ Nassourdine Imavov │ 1618 │     11 │  2025-09-06 │
│    4 │ Dricus Du Plessis  │ 1617 │     10 │  2025-08-16 │
│    5 │ Gegard Mousasi     │ 1613 │     11 │  2017-04-08 │
│    6 │ Demian Maia        │ 1612 │     13 │  2012-01-28 │
│    7 │ Gregory Rodrigues  │ 1608 │     13 │  2026-03-07 │
│    8 │ Elias Theodorou    │ 1601 │     11 │  2019-05-04 │
│    9 │ Thiago Santos      │ 1595 │     15 │  2018-08-04 │
│   10 │ Yushin Okami       │ 1590 │     18 │  2013-09-04 │
│   11 │ Nursulton Ruziboev │ 1583 │      4 │  2025-05-17 │
│   12 │ Joe Pyfer          │ 1582 │      8 │  2026-03-28 │
│   13 │ Alex Pereira       │ 1580 │      5 │  2023-04-08 │
│   14 │ Frank Shamrock     │ 1579 │      4 │  1999-09-24 │
│   15 │ Anthony Smith      │ 1575 │      7 │  2018-02-03 │
└──────┴────────────────────┴──────┴────────┴─────────────┘
```

> **What you should see:** the top-15 Elo ranking for the requested division.
> All-time rankings include historic fighters, so expect a mix of active and
> retired competitors.

Rankings are sorted by the multi-domain Elo: striking output, takedown
efficiency, finish rate, and strength of schedule are folded into a single
per-division rating. Small-sample fighters (`Fights < 8`) can rank high
because Elo rewards quality wins regardless of volume — cross-reference the
`Last Active` column when building a fantasy card.

## 6. (Optional) Serve the FastAPI API

Run the API when you need HTTP access for downstream services, partner
integrations, or Swagger exploration. The CLI commands above are always
available without the server.

```bash
UFC_API_KEYS="dev:devsecret" UFC_ENV="dev" \
  uv run uvicorn --factory ufc_prediction.api.app:create_app --port 8000
```

Then call the predict endpoint with `curl`. The `X-API-Key` header takes the
full `partner:secret` entry (not just the bare secret) — this matches the
`auth.py` contract.

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: dev:devsecret" \
  -H "Content-Type: application/json" \
  -d '{"fighter_a": "Israel Adesanya", "fighter_b": "Sean Strickland"}'
```

```
{"schema_version":"1.2.0","win_probability":0.4931,"fighter_a":"Israel Adesanya","fighter_b":"Sean Strickland",...}
```

> **What you should see:** HTTP 200 with a JSON body matching the
> `PredictorOutputV1` contract — same `win_probability` semantics as the
> `ML Model` row in the step-5 matchup table.

For the full API surface, open `http://localhost:8000/docs` — the Swagger UI
is gated on `UFC_ENV=dev` and exposes every endpoint with try-it-now forms.

## Troubleshooting

- **`psql: connection refused`** → confirm `docker ps` lists `ufc-pg` as `Up`; if not, `docker start ufc-pg`.
- **`uv: command not found`** → install with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then re-open the shell.
- **`ufc db seed` says "DATABASE_URL is not set"** → step 3 didn't run or `.env` wasn't sourced. Re-export `DATABASE_URL` and retry.
- **`ufc db seed` says "Pre-flight 5/5 FAILED"** → your local DB already has data. Pass `--force` to restore over it, or drop the DB first with `docker compose down -v` then `docker compose up -d db` and retry.
- **`✓ ModelPredictor v2 instantiated` line is missing / red** → the canonical xgb_v2 model file is missing or corrupted. Verify `models/xgb_v2.joblib` is checked out (`git status`); the AUDIT-01 contract guarantees its SHA across the milestone.
- **401 from `POST /api/v1/predict`** → the `X-API-Key` header must be the full `dev:devsecret` entry, not just the bare secret. Restart the server with `UFC_API_KEYS="dev:devsecret"` set inline.

## Next steps

- **Partner-facing contract:** [`docs/PARTNER-CONTRACT.md`](./PARTNER-CONTRACT.md) — predictor output schema, PARTNER v1.x versioning, deprecation policy.
- **Methodology summary:** [`METHODOLOGY_CLIENT.md`](../METHODOLOGY_CLIENT.md) — how the model is trained, gate methodology, calibration approach.
- **Known issues + scraper status:** [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) — per-scraper status table + prediction-degradation timeline.
- **Architecture deep-dive:** [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md).
