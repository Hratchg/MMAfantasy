# UFC Fight Prediction — Technical Handoff

A self-contained predictor that estimates win probabilities for UFC fights. This document hands the project to a new technical owner.

**Shipped:** v3.0 (2026-06-10) + v3.0.1 patch (2026-06-14) — see git tags `v3.0` and `v3.0.1`.

---

## 1. What this is

You give it two fighter names. It gives you back a win probability for each.

Under the hood it's a Python service that combines three signals:

1. **Elo ratings.** A rolling skill rating per fighter, updated after each fight. Separate ratings for overall, striking, and grappling.
2. **An XGBoost model** trained on 72 features (physical mismatches, recent form, finish rates, days since last fight, etc.).
3. **A logistic regression on top** that mixes the XGBoost output with closing betting odds to produce the final probability.

The combined model is called **META-V22**. It's the canonical model — the one you should use.

You can talk to it two ways:

- **Command line:** `ufc predict matchup "Israel Adesanya" vs "Sean Strickland"`
- **HTTP:** `POST /api/v1/predict` against a FastAPI server you run locally or deploy.

---

## 2. How accurate is it?

Honest answer: **about 70% accurate (~0.20 Brier)**, which is roughly where the betting market is too. That's the ceiling on this sport — UFC fights have a lot of single-punch randomness.

> **Correction (2026-07-01):** the table below reports the **META-V22** stacker on a *deduplicated Phase-26 substrate*. Those figures **do not reproduce on the current corpus** and are retained only for lineage. A 2026-07-01 re-measurement found META-V22 provides **no lift over the base `xgb_v2`**: frozen on the current substrate it regresses (0.42 Brier, scaler-OOD confound), and a clean refit only matches base (0.207 vs 0.193). The base model — what the CLI/API actually serve — is **~70% accuracy / ~0.20 Brier**. Full evidence in `KNOWN_ISSUES.md` → "Model performance clarification."

On a held-out test set of recent fights (META-V22, deduplicated Phase-26 substrate — see correction above):

| Slice | Accuracy | Brier score (lower = better) |
|---|---|---|
| Most recent 12 months | ~75% (dedup substrate) | 0.161 |
| Most recent 24 months | ~75% (dedup substrate) | 0.165 |
| Random 15% holdout | ~75% (dedup substrate) | 0.154 |

The model's real value isn't beating the market — it's giving you a probability *and an explanation* (which fighter has the Elo edge, which has the striking edge, etc.) that you can show to fantasy users.

**Important caveat:** any UFC predictor claiming >70% should be looked at skeptically. The closing betting line itself hits ~70-75%. Most of our accuracy comes from incorporating those closing odds. Without odds, the model is closer to 65%.

---

## 3. Quick start

Full walkthrough in **`docs/INSTALL.md`** (one page, tested). Short version, on a fresh Mac:

```bash
# 1. Clone and install Python deps (requires Python 3.13)
git clone <repo-url>
cd ufc-fight-prediction
uv sync

# 2. Install pg_restore (needed by `ufc db seed`)
brew install libpq
# libpq is keg-only on macOS, so add its bin dir to your PATH:
echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
# Verify:
pg_restore --version       # should print "pg_restore (PostgreSQL) 16.x" or similar

# 3. Start Postgres in Docker (canonical: docker compose, host port 5433, postgres:18)
docker compose up -d db
# ...or the raw-docker equivalent:
#   docker run -d --name ufc-db -p 5433:5432 \
#     -e POSTGRES_USER=ufc -e POSTGRES_PASSWORD=ufc -e POSTGRES_DB=ufc_prediction \
#     postgres:18-alpine
# Use Postgres 17+: a modern pg_restore emits `SET transaction_timeout`, which a
# Postgres 16 server rejects and the `ufc db seed` restore fails.

# 4. Tell the CLI how to reach it
export DATABASE_URL='postgresql+psycopg://ufc:ufc@localhost:5433/ufc_prediction'

# 5. Load the shipped corpus dump (~10 MB pg_dump file in data/seed/)
uv run ufc db seed

# 6. Predict a matchup
uv run ufc predict matchup "Israel Adesanya" vs "Sean Strickland"
```

You should see a Rich-formatted table with win probability for each fighter, Elo ratings, and the top contributing features.

**The canonical host port is `5433`** (chosen so it does not collide with a stock `5432` Postgres — ssh tunnels or another local Postgres). If `5433` is also taken, pick another host port (`-p 5440:5432`) and update `DATABASE_URL` to match.

---

## 4. How it works (architecture)

Three layers, each one calling into the next.

### Layer 1: Data in Postgres

Twelve tables hold the corpus:

| Table | Rows | What it holds |
|---|---|---|
| `events` | 1,872 | One row per UFC event |
| `fights` | 16,902 | One row per fight in those events |
| `fighters` | 6,820 | Fighter identity + biographical data |
| `fighter_aliases` | 399 | Name variants for matching |
| `round_stats` | 68,886 | Per-round strikes, takedowns, control time |
| `fight_odds` | 25,632 | Closing + opening betting lines (from BFO) |
| `elo_snapshots` | 89,988 | Pre-computed Elo for every fighter at every fight date |
| `computed_features` | 28,624 | Pre-computed feature vectors per fight (caching) |
| `referees` + `venues` + `model_runs` + `alembic_version` | small | Metadata |

The dump (`data/seed/ufc_corpus_v30.dump`, 10.42 MB) is committed to the repo. `ufc db seed` calls `pg_restore` under the hood to load it.

### Layer 2: Feature engineering

For any two fighters and a target date, the code builds a 72-element feature vector:

- Elo differences (overall, striking, grappling)
- Physical mismatches (reach, height, age, weight)
- Recent form (win streak, finish rate, takedown rate over last N fights)
- Closing betting odds from BFO if available
- A handful of contextual features (days since last fight, debutant flag, etc.)

The important rule: **only data from before the fight is used.** No leakage. There are unit tests that enforce this temporal constraint.

### Layer 3: The model

Two trained artifacts you should know about, both in `models/`:

- `xgb_v2.joblib` — the base XGBoost classifier (72 features → win probability)
- `meta/meta_v2.joblib` — the meta-learner on top (mixes XGBoost output + closing odds + Elo)

At predict time the system tries to use the meta-learner. If closing odds aren't available for that matchup, it falls back to XGBoost only. The output tells you which path was used via a `meta_skipped_reason` field.

Both files are protected by a pre-commit hook — they should never be modified by hand. There's an audit chain (AUDIT-01) that verifies their SHA256 hasn't drifted across the last 90 phases of work.

---

## 5. Project structure

```
ufc-fight-prediction/
├── docs/                          ← operator-facing docs (start here)
│   ├── INSTALL.md                 ← install + use walkthrough (v3.0)
│   ├── PARTNER-CONTRACT.md        ← what the API promises consumers
│   └── ARCHITECTURE.md            ← deeper architecture writeup
├── src/ufc_prediction/
│   ├── cli/                       ← the `ufc` command and its subcommands
│   ├── api/                       ← FastAPI server (POST /api/v1/predict)
│   ├── ml/                        ← model loading, prediction, feature assembly
│   │   ├── predictor.py           ← ModelPredictor — the main entry point
│   │   ├── feature_matrix.py      ← builds the 72-feature vector
│   │   └── persistence.py         ← loads .joblib model files
│   ├── scraper/                   ← scrapers for UFCStats, BFO, Sherdog, etc.
│   ├── elo/                       ← Elo computation engine
│   ├── features/                  ← feature engineering primitives
│   ├── db/                        ← SQLAlchemy models, session
│   └── ingest/                    ← raw scrape → normalized DB row
├── migrations/                    ← Alembic database migrations
├── tests/                         ← unit + integration tests
├── models/
│   ├── xgb_v2.joblib              ← production base model (PROTECTED — don't touch)
│   └── meta/meta_v2.joblib        ← production meta-learner (PROTECTED — don't touch)
├── data/seed/
│   └── ufc_corpus_v30.dump        ← portable corpus snapshot (10.42 MB pg_dump)
├── pyproject.toml                 ← Python deps (uv-managed)
├── KNOWN_ISSUES.md                ← scraper status + known regressions
└── README.md                      ← short intro + Docker deploy guide
```

---

## 6. What works and what's broken

### Works
- Loading the corpus dump and predicting matchups end-to-end (see `docs/INSTALL.md`)
- BFO odds scraper (`src/ufc_prediction/scraper/bfo_scraper.py`)
- The Elo engine, feature assembly, and the trained model itself
- The FastAPI server, including authentication and rate limiting
- Containerized deployment (working `Dockerfile`, runs on any container host)

### Broken or blocked
- **Most other scrapers.** UFCStats blocks us with anti-bot protection. Sherdog and Tapology block us via Content-Signal `ai-train=no`. Oddsportal blocks us via robots.txt. **Only BFO works for refreshing data.** Full status per scraper is in `KNOWN_ISSUES.md`.

### Recently fixed (v3.0.1 patch, 2026-06-14)

Two regressions were found during v3.0 close and resolved in the v3.0.1 patch — they no longer affect the install or the API:

- **`ufc predict matchup` default-version regression.** The CLI and API both defaulted to an unpromoted candidate model that mismatched the meta-learner; both entry points now explicitly pin `version="v2"`.
- **Integration test port-binding quirk on macOS.** `tests/integration/test_db_seed.py` now requests an ephemeral port from the OS instead of hardcoding `55555`.

Full disposition in `KNOWN_ISSUES.md`.

---

## 7. How the prediction stays accurate over time

You loaded a snapshot. As new fights happen, the snapshot gets stale. Rough timeline:

- **0–3 months:** predictions stay accurate for established fighters (those with 5+ UFC fights). New fighters (debutants) drift immediately because we have no Elo prior for them — the system uses a flat default.
- **3–6 months:** established fighters start drifting too. New fights are happening that the model hasn't seen.
- **6+ months:** predictions are materially stale. Either refresh the data or accept reduced accuracy.

To refresh, you need to re-ingest data. BFO works. UFCStats doesn't. That's the central problem for whoever takes this over.

---

## 8. The 'never modify these files' list

Six files are byte-frozen and a pre-commit hook enforces it:

- `models/xgb_v2.joblib`
- `models/meta/meta_v2.joblib`
- `src/ufc_prediction/ml/feature_matrix.py`
- `src/ufc_prediction/ml/persistence.py`
- `src/ufc_prediction/ml/predictor.py`
- `src/ufc_prediction/ml/trainer.py`

If you modify any of these, the model's predictions could silently change for the same input — there's no automated test that catches drift in a small number of bytes. The pre-commit hook checks SHA256 on every commit. If you genuinely need to change one, document why, run the full evaluation suite, and update the protected-file list.

---

## 9. Suggested next steps for the new owner

In rough priority order:

1. **Run through `docs/INSTALL.md` on a fresh machine.** Make sure the walkthrough works for you. If anything's broken on your system, log it.
2. **Read `KNOWN_ISSUES.md`.** The two v3.0 regressions were resolved in v3.0.1; the only remaining queued item is `OPS-V30-01` (pre-commit hook restore on fresh clones).
3. **Decide what to do about data ingest.** This is the actual strategic question. Options:
   - Live with stale data (predict only on the existing corpus; useful for backtest / replay)
   - Get a licensed UFC data feed (paid)
   - Negotiate access to the blocked scrapers
   - Build a fresh scraper against a source that doesn't block (good luck)
4. **Decide what to do about predictions getting stale.** Same question, different framing.
5. **If you want to retrain:** the model files are versioned (`xgb_v2`, `meta_v2`). You can train a `v3` and ship it as a sibling. The promotion gate is documented in `.planning/gate_methodology_v2.7.md` — it has a substrate-drift-immune dual-test methodology, you don't want to skip that.

---

## 10. Reference

### Key commands

```bash
# Install
uv sync
brew install libpq                      # for pg_restore (needed by `ufc db seed`)
# add /opt/homebrew/opt/libpq/bin to your PATH (libpq is keg-only on macOS)

# Database
uv run ufc db seed                      # load corpus from data/seed/
uv run ufc db status                    # show table row counts

# Predict
uv run ufc predict matchup "A" vs "B"
uv run ufc fighter lookup "Israel Adesanya"
uv run ufc fighter rankings Middleweight
uv run ufc fighter rankings Middleweight --top 25

# Serve API locally
uv run uvicorn --factory ufc_prediction.api.app:create_app --port 8000

# Tests
uv run pytest                           # unit tests
uv run pytest -m integration            # integration tests (needs Docker)
```

### Required environment

| Variable | Purpose | Example |
|---|---|---|
| `DATABASE_URL` | Postgres connection (use `postgresql+psycopg://` scheme, not `postgres://`) | `postgresql+psycopg://ufc:ufc@localhost:5433/ufc_prediction` |
| `UFC_API_KEYS` | API auth (for FastAPI). The whole `key:secret` string is what clients send in the `X-API-Key` header. | `dev:devsecret,partner:realsecret` |
| `UFC_ENV` | Gates `/docs` and `/redoc` (set to `dev` in dev) | `dev` |
| `UFC_CORS_ORIGINS` | CORS allow-list | `https://your-frontend.example.com` |
| `SENTRY_DSN` | Optional error reporting | `https://...@sentry.io/...` |

### Calling the API

Quick example (after starting the server above):

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev:devsecret' \
  -d '{"fighter_a": "Israel Adesanya", "fighter_b": "Sean Strickland"}'
```

As of v3.0.1, this returns HTTP 200 with a `PredictorOutputV1` JSON body.

### Key files to read first

- `docs/INSTALL.md` — install walkthrough
- `KNOWN_ISSUES.md` — scraper + regression status
- `README.md` — short intro + Docker deploy
- `docs/ARCHITECTURE.md` — system architecture writeup
- `docs/PARTNER-CONTRACT.md` — API contract for downstream consumers
- `.planning/RETROSPECTIVE-v3.0.md` — what was shipped in v3.0
- `pyproject.toml` — Python deps and version pins

---

## Questions to ask the previous owner

If you can still reach the previous owner, things worth confirming:

1. Are there any partners actively consuming the API? If yes, what's their contact and what version of the schema are they on?
2. Is the `v3.0` git tag the canonical handoff point, or has work continued past it?
3. Are there credentials (BFO API, Sentry, container host / DB) that need to be transferred?
4. Are there any v3.0.1 patch follow-ups (e.g., regression tests covering the default-version pinning)?
5. What's the practical plan for ongoing data refresh?

---

*Last updated 2026-06-10 at v3.0 close.*
