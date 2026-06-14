# Architecture — UFC Fight Prediction

One-page tour of the system. Read this before diving into the source.

## Data flow

```
External sources                Local pipeline                   Serving surface
─────────────────              ──────────────                   ────────────────

  ┌────────────┐               ┌──────────────┐
  │ UFCStats   │──scrape──────▶│ events,      │
  │ (canonical │               │ fights,      │
  │  fights)   │               │ fight_stats, │
  └────────────┘               │ referees,    │              ┌──────────────────┐
                               │ venues       │──features───▶│ feature_matrix   │
  ┌────────────┐               │ (PostgreSQL) │              │ (chronological,  │
  │ Sherdog    │──scrape──────▶│              │              │  pre-fight only) │
  │ (camps,    │               │              │              └──────────────────┘
  │  career)   │               │              │                       │
  └────────────┘               │              │                       ▼
                               │              │              ┌──────────────────┐
  ┌────────────┐               │              │              │ Multi-domain     │
  │ BFO        │──scrape──────▶│              │              │ Elo ratings      │
  │ (closing   │               │              │              │ (overall/        │
  │  odds)     │               │              │              │  striking/       │
  └────────────┘               │              │              │  grappling)      │
                               └──────────────┘              └──────────────────┘
                                                                       │
                                                                       ▼
                                                              ┌──────────────────┐
                                                              │ xgb_v2.joblib    │
                                                              │ (XGBoost on 72   │
                                                              │  feature cols;   │
                                                              │  byte-identity   │
                                                              │  locked)         │
                                                              └──────────────────┘
                                                                       │
                                                                       ▼
                                                              ┌──────────────────┐
                                                              │ META-V22         │      ┌──────────┐
                                                              │ meta_v2.joblib   │─────▶│ FastAPI  │
                                                              │ (Logistic stack: │      │ /api/v1  │
                                                              │  xgb_oof_prob +  │      └──────────┘
                                                              │  elo + closing   │             │
                                                              │  odds + style)   │             │      ┌────────┐
                                                              └──────────────────┘             ├─────▶│ CLI    │
                                                                                               │      │ `ufc`  │
                                                                                               │      └────────┘
                                                                                               │
                                                                                               ▼
                                                                                       Partner consumers
                                                                                       (fantasy MMA app)
```

## Layered architecture

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Ingestion** | `src/ufc_prediction/scraper/`, `scripts/scrape_*.py` | Pull fight data from UFCStats, Sherdog, BFO; write to PostgreSQL |
| **Dedup** | `src/ufc_prediction/dedup/`, `scripts/backfill_fighter_aliases_from_dedup_recon.py` | Cross-source fight + fighter resolution (6-tier alias matcher shipped in Phase 28-04) |
| **Features** | `src/ufc_prediction/features/`, `feature_matrix.py` | Transform normalized DB rows into model-ready feature vectors with strict pre-fight temporal ordering |
| **Elo** | `src/ufc_prediction/elo/` | Multi-domain rating engine; append-only chronological updates |
| **ML primitives** | `src/ufc_prediction/ml/` | `predictor.py`, `evaluator.py`, `variance.py`, `gate_contract.py`, `oof.py` |
| **Models** | `models/xgb_v2.joblib`, `models/meta/meta_v2.joblib` | Pinned, AUDIT-01 protected production artifacts |
| **API** | `src/ufc_prediction/api/` | FastAPI app; `/api/v1/predict` (matchup), `/fighters/{name}`, `/matchup`, `/rankings/{division}` |
| **CLI** | `src/ufc_prediction/cli/` | Typer-based `ufc` commands (ingest / elo / fighter / scrape / features / matchup / export / predict) |
| **Contracts** | `src/ufc_prediction/contracts/` | Partner-facing JSON schemas (`predictor.schema.v1.0.0.json`, `.v1.1.0.json`) + OpenAPI specs |

## Key invariants

These are **load-bearing** — break them and prediction integrity is at risk.

### 1. AUDIT-01 byte-identity (xgb_v2 + meta_v2)

`models/xgb_v2.joblib` SHA-256 `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099` is preserved end-to-end across **all** releases since v2.1. The pre-commit hook on `feature_matrix.py / persistence.py / predictor.py / train.py / models/xgb_v2.joblib` blocks accidental edits. Per-phase MID + END SHA checkpoints written to `.planning/phases/*/...-XGB-V2-SHA-*.txt`.

**Why this matters:** Reproducibility. Anyone who downloads the repo at any commit since v2.1 gets the exact same model byte-for-byte. Predictions are deterministic across time.

**How to extend the model:** Don't edit `xgb_v2.joblib`. Train a new candidate (`scripts/retrain_xgb_v3.py`), gate-test it against `.planning/gate_contract_v2.3.json`, and promote only if it clears the triple-gate (gate clearance + ≥0.003 Brier margin + per-step ≥0.003 hurdle). See [`GLOSSARY.md`](GLOSSARY.md#audit-01-chain) for chain mechanics.

### 2. Temporal integrity (no leakage)

All features computed with explicit pre-fight boundaries. Feature computation functions accept fight_date as parameter; functions validate they use only data from before that date. Each fighter's Elo rating is immutable once computed for a fight date.

**Why this matters:** Models trained on leakage look great in offline eval and fail in production. CLAUDE.md treats temporal integrity as the non-negotiable cross-cutting constraint.

### 3. Gate contract immutability (formula hash D-18)

The promotion-gate formula hash `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a` is LOCKED across v2.1 → v2.2 → v2.3. Each version emits a new gate contract (`gate_contract_v2.3.json`) with new per-slice thresholds, but the formula that produces those thresholds doesn't change. No post-measurement renegotiation.

**Why this matters:** Governance. A model that "barely cleared the gate" doesn't get to relax the gate in retrospect. The contract is binding.

### 4. PARTNER schema forward-compat (v1.0.0 byte-frozen)

`src/ufc_prediction/contracts/predictor.schema.v1.0.0.json` has not been modified since Phase 25 (v2.2). Phase 32 (v2.3) added v1.1.0 as a sibling with 3 additive optional fields. v1.0.0 partners see byte-identical response shape. Phase 25's forward-compat lock is binding.

**Why this matters:** Partners don't break on minor releases. Every v1.x bump is additive-only.

## Three data refresh cadences

| Cadence | Trigger | Operation |
|---------|---------|-----------|
| **Per-event** (~weekly) | New UFC card lands | Scrape new fights; Elo ratings update chronologically (append-only); fighter career aggregates refresh |
| **Per-card** (~daily, day-of) | Upcoming card | Pull BFO closing odds; recompute features for upcoming fighters; run predictions |
| **Per-season** (~quarterly) | Drift detection OR planned upgrade | Operator-binding: retrain xgb_v2 candidate; run gate spike; promote only if triple-gate clears |

**Per-season retrains are NEVER automated.** Cron is for data; promotion is always manual.

## Versioning + lifecycle

- **Project** versioning is milestone-based (v1.0 MVP → v1.1 BFO + dedup → v2.0 → v2.1 → v2.2 → v2.3). Each milestone has its own ROADMAP + REQUIREMENTS archived under `.planning/milestones/v{X.Y}-{ROADMAP,REQUIREMENTS}.md` + phase artifacts under `.planning/milestones/v{X.Y}-phases/`.
- **PARTNER schema** versioning is semver (v1.0.0, v1.1.0, …). v1.x bumps are additive-only by binding contract.
- **Model** versioning is suffix-based (`xgb_v1`, `xgb_v2`, ...; `meta_v2`, ...). The pre-commit hook + AUDIT-01 chain ensure we never accidentally lose lineage on the byte-identity contract.

## Where to read more

- **Why a specific design choice was made:** the relevant phase's `*-CONTEXT.md` + `*-SUMMARY.md` under `.planning/milestones/v{X.Y}-phases/`.
- **What the gate contract numerics mean:** [`GLOSSARY.md`](GLOSSARY.md).
- **How to integrate as a partner:** [`QUICKSTART-PARTNER.md`](QUICKSTART-PARTNER.md) + [`PARTNER-RELEASE-v2.3.md`](PARTNER-RELEASE-v2.3.md).
- **How to contribute code:** [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
