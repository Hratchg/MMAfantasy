# MMAfantasy — UFC fight-prediction pipeline

Python 3.13 · uv · Postgres (colima) · XGBoost/sklearn · AUDIT-01 frozen-model discipline.

## Environment (required for almost every command)
```bash
export DOCKER_HOST="unix:///Users/hratchghanime/.colima/default/docker.sock"
export DATABASE_URL="postgresql+psycopg://ufc:ufc@localhost:5433/ufc_prediction"
export TESTCONTAINERS_RYUK_DISABLED=true
uv sync --frozen
```
These three env vars are also set in `.claude/settings.json` so Bash tool calls inherit them.

## Database
- Postgres runs in the **`mmafantasy-db-1`** container on host port **5433** (user/pass/db all `ufc`).
- **The host has no `psql`.** Query via the container: `docker exec mmafantasy-db-1 psql -U ufc -d ufc_prediction -tA -c "…"`. The `/corpus-stats` skill wraps the common queries.
- Schema notes: `events.date` (not `event_date`); `fight_odds` has no `id` (composite PK `fight_id,fighter_id`); event `source ∈ {ufcstats, kaggle-mdabbert, kaggle-rajeevw}`.

## Verification gates (run before declaring done)
```bash
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest -q -m "not slow"
```
`.claude/hooks/ruff_format.py` (PostToolUse) auto-formats `.py` edits so `ruff format --check` stays clean.

**Known-expected failures (NOT your bug):**
- `tests/integration/test_db_seed.py::test_round_trip_seed_against_disposable_postgres` — host lacks `pg_restore` (passes in CI).
- `tests/integration/test_train_meta_v22_real_data.py::test_meta_v2_joblib_not_promoted_yet` — stale Plan-26 test; `meta_v2.joblib` is a shipped AUDIT-01 artifact.
- **`tests/**/test_compose_v23_*.py` (3 files) — module-level `META_V22_BASELINE_BRIER` drift assertion that ERRORS AT COLLECTION and aborts the whole run.** To get a signal on everything else, add `--ignore=tests/integration/test_compose_v23_path_a.py --ignore=tests/unit/ml/test_compose_v23_stepwise.py --ignore=tests/unit/ml/test_compose_v23_triple_gate.py`.

## AUDIT-01 frozen-model discipline (critical)
Byte-identity is enforced on a protected set (`scripts/check_audit01_protected_files.py::PROTECTED_FILES`): the frozen models (`models/xgb_v2.joblib`, `models/meta/meta_v2.joblib`, `models/meta/meta_v2_dedup.joblib`), the spike scripts (`spike_noise_floor_v2{2,3}.py`, `train_meta_v22.py`), core ML source (`predictor.py`, `feature_matrix.py`, `persistence.py`, `train.py`), and the predictor schema.
- A **pre-commit hook** blocks commits touching these unless `AUDIT01_OVERRIDE=1`.
- `.claude/hooks/audit01_guard.py` (PreToolUse) blocks *edits* to them at edit time — same bypass.
- `spike_noise_floor_v22.py` is additionally **D-03 byte-locked** by `tests/integration/test_variance_harness.py` (must be byte-identical to HEAD).
- Frozen SHAs: `xgb_v2.joblib` = `6e7641…ba099`, `meta_v2.joblib` = `77076d3b…f9196`. **Verify unchanged at start and end of any model work.**

## Corpus & model facts
- `load_fight_records` filters `Event.source=='ufcstats'` (Plan 28-04 dedup) → ~8,581 fights. The frozen `xgb_v2` was trained on the PRE-dedup **1.95×-inflated** cross-source corpus (16,641 rows). When comparing a dedup-trained candidate to frozen, gate against a **dedup-refit baseline** (frozen config refit on the same clean substrate), not the inflated frozen artifact.
- Elo debutant seeding reads `data/sherdog/pre_ufc_records.csv` (regenerate via `uv run python scripts/ingest_pre_ufc_records_v25.py` — a ~1.5h Sherdog scrape). **If that file is absent, `elo compute` silently falls back to flat-1500 seeds and degrades the training substrate.**
- BFO odds: use the **fighter-profile / date-matched** path scoped to the relevant fighters. Do NOT use the BFO event-URL name-search — it fuzzy-matches wrong older events.

## Retrain / promote workflow
`/retrain-gate` (skill) runs: assemble dedup corpus → refit candidate across seeds → dedup-refit baseline noise-floor gate → hard-gate check (`_enforce_accuracy_gate`: Brier ≤ 0.2202, acc ≥ 0.6391) → verify frozen SHAs → **STOP before promotion**. The `ml-gate-reviewer` subagent independently reviews a candidate against this discipline. Never promote (swap the frozen file) without explicit operator approval.

## CLI
`uv run ufc {ingest,elo,features,scrape,gate,predict,db,…}` — see `uv run ufc --help`. Key: `elo compute`, `features compute`, `predict train`, `predict gate-spike`, `gate verify`, `scrape {odds,sherdog}`.
