# MMAfantasy — UFC fight-prediction pipeline

Python 3.13 · uv · Postgres (colima) · XGBoost/sklearn · AUDIT-01 frozen-model discipline.

## Environment (required for almost every command)
```bash
export DOCKER_HOST="unix:///Users/hratchghanime/.colima/default/docker.sock"
export DATABASE_URL="postgresql+psycopg://ufc:ufc@localhost:5439/ufc_prediction"
export TESTCONTAINERS_RYUK_DISABLED=true
uv sync --frozen
```
These three env vars are also set in `.claude/settings.json` so Bash tool calls inherit them.

## Database
- Postgres runs in the **`mmafantasy-db-1`** container on host port **5439** (user/pass/db all `ufc`).
- **Port note:** the repo's `docker-compose.yml` (and CI, README, `docs/INSTALL.md`, the `config.py` default) use the canonical host port **5433** — that is correct for fresh installs. On this machine 5433 is held by an unrelated container (`gco-test-pg`, postgres:16), so `mmafantasy-db-1` was brought up with a compose override (`services.db.ports: !override ["5439:5432"]`). Nothing on 5433 is this DB — pointing `DATABASE_URL` there silently hits the wrong Postgres. If you ever recreate the container, pass the same override (`docker compose -f docker-compose.yml -f <override>.yml up -d db`); a plain `docker compose up` will try to bind 5433 and collide. Confirm with `docker ps --format '{{.Names}} {{.Ports}}' | grep mmafantasy`.
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
- **Drift guard, not an expected failure:** `tests/**/test_compose_v23_*.py` (4 files) import `scripts/compose_v23_meta.py`, which asserts at import time that `META_V22_BASELINE_BRIER` matches the gitignored `.planning/phases/26-…/META_V22_SPIKE.json`. That JSON is regenerated on the live DB by `tests/regression/test_eval_slice_sizes.py`. On a mismatch the error happens **at collection and aborts the whole run**. The constants were re-anchored on 2026-10-01 (D5 leak fix + D6-B re-baseline), and the files pass. If the substrate or `xgb_v2` changes, re-anchor the constants, `tests/unit/ml/test_compose_v23_triple_gate.py` and `spike_noise_floor_v23.PLAN_29_03_RANDOM_15PCT_BRIER`; do not ignore the files.

## AUDIT-01 frozen-model discipline (critical)
Byte-identity is enforced on a protected set (`scripts/check_audit01_protected_files.py::PROTECTED_FILES`): the frozen models (`models/xgb_v2.joblib`, `models/meta/meta_v2.joblib`, `models/meta/meta_v2_dedup.joblib`), the spike scripts (`spike_noise_floor_v2{2,3}.py`, `train_meta_v22.py`), core ML source (`predictor.py`, `feature_matrix.py`, `persistence.py`, `train.py`), and the predictor schema.
- A **pre-commit hook** blocks commits touching these unless `AUDIT01_OVERRIDE=1`.
- `.claude/hooks/audit01_guard.py` (PreToolUse) blocks *edits* to them at edit time — same bypass.
- `spike_noise_floor_v22.py` is additionally **D-03 byte-locked** by `tests/integration/test_variance_harness.py` (must be byte-identical to HEAD).
- Frozen SHAs: `xgb_v2.joblib` = `760307…5677a`, `meta_v2.joblib` = `e04454…2502a8` (full values in `scripts/spike_noise_floor_v23.py::EXPECTED_XGB_V2_SHA` and `src/ufc_prediction/cli/predict.py`). **Verify unchanged at start and end of any model work.**

## Corpus & model facts
- `load_fight_records` filters `Event.source=='ufcstats'` (Plan 28-04 dedup) → ~8,581 fights. Since the 2026-07 re-baseline the frozen `xgb_v2` is trained on this dedup corpus. D6-B (2026-10-01) used 6,792 train and 1,789 test fights at the 2023-01-01 cutoff. Models before July used the **1.95×-inflated** cross-source corpus (16,641 rows). Gate a candidate against the **dedup-refit baseline** (the frozen config refit across seeds 42–51 on the same substrate), not against the single frozen artifact. One seed's score is not a bar.
- Elo debutant seeds come from the **`debutant_seed_inputs` table** (migration `9fc47fdd75b6`; 1,673 rows on the live corpus) — the single source of truth for `elo compute` and the serve path (`ml/inference_features`), so the Docker image needs no CSV. `data/sherdog/pre_ufc_records.csv` (untracked) is only the backfill input/export: load it with `uv run ufc db backfill-pre-ufc-seeds --csv data/sherdog/pre_ufc_records.csv` (idempotent; `--dry-run` validates read-only). `scripts/ingest_pre_ufc_records_v25.py` (~1.5h Sherdog scrape) writes the table and the CSV export; `--only-missing` scrapes just new debutants (after `ufc scrape sherdog` has found their URL). **If the table is empty or missing, `elo compute` exits 1 before any write** (pass `--allow-unseeded` only for an intentional flat-1500 run, which degrades the training substrate). Serving with an empty/missing table continues with flat-1500 debutants and logs an ERROR once per process (the empty result is not cached, so a backfill takes effect without a restart).
- Substrate regeneration order after a feature/Elo code change: (seeds present: `SELECT count(*) FROM debutant_seed_inputs` = 1,673) → `elo compute` → `features compute` → re-run the retrain gate → re-anchor the pinned baselines listed in `.claude/skills/retrain-gate/SKILL.md` → regenerate `data/seed/ufc_corpus_v30.dump` (recipe in `data/seed/PROVENANCE.md`).
- BFO odds: use the **fighter-profile / date-matched** path scoped to the relevant fighters. Do NOT use the BFO event-URL name-search — it fuzzy-matches wrong older events.

## Retrain / promote workflow
`/retrain-gate` (skill) runs: assemble dedup corpus → refit candidate across seeds → dedup-refit baseline noise-floor gate → hard-gate check (`_enforce_accuracy_gate`: Brier ≤ 0.2202, acc ≥ 0.6391) → verify frozen SHAs → **STOP before promotion**. The `ml-gate-reviewer` subagent independently reviews a candidate against this discipline. Never promote (swap the frozen file) without explicit operator approval.

## CLI
`uv run ufc {ingest,elo,features,scrape,gate,predict,db,…}` — see `uv run ufc --help`. Key: `elo compute`, `features compute`, `predict train`, `predict gate-spike`, `gate verify`, `scrape {odds,sherdog}`.
