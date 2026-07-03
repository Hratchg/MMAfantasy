# Corrected-Corpus Retrain — Handoff Plan

**Branch:** `fix/odds-recompute-and-retrain` (start here; nothing pushed/merged).
**Why:** solve two train/serve issues at the root by re-promoting a corrected `xgb_v2` baseline.
1. **Odds-math skew** — PR #12 fixed closing-consensus math (probability-space) but the frozen `xgb_v2` was trained on the OLD stored `closing_implied_prob`; inference now computes the new formula → skew.
2. **Base `days_since_last_fight_diff` inference gap** — the frozen base was trained WITH this feature but gets NaN at inference (never populated in `inference_features._populate_meta`).

**Goal is CORRECTNESS (train/serve consistency), not a performance gain.** The correction is negligible on most fights (median |Δclosing_prob|=0.0005) and material only on ~1,066 near-even/straddle-zero fights (up to 0.46). Expect the retrained model to perform ≈ frozen; re-promote it as the *corrected* baseline if it doesn't regress and the gate is clean.

**META-V22 stays OFF.** It adds no lift (see `KNOWN_ISSUES.md` → "Model performance clarification"). Do NOT enable/retrain the meta. ⚠️ The `days_since` fix in step 3 un-starves the meta (it would then run and REGRESS) — so step 3 MUST also add an explicit guard keeping the meta skipped.

## Environment
```
export DOCKER_HOST="unix:///Users/hratchghanime/.colima/default/docker.sock"
export DATABASE_URL="postgresql+psycopg://ufc:ufc@localhost:5433/ufc_prediction"
export TESTCONTAINERS_RYUK_DISABLED=true
cd /Users/hratchghanime/MMAfantasy && uv sync --frozen
# DB already seeded (6820 fighters / 16902 fights / 25632 fight_odds). Re-seed if needed:
# uv run ufc db seed --from data/seed/ufc_corpus_v30.dump --no-migrate --force
```

## Frozen baseline (current, pre-retrain)
- `xgb_v2.joblib` sha256 `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099`
- `meta_v2.joblib` sha256 `77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196`
- baseline file: `.planning/AUDIT-01-BASELINE-SHA.txt` (xgb SHA)
- xgb_v2 config (`models/xgb_v2_meta.json`): cutoff `2023-01-01`, n_train 13315, n_test 3326, 72 features, xgboost 3.2.0 / sklearn 1.8.0. best_params: n_estimators=253, max_depth=7, lr=0.013116743875697326, subsample=0.6649190225778725, colsample_bytree=0.7330702307222914, min_child_weight=6, gamma=4.585236505363638, reg_alpha=2.9183206987079522e-06, reg_lambda=4.437482739059479e-05.

## Steps

### 1 — Re-ingest (recompute stored implied probs from raw MLs) ✅ tooling committed (523b0f8)
```
uv run python scripts/recompute_fight_odds_implied_probs.py            # dry run (already verified)
uv run python scripts/recompute_fight_odds_implied_probs.py --apply    # persist to DB
```
Verify: closing_implied_prob now matches `devig_closing_range` from the raw MLs; near-even fights corrected. No fabrication (invalid MLs → NULL).

### 2 — Base `days_since_last_fight_diff` inference-parity fix (+ meta OFF guard)
In `src/ufc_prediction/ml/inference_features.py::_populate_meta`, populate `days_since_last_fight_diff` to match training EXACTLY (`feature_matrix.py` career block, `FEATURE_COLUMNS_NO_NET[32]`):
```python
# prior_a / prior_b already queried here via _query_fighter_prior_fight_date (strict <)
days_a = float((event_date_val - prior_a).days) if prior_a is not None else float("nan")
days_b = float((event_date_val - prior_b).days) if prior_b is not None else float("nan")
feats["days_since_last_fight_diff"] = (
    days_a - days_b if (days_a == days_a and days_b == days_b) else float("nan")
)  # UNCLIPPED, NaN-for-debut, A-B, NaN-propagating. (This exact patch was verified
   # earlier: magnitude parity with training; sign flip is training's md5 A/B swap.)
```
Because this un-starves the 13-col meta (which regresses), also **explicitly disable the meta** at inference (e.g. in `predictor.predict()` set `meta_skipped_reason="disabled_no_lift"` before the dispatch, or construct the predictor with `meta_dir=None`). Document why (KNOWN_ISSUES no-lift finding). This is a base-model inference change → gate-validate (step 4).

### 3 — Retrain `xgb_v2` on the corrected corpus
Do a clean apples-to-apples refit FIRST: same locked `best_params` (above), same cutoff/features, corrected data. This isolates the effect of corrected odds + days_since from any hyperparameter change. (Optionally a second Optuna candidate, but the locked-param refit is the canonical comparison.)
- Entry points: `src/ufc_prediction/ml/trainer.py::ModelTrainer` / `ufc predict train` (predict.py:~999, Optuna + hard accuracy gate). To fix params, fit `XGBClassifier(**best_params)` directly per `trainer.train`'s final-fit path, then calibrate (`CalibratedClassifierCV(FrozenEstimator(model), method=isotonic if n_calib>=1000 else sigmoid)`), matching `trainer.py`.
- Produce `models/xgb_v2_corrected.joblib` (candidate name — do NOT overwrite the frozen file yet).

### 4 — Gate-validate the candidate vs frozen baseline
- Multi-seed noise-floor spike: `scripts/spike_noise_floor_v22.py` (10 seeds × 3 slices) → thresholds.
- Contract: confirm no `.planning/gate_contract*.json` violation (`gate_contract_v2.3.json`, `v2.6.json`).
- Gate verifier: `src/ufc_prediction/ml/gate_verifier.py::verify_candidate_vs_canonical` (refit_baseline strategy) — MUST NOT emit `confound_block`. Also `ufc gate verify`.
- Base-inference deltas from step 2 gate-checked (days_since now populated).
- **Acceptance:** candidate Brier ≤ frozen (within noise) on all 3 slices AND no confound AND no contract violation. **This is a correctness re-promote, so parity (not strictly better) is acceptable — but a REGRESSION means STOP and report; do not force.**
- Note: `scripts/remeasure_meta_v22_v23.py` needs `.planning/phases/34-trust-baseline-no-odds-fallback/34-META-V2-SHA-PHASE-34-START-PLAN-01.txt` (the frozen meta SHA) — that anchor is missing from this checkout; recreate it with the meta_v2 SHA if you run that script (meta is OFF, so this is only if you re-measure meta for the record).

### 5 — Re-promote (only if step 4 clean) — STOP FOR OPERATOR APPROVAL FIRST
- Replace `models/xgb_v2.joblib` with the corrected model; regenerate `models/xgb_v2_meta.json`.
- Update `.planning/AUDIT-01-BASELINE-SHA.txt` to the NEW sha256 (this deliberately breaks the old byte-identity chain — record it).
- Regenerate the shipped corpus dump `data/seed/ufc_corpus_v30.dump` from the corrected DB (+ update `data/seed/PROVENANCE.md` SHA).
- Update CHANGELOG.md, KNOWN_ISSUES.md (retire the odds-skew + days_since items), and record a superseding `D-{N}` for the D-01 closing-consensus method change (per `bfo_math.py` docstring).
- Atomic commits; do NOT push/merge/promote without explicit operator sign-off.

## Guardrails (from operator)
- Never fabricate/impute odds (invalid MLs → NULL only).
- Never skip/xfail/delete a test to get green.
- Gate-validate everything; **if the gate regresses, STOP and report — do not force.**
- Frozen-model replacement (step 5) is a deliberate, approved re-promote — but STILL stop for sign-off before it lands.
- Verify: `uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest -q -m "not slow"` (local `test_db_seed …disposable_postgres` fails only because the host lacks `pg_restore` — passes in CI).

## Status at handoff
- Step 1 tooling committed (523b0f8), dry-run verified. DB NOT yet mutated.
- Steps 2–5 pending. Frozen models intact. `main` untouched.
