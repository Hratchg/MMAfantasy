---
name: retrain-gate
description: Retrain the xgb_v2 candidate on the current corrected/dedup corpus and gate it fairly against a dedup-refit baseline (the frozen config refit on the same substrate). Runs the refit across seeds, reports per-slice Brier vs frozen and vs the dedup-refit noise floor, checks the hard operator gate, verifies frozen SHAs, and STOPS before promotion. Use when re-promoting a base model after an odds/elo/corpus change.
disable-model-invocation: true
---

# retrain-gate

Repeatable, STOP-before-promotion retrain + fair gate for `xgb_v2`. **Never swaps the frozen model** — it writes a candidate and reports a verdict for operator approval.

## Preconditions
- Env set (see CLAUDE.md): `DOCKER_HOST` (colima socket), `DATABASE_URL` (host port **5439** on this machine — whatever host port `mmafantasy-db-1` is bound to; the compose-file default 5433 is held by an unrelated container here, see CLAUDE.md "Database"), `TESTCONTAINERS_RYUK_DISABLED=true`; `uv sync --frozen`.
- Substrate current: run `ufc elo compute` + `ufc features compute` after any ingest/scrape, and confirm `data/sherdog/pre_ufc_records.csv` exists (else `elo compute` exits 1; `--allow-unseeded` would fall back to flat-1500 and degrade the candidate).

## Run
```bash
uv run python scripts/retrain_and_gate.py FINAL
```
This: assembles the dedup ufcstats corpus (72-col `FEATURE_COLUMNS_NO_NET`), refits the locked `best_params` candidate across seeds 42–51 (the **dedup-refit baseline distribution**), saves seed-42 to `models/xgb_v2_corrected.joblib` (frozen untouched; `save_model` refuses to overwrite, so move or delete a previous candidate first), evaluates the frozen model on the same test set, and prints overall + per-slice (`most_recent_12mo`, `most_recent_24mo`, `random_15pct`) Brier with the z of frozen within the dedup-refit distribution.

The script's `frozen` column is the currently promoted `xgb_v2`. Since the 2026-07 re-baseline that model was itself trained on the dedup corpus, so the comparison is same-substrate; the script's older "inflated" wording predates that. Recency slices are cut relative to the latest test-set event date (`evaluator.gate_reference_date`, printed as `slice_anchor`), so per-slice numbers only move when the corpus changes, not with the calendar.

## Interpret / report
- **Fair gate:** candidate must be parity-or-better vs the dedup-refit baseline on all 3 slices (frozen `|z| ≲ 2` = parity; the frozen model was trained on the inflated corpus, so it is NOT the bar).
- **Hard gate:** candidate overall Brier ≤ 0.2202 and accuracy ≥ 0.6391 (`_enforce_accuracy_gate`).
- **Integrity:** afterward run `shasum -a 256 models/xgb_v2.joblib models/meta/meta_v2.joblib` and confirm the frozen SHAs (`0b0b40…f6fecd`, `e04454…2502a8`; full values in CLAUDE.md / `scripts/spike_noise_floor_v23.py::EXPECTED_XGB_V2_SHA`) are unchanged, and `git diff --stat HEAD -- scripts/spike_noise_floor_v22.py` is empty.
- **Anchors:** if the substrate was regenerated (`elo compute` / `features compute`), the pinned baselines move too: `PLAN_29_03_RANDOM_15PCT_BRIER` (spike_v23 + `tests/integration/test_variance_harness.py`), `META_V22_BASELINE_BRIER` (`scripts/compose_v23_meta.py` + `tests/unit/ml/test_compose_v23_triple_gate.py`, from a fresh `META_V22_SPIKE.json`), the v22 Pitfall E drift (`EXPECTED_XGB_V2_BRIER`, tolerance 0.005) and the shipped seed dump (`data/seed/PROVENANCE.md`). Re-anchor them in the same PR.

Report a compact verdict: PROMOTABLE (all gates pass) or BLOCKED (name the failing slice/gate). End by stating that promotion (swap `xgb_v2.joblib`, regenerate meta + dump, update the AUDIT-01 SHA and re-derive `EXPECTED_XGB_V2_BRIER` in the byte-locked `spike_noise_floor_v22.py` via `AUDIT01_OVERRIDE`) is reserved for explicit operator approval.
