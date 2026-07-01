# Phase 64: TRAVEL FEAT-V261-01 verifier run - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Smart discuss — batched grey areas, operator accepted all 4 recommendations 2026-06-04

<domain>
## Phase Boundary

Re-run the substrate-drift–safe verifier (`verify_candidate_vs_canonical(...)` from Phase 55) against the TRAVEL composition candidate (`models/meta/meta_v22_travel.joblib`) using Phase 63's substrate-snapshot loader. Replaces the Phase 58 PROVISIONAL Path B with a real GATE-V26-02 verdict and executes the path-conditional close.

**In scope:**
- New module `scripts/build_travel_substrate_v261.py` that generates the TRAVEL substrate-snapshot parquet (15-wide feature vectors: META-V22 13 + `travel_distance_km` + `tz_shift_hours`) from Phase 42 source logic across three slices (`most_recent_12mo`, `most_recent_24mo`, `random_15pct`) and writes to `data/intermediate/travel_substrate_v261.parquet` per Phase 63 D-01 format.
- Surgical guard in `src/ufc_prediction/ml/gate_verifier.py::verify_candidate_vs_canonical(...)`: when `canonical_pipeline.n_features_in_` ≠ candidate's feature width, skip the raw canonical predict call, set `raw_baseline_brier=None`, force `verdict="confound_block"` with rationale `width_mismatch_drift`. Strict superset of existing behavior (no regression for width-matched cases).
- Invoke `ufc gate verify --candidate models/meta/meta_v22_travel.joblib --substrate-parquet data/intermediate/travel_substrate_v261.parquet --out results/travel_promotion_gate_v261.json` end-to-end.
- Emit `results/travel_promotion_gate_v261.md` (operator writeup) alongside the JSON sidecar; MD rendered via inline CLI helper or `scripts/render_gate_verdict_md.py` (planner's call).
- Update `scripts/run_travel_gate_v26.py` to delegate to `ufc gate verify` (back-compat shim).
- Path-conditional close:
  - **Path A** (`path_a_promote`): `cp models/meta/meta_v22_travel.joblib models/meta/meta_v2_travel.joblib` (sibling COPY, not rename); commit both.
  - **Path B** (`path_b_reject`): `git mv models/meta/meta_v22_travel.joblib .planning/phases/64-travel-feat-v261-01-verifier-run/archive/meta_v22_travel.joblib`.
  - **confound_block** (including `width_mismatch_drift` sub-case): artifact untouched; verdict + rationale documented in the MD; v2.7+ backlog row added.
- Unit tests for the substrate builder (round-trip parquet via Phase 63 loader, slice membership counts, deterministic SHA per slice).
- Integration test that runs the full `ufc gate verify` pipeline end-to-end against a synthetic 15-wide substrate fixture and asserts the width-mismatch guard fires (since the synthetic canonical is 13-wide).

**Out of scope (NOT this phase):**
- REF / NET / odds-drift substrate generation (Phases 65–67 each own their own).
- Verifier core logic changes beyond the surgical width-guard (the meta-gate confound math, formula-gate, methodology selection all stay LOCKED).
- Any change to `EvalSlice`, gate contract, META-V22, xgb_v2, meta_v2 model artifacts.
- Modifying the Phase 58 outputs (`results/travel_promotion_gate_v26.{md,json}`) — those remain as the v2.6 PROVISIONAL audit trail; v2.6.1 emits the parallel `_v261` artifacts.

</domain>

<decisions>
## Implementation Decisions

### Width-mismatch handling (Q1)
- **D-01:** Add a surgical width-detect guard in `verify_candidate_vs_canonical`. When `canonical_pipeline.n_features_in_ != len(sl.feature_vectors[0])`, skip the raw canonical `predict_proba` call, set `raw_baseline_brier[slice]=None` for all slices, force `verdict="confound_block"` with `confound_rationale="width_mismatch_drift"` (or extend the verdict's existing rationale field). The refit_baseline path still runs (refit pipeline gets retrained on candidate width → no width error → aligned numbers still meaningful for the audit trail).
  - Rationale: TRAVEL is a feature-WIDENED candidate (15 cols vs canonical 13). Any feature-widened candidate would hit the same wall. The methodology spec §3.1 implicitly assumes width-matched candidate+canonical; widening is itself the most extreme form of substrate drift, and detecting it should be auto-confound, not crash. Surgical guard is non-regressive: width-matched cases (Phase 65 REF refit at 13-wide, Phase 66 NET in-place change, etc.) take the unchanged code path.
  - **Phase 63 scope-boundary note:** Phase 63's "no verifier changes" was scoped to that phase's loader work, not a permanent freeze on the verifier file. Phase 64 explicitly opens the verifier for this one surgical addition. The new behavior is gated by a single `if` and ships with a dedicated unit test.
  - Rejected: (a) substrate-at-13-wide → meta_v22_travel.predict_proba would itself crash; doesn't satisfy "real verdict" success criterion. (b) Two-substrate API change → out of scope, large surface. (c) Refit canonical to 15-wide and use it as both raw+aligned baseline → loses the raw-vs-aligned delta signal the meta-gate depends on.

### Substrate parquet source + storage (Q2)
- **D-02:** Build the TRAVEL substrate via `scripts/build_travel_substrate_v261.py`.
  - **Source:** reuse the eval-matrix-building logic from `scripts/compose_v25_travel.py` (Phase 42 verified path). Produces the same 15-wide feature vectors per fight that meta_v22_travel was trained on.
  - **Slices:** `most_recent_12mo`, `most_recent_24mo`, `random_15pct` — matching Phase 42's `TRAVEL_COMPOSITION_V25_REPORT.json` slices for direct comparability.
  - **Per-slice `substrate_sha`:** `hashlib.sha256()` over the slice's `(feature_vector, outcome)` rows sorted deterministically — UNIQUE per slice (Phase 63 D-03 requires distinct SHAs across slices).
  - **Output:** `data/intermediate/travel_substrate_v261.parquet` — gitignored (regeneratable from script + Phase 42 data). Script itself + tests committed.
  - **Determinism:** fixed numpy seed for the `random_15pct` slice; sorted feature ordering matching `meta_v22_travel_meta.json::feature_columns` order.
  - Rejected: (a) commit parquet as fixture → MB-scale bloat. (b) embed generation in run_travel_gate wrapper → tangles generation+invocation. (c) reuse `oof_predictions_v22.parquet` → wrong width and wrong content (xgb_v3 OOF, not 15-wide TRAVEL feature matrix).

### CLI invocation path (Q3)
- **D-03:** The Phase 63 `ufc gate verify` CLI is the canonical v2.6.1 invocation surface. Phase 64 uses it end-to-end:
  ```
  ufc gate verify \
    --candidate models/meta/meta_v22_travel.joblib \
    --substrate-parquet data/intermediate/travel_substrate_v261.parquet \
    --out results/travel_promotion_gate_v261.json
  ```
- **D-03a:** Update `scripts/run_travel_gate_v26.py` to delegate to `ufc gate verify` (back-compat shim with deprecation note in the module docstring). Removes the v2.6 placeholder logic that returned rc=2.
  - Rationale: Phase 63 IS the v2.6.1 invocation surface. Phase 58's wrapper was a v2.6 scaffold; keeping two parallel surfaces invites drift.

### Output emission + Path A/B execution (Q4)
- **D-04 (JSON sidecar):** Use `emit_verdict_json(verdict, out_path)` (Phase 55, byte-stable). Written to `results/travel_promotion_gate_v261.json`. **DO NOT** overwrite Phase 58's `travel_promotion_gate_v26.json` — that file stays as the v2.6 PROVISIONAL audit trail.
- **D-05 (MD writeup):** Emit `results/travel_promotion_gate_v261.md`. Sections (template mirrors Phase 58 doc):
  1. Header (Phase 64 FEAT-V261-01, date, verdict, methodology).
  2. Verdict + rationale (auto-populated from verdict fields).
  3. Per-slice metric table (raw/aligned/candidate Brier; raw=None on confound_block width-mismatch case).
  4. AUDIT-01 anchor check (xgb_v2 + meta_v2 SHAs unchanged).
  5. Path-conditional next steps.
  6. References (gate_methodology_v2.6.md §3.1, Phase 55 verifier, Phase 63 loader, Phase 58 PROVISIONAL writeup it supersedes).
- **D-06 (Path A — `path_a_promote`):** Sibling **COPY** (not rename):
  - `cp models/meta/meta_v22_travel.joblib models/meta/meta_v2_travel.joblib`
  - Author `models/meta/meta_v2_travel_meta.json` (sibling metadata; references `path_a` lineage).
  - Commit both. `meta_v22_travel.joblib` stays in place — preserves the Phase 42 lineage trace for the audit chain.
- **D-07 (Path B — `path_b_reject`):**
  - `git mv models/meta/meta_v22_travel.joblib .planning/phases/64-travel-feat-v261-01-verifier-run/archive/meta_v22_travel.joblib`
  - Also move `models/meta/meta_v22_travel_meta.json` alongside.
  - Add a `archive/README.md` documenting why archived + reference to the verdict file.
- **D-08 (confound_block, including width_mismatch_drift sub-case):** Artifact UNTOUCHED. MD documents the confound; v2.7+ backlog row added in `.planning/REQUIREMENTS.md` (or a follow-on REQ file) with re-eligibility criteria (e.g., "if/when meta_v3-style 15-wide canonical lands, re-run").
- **D-09 (AUDIT-01 invariants):** In ALL three paths, `xgb_v2.joblib` SHA `6e7641…0099` AND `meta_v2.joblib` SHA `77076d3b…9196` MUST stay byte-identical end-to-end. Verified at phase-end SHA snapshot (`64-AUDIT-01-END.txt`).

### Claude's Discretion
- Exact module-internal naming for the width-guard (`_check_pipeline_width_match`, `_detect_width_mismatch`, etc.) — picker's call.
- Whether to add a separate `scripts/render_gate_verdict_md.py` helper or inline the MD rendering inside `cli/main.py::gate_verify` — picker chooses based on code-organization preference; both are acceptable.
- Exact phrasing of the `width_mismatch_drift` rationale string in the verdict (must include the actual widths for debuggability).
- Synthetic-substrate test fixture details for the unit test (size, seed, etc.).
- Whether to add a `--force-raw-baseline` CLI escape hatch (default OFF) — picker MAY add if useful for debugging; not required.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 64 contract sources
- `.planning/REQUIREMENTS.md` — v2.6.1 REQ list; FEAT-V261-01 is Phase 64's requirement.
- `.planning/ROADMAP.md` — Phase 64 §Scope is authoritative.
- `.planning/phases/63-substrate-snapshot-loader-crit-v261-01/63-CONTEXT.md` — D-01 (parquet schema) and D-03 (validation rules) are the contract this phase's substrate parquet MUST satisfy.

### EvalSlice + verifier contracts (Phase 55 LOCKED; Phase 64 adds ONE width-guard)
- `src/ufc_prediction/ml/gate_verifier.py` — `verify_candidate_vs_canonical(...)` line 222, `EvalSlice` lines 54-70, `emit_verdict_json(...)` line 458. Phase 64 ADDS a width-mismatch guard before line 320; all other logic unchanged.
- `.planning/gate_methodology_v2.6.md` §3.1 (refit_baseline methodology), §6.1 (input contract), §6.2 (verdict shape).

### TRAVEL candidate + Phase 42 source data
- `models/meta/meta_v22_travel.joblib` — Phase 42 advisory-only sibling; 15-wide.
- `models/meta/meta_v22_travel_meta.json` — feature_columns (15-wide), slice_metrics (Phase 42), operator_caveat.
- `scripts/compose_v25_travel.py` — Phase 42 composition script; substrate-builder reuses its eval-matrix logic.
- `.planning/milestones/v2.5-phases/42-travel-feature-engineering-closeout/TRAVEL_COMPOSITION_V25_REPORT.json` — Phase 42 per-slice ground truth metrics for comparability.
- `results/travel_oof_verification_v25.md` — Phase 45 Plan 45-01 OOF-source-divergence evidence; the "expected confound" priors.

### Phase 63 CLI surface
- `src/ufc_prediction/cli/main.py::gate_verify` — Phase 63 wired-up CLI; Phase 64 invokes it end-to-end (no CLI changes needed unless the MD-rendering helper is inlined).
- `src/ufc_prediction/ml/substrate_loader.py::load_substrate_snapshot` — Phase 63 loader; the substrate parquet built in Phase 64 MUST round-trip through this loader without error.

### Phase 58 supersession context
- `scripts/run_travel_gate_v26.py` — Phase 58 v2.6 wrapper; Phase 64 D-03a updates to delegate to `ufc gate verify`.
- `results/travel_promotion_gate_v26.{md,json}` — Phase 58 PROVISIONAL Path B writeup; NOT modified; Phase 64 emits parallel `_v261` files.

### v2.6.1 invariants
- `.planning/STATE.md` §Cross-Cutting Invariants — xgb_v2 + meta_v2 SHAs byte-identical end-to-end.
- `.planning/AUDIT-01-BASELINE-SHA.txt` — current canonical SHAs.
- Pre-commit guard `scripts/check_audit01_protected_files.py` (if present) — Path A's `meta_v2_travel.joblib` is a NEW sibling, not a modification of canonical; Path B's archive move is an artifact-relocation, not a modification.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 63 substrate loader** (`src/ufc_prediction/ml/substrate_loader.py`) — round-trip target for the new substrate parquet.
- **Phase 63 CLI** (`src/ufc_prediction/cli/main.py::gate_verify` with `--substrate-parquet` + `--out`) — invocation surface; Phase 64 does NOT modify it.
- **`emit_verdict_json(verdict, out_path)`** (`gate_verifier.py:458`) — JSON sidecar emitter; byte-stable.
- **`verify_candidate_vs_canonical(...)`** (`gate_verifier.py:222`) — Phase 64 ADDS a single width-guard `if` before the prediction loop; all other logic unchanged.
- **`scripts/compose_v25_travel.py`** — Phase 42 TRAVEL composition logic; substrate-builder reuses the eval-matrix path that produces 15-wide feature vectors per slice.
- **Phase 58 verdict-MD template** (`results/travel_promotion_gate_v26.md`) — section layout reused for the `_v261` MD.

### Established Patterns
- **Verdict doc + JSON sidecar** (Phase 45 Plan 45-04, Phase 58) — `.md` for operators + `.json` for audit trail.
- **Sibling COPY pattern** (Phase 42 D-10) — Path A produces a sibling `*_<purpose>.joblib` alongside the canonical; never overwrites.
- **`scripts/build_*.py` for data artifact generation + `tests/unit/scripts/test_*.py` for round-trip + slice-count assertions** — common pattern across `compose_v25_travel.py`, `compose_v23_meta.py`.
- **Deterministic SHA per slice** via `hashlib.sha256(canonical_bytes)` after sorted serialization — Phase 23/Phase 26 substrate-SHA pattern.

### Integration Points
- **`src/ufc_prediction/ml/gate_verifier.py`** — single new `if`-guard added before line 320. Existing tests must still pass; new dedicated unit test exercises the width-mismatch path.
- **`scripts/build_travel_substrate_v261.py`** (new) — builds the parquet; gitignored output; CLI: `python scripts/build_travel_substrate_v261.py [--output PATH]`.
- **`scripts/run_travel_gate_v26.py`** — delegates to `ufc gate verify` subprocess; deprecation note in docstring; preserves CLI flags for back-compat.
- **`tests/unit/ml/test_gate_verifier_width_guard.py`** (new) — synthetic 13-wide canonical + 15-wide candidate; assert verdict.verdict == "confound_block" + rationale contains "width_mismatch".
- **`tests/unit/scripts/test_build_travel_substrate_v261.py`** (new) — round-trip via Phase 63 loader; slice counts match Phase 42 ground truth; per-slice SHAs distinct.
- **`tests/cli/test_gate_verify_travel_e2e.py`** (new) — end-to-end CLI run against the built substrate; asserts verdict file exists and is one of the three valid verdicts.
- **`.gitignore`** — add `data/intermediate/travel_substrate_v261.parquet` (regeneratable).
- **`results/travel_promotion_gate_v261.{md,json}`** — actual phase outputs; committed.
- **Path A:** `models/meta/meta_v2_travel.joblib` + `models/meta/meta_v2_travel_meta.json` committed.
- **Path B:** `.planning/phases/64-travel-feat-v261-01-verifier-run/archive/meta_v22_travel.joblib` + meta JSON + README.

</code_context>

<specifics>
## Specific Ideas

- The width-guard error path emits a verdict with `aligned_*_brier` populated (refit_baseline path runs cleanly on 15-wide substrate) and `raw_baseline_brier=None`; downstream consumers can rely on `verdict.verdict == "confound_block" and verdict.confound_rationale.startswith("width_mismatch")` for programmatic detection.
- The substrate builder MUST emit per-slice `substrate_sha` values that are STABLE under re-run with the same Phase 42 inputs (deterministic). This enables a future re-verification audit to compare SHAs across runs.
- Phase 64 should ship a `64-AUDIT-01-START.txt` (snapshot of xgb_v2 + meta_v2 SHAs at phase entry) and `64-AUDIT-01-END.txt` (snapshot at phase exit); the END file MUST byte-match START for the canonical anchors regardless of path (A/B/confound).
- Phase 64's plan list should follow the Phase 63 4-plan pattern: (1) verifier width-guard + test, (2) substrate builder script + test, (3) CLI integration test + Phase 58 wrapper delegation, (4) actual run + verdict emission + path-conditional close.
- Path A and Path B both produce a git diff; confound_block produces only `results/travel_promotion_gate_v261.{md,json}` + the AUDIT-01 END snapshot (no model artifact movement).

</specifics>

<deferred>
## Deferred Ideas

- **REF / NET / odds-drift substrate generators** — Phases 65–67 own their own (each has different feature set).
- **Generalized `scripts/build_substrate_v261.py` factory** — could DRY out the four FEAT-V261 substrate builders. Defer to Phase 67 close-out if the duplication becomes obvious.
- **`--force-raw-baseline` CLI escape hatch on `ufc gate verify`** — at Claude's discretion in Phase 64; not required. If added, OFF by default.
- **Auto-PR template for Path A/B/confound** — manual commit is fine for v2.6.1; automation if Phase 65–67 prove the workflow is repetitive enough.
- **Cross-phase substrate caching** — if Phase 64's substrate parquet generation is slow, caching could amortize across re-runs. Profile first; defer optimization.
- **`meta_v2_travel.joblib` predictor.py wiring** (if Path A) — promoting the sibling to the live prediction path is a separate phase decision (would require operator approval per §7 of methodology spec). Phase 64 only ships the sibling artifact.

</deferred>

---

*Phase: 64-travel-feat-v261-01-verifier-run*
*Context gathered: 2026-06-04 via smart-discuss batch (operator accepted all 4 recommendations)*
