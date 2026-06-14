# Contributing

How to work in this codebase without breaking the load-bearing invariants. This guide also documents the v2.4 milestone contribution requirements (HYGIENE-V24-05, Phase 38).

## Branch Protection Requirements

Branch protection on `master` is operator-managed via GitHub Settings → Branches. The required configuration is:

- **Required status checks** — the `ci.yml` workflow (see [`.github/workflows/ci.yml`](./.github/workflows/ci.yml)) must pass on every pull request before merge.
- **Required reviewers** — at least one approving review.
- **Signed commits** — encouraged but not required for v2.4. (v2.5+ may upgrade to required.)
- **Linear history** — recommended (squash-merge or rebase-merge).
- **Force pushes** — disabled on `master`.

These settings are NOT enforced via repo-checked-in configuration in v2.4 (Terraform / repo-settings YAML is deferred per CONTEXT.md `<deferred>` "Branch protection automation"). Operator action required after merging Phase 38.

## Before you start

Read these in order:
1. [`README.md`](README.md) — project quickstart
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system overview + invariants
3. [`docs/GLOSSARY.md`](docs/GLOSSARY.md) — terminology
4. [`CLAUDE.md`](CLAUDE.md) — project-level constraints (non-negotiables)

## Local Development

- **Python 3.14+** (`pyproject.toml` pins `>=3.14.0,!=3.14.1`)
- **Package manager:** [uv](https://github.com/astral-sh/uv) — required (not pip/poetry)
- **Tests:** pytest 8.x (`uv run pytest -q`)
- **Linter / formatter:** [ruff](https://docs.astral.sh/ruff/) (`uv run ruff check`, `uv run ruff format --check`)
- **Type checking:** not currently configured; manual review (mypy strict deferred to v2.5+)

```bash
# First-time setup
uv sync --frozen

# Fast test suite (skip slow / integration tests)
uv run pytest -q -m "not slow"

# Full test suite (includes slow + regression-tier tests)
uv run pytest -q

# Lint + auto-fix
uv run ruff check --fix

# Format check (CI enforces this)
uv run ruff format --check

# App entry points
uv run ufc --help                                      # CLI
uv run uvicorn ufc_prediction.api.app:app --reload     # API server
```

The CI workflow (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, and `pytest -q -m "not slow"` on every pull request — see Plan 38-04 for the workflow definition. Run these locally before pushing to avoid round-trip CI failures.

Tip: when modifying the entertainment disclaimer in [`src/ufc_prediction/api/disclaimer.py`](./src/ufc_prediction/api/disclaimer.py), remember the same body is QUOTED VERBATIM in the README.md `## Disclaimer` section. Both surfaces must move together; the `tests/contracts/test_forward_compat_v1_2_0.py::test_disclaimer_field_additive_pins_byte_shape_for_v100_v110` test guards the API-side responses but not the README quote.

## AUDIT-01 protected files (DO NOT casually edit)

The following files are byte-identity locked by a pre-commit hook. Edits require deliberate operator action, typically as part of a planned phase that documents the change in PROJECT.md as a `D-{N}` decision row.

```
models/xgb_v2.joblib                          # production base model
models/meta/meta_v2.joblib                    # production meta-learner
models/meta/meta_v2_dedup.joblib              # dedup variant
src/ufc_prediction/ml/predictor.py            # inference dispatch
src/ufc_prediction/ml/feature_matrix.py       # feature engineering
src/ufc_prediction/ml/persistence.py          # model load/save
src/ufc_prediction/ml/train.py                # training entry
scripts/spike_noise_floor_v22.py              # v2.2 spike (preserved for audit lineage)
scripts/spike_noise_floor_v23.py              # v2.3 spike
scripts/train_meta_v22.py                     # v2.2 meta trainer
src/ufc_prediction/contracts/predictor.schema.v1.0.0.json  # PARTNER v1.0.0 byte-frozen
```

If your change needs to touch any of these, **stop and check with the operator first.** There's almost always a fork-not-mutate path (e.g., `compose_v23_meta.py` was a fork of `train_meta_v22.py`; `spike_noise_floor_v23.py` forked from `spike_noise_floor_v22.py`).

If you hit the pre-commit hook unexpectedly, that's the hook doing its job — you're touching a protected file. Don't bypass with `--no-verify`. Investigate.

## The planning system (`.planning/`)

This project uses a structured planning framework (GSD — "Get Shit Done"). All non-trivial work flows through:

```
ROADMAP (milestone-level)
  └── PHASE (a coherent unit of work; e.g., "Phase 28: Referee + Venue Ingestion")
        ├── CONTEXT.md      (locked decisions D-01, D-02, ...)
        ├── RESEARCH.md     (technical patterns + pitfalls)
        ├── PLAN.md         (executable task breakdown)
        ├── SUMMARY.md      (post-execution outcome)
        └── VERIFICATION.md (goal-backward audit)
```

The `.planning/` directory is **history**, not current state. Read `src/` for current truth. Read `.planning/` to understand *why* the code is the way it is.

### Adding new functionality

For non-trivial additions (more than a bug fix or doc edit), use a GSD command:

```bash
# Small fix or doc edit
/gsd-quick

# Bug investigation
/gsd-debug

# Planned new feature
/gsd-discuss-phase N   # capture context first
/gsd-plan-phase N      # then create plan
/gsd-execute-phase N   # then execute
```

If you're not using Claude Code / GSD, follow the same shape manually: write a CONTEXT.md capturing decisions, draft a PLAN.md before coding, then commit atomically with task references.

## Coding conventions

### Naming

- **Modules:** lowercase_with_underscores (`feature_matrix.py`, `gate_contract.py`)
- **Functions:** verb-first, snake_case (`compute_elo`, `load_gate_contract`, `bootstrap_resample`)
- **Constants:** UPPERCASE_WITH_UNDERSCORES (`META_V22_FEATURE_COLUMNS`, `SEEDS_DEFAULT`)
- **Versioned artifacts:** suffix-based (`xgb_v2`, `meta_v2`, `meta_v2_dedup`, `predictor.schema.v1.1.0.json`)
- **Phase artifacts:** zero-padded prefix (`28-01-PLAN.md`, `32-XGB-V2-SHA-PHASE-32-MID-PLAN-01.txt`)

### Module docstrings

Every non-trivial module has a docstring linking back to the planning artifact that justified its existence. Example from `src/ufc_prediction/ml/variance.py`:

```python
"""Phase 30 bootstrap-variance harness — canonical multi-seed entry point.

D-01 / D-02 / D-05 / D-08 binding (per
.planning/.../30-CONTEXT.md):
  D-01: variance is injected by bootstrap resampling...
  D-02: this module's public surface is locked at exactly four functions:
        bootstrap_resample / multi_seed_metrics / aggregate_variance / assert_distinct_seed_brier
  ...
"""
```

When adding a new module:
- Open with one-paragraph "what + why"
- Cite the phase + decision number that justified it
- Mark any locked public surface as "D-{N} binding"

### Imports

- Standard library → third-party → local, separated by blank lines
- Absolute imports preferred: `from ufc_prediction.ml.variance import bootstrap_resample`
- No `from x import *`

### Testing

- Unit tests in `tests/unit/`; integration in `tests/integration/`; regression in `tests/regression/`; partner contract drift in `tests/contracts/`
- Each new feature must come with at least one test that exercises the public surface
- For ML primitives: include a test that exercises the *math*, not just "function runs without error"
- Fixtures live under `tests/fixtures/{feature_area}/`

### No watch-mode flags

Tests use one-shot pytest invocations. No `pytest-watch`, no `--watch`, no hot-reload patterns that obscure feedback latency.

## Commit Conventions

This repository follows [Conventional Commits](https://www.conventionalcommits.org/), matching the existing repo log style:

- Atomic per task — no batching across logical units
- TDD-style: `test(N-NN):` RED commit followed by `feat(N-NN):` or `fix(N-NN):` GREEN commit
- Phase-scoped: include the phase or plan ID in the scope (`feat(30-01): GREEN — implement variance.py per D-01`)

Prefix conventions:
- `feat` — new functionality.
- `fix` — bug fix.
- `docs` — documentation-only changes (this file, README, PARTNER-RELEASE notes).
- `refactor` — code change that neither adds a feature nor fixes a bug.
- `test` — test-only changes.
- `chore` — build / tooling / .gitignore / CI workflow changes.

Examples from the existing log:
- `feat(34-06): predictor no-odds fallback routing + tests + AUDIT-01 MID/END`
- `docs(37): VERIFICATION — Phase 37 PASSED`
- `fix(28-04): cross-source dedup matcher null-name handling`

Phase-bound commits append the phase + plan number in parentheses (e.g., `(34-06)` for Phase 34 Plan 06). This is enforced by GSD workflow convention, not by a commit hook in v2.4.

## Temporal integrity

All features must be computed with explicit pre-fight boundaries. If you're adding a new feature:

1. The function must accept `fight_date` (or equivalent) as a parameter
2. Internal queries must filter to data strictly before that date
3. Document this in the docstring with an example
4. Add a test that constructs synthetic future data and verifies the feature ignores it

This is non-negotiable per CLAUDE.md. Leakage bugs in sports ML look fine in offline eval and fail in production.

## Three model invariants

When working with model artifacts, internalize these:

1. **AUDIT-01 byte-identity** — `xgb_v2.joblib` SHA-256 `6e7641…ba099` is preserved across all releases since v2.1. Don't retrain unless explicitly planned.
2. **Gate contract immutability (formula hash D-18)** — `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a` is LOCKED. Per-slice numerics change between versions; the formula does not. No post-measurement renegotiation.
3. **PARTNER schema additive-only (Phase 25 lock)** — `predictor.schema.v1.0.0.json` is byte-frozen. v1.1.0+ adds optional fields only; never modifies or removes existing ones.

If you can't satisfy a feature request without breaking one of these, escalate to the operator. There is almost always a workaround (new sibling artifact, new optional field, new fork).

## Pull Request Workflow

1. Branch from `master`: `git switch -c feat/<short-description>` or `fix/<short-description>`.
2. Make your changes; run `uv run pytest -q` + `uv run ruff check` locally.
3. Push the branch: `git push -u origin feat/<short-description>`.
4. Open a PR against `master` via the GitHub UI (or `gh pr create`).
5. Wait for CI to run — `.github/workflows/ci.yml` reports back within ~2-5 minutes.
6. Request review from at least one approver (branch protection enforces ≥1).
7. Once CI is green AND review is approved, squash-merge or rebase-merge to `master`.
8. Delete the feature branch after merge.

Do not push directly to `master`; branch protection rejects it.

For larger / phase-scope changes, the GSD workflow (`/gsd-plan-phase` → `/gsd-execute-phase` → `/gsd-verify-work`) applies; see [`CLAUDE.md`](./CLAUDE.md).

### Pre-PR checklist

Before opening a PR:

- [ ] Tests pass: `uv run pytest -q`
- [ ] Lint clean: `uv run ruff check`
- [ ] Format check: `uv run ruff format --check`
- [ ] No AUDIT-01 protected files modified (or, if modified, explicit operator approval cited in PR description)
- [ ] New modules have docstrings citing planning artifact
- [ ] New features have tests in `tests/unit/` or `tests/integration/`
- [ ] No leakage: temporal integrity preserved
- [ ] Commit history is atomic + conventional (squash if needed before merge)
- [ ] If this addresses a v2.4+ backlog item, the relevant phase or CONTEXT.md references the item

## Where to look for examples

- **Adding a new ML primitive:** look at `src/ufc_prediction/ml/variance.py` (Phase 30 — small, focused, well-documented)
- **Adding a new CLI subcommand:** look at `src/ufc_prediction/cli/predict.py` (Phase 31 added `gate-spike` here)
- **Adding a new API field:** look at `src/ufc_prediction/api/v1/models.py` + `scripts/emit_partner_contracts.py` (Phase 32 added the v1.1.0 additive trio)
- **Adding a new scraper:** look at `scripts/scrape_referees_full.py` (Phase 28)
- **Adding a new schema version:** look at `src/ufc_prediction/contracts/predictor.schema.v1.1.0.json` (Phase 32)

When in doubt, find the most recently-added analog and mirror its shape.
