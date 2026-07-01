# Changelog

All notable changes to this project. Each version corresponds to a git tag (`git tag -l 'v*'`). For full per-phase detail, see the linked retrospective or archived `ROADMAP.md` under `.planning/milestones/`.

The project follows a continuous-milestone format. Each milestone delivers a coherent slice of work and ends with a git tag, a retrospective, and an AUDIT-01 byte-identity proof that the canonical model files are unchanged.

---

## Unreleased

**Fixed:**
- Matchup predictions are now **order-invariant**. The raw predictor was asymmetric — swapping `fighter_a`/`fighter_b` shifted the probability by ~10 points on close fights (caused by three non-anti-symmetric style/stance features: `stance_matchup`, `a_striking_vs_b_grappling`, `a_grappling_vs_b_striking`). A new `ufc_prediction.ml.order_invariant.predict_order_invariant` wrapper averages both orderings; the CLI and API now call it, so whoever you list first no longer changes the result. The AUDIT-01-locked `predictor.py` and the model artifacts are untouched.
- **CI: pinned uv to Python 3.13** so `lxml` installs from its prebuilt wheel instead of a doomed source build (`UV_PYTHON=3.13` in `ci.yml` + `refresh-odds.yml`); also dropped the invalid `python-version` input from `setup-uv@v3` that was emitting an `Unexpected input(s)` annotation on every run.
- **Scraper resilience:** `ScraperClient` now retries the Cloudflare `520`–`524` origin-error family (e.g. `522` "Connection Timed Out") with the same backoff it already applied to `429`/`503`. A single transient `522` from BestFightOdds had been aborting the entire weekly odds refresh.
- **`OPS-V30-01`** — the pre-commit git hook was missing on fresh clones because setup only ran `uv sync`. `pre-commit` now ships in the dev dependency group (so `uv sync` provides it) and CONTRIBUTING.md's first-time setup documents `uv run pre-commit install`. Correctness was never at risk — the CI `pre-commit` job runs every hook on each PR — but the local AUDIT-01 guard is now reliably installable in two documented steps.

---

## v3.0.1 — Regression Patch · 2026-06-14

Resolves the two regressions surfaced during v3.0 close before handoff to a new owner. Both fixes are small (handful of lines) and touch only non-protected files.

**Fixed:**
- `REG-V30-01` — CLI and API both defaulted to an unpromoted candidate model. The default model version is now explicitly pinned to `v2` in `src/ufc_prediction/api/v1/predict.py` (API) and `src/ufc_prediction/cli/predict.py` (CLI). `ufc predict matchup` and `POST /api/v1/predict` now succeed out of the box.
- `REG-V30-02` — `tests/integration/test_db_seed.py` now requests an ephemeral port from the OS via `socket.bind(("127.0.0.1", 0))` instead of hardcoding port `55555`, fixing the port-binding quirk on some macOS Docker Desktop installs.

No model artifacts touched; AUDIT-01 byte-identity contract preserved.

---

## v3.0 — Handoff Package · 2026-06-10

The handoff-package milestone. No new model work — package the project for delivery to a downstream technical owner.

**Shipped:**
- `docs/INSTALL.md` (293 lines) — one-page operator walkthrough from `git clone` to first prediction
- `data/seed/ufc_corpus_v30.dump` (10.42 MB pg_dump custom-format) — portable corpus snapshot committed to the repo
- `ufc db seed` + `ufc db status` CLI commands — one-step DB bootstrap
- `KNOWN_ISSUES.md` (247 lines) — per-scraper status, prediction-degradation timeline, v3.0 regressions, v3.1 queue
- `.planning/RETROSPECTIVE-v3.0.md` (143 lines)

**Known issues queued for v3.1:**
- `REG-V30-01` — `ufc predict matchup` defaults to wrong model version (workaround: `--version v2`)
- `REG-V30-02` — Integration test Docker port 55555 binding quirk on some macOS installs


---

## v2.7 — Substrate-Drift-Immune Methodology + Re-Verification · 2026-06-07

Shipped a dual-substrate gate methodology that can distinguish "candidate genuinely worse" from "measurement artifact". Re-ran the 3 v2.6.1 confound-blocked candidates (TRAVEL / REF / NET) under the new methodology — all confirmed worse than META-V22. mypy strict coverage 80.7% → 90.8%. 5 deferred items formally retired at v3.0 open.


---

## v2.6.1 — Carryover Closure Sprint · 2026-06-06

Closure sprint for v2.6 carryovers. Shipped the substrate-snapshot loader (unblocked 4 FEAT verifier runs — all confound-blocked / Path B). Shipped v1.3.0 partner contracts + TypeScript SDK codegen + CI workflow. mypy 51.7% → 80.7%. 13/16 REQs shipped; 3 deferred.


---

## v2.6 — Full Tech-Debt Drain + Gate Methodology Reset · 2026-06-03

Tech-debt drain milestone. Shipped pre-commit framework + CI gate, mypy strict pilot, ProblemDetails (RFC 7807) error wrapper, TypeScript SDK scaffold, and the substrate-drift–safe gate methodology spec + verifier (operator-approved). 27 REQs across 7 categories. AUDIT-01 chain 47-of-N → 62-of-N.


---

## v2.5 — Corpus Growth + Substrate Completion + meta_v3 Candidate · 2026-06-03

Scrape-forward corpus growth through 2026 events. BFO disambiguation root-cause fix shipped +43pp closing-odds coverage. Sherdog debutant Elo seed shipped +0.0032 Brier on debutant cohort. xgb_v3 + meta_v3 sibling candidates trained but did NOT meet promotion gate → META-V22 stays canonical. 30 REQs (29 satisfied + 1 N/A).


---

## v2.4 — Audit Remediation + Partner-Readiness Hardening · 2026-05-27

Partner-deployment hardening. Closed 12 P0 + 11 P1 external audit findings. Shipped API-key auth, rate limiting, structured logging, Sentry, `/health` + `/ready` endpoints, Dockerfile + Fly.io deploy, LICENSE + 200-word disclaimer + data-source attribution. PARTNER schema v1.2.0 additive. 34/34 REQs. Production-ready.


---

## v2.3 — Data Substrate Completion + Ship-Ready Re-Derivation · 2026-05-22

**First public partner release.** Backfilled `events.referee_id` + `events.venue_id` (770/770 ufcstats events; 41.1% of all event rows). Honest multi-seed bootstrap variance harness. Forward-stepwise composition cleared META + CALIB; REF + TRAVEL didn't add ≥0.003 Brier improvement (meta_v3 NOT promoted). PARTNER schema v1.1.0 additive trio. 25 REQs.


---

## v2.2 — Data Investment + Backlog Pull + Partner-Ready Predictor · 2026-05-17

META-V22 (`meta_v2.joblib`) promoted as the new top-of-stack blender. Partner-facing contracts locked at v1.0.0 BEFORE candidate promotion. CALIB + REF + TRAVEL stepwise composition. xgb_v2 byte-identical end-to-end. AUDIT-01 chain extends 4-of-4 → 20-of-N.


---

## v2.1 — Empirical Gate + Backlog Drain · 2026-05-11

Empirically derived promotion gate from 10-seed × 3-slice noise-floor spike. 3-of-3 META candidates gate-failed; xgb_v2 stays canonical.


---

## v2.0 — Architectural Lift (Partial) · 2026-05-03

Closed-partial. xgb_v3 measurably better than xgb_v2 but didn't clear the v2.0 binding gate → NOT promoted. NET-* features (PageRank + 2-hop SoS) measurably hurt; LIVE-only path delivered.


---

## v1.1 — BFO Integration + Dedup · 2026-05-02

BestFightOdds integration. xgb_v2 measured at Brier 0.221 / Acc 65%. Cross-source dedup.


---

## v1.0 — MVP · 2026-04-29

First working predictor. Phases 1–13. Brier 0.230 / Acc 62%.


---

## Cross-cutting invariants

A few invariants have held across the entire project history:

- **`xgb_v2.joblib` byte-identical** since v2.1. SHA-256: `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099`.
- **`meta_v2.joblib` byte-identical** since v2.2. SHA-256: `77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196`.
- **AUDIT-01 chain** verifies these SHAs at every phase boundary. As of v3.0 close, the chain is 90-of-N FINAL.
- **D-18 formula gate** is LOCKED — no post-measurement renegotiation of promotion criteria.

A pre-commit hook enforces the byte-identity contract on a defined set of protected files. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.
