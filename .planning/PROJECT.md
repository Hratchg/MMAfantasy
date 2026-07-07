# UFC Fight Prediction

## What This Is

A Python prediction pipeline for fantasy MMA players that combines a multi-domain Elo rating system with an XGBoost win-probability model trained on real UFC performance data. The differentiator over win/loss-based fantasy tools is a domain-aware Elo (overall + striking + grappling) layered with rolling-window performance, physical mismatch, and historical betting-odds features — all served through a deterministic CLI (`ufc predict matchup`, `ufc fighter lookup`, `ufc fighter rankings`).

## Core Value

Fantasy MMA players can reference a fighter's Elo rating and a calibrated XGBoost win probability — both grounded in real performance and historical betting-market data — to make informed picks.

## Current Milestone: v3.0 Handoff Package

**Goal:** Package the project as a self-contained predictor that a technical user can install, run, and use to predict UFC fights from the existing corpus. Documentation + DB dump + handoff materials only. **No new model work. No new scrapers. No re-verification.** xgb_v2 + meta_v2 byte-identical end-to-end (AUDIT-01 chain continues from 86-of-N FINAL into v3.0). Operator scope: prepare clean handoff to downstream technical users who will run their own data ingest going forward.

**Target buckets:**

- **A — Install + Use Walkthrough** — One-page operator-facing guide from `git clone` to `ufc predict matchup` output. Covers Python + DB + ufc CLI setup.
- **B — DB Dump Packaging** — Ship a portable corpus snapshot (parquet bundle OR pg_dump) so downstream users don't have to re-scrape from scratch.
- **C — Scraper Handoff Docs (`KNOWN_ISSUES.md`)** — Per-source status of each scraper (UFCStats anti-bot blocked, BFO works, Sherdog/Tapology Content-Signal); what works, what's blocked, what downstream users need to solve themselves; degradation timeline for predictions without fresh ingest.
- **D — DOC Close** — v3.0 RETROSPECTIVE + tag v3.0.

**Out of v3.0 scope** (formally retired — NOT carried forward):
- ALL FEAT-V27 / FEAT-V30 candidate re-verifications (TRAVEL/REF/NET) — 5 milestones of evidence; META-V22 is final
- DATA-V30-01 Supabase refresh — downstream user's job
- CORPUS-V30-01/02 alt-events ingest — downstream user's job; Content-Signal blocked anyway
- METH-V30-01 gate recalib --apply — corpus trigger will never fire
- DX-V30-01 mypy 90% → 95% — current 90.8% is good enough
- Any methodology refinement (substrate-realignment, shadow-traffic) — no candidate to promote
- New feature work (NN, coaches, weigh-ins) — not in handoff scope
- Frontend / partner onboarding — explicitly out per operator (2026-06-08)

**Invariants carried forward (from v2.7):**
- `xgb_v2.joblib` SHA `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099` BYTE-IDENTICAL through v3.0 (AUDIT-01 chain extends 86-of-N FINAL → end-of-v3.0 FINAL). All v2.6.1+v2.7 SIBLING artifacts UNTOUCHED.
- `meta_v2.joblib` SHA `77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196` BYTE-IDENTICAL through v3.0. META-V22 canonical.
- Formula hash `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a` LOCKED per D-18. Methodology layer not modified in v3.0.
- PARTNER schema v1.0.0 + v1.1.0 + v1.2.0 + v1.3.0 byte-frozen. No v1.4.0 in v3.0.
- Pre-commit hook on protected files + SOFT-PROTECT on `meta_v2_refit_v*` siblings STAYS active.

**Path-conditional outcomes:**
- **Path A (default):** All 3 handoff phases ship; downstream users receive working repo + DB dump + docs; project formally complete and shippable. v3.0 = final milestone.
- **Path B (if downstream users surface real blockers during handoff):** Document findings + minor fixes; v3.1 hotfix milestone scoped to actual reported issues.

<details>
<summary>Previous v2.5 goal (collapsed; full text in milestones/v2.5-ROADMAP.md)</summary>

**v2.5 shipped 2026-06-03** — Corpus Growth + Substrate Completion + meta_v3 Candidate (29/30 REQs SATISFIED + META3-V25-04 N/A under Path B; AUDIT-01 chain 47-of-N FINAL; META-V22 canonical preserved; PARTNER schema v1.2.0 unchanged under Path B). Largest delta: BFO disambiguation root-cause fix shipped +43pp closing-odds coverage. Sherdog debutant Elo seed shipped +0.0032 Brier on debutant cohort. xgb_v3 + meta_v3 + meta_v22_travel candidates shipped sibling-only; canonical xgb_v2 + meta_v2 byte-identical to v2.3 / v2.4. Multi-source resilience spike Path B both axes (alt-odds + alt-events deferred to v2.6+). Methodology learning: substrate-drift confound in gate-design surfaced; gate redesign required before next META-V retrain.

**Archive:** [milestones/v2.5-ROADMAP.md](./milestones/v2.5-ROADMAP.md) · [milestones/v2.5-REQUIREMENTS.md](./milestones/v2.5-REQUIREMENTS.md) · [milestones/v2.5-MILESTONE-AUDIT.md](./milestones/v2.5-MILESTONE-AUDIT.md) · [retrospective](./milestones/v2.5-phases/47-partner-documentation-retrospective/47-RETROSPECTIVE.md)

**Operator action items pre-public-push:** annotated `v2.5` tag created at milestone close — operator pushes (`git push origin v2.5`) when ready.

</details>

<details>
<summary>Previous v2.5 goal (collapsed; full text in milestones/v2.5-ROADMAP.md)</summary>

## Previous Milestone: v2.5 Corpus Growth + Substrate Completion + meta_v3 Candidate

**Goal:** Grow the data substrate (scrape-forward to 2026, fix scraper hygiene, close TRAVEL feature engineering, resolve BFO disambiguation anomalies, replace the flat 1500 debutant Elo default with Sherdog-derived seeds), then train a meta_v3 candidate on the populated substrate and gate-promote only if it clears v2.3 floors + ≥0.003 Brier hurdle. Carry the 4 v2.4 tech-debt items.

**Target features:**

- **Corpus growth** — scrape-forward UFC events through 2026 + scrape_event_urls source-fix + fighters_names.csv refresh.
- **TRAVEL close-out** — FeatureMatrixAssembler `travel_distance` + `tz_shift` primitives; recompose META → CALIB → TRAVEL on populated substrate.
- **BFO disambiguation anomaly** — resolve 2011/2013/2019/2020 probe-strategy clustering.
- **Sherdog debutant Elo seed** — replace flat 1500 default with Sherdog-derived seed for true UFC debutants.
- **Alternative odds source spike** — 1-day discovery of ESPN BET / FanDuel / oddsportal / kambi historical coverage for pre-2021 events; commit-or-defer at spike close.
- **meta_v3 candidate retrain** — train xgb_v3 + meta_v3 on the populated corpus; **gate-promote only if** clears v2.3 floors (0.70 on all 3 widened slices) **AND** ≥0.003 Brier hurdle over META-V22. If gate-cleared, partner schema bumps to v1.3.0 (additive `model_lineage` field).
- **v2.4 tech-debt closure** — stale `_meta` integration-test assertion (TD-v24-A), unregistered `pytest.mark.integration` (TD-v24-B), REQUIREMENTS.md + ROADMAP.md stale checkboxes (TD-v24-C, TD-v24-D).

**Out of v2.5 scope** (explicit exclusions): REF feature redesign (v2.6+ — pending broader referee corpus growth); TypeScript SDK codegen (v2.6+); ProblemDetails error wrapper (v2.6+); pre-commit hooks framework + mypy strict (v2.6+); multi-region Fly deployment (v2.6+ if SLA crosses 99.5%→99.9%); NN base learner v3.x (v3.x); coach features (v2.6+); weigh-in / medical suspension data (v2.6+).

**Invariants carried forward (from v2.4):**
- `xgb_v2.joblib` SHA `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099` BYTE-IDENTICAL through v2.5 (AUDIT-01 chain extends 39-of-N → ~46-of-N). meta_v3 is a **candidate** sibling — does NOT supersede META-V22 unless gate-promoted.
- `meta_v2.joblib` SHA `77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196` BYTE-IDENTICAL through v2.5.
- Formula hash `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a` LOCKED per D-18 (no post-measurement renegotiation).
- PARTNER schema v1.0.0 + v1.1.0 + v1.2.0 byte-frozen (Phase 25 forward-compat lock). v1.3.0 additive only if meta_v3 promotes.
- Pre-commit hook on protected files (`feature_matrix.py`, `persistence.py`, `predictor.py`, `train.py`, `models/xgb_v2.joblib`, `models/meta/meta_v2.joblib`) STAYS active. New model files (`models/xgb_v3.joblib`, `models/meta/meta_v3.joblib`) are NOT in the protected list during v2.5 (mutable until promoted).

**Path-conditional outcomes:**
- **Path A — meta_v3 gate-promotes:** v1.3.0 schema additive + model_lineage field in partner schema; predictor.py exposes model-selector; META-V22 stays canonical alongside; v2.5 SHIPS as model-improvement milestone.
- **Path B — meta_v3 doesn't clear hurdle:** Ship corpus + substrate completion only; meta_v3 spike findings logged to v2.6 backlog; META-V22 remains canonical; v2.5 SHIPS as data-substrate milestone.
- Operator decides at Phase 45 close based on actual gate verification results.

</details>

## Previous Milestone: v2.4 Audit Remediation + Partner-Readiness Hardening (shipped 2026-05-27)

**Shipped:** 34/34 REQs; AUDIT-01 chain 32-of-N → 39-of-N FINAL; partner-deployable. Closes 12 P0 + 11 P1 external due-diligence audit findings under pure-fantasy framing. `xgb_v2.joblib` + `meta_v2.joblib` byte-identical end-to-end. Production API surface hardened (API-key auth + slowapi rate-limit + CORS lockdown + /health/ready + structured logging + Sentry + v1.2.0 prediction_metadata additive). Fly.io + Python 3.13 + Dockerfile deployment shipped. LICENSE + 200-word entertainment disclaimer + UFCStats/Sherdog/BFO attribution + CI workflow + DATA_STRATEGY.md TOS section + TECHNICAL_HANDOFF.md `## v2.4 Outcomes` + `## Known Limitations (P0/P1/P2)`. Sibling fallback `xgb_v2_no_odds.joblib` closes BFO single-point-of-failure WITHOUT touching xgb_v2 bytes.

**Archive:** [milestones/v2.4-ROADMAP.md](./milestones/v2.4-ROADMAP.md) · [milestones/v2.4-REQUIREMENTS.md](./milestones/v2.4-REQUIREMENTS.md) · [milestones/v2.4-MILESTONE-AUDIT.md](./milestones/v2.4-MILESTONE-AUDIT.md) · [retrospective](./milestones/v2.4-phases/39-partner-documentation-handoff/39-RETROSPECTIVE.md)

**Operator action items pre-public-push:** annotated `v2.4` tag created at milestone close — operator pushes (`git push origin v2.4`) when ready.

**v2.5+ backlog seed items absorbed into v2.5:** items 1 (TRAVEL close-out), 8 (Sherdog debutant Elo seed), 10 (scrape-forward 2026), 13 (meta_v3 retrain candidate), 20 (alt-odds spike), 21 (scrape_event_urls fix), 22 (BFO disambiguation anomaly). Remaining v2.6+ backlog: 15 items (REF, SDK, ProblemDetails, pre-commit, mypy, NN v3.x, multi-region, coach features, weigh-in/medical, GATE-RECALIB-PERIODIC, etc. — see archived v2.4-ROADMAP.md).

<details>
<summary>Previous v2.4 goal (collapsed; full text in milestones/v2.4-ROADMAP.md)</summary>

**Goal was:** Close the 12 P0 + 11 P1 findings from the external due-diligence audit so the system is genuinely partner-deployable (pure-fantasy framing). xgb_v2 + meta_v2 model artifacts stay byte-identical; v2.4 adds the trust-hardening + production wrapper the audit flagged as missing.

**Target features:**

- **Trust Hardening** — 4-baseline benchmark (market-odds-only / Elo-only / no-odds / coin-flip) on v2.3 slices; META-V22 inference-only re-measurement on v2.3 widened slices (no retrain); SHAP + permutation importance for META-V22 and xgb_v2 (no retrain); calibration report (per-slice ECE + reliability plots + Platt-vs-isotonic A/B).
- **Production Wrapper** — API-key auth + slowapi rate limit + restricted CORS; `/health` + `/ready` endpoints; structured logging + Sentry SDK; `/docs` and `/redoc` gated behind dev env; app version bump to `2.3.0` in `pyproject.toml` + `app.py` (schema_version stays separate); Python pin drop to `>=3.13.0,!=3.13.1`; ruff target → `py313`; Dockerfile + Fly.io `fly.toml`.
- **Contract & Resilience** — fix `models/meta/meta_v2-contract.json` (`gate_contract_ref` → v2.3, `candidate_or_promoted` → `promoted`); `PredictorOutputV1` v1.2.0 additive `prediction_metadata` block (`odds_source`, debutant flags); train + persist `xgb_v2_no_odds.joblib` fallback + predictor routing; weekly GH Actions BFO refresh.
- **Hygiene & Legal** — LICENSE + 200-word entertainment disclaimer + UFCStats/Sherdog/BFO attribution; repo cleanup (`.gitignore` data dumps, prune worktrees, resolve uncommitted diffs); `.github/workflows/ci.yml` (ruff + pytest); fix `load_computed_features()` `as_of_date <= event_date` filter + regression test; correct "events.referee_id 100%" claim in STATE.md / partner docs to "100% of ufcstats events; 41.1% of all event rows".
- **Documentation** — v2.2→v2.3 metric delta attribution in `PARTNER-RELEASE-v2.3.md`; train-vs-serve closing-odds semantics; partner deprecation policy doc (12-month claim currently asserted at D-22, not documented); TOS section in `DATA_STRATEGY.md`; fold v2.4 progress into `TECHNICAL_HANDOFF.md` for hiring-party delivery.

**Out of v2.4 scope** (explicit exclusions): TRAVEL feature-engineering close-out (v2.5+ — only if calibration report or stepwise composition shows it would compose above CALIB); REF feature redesign (v2.5+); secondary odds source (Pinnacle/OddsAPI) beyond the `xgb_v2_no_odds` fallback (v2.5+); frontend/UI work (still deferred as separate milestone); META-02 NN escalation (still AF-10 gated); corpus growth / scrape-forward through 2026 events (v2.5+); meta_v3 retraining (META-V22 stays canonical for v2.4); any change to xgb_v2.joblib or meta_v2.joblib bytes.

**Invariants carried forward:** xgb_v2 SHA `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099` byte-identical through v2.4 (AUDIT-01 chain extends 32-of-N → 39-of-N FINAL); meta_v2 SHA preserved; formula hash `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a` LOCKED per D-18 (no post-measurement renegotiation); PARTNER schema v1.0.0 byte-frozen + v1.1.0 additive lock (any v1.2.0 metadata extension must remain additive); pre-commit hook on protected files stays active.

**Operator-confirmed context for v2.4:** use case is **pure fantasy MMA** (gambling-reg load = zero; legal scope reduced to LICENSE + disclaimer + attribution); BFO commercial use **assumed contractually OK** per operator; deployment target = **Fly.io + Python 3.13 + Dockerfile** (third party can substitute); META-V22 is **remeasured, not retrained**; TECHNICAL_HANDOFF.md is the deliverable for the hiring party.

</details>

## Previous Milestone: v2.3 Data Substrate Completion (shipped 2026-05-22)

**Shipped:** v2.3 Data Substrate Completion + Ship-Ready Re-Derivation — first public partner release tag. META-V22 canonical; PARTNER schema v1.1.0 additive trio; populated REF/TRAVEL substrate; gate_contract_v2.3.json (Path A — 0.70 floor cleared on all 3 slices). AUDIT-01 chain leaf 32-of-N. See [milestones archive](./milestones/v2.3-ROADMAP.md) + [retrospective](./phases/33-close-retrospective-public-release/33-RETROSPECTIVE.md).

**Operator action items pre-public-push:**
- Run `/gsd-verify-work` conversational UAT (PARTNER-V23-04 operator-pending)
- Verify v2.3 tag annotation (`git show v2.3`)
- `git push origin v2.3` AFTER UAT GREEN

**v2.4+ backlog seed items** (from v2.3 retrospective):
1. TRAVEL feature-engineering gap — substrate populated but FeatureMatrixAssembler missing travel_distance/tz_shift primitives
2. REF feature investigation — REF didn't compose above CALIB even with populated substrate
3. predictor.py:163-171 cols-dispatch generalization (meta_v3+ readiness)
4. random_15pct slice floor deviation (re-widen or split differently)
5. CAMP re-eligibility trigger (top-30 Sherdog coverage ≥60%)

To open v2.4+: run `/gsd-new-milestone v2.4`. Fresh REQUIREMENTS.md will be created at that step.

<details>
<summary>Previous v2.3 goal (collapsed; full text in milestones/v2.3-ROADMAP.md)</summary>

**Goal was:** Close the data-substrate gaps surfaced at v2.2 close (events.referee_id + events.venue_id at 0% populated) so REF + TRAVEL features actually carry signal, re-derive the gate on the populated substrate with genuine multi-seed variance, then promote a ship-ready candidate that partners can build against without caveats. v2.3 is the **first public-facing partner release** — same code architecture as v2.2 with materially better empirical evidence behind it. PARTNER schema bumps to v1.1.0 (additive only — Phase 25's forward-compat lock is binding).

</details>

**Target features:**

- **Referee + Venue Ingestion Pipeline** — backfill `events.referee_id` + `events.venue_id` across the 5,799-fight corpus using the Phase 22 referee scraper + `data/venues.csv` lookup + fuzzy match for unmapped venue strings. Emit ingestion coverage reports (targets: referee_id ≥90% of post-2007 events; venue_id ≥95%). Closes Q6 finding from v2.2.

- **CAMP Re-Audit** — re-run Phase 20 Sherdog `Association:` coverage audit on the (possibly expanded) corpus to check if top-30 coverage now crosses 60% threshold. If yes, CAMP migration + features re-eligible; if no, defer again with pre-registered v2.4+ trigger.

- **Genuine Multi-Seed Variance Harness** — replace v2.2's deterministic 5-seed META spike with bootstrap resamples of training set (or stratified random splits per seed) so 5 seeds produce 5 different metrics. Report Brier/Acc CIs honestly; gate contract emits real variance + CI half-widths.

- **Larger Eval Sets** — widen post-NaN-drop eval slices (currently 274 fights) by per-feature NaN handling (instead of symmetric drop), pooling slices, or extending random_15pct sample. Target: ≥500 fights per slice post-drop. Resolves v2.2's 12mo/24mo slice collapse.

- **Gate Re-Derivation on Populated Substrate** — Phase 24 equivalent: re-run noise-floor spike using same formula hash on the populated REF/TRAVEL substrate. Commit at milestone open to whether the original 0.70 operator floor is reachable empirically OR what the new empirical floor is. Emit `gate_contract_v2.3.json` (v2.1 + v2.2 contracts PRESERVED for audit lineage).

- **Forward-Stepwise Re-Composition** — Phase 26 equivalent on the populated substrate: META → CALIB → REF → TRAVEL (+ conditional CAMP). Promote candidate that beats BOTH the v2.3 gate AND ≥0.003 Brier hurdle on all 3 slices. Pre-template "REF/TRAVEL still don't contribute meaningful signal" outcome path.

- **Ship-Readiness Review + Public Release** — after gate clears, run `/gsd-verify-work` conversational UAT; prepare partner release notes (`docs/PARTNER-RELEASE-v2.3.md`); bump partner schema to v1.1.0 (additive only — version field + new optional metadata, no breaking changes); git tag `v2.3` as the **first public partner release** (v2.2 stays as internal milestone).

**Out of v2.3 scope** (explicit exclusions): corpus growth beyond what referee/venue ingestion incidentally requires (no scrape-forward through 2026 events; that's v2.4); frontend/UI work; weigh-in/medical suspension data sources; new ML architectures (NN base learner replacing XGBoost is v3.x); Tapology/ESPN API integration; automated cron scheduling; real-time event tracking; META-02 NN escalation (still AF-10 gated — only if META-V22 logistic gets beaten by ≥0.003 Brier).

**Target features:**

- **Gate Recalibration v2.2 (front of milestone)** — re-run 10-seed × 3-slice noise-floor spike on the new feature column space (xgb_v2 cols + new Level-1 META + new contextual features). Mechanically re-derive BOTH `brier_max` AND `accuracy_min` per slice using D-18's `median ± 1·max(seed_std, bootstrap_BCa_68pct_CI_half)` formula. Operator-stated empirical floor: `accuracy_min ≥ 0.70` on all 3 slices. Output: `gate_contract.json` v2.2 with new formula_hash. AF-1 (no hparam retuning) and AF-2 (no feature engineering) banned during spike.

- **BFO Coverage Backfill** — close the Phase 15.1 coverage gap (60%+ of post-cutoff fights have NaN `closing_prob_diff` per D-20 honest read). `closing_prob_diff` is the model's #1 training feature and the root cause of META-01's failure on `random_15pct`. Targets: bring BFO coverage from ~15% on training set to a measurably higher floor; pre-register the coverage target and the new train/eval distribution match strategy. Plan must include BFO archive reachability spike before committing to a coverage target.

- **Rich Level-1 META Features (D-02(P19) deferred set)** — matchup metadata, division priors, layoff, age, Elo-deltas (overall + striking + grappling) added to the meta-learner's Level-1 input vector. Retry META-01 on the BFO-backfilled corpus with the rich Level-1 set; META-02 NN escalation still gated by ≥0.003 Brier improvement over META-01 per D-13(v2.0) carry-forward.

- **Sherdog Association Audit + CAMP (Fight Camp)** — Phase 19 v2.0 carry-forward. CAMP-00 audit (1-day spike) MUST run before CAMP-01..03 scope; audit gates whether CAMP feature set is viable from Sherdog `Association:` coverage. If audit clears: CAMP-01 (aliases normalization table), CAMP-02 (per-camp performance aggregates), CAMP-03 (camp-vs-camp matchup features). 1 Alembic migration.

- **Referee + Travel + Calibration (REF/TRAVEL/CALIB)** — Phase 18 v2.0 carry-forward. REF-00..02 (referee tendency features — finish rate per referee, no-action rate, scorecard tendencies), TRAVEL-01..03 (timezone shift, travel distance, days-since-travel), CALIB-01..02 (Platt default explicit, isotonic conditional on `len(X_calib) ≥ 1000` per D-14(v2.0)). 3 Alembic migrations.

- **Partner-Ready Contracts (last phase of milestone)** — versioned `predictor.json` output schema with semver + deprecation policy; versioned `*.joblib` + `*-meta.json` artifact contract (feature_columns, version, metrics, cutoff_date, n_features, gate_contract reference); OpenAPI 3.x spec for FastAPI prediction routes so the third-party UI partner can codegen clients. Includes write-side stability tests (output snapshot tests) and read-side integration docs.

**Out of v2.2 scope** (explicit exclusions): corpus growth (scrape-forward through 2026 events), frontend/UI work, weigh-in/medical suspension data sources (Phase 20 still deferred), new ML architectures (NN base learner replacing XGBoost is v3.x), Tapology/ESPN API integration, automated cron scheduling, real-time event tracking, coach features (likely subsumed by CAMP signal — re-evaluate after CAMP-00 audit; trimmed at scope confirmation), time-decayed PageRank revisits (v2.1 Phase 18 tested 2 variants both gate-failed; trimmed at scope confirmation), opening-to-closing odds drift feature (untested signal; trimmed at scope confirmation).

## Historical State Notes (pre-v2.1)

**Shipped:** v2.0 Architectural Lift (Partial — Foundation Only) (closed partial 2026-05-03) · [archive](./milestones/v2.0-ROADMAP.md) · [requirements](./milestones/v2.0-REQUIREMENTS.md)

**Production model:** `models/xgb_v2.joblib` — Brier 0.2206 / Acc 0.6506 / AUC 0.6981. UNCHANGED from v1.1 close. xgb_v3 5-seed median improved to {Brier 0.2156-0.2222, Acc 0.6582-0.6681} across the 3 slices (strictly better than xgb_v2) but did not clear the v2.0 binding gate (Brier ≤ 0.215 AND Acc ≥ 0.67) on any slice. Per D-15+D-17, NO relaxation: xgb_v3 NOT promoted; xgb_v3.joblib NOT persisted. xgb_v2 stays.

**xgb_v2 SHA-256 baseline (rollback path):** `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099`. Byte-identity preserved across v1.1 → v2.0 (3-of-3 audit checkpoints in Phase 16 all OK).

**v2.0 delivery:** Phase 16 only of planned 4 phases. 15 of 30 v2.0 REQs delivered (HOUSE-* + LIVE-* + NET-* + MODEL-02); 1 explicitly gate-failed (MODEL-01); 14 deferred to v2.1+ (META-* + CALIB-* + REF-* + TRAVEL-* + CAMP-*). The 44/72 NaN-padded inference-feature gap is closed end-to-end — `closing_prob_diff` (the model's #1 training feature) now populates at predict time via `bfo_live.py` + `inference_features.py`; pre-Phase-16 it was ALWAYS NaN.

**Phase 16 honest read:** xgb_v3 IS strictly better than xgb_v2 (Brier improves 0.0050, Acc improves 0.0175, AUC improves 0.0182). v2.0 tightened the v1.1 gate by ~0.005 Brier and ~0.031 Acc, putting xgb_v3 in a between-gates band: would have shipped under v1.1, does not clear v2.0. NET-* features (PageRank + 2-hop SoS) measurably HURT performance on every slice — the v2.0 gate may be aspirational from a 5,799-fight corpus.

**Data:** 5,799 historical UFC fights × 12,046 BFO odds rows; 3,375 fighters with Sherdog pre-UFC career data. UNCHANGED from v1.1.

**Active CLI surface:** `ufc predict matchup`, `ufc predict train`, `ufc predict coverage`, `ufc fighter lookup`, `ufc fighter rankings`, `ufc scrape odds`. UNCHANGED from v1.1; `ufc predict matchup` now delivers the full 72-feature vector at predict time when BFO is reachable.

## Next Milestone Goals (v2.1+)

**First decision before any feature work:** Re-evaluate the v2.0 promotion gate (Brier ≤ 0.215 AND Acc ≥ 0.67). The Phase 16 result demonstrates the gate may be aspirational from current corpus size. Either tighten data first (more years, better coverage) or relax the gate to a level that's measurably crossable.

**Carry-forward backlog (from v2.0):**

1. **NET-* redesign or removal** — Phase 16 finding: PageRank + 2-hop SoS hurt model performance. Code stays committed (infrastructure has independent value for ranking displays). v2.1 options: per-division PageRank, time-decayed weighting (`0.98^days`), Sherdog pre-UFC seed, or feature column ablation.
2. **Phase 17 — Meta-Learner** — META-01/02/03. D-16(v2.0) "gates against xgb_v3" is now vacuous (no xgb_v3 was promoted); v2.1 must re-decide gate baseline.
3. **Phase 18 — Referee + Travel + Calibration** — REF-00..02, TRAVEL-01..03, CALIB-01..02. Three Alembic migrations. Platt default, isotonic conditional.
4. **Phase 19 — Fight Camp** — CAMP-00..03. Sherdog `Association:` coverage audit gates scope.
5. **Phase 20 — Weigh-In and Medical Suspension Data** — revisit data-source viability.
6. **Coach features, time-decayed PageRank, opening-to-closing odds drift** — long-tail v2.1+ ideas.

## Requirements

### Validated

- ✓ Scrape and store historical UFC fight data (round-by-round stats from UFCStats) — v1.0 Phase 5
- ✓ Ingest Kaggle datasets for rapid prototyping — v1.0 Phase 2
- ✓ Compute domain-specific Elo (overall + striking + grappling, per-division) — v1.0 Phases 3, 6
- ✓ All-time historical Elo for every fighter since ~2005 — v1.0 Phase 3
- ✓ Full matchup matrix (offensive-vs-defensive differentials) — v1.0 Phases 7, 8
- ✓ Physical matchup modifiers (reach, height, leg reach) — v1.0 Phase 8
- ✓ Style tagging (striker / grappler / balanced) — v1.0 Phase 7
- ✓ REST API for fighter ratings, rankings, matchup comparisons — v1.0 Phase 9
- ✓ CLI for lookups + CSV/JSON exports — v1.0 Phases 4, 9
- ✓ Manual scrape + recompute trigger — v1.0 Phase 5
- ✓ PostgreSQL database for fighter / fight / rating data — v1.0 Phase 1
- ✓ Per-round data ingestion (pace decay features) — v1.0 Phase 11
- ✓ Sherdog pre-UFC career records (debutant Elo init) — v1.0 Phase 12
- ✓ XGBoost win-probability model with calibrated output — v1.0 Phase 10 (xgb_v1: Brier 0.2302, Acc 61.9%)
- ✓ BestFightOdds historical data integration — v1.1 Phase 13 (code) + Phase 15 (data: 12,046 rows)
- ✓ Fighter de-duplication across Kaggle + UFCStats sources — v1.1 Phase 14 (DEDUP-01/02/03)
- ✓ BFO scrape resilient on Python 3.14 (no PicklingError) — v1.1 Phase 15 (ODDS-01)
- ✓ Retrained model improving over xgb_v1 — v1.1 Phase 15.1 (xgb_v2: Brier 0.2206, Acc 0.6506; ODDS-03 closed via D-04(P15.1) relaxation + D-12 spirit override)
- ✓ v1.1 audit-derived housekeeping closed — v2.0 Phase 16 (HOUSE-01..06: n_features metadata, dedup unification, retro VERIFICATION.md, Python 3.14.1 blocklist pin, METHODOLOGY refresh)
- ✓ 44/72 NaN-padded inference-feature gap closed end-to-end — v2.0 Phase 16 (LIVE-01..03: bfo_live.py + inference_features.py + load-time FEATURE_COLUMNS assertion; `closing_prob_diff` populates at predict time)
- ✓ Opponent-network feature infrastructure (PageRank + 2-hop SoS, as-of-fight-date, no temporal leakage) — v2.0 Phase 16 (NET-00..03; **empirical caveat:** features measurably hurt model performance, carry-forward to v2.1 for redesign or removal)
- ✓ xgb_v2 byte-identity preservation across architectural lift — v2.0 Phase 16 (MODEL-02; 3-of-3 SHA audit checkpoints OK; rollback path certified)

### Gate-Failed (Carry-Forward)

- ⚠ Promote `xgb_v3.joblib` under v2.0 binding gate (Brier ≤ 0.215 AND Acc ≥ 0.67 on all 3 slices) — v2.0 Phase 16 MODEL-01: gate failed on all 3 slices despite xgb_v3 strictly improving on xgb_v2. Per D-15+D-17, NO relaxation: xgb_v2 stays. v2.1 first decision: re-evaluate gate thresholds before committing to NN meta-learner / referee-travel / camp data work.

### Active

Active v2.6 requirements live in `.planning/REQUIREMENTS.md` — categories: HYGIENE (A), DX (B), API (C), GATE-METHOD (D), FEAT-DEBT (E). v2.5 REQUIREMENTS.md archived to `.planning/milestones/v2.5-REQUIREMENTS.md`; v2.4 at `.planning/milestones/v2.4-REQUIREMENTS.md`; v2.3 at `.planning/milestones/v2.3-REQUIREMENTS.md`; v2.2 at `.planning/milestones/v2.2-REQUIREMENTS.md`; v2.1 at `.planning/milestones/v2.1-REQUIREMENTS.md`.

v2.2 historical scope summary (closed 2026-05-17):
- **GATE-V22-* requirements** — v2.2 gate recalibration spike on the new feature column space; mechanically derive both `brier_max` AND `accuracy_min` per slice; empirical floor `accuracy_min ≥ 0.70` on all 3 slices
- **BFO-V22-* requirements** — BFO coverage backfill; close the Phase 15.1 coverage gap; pre-register coverage target + train/eval distribution match strategy
- **META-V22-* requirements** — rich Level-1 META feature set (D-02(P19) deferred); META-01 retry on backfilled corpus; META-02 NN escalation gated by ≥0.003 Brier improvement (D-13(v2.0) carry-forward)
- **CAMP-* requirements** — Sherdog Association coverage audit (CAMP-00 spike) + CAMP-01..03 if audit clears; 1 Alembic migration
- **REF-* + TRAVEL-* + CALIB-* requirements** — referee tendency features (REF-00..02), travel/timezone features (TRAVEL-01..03), explicit calibration wiring (CALIB-01..02); 3 Alembic migrations; Platt default, isotonic conditional on `len(X_calib) ≥ 1000`
- **PARTNER-* requirements** — versioned predictor JSON output schema (semver + deprecation policy); versioned model + metadata artifact contract; OpenAPI 3.x spec for FastAPI surface

### Future Requirements

- **Re-evaluate v2.0 promotion gate thresholds** — first v2.1 decision. Brier ≤ 0.215 AND Acc ≥ 0.67 may be aspirational from current ~5,799-fight corpus.
- **NET-* redesign or removal** — Phase 16 finding: PageRank + 2-hop SoS hurt model performance. v2.1 candidates: per-division PageRank, time-decayed weighting, Sherdog pre-UFC seed, or full ablation.
- **Phase 17 (Meta-Learner)**, **Phase 18 (Referee + Travel + Calibration)**, **Phase 19 (Fight Camp)** — v2.0 phases never started; deferred to v2.1+. See ROADMAP.md "v2.1+ Backlog" for full carry-forward.
- **Phase 20 — Weigh-In and Medical Suspension Data** — weight-cut tracking, injury / suspension history. Revisit for v2.1+ once data-source viability is assessed.
- **Coach features** — `Fighter.coach_tag` and per-coach behavior aggregates. Deferred from v2.0 because camp signal likely subsumes most coach signal at v2.0 dataset size.

### Out of Scope

- **Frontend/web UI** — still deferred. Data engine is production-grade; UI work is a meaningful new milestone, not a v2.x stretch.
- **Automated scheduling/cron** — still manual; the scrape pipeline is fast enough (~2.5h via `260423-agz`) that operator-triggered runs remain practical.
- **Tapology / ESPN API** — UFCStats + Sherdog + BFO is sufficient for v2.x prediction quality. Revisit if a feature audit identifies a coverage gap not addressable from current sources.
- **Additional ML architectures beyond NN meta-learner** — XGBoost remains base learner. Phase 17 NN is a blender, not a replacement.
- **Real-time event tracking** — historical-only is the design; live in-fight prediction is a separate product.

## Context

- **Codebase:** ~31,400 LOC Python (src + tests + scripts) at v1.1 close; v2.0 added +17,081 / −2,203 LOC across 71 files (Phase 16 only).
- **Tech stack:** Python `>=3.14.0,!=3.14.1` (HOUSE-06 pin: NetworkX 3.6.1 blocklists 3.14.1), SQLAlchemy + psycopg + Postgres (Alembic migrations), Typer CLI, FastAPI surface, XGBoost / scikit-learn / pandas / NumPy / NetworkX (added in v2.0 for opponent-network features). Custom `ScraperClient` for HTTP (no `multiprocessing` after v1.1 — Py 3.14 doesn't allow closure pickling under spawn).
- **Data:**
  - 5,799 unique UFC fights (Kaggle + UFCStats merged; Sherdog supplements debutant Elo) — UNCHANGED from v1.1
  - 12,046 BFO odds rows / 5,799 fights — 100% UFCStats fight coverage but only 15.1% training-set coverage (BFO archive starts ~2007) — UNCHANGED from v1.1
  - 3,375 fighters with pre-UFC Sherdog records — UNCHANGED from v1.1
  - Source priority: ufcstats > sherdog > kaggle (now centralized in `src/ufc_prediction/dedup/source_priority.py` per HOUSE-04)
- **Production model:** `models/xgb_v2.joblib` — 72-feature column space; **all 72 features now computed live at inference** (was 28/72 pre-Phase-16). xgb_v1 retained byte-identical as rollback path. xgb_v3 NOT promoted (gate failed); Phase 16 candidate bundle was in a now-removed worktree — any v2.1 retrain restarts from scratch.
- **v2.0 carry-forward findings:**
  - **NET-* features hurt performance** (Δ Brier vs no-NET baseline = -0.0007 / -0.0007 / -0.0013 across 3 slices). Code stays committed (infrastructure value for ranking displays); ML feature columns can be ablated in v2.1's first plan.
  - **v2.0 gate may be aspirational** from current corpus. Brier ≤ 0.215 AND Acc ≥ 0.67 was tightened from v1.1 by ~0.005 Brier and ~0.031 Acc. xgb_v3 fell in the between-gates band: would have shipped under v1.1, did not clear v2.0.
  - **75-feature column space designed and integrated** — `pagerank_diff`, `sos_2hop_diff`, `is_debutant_in_graph_diff` added to `FEATURE_COLUMNS`. Production xgb_v2 still on 72 features (xgb_v3 was 75, never persisted).
- **Open backlog:** none active (last item 999.4 resolved via `260501-u9u`). v2.1+ carry-forward tracked in ROADMAP.md "v2.1+ Backlog" section.

## Constraints

- **Tech stack:** Python 3.14 backend; SQLAlchemy + Postgres; Typer CLI; FastAPI for API surface
- **Database:** PostgreSQL for fighter / fight / rating / odds data
- **Data integrity:** All Elo and ML features must use only pre-fight data (strict temporal ordering enforced by `Event.event_date` filtering and explicit cutoff in `train_test_split`)
- **Data quality:** Significant-strike classification is subjective (ringside staff judgment); BFO archive sparse pre-2007. Design for noise tolerance.
- **Sample size:** Many fighters <8 UFC fights; Bayesian shrinkage and minimum-N gating in EWMA / rolling features
- **Python 3.14 compatibility:** `multiprocessing` with closure-based workers raises `PicklingError` under spawn — all scrapers must use `ScraperClient`'s `ThreadPool` instead
- **Model promotion contract (D-09 / D-10 from Phase 15):** xgb_v1 byte-identity must be preserved across any retrain; `*-meta.json` schema must include `feature_columns`, `version`, `metrics`, `cutoff_date`, `n_training_fights`, `n_test_fights` (and ideally `n_features` — currently absent from `save_model`, present in xgb_v2_meta.json via manual edit)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Domain-specific Elo (striking + grappling + overall) | MMA is multi-domain; a single rating flattens style differences | ✓ Validated — v1.0 |
| Per-division Elo pools | Fantasy players compare within weight classes; cross-class Elo is noise | ✓ Validated — v1.0 |
| Body metrics as matchup modifiers, not Elo inputs | Keep Elo as a pure skill signal | ✓ Validated — v1.0 |
| Kaggle + scraper in parallel | Kaggle for fast iteration, scraper for data ownership | ✓ Validated — v1.0 |
| Manual update trigger for v1 | Simplicity first | ✓ Validated — v1.0 |
| Full matchup matrix in v1 | Style interactions are highest-value per DATA_STRATEGY.md | ✓ Validated — v1.0 |
| Source priority filter in `_resolve_fighter` only (Phase 14) | No schema migration; ufcstats > sherdog > kaggle | ✓ Validated — Phase 14 |
| Model metadata `feature_columns` authoritative for live prediction | Code FEATURE_COLUMNS grows; model only knows training-time cols | ✓ Validated — Phase 14 (caught 67 vs 70 mismatch in UAT) |
| Replace `ufcscraper.BestFightOddsScraper` with native `ScraperClient`-based BFOScraper (Phase 15) | `ufcscraper` 1.1.0 uses closure-based workers that fail under Py 3.14 spawn; HTML structure also drifted | ✓ Validated — Phase 15 (no PicklingError; 12,046 rows ingested) |
| Hard accuracy gate before `save_model` (Phase 15 D-07) | Refuse to promote a model that doesn't meet target metrics; xgb_v1 stays as rollback | ✓ Validated — Phase 15.1 (gate held under strict thresholds; relaxed via documented operator override only) |
| Path B (relaxed-gate derivation) over Path A (window-narrowed retrain) for ODDS-03 closure | Per-year coverage diagnostic showed Path A lever-1 structurally infeasible (max 36.8% << 70% threshold) | ✓ Validated — Phase 15.1 (Brier noise floor confirmed; spirit override applied via D-12) |
| Closest-date matching in `BFOOddsIngester._resolve_fight` (999.4 / 260501-u9u) | Rematch overwrite bug silently dropped 117 fights' odds; `event_date` is parsable from composite `BFOOddsRow.fight_id` | ✓ Validated — quick task `260501-u9u` (rematch_fights_with_odds 125 → 239) |
| Substring + exact-match dedup unified in `_resolve_fighter` (260501-uyd) | Phase 14's tests only covered exact match; substring queries (Cormier, Topuria) bypassed source-priority filter | ✓ Validated — quick task `260501-uyd` |
| D-13(v2.0): Phase 17 starts with logistic + interactions; NN escalates only if Brier improves ≥0.003 over plain logistic | Stacking literature (Niculescu-Mizil & Caruana ICML 2005); 5,799-fight dataset is at low end of NN sweet spot | Pending — Phase 17 |
| D-14(v2.0): Phase 18 calibration defaults to Platt; isotonic conditional on `len(X_calib) ≥ 1000` | Niculescu-Mizil 2005 1,000-sample threshold; calibration set sits at ~580–870 fights at v2.0 cutoff | Pending — Phase 18 |
| D-15(v2.0): Phase 16 includes 1–2 day NET-00 graph design spike before implementation | 7 of 13 documented pitfalls touch Phase 16; per-division vs pan-MMA has no published MMA precedent | ✓ Validated — Phase 16 (16-03 Task 1 spike + operator gsd-checkpoint resolved `approve pan-mma-mov`; 16-04 D-08 sanity floor caught the empirical NET-* regression — Δ Brier < 0 on every slice — confirming the spike-then-validate-empirically discipline was correct) |
| D-16(v2.0): Phase 17 NN gates against Phase-16-retrained xgb_v3, not xgb_v2 | Otherwise Phase 16 lift is laundered as Phase 17's win | Pending — Phase 17 |
| D-17(v2.0): Promotion gate (Brier ≤ 0.215 AND Acc ≥ 0.67) is binding — NO mid-milestone relaxation pathway | Avoids Phase 15.1 D-04(P15.1) precedent; either pass and ship, or fail and stop for operator decision | ✓ Exercised — Phase 16 (xgb_v3 5-seed median FAILED on all 3 slices: Brier {12mo:0.2156, 24mo:0.2167, rand15:0.2222} — strictly better than xgb_v2 (0.2206) but the v2.0 gate was tightened from v1.1 by ~0.005 Brier and put xgb_v3 in a between-gates band. Per D-15+D-17 NO relaxation: xgb_v3 NOT promoted; xgb_v2 stays in production; gate held. See 16-04-XGB-V3-REPORT.md.) |
| D-13(P16): 3-slice gate (most-recent-12mo / 24mo / random-15%-temporal) — ALL must clear Brier ≤ 0.215 AND Acc ≥ 0.67 | Stricter than median-only or primary-slice-only; protects against "works on most-recent-12mo but fails on 24mo" patterns | ✓ Validated — Phase 16 (3-slice harness fired correctly on all 3 slices in 16-04 Task 5; identified per-slice failure margins for operator decision; see 16-04-XGB-V3-REPORT.md) |
| D-14(P16): Top-N ranking-stability is a SOFT flag (Spearman ρ < 0.7 = operator review), not a hard gate | Catches catastrophic ranking flips without rejecting genuine improvement | Deferred — Phase 16 (skipped in FAIL path per plan's gate-decision tree; would have run on PASS path. Soft-check semantics validated by absence of premature rejection) |
| D-16(P16): 5 random_seeds (42, 43, 44, 45, 46), median (not mean) wins | Phase 15.1 saw ±0.0006 seed-to-seed Brier swing; 5 seeds gives stable median; median (not mean) avoids one bad seed dragging the verdict | ✓ Validated — Phase 16 (per-seed Brier std ≤ 0.0009 on every slice; median verdict insensitive to seed choice; per-seed array preserved in 16-04-XGB-V3-REPORT.md "Per-Seed Sub-Rows") |
| **D-18(v2.1, GATE)**: v2.1 promotion gate empirically derived from 10-seed × 3-slice noise-floor spike on `FEATURE_COLUMNS_NO_NET` (72 cols, xgb_v2's exact column space per D-01(P17, corrected 2026-05-04 — Option B)). Formula `median ± 1·max(seed_std, bootstrap_BCa_68pct_CI_half)` per slice per metric, operator-approved at sha256 `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a`. Per-slice thresholds: `most_recent_12mo: {brier ≤ 0.2102, acc ≥ 0.6870}`, `most_recent_24mo: {brier ≤ 0.2134, acc ≥ 0.6673}`, `random_15pct: {brier ≤ 0.2151, acc ≥ 0.6809}`. Supersedes D-13(P16) and D-17(v2.0). Carries forward D-17 "no relaxation" + adds **"no post-measurement renegotiation"** clause. | The v2.0 mistake was setting thresholds before measuring corpus noise floor. Phase 17 inverts the ordering: measure noise first via 10-seed retrain on xgb_v2's exact column space, derive thresholds mechanically from operator-approved formula. Empirical finding: the gate is **tighter** than v2.0's aspirational `0.215 / 0.67` on every slice (Brier tighter on all 3; Acc tighter on most_recent_12mo and random_15pct), confirming the v2.0 gate was less aspirational than first thought — but achievable on most_recent_12mo, not all slices. | ✓ Validated — Phase 17 (`.planning/gate_contract.json` + `17-NOISE-FLOOR-REPORT.md`) |
| **D-19(v2.1, NET)**: NET-V2 redesign empirically tested via Plan 18 5-seed × 3-slice harness (single-variant pre-commit per D-01(P18); ablation + time-decayed `0.98^days` per D-02(P18); ≥0.003 Brier improvement-margin tie-breaker per D-03(P18)). Outcome: `neither_clears_xgb_v2_stays` — neither variant cleared the v2.1 gate (`gate_contract.json` formula_hash `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a`) on all 3 slices. Per-slice gate verdicts: ablation `12mo:✗(0.2145>0.2102) 24mo:✗(0.2155>0.2134) rand:✗(0.2209>0.2151)`; time_decayed `12mo:✗(0.2144>0.2102) 24mo:✗(0.2155>0.2134) rand:✗(0.2207>0.2151)`. Tie-breaker margin (0.003 Brier across all 3 slices) moot — neither variant cleared the gate to begin with. xgb_v2 retained as v2.1 base model (sha256 `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099`); no canonical `xgb_v2_1.joblib` alias exists. Both variant joblibs archived to `.planning/phases/18-net-redesign-or-removal/archive/` for empirical record (AUDIT-03; ablation sha256 `a435c1c4f5fd103afcf53de02ab8d2b7780ad39b75609eda2fe05cf469fe52a3`, time_decayed sha256 `34afef69e90137c15425525f898080d3c5b6f88f115c47e045925fdb024ad489`). `network_v2.py` retained as preserved-but-unused infrastructure (parallel to `network.py` retained for ranking-display use); ML feature columns remain at 75 cols with NET-* but xgb_v2 (72-col, no NET-*) is the production model. APPLIES D-18(v2.1, GATE) — does NOT supersede it. Adds **single-variant pre-commit precedent** for v2.2+ feature redesigns. | The v2.1 gate is rigorous: Phase 17 measured corpus noise floor at median Brier 0.2147 / std 0.0045, set gate at median − 1·std = 0.2102 on most_recent_12mo; ablation produced Brier 0.2145 (within noise of xgb_v2's reproduce 0.2147 → 0.2179, well above gate). Time-decayed variant did not improve sufficiently over ablation. The pattern validates v2.1's empirical-first discipline: the gate is set to a level that requires real improvement, and "no improvement clears it" is a valid empirical answer (1-of-5 pre-approved outcome paths per D-05(P18); Pitfall #10 explicit pre-templating). Phase 19 META-01 will stack on xgb_v2 (NOT on a Phase-18-promoted candidate). | ✓ Validated — Phase 18 (`18-NET-DESIGN-COMPARISON.md` + `archive/` bundle + variant meta JSONs + `18-VERIFICATION.md`) |
| **D-20(v2.1, META-01)**: META-01 (logistic + interactions blender on xgb_v2 base) empirically tested via Plan 19-02 5-seed × 3-slice harness (three-way split per D-01(P19); minimal Level-1 features `[xgb_oof_prob, elo_prob, closing_prob_diff]` + PolynomialFeatures(degree=2, interaction_only=True) per D-02(P19); meta_train=2398, meta_eval=928 from 16641 fights; symmetric NaN-drop on train + eval per Rule 1 fix at e26a7f8). Outcome: `meta01_does_not_clear_gate` — META-01 cleared `most_recent_12mo` (Brier 0.2043 ≤ 0.2102 ✓; Acc 0.7085 ≥ 0.6870 ✓) and `most_recent_24mo` (identical metrics; surviving rows after NaN-drop on closing_prob_diff all fall within most-recent-12mo window) with significant margin, but **failed `random_15pct` decisively** (Brier 0.2343 > 0.2151; Acc 0.6250 < 0.6809). Per D-13(P16) all-3-slices-must-pass discipline, single-slice failure = no ship. Per D-07(P15) hard-gate-then-save, `models/meta/meta_v1.joblib` NOT promoted; META-01 candidate persisted as `meta_v1_candidate.joblib` and archived to `.planning/phases/19-meta-learner/archive/meta_v1_922621bd.{joblib,_meta.json,.sha256}` for empirical record (AUDIT-03). xgb_v2 retained as canonical (SHA `6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099` byte-identity preserved end-to-end via 4-checkpoint AUDIT-01 chain spanning Phase 17 start → Phase 19 end). META-02 escalation NOT contemplated (no baseline to beat); Plan 19-02b not triggered. APPLIES D-18(v2.1, GATE) — does NOT supersede it. Honest read: META-01 is overfitting to BFO-rich recent fights (60%+ of post-cutoff has NaN closing_prob_diff per Phase 15.1 coverage gap); doesn't generalize to the temporally-diverse random_15pct sample. v2.2+ retry candidates: rich Level-1 feature set (D-02(P19) deferred), BFO coverage backfill, corpus growth >10% triggering GATE-RECALIB-PERIODIC. | Phase 19 is the third v2.1 candidate to fail the gate (after Phase 17 setting it; Phase 18 NET-V2 ablation/time-decay both failing; Phase 19 META-01 failing on random_15pct). The pattern validates v2.1's empirical-first discipline: 3-of-3 candidates failed honest measurement; the production model (xgb_v2) is the local optimum at the current corpus + feature scale. Future improvements require corpus growth (GATE-RECALIB-PERIODIC backlog), richer Level-1 features (D-02(P19) deferred), targeted BFO backfill, or genuinely new architectures (v3.x). Production behavior unchanged for end users; predictor JSON gains observability fields per D-05(P19) but `win_probability` is unchanged from xgb_v2 base. | ✓ Validated — Phase 19 (`.planning/phases/19-meta-learner/19-META-LEARNER-REPORT.md` + `19-VERIFICATION.md` + `19-SUMMARY.md` + `archive/` bundle + candidate meta JSON) |
| **D-21(v2.2, GATE)** | Phase 24 GATE recalibration on the 90-col v2.2 column space (`FEATURE_COLUMNS_V22`) uses the **same formula hash `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a`** (D-18 binding — no post-measurement renegotiation). Thresholds re-derived mechanically via 10-seed × 3-slice noise-floor harness. Operator empirical floor `accuracy_min ≥ 0.70` applied at spike time via `max(formula_output, 0.70)` per slice. HALT-AND-DECIDE protocol if any slice formula_output < 0.70 (no auto-relaxation). NEW v2.2 contract fields: `feature_columns_hash` (SHA-256 of v2.2 column list), `bfo_backfill_committed_at` (Phase 21 commit ISO timestamp `2026-05-15T15:06:49-07:00`). v2.1 contract at `.planning/gate_contract.json` PRESERVED for audit lineage. Per-slice thresholds (materialized 2026-05-16): `most_recent_12mo` accuracy_min=0.6798 / brier_max=0.2132; `most_recent_24mo` accuracy_min=0.6587 / brier_max=0.2164; `random_15pct` accuracy_min=0.6648 / brier_max=0.2172. **Operator decision: ACCEPT EMPIRICAL TRUTH** (v2.2_gate_breaks_floor_accept_truth) — all 3 slices formula_output < 0.70; un-floored thresholds carried forward per D-18; operator floor MOVES TO v2.3+ as a tighter commitment after BFO/referee/venue ingestion substrate fills. | Locking the formula + floor + protocol BEFORE measurement prevents Pitfall #1 (operator post-hoc renegotiation). The v2.2 column space adds 18 cols (3 REF + 6 TRAVEL + 9 META) to xgb_v2's 72-col baseline; xgb_v2 is NOT retrained — the spike measures xgb_v2-shape variance on the wider matrix using locked best_params (AF-1) and locked feature set (AF-2). The looser thresholds reflect Q6 finding: events.referee_id + venue_id at 0% populated rendered REF/TRAVEL features inert (Bayesian fallback + NaN propagation). | ✓ Validated — Phase 24 (`gate_contract_v2.2.json` + `24-NOISE-FLOOR-REPORT.md` + `24-HALT-AND-DECIDE.md` + `24-SUMMARY.md` + `tests/regression/test_gate_contract_phase24.py`) |
| **D-24(v2.3, GATE)** | Phase 31 GATE re-derivation on the POPULATED REF/TRAVEL substrate (post-Phase 28 ingestion) uses the **same formula hash `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a`** (D-18 binding — no post-measurement renegotiation). Thresholds re-derived mechanically via 10-seed × 3-slice bootstrap-variance harness (Phase 30 deliverable `variance.py` 4-fn surface). Operator empirical floor `accuracy_min ≥ 0.70` PRE-COMMITTED per CONTEXT D-01 Path A — `accuracy_min = max(formula_output, 0.70)` per slice at spike time. HALT-AND-DECIDE protocol if any slice `formula_output < 0.70` (no auto-relaxation; Path D materializes at partner-readiness gate). NEW v2.3 contract fields: `ingest_completed_at` (Phase 28 INGEST_COVERAGE.json commit ISO timestamp `2026-05-20T01:59:08-07:00`); `feature_columns_hash` (SHA-256 of v2.3 column list — currently identical to v2.2 column space; substrate fill, not schema change); `bfo_backfill_committed_at` (carry-forward `2026-05-15T15:06:49-07:00` from v2.2). v2.1 contract at `.planning/gate_contract.json` PRESERVED for audit lineage; v2.2 contract at `.planning/gate_contract_v2.2.json` PRESERVED for audit lineage. Per-slice thresholds materialized 2026-05-22 (post-spike): Pending — materialized post-spike by Plan 31-02. Supersedes `D-21(v2.2, GATE)` for the v2.3-onwards GATE invariant; v2.2 binding remains historical record. AUDIT-01 chain extends 26-of-N → 28-of-N via Phase 31 MID + END SHA artifacts. | Locking the formula + floor + protocol BEFORE measurement prevents Pitfall #1 (operator post-hoc renegotiation). v2.3 IS the first public partner release per ROADMAP; Path A 0.70 floor reinforces the partner-facing pitch. The v2.3 substrate (post-Phase 28 referee + venue ingestion + Phase 29 widened eval slices) is the v2.2 "data wasn't there yet" fix — Phase 31 re-tests the empirical claim on the populated corpus. | ⏳ Pending — Phase 31 (`gate_contract_v2.3.json` + `31-NOISE-FLOOR-REPORT.md` + `31-HALT-AND-DECIDE.md` conditional + `31-SUMMARY.md` + `tests/regression/test_gate_contract_phase31.py`) |
| **D-22(v2.2, PARTNER)** | Date: 2026-05-17. What: Partner-facing contracts lock to xgb_v2's current output shape at Position 5a (BEFORE Phase 26 candidate promotion). MANDATORY forward-compat fuzz test against mocked META-V22-active response. URL-path major versioning + envelope minor versioning. 12-month deprecation window. `*-contract.json` is a SIBLING artifact (NOT a sub-object of `*-meta.json`) — D-09(P15) carry-forward. Pydantic v2 `PredictorOutputV1` is single source of truth; emits Draft 2020-12 JSON Schema via `.model_json_schema()`. OpenAPI 3.1.0 auto-emitted via `create_app().openapi()`. Hypothesis @given 100-sample property-based round-trip + xgb_v2-only mock + hand-constructed META-V22-active mock (meta_learner_version='v22.1', meta_prob=0.7234, base_prob=0.6512 ≠ win_probability) all validate against committed schema at lock time. openapi-spec-validator>=0.8.5 CI gate (positive + negative tests per Pitfall #7). | Pitfall #5 resolution: lock schema BEFORE first promotion so future META-active responses are forward-compatible by construction, not by discovery. Position 5a build-order resolves the schema-vs-data ordering ambiguity that Phase 26 would otherwise inherit. Phase 26 Wave-0 RED test will assert every promoted candidate's predictor output validates against `predictor.schema.v1.0.0.json` + `meta_v22_active_mock` fixture (per CONTEXT.md D-09). | References: CONTEXT.md D-01..D-12; RESEARCH.md §Architecture Patterns 1-4; ROADMAP.md Phase 25 success criteria 1-6; REQUIREMENTS.md PARTNER-V22-01..06. |
| **D-23(v2.2, META-V22 OUTCOME)** | Phase 26 Forward-Stepwise Candidate Promotion materialized **Path B**: META-V22 logistic+poly2 blender on 13-col rich Level-1 (BFO-backfilled corpus per Phase 21) CLEARS the v2.2 gate on all 3 slices (Brier 0.2131/0.2131/0.1867 vs gate 0.2132/0.2164/0.2172; acc ~0.70). REF stepwise gate-clears but ≥0.003 Brier hurdle FAILS (delta 0.0007-0.0011 < D-13(v2.0)); TRAVEL DEGENERATE (0% events.venue_id coverage per Phase 23 Q6); CAMP dropped end-to-end. `models/meta/meta_v2.joblib` promoted per D-07(P15) hard-gate-then-save (replaces v1 which failed Phase 19 random_15pct). xgb_v2.joblib SHA byte-identical (AUDIT-01 19-of-N). predictor.py LIVE-03 1-line discovery delta (`get_latest_meta_version` helper). META-02 NN escalation deferred to v2.3+. REF/TRAVEL composition deferred to v2.3+ after referee_id + venue_id ingestion fills (Q6 root cause). | The BFO backfill (Phase 21 — root-cause fix for META-01 random_15pct failure) + rich Level-1 features (Phase 23 — D-02(P19) deferred set) + looser v2.2 gate (operator accept_truth at Phase 24) closed the META-01 failure mode. META-V22 achieved Brier 0.1867 on random_15pct vs META-01's 0.2343 — a 0.0476 improvement (>15x the ≥0.003 hurdle). Pre-templated Path B materialized without operator renegotiation (D-18 binding). | ✓ Validated — Phase 26 (`META_V22_SPIKE.json` + `META_V22_COEFFICIENT_STABILITY.json` + `REF_STEPWISE.json` + `TRAVEL_STEPWISE.json` + `26-COMPOSITION-REPORT.md` + `26-SUMMARY.md` + `models/meta/meta_v2.joblib`) |
| **D-25(UFCStats browser scraper, SUPERSEDES "Option A — no bypass")** | Branch `feat/ufcstats-browser-scraper`. What: The operator has REVERSED the prior UFCStats posture ("Option A — honor the JS proof-of-work challenge, no active bypass; freeze the corpus"). Active challenge-solving is now authorized via a headless-browser fetcher `BrowserFetcher` (Playwright/Chromium) in `src/ufc_prediction/scraper/browser_fetch.py`, injected behind the existing `ScraperClient.get`/`.map` seam and selected with `ufc scrape all|latest --backend browser [--proxy URL]`. Anti-bot detection factored into shared `src/ufc_prediction/scraper/antibot.py` (`detect_antibot`, reused from `scripts/ingest_pre_ufc_records_v25.py`). Parsers, diff/upsert, and ingest orchestration are UNCHANGED — this is a surgical fetch-layer swap only. Politeness/ToS posture: single worker, serial `map`, ≥1.5s delay, browser context+cookies solved once & reused, exponential backoff, optional residential proxy (`UFC_SCRAPE_PROXY`). Honest-halt contract: persistent challenge after retries raises `AntiBotChallengeError` — NEVER fabricates/returns stub data. `playwright` added to deps (`uv run playwright install chromium` one-time). Supersedes the "Formally Retired / do not re-enable UFCStats" guidance in `KNOWN_ISSUES.md` and `.planning/REQUIREMENTS.md` for the fetch layer. | UFCStats is the only source of trainable round-level fight stats (`round_stats` → `computed_features`); the corpus has been frozen since ~2026-05-27. Reviving `ufc scrape latest` is the prerequisite for adding recent fights and retraining (RETRAIN-PLAN.md). Operator-accepted tradeoff: headless circumvention is brittle and may be re-blocked within hours/days, possibly requiring a residential proxy and ongoing maintenance; fallback remains a licensed/API source or odds-only. | Live-smoke result recorded in the feature branch report; unit-tested (mocked Playwright, `detect_antibot` fixtures, injected-fetcher ingest parity). |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-02 — v2.6 milestone opened (Full Tech-Debt Drain + Gate Methodology Reset). Scope derived from operator directive "catch up on ALL the tech debt": all 27 P2 items in TECHNICAL_HANDOFF.md (scope-eligible subset) across 5 buckets — Housekeeping (A: meta_v3 candidate rename, scrape_event_urls fix, fighters_names refresh), DX hardening (B: pre-commit framework, mypy strict), API hardening (C: ProblemDetails, TypeScript SDK codegen), Methodology debt (D: substrate-drift–robust gate redesign, GATE-RECALIB-PERIODIC), Feature-debt drain (E: REF redesign, TRAVEL composition close-out, bias audit, time-decayed PageRank, opening-to-closing odds drift). No new product features. xgb_v2 + meta_v2 byte-identical through v2.6 (AUDIT-01 chain continues from 47-of-N FINAL). Bucket E composition is gate-conditional on Bucket D redesign clearing. Phase numbering continues from 48 (v2.5 ended at 47).*
