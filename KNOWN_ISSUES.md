# Known Issues — UFC Fight Prediction v3.0

This document is the canonical operator-facing reference for what works, what's
blocked, when predictions start drifting, and what's queued for the v3.1 hotfix
milestone. It reflects state at v3.0 milestone close. Last updated 2026-06-10.

If you are setting up the project for the first time, read `docs/INSTALL.md`
first; the install walkthrough forward-links here from steps 5 and 6 and from
its "Next steps" section.

## Scraper status — summary

| Source | Status | Last verified | Downstream impact (one line) |
|---|---|---|---|
| UFCStats | Blocked | v2.5 Phase 40 close (Apr 2026) | No fresh fight-stats refresh; predictions drift on active roster within 3–6 months. |
| BFO / BestFightOdds | Works | v2.5 close (100% post-2021 coverage) | Odds backfill operational for post-2021 fights; pre-2021 gap unresolved. |
| Sherdog | Blocked (Content-Signal `ai-train=no`) | v2.7 Phase 81 (Jun 2026) | No alt-source backfill for events missing from UFCStats; net-new ingest is downstream user's concern. |
| Tapology | Blocked (Content-Signal `ai-train=no`) | v2.7 Phase 82 (Jun 2026) | Same as Sherdog — alt-source backfill unavailable for handoff users. |
| oddsportal | Blocked (`robots.txt` Disallow) | v2.7 Phase 79 (Jun 2026) | BFO opening-line pre-2021 backfill unreachable; ≤80% coverage gate not closeable from this source. |
| Sherdog debutant Elo seed | One-time backfill | v2.5 Phase 43 (Apr 2026) | Active; new debutants without a Sherdog career profile use the flat 1500 default. |

Status taxonomy: **Works** | **Blocked** | **Partial** | **One-time backfill**.

## Per-scraper detail

### UFCStats — `src/ufc_prediction/scraper/client.py`, `parse_event_list.py`, `parse_event_detail.py`, `parse_fight_detail.py`, `parse_fighter.py`

**What it does:** Primary source of UFC fight, fighter, round, and event records. Drives the canonical corpus that META-V22 + xgb_v2 are trained against.

**Status:** Blocked since v2.5 Phase 40 close (Apr 2026).

**Root cause:** Cloudflare-style anti-bot challenge on the public site. Direct `requests.get(...)` returns 403; headless-browser circumvention is brittle and gets re-blocked within hours. The behavior was first flagged at v2.4 close, confirmed at v2.5 Phase 40, and left as a Bucket I deferral through v2.6, v2.6.1, and v2.7 before being formally retired at v3.0 open (`.planning/REQUIREMENTS.md` "Formally Retired" table).

**Suggested workaround for downstream users:** None reliable through this scraper. Use the shipped corpus dump (`data/seed/ufc_corpus_v30.dump`, loadable via `uv run ufc db seed`) as your baseline, then investigate licensed alternatives or accept stale data going forward.

**Milestone history:** v2.4 close → flagged → v2.5 Phase 40 confirmed blocked → carried as Bucket I deferral through v2.6 → v2.6.1 → v2.7 → formally retired at v3.0 open.

> **Downstream TL;DR:** Don't try to re-enable UFCStats. Pin yourself to the shipped corpus and refresh when the underlying access situation changes.

### BFO / BestFightOdds — `src/ufc_prediction/scraper/bfo_scraper.py`, `bfo_ingest.py`, `bfo_classify.py`, `bfo_matcher.py`, `bfo_math.py`, `bfo_models.py`, `bfo_live.py`

**What it does:** Fight-odds scraper for both pre-fight opening lines and live close-out lines. Feeds `fight_odds` table; informs the META-V22 stacker's market-aware features.

**Status:** Works. 100% coverage on post-2021 fights as of v2.5 close. Overall corpus coverage at 72.52% (18,588 / 25,632 rows; pre-2021 gap is the residual).

**Root cause of the residual gap:** BFO opening-line data thins out for fights before ~2021; the secondary-source backfill attempt against oddsportal (v2.7 Phase 79) was structurally blocked by `robots.txt` Disallow.

**Suggested workaround for downstream users:** Use BFO as-is for ongoing ingest. For the pre-2021 backfill gap, see FEAT-V30-04 in `.planning/REQUIREMENTS.md` (Bucket I row) — 5 prioritized alternative sources (ESPN BET historical, DraftKings archive, Action Network, direct sportsbook API partnerships, oddsportal re-evaluation if the policy changes) are pre-enumerated for a future v3.x milestone.

**Milestone history:** Productionized v2.4–v2.5. Coverage lock at Phase 67 (72.52%). v2.7 Phase 79 confirmed the residual gap is not closeable from oddsportal.

> **Downstream TL;DR:** BFO works. Run it on the same cadence as your fight-record ingest. Don't chase the pre-2021 gap with oddsportal.

### Sherdog — `src/ufc_prediction/scraper/sherdog.py`, `sherdog_models.py`

**What it does:** Originally intended as an alt-source for fight events missing from UFCStats and as the source of historical career data used to seed debutant Elo ratings.

**Status:** Blocked since v2.7 Phase 81 (Jun 2026). Content-Signal header advertises `ai-train=no`.

**Root cause:** Operator policy decision at v2.7 Phase 81 (Bucket E retired). The Sherdog response carries a `Content-Signal` header with `ai-train=no`. The project honors the directive and does not ingest alt-event data from Sherdog.

**Note on the debutant Elo seed sub-feature:** The one-time backfill shipped at v2.5 Phase 43 (using Sherdog career data captured before the policy declaration) is still active in the canonical corpus and is documented separately below. The blocker applies to *new* alt-event ingestion, not to the existing seed.

**Suggested workaround for downstream users:** None. Net-new corpus growth from Sherdog is a downstream user's concern; re-eligibility requires either a policy change on Sherdog's side or a licensed-access arrangement.

**Milestone history:** v2.5 Phase 43 → debutant Elo backfill shipped (pre-policy). v2.7 Phase 81 → Content-Signal `ai-train=no` confirmed; Bucket E retired. v3.0 open → formally retired as CORPUS-V30-01.

> **Downstream TL;DR:** Sherdog alt-events are not available. The pre-existing debutant seed in the shipped corpus is unaffected.

### Tapology — _(no module shipped; v2.7 Phase 82 evaluation only)_

**What it does:** Considered at v2.7 Phase 82 as a second alt-source for events missing from UFCStats.

**Status:** Blocked since v2.7 Phase 82 (Jun 2026). Same Content-Signal `ai-train=no` policy as Sherdog.

**Root cause:** Operator policy decision at v2.7 Phase 82 (Bucket E retired). No production scraper module was ever shipped; the evaluation closed before any ingest code landed.

**Suggested workaround for downstream users:** None. Same disposition as Sherdog.

**Milestone history:** v2.7 Phase 82 → evaluated → Content-Signal `ai-train=no` confirmed → Bucket E retired. v3.0 open → formally retired as CORPUS-V30-02.

> **Downstream TL;DR:** Tapology is not a viable alt-source. Don't write a scraper for it.

### oddsportal — _(no module shipped; v2.7 Phase 79 probe only)_

**What it does:** Considered at v2.7 Phase 79 as the secondary source to backfill the BFO pre-2021 opening-line gap.

**Status:** Blocked since v2.7 Phase 79 (Jun 2026). `robots.txt` Disallow covers `*-1998*` through `*-2024*` for ALL user-agents — the entire 2007–2020 backfill window.

**Root cause:** `https://www.oddsportal.com/robots.txt` explicitly disallows the year-suffixed URL patterns that would have been needed for backfill. The project honors `robots.txt`. The probe-evidence file (`results/bfo_oddsportal_probe_v27.md`) preserves verbatim Disallow strings as the audit trail.

**Suggested workaround for downstream users:** None through oddsportal. See FEAT-V30-04 in `.planning/REQUIREMENTS.md` for 5 prioritized v3.x alt-source candidates. Re-eligibility trigger is locked at ≥80% overall corpus coverage.

**Milestone history:** v2.7 Phase 79 (operator-selected at /gsd-discuss-phase 2026-06-07) → probe → Disallow confirmed → Path B documentary close. No scraper module shipped (no callable execution path). v3.0 open → formally retired as FEAT-V30-04.

> **Downstream TL;DR:** oddsportal is structurally blocked. Re-evaluate only if their `robots.txt` policy changes.

### Sherdog debutant Elo seed — one-time backfill at v2.5 Phase 43

**What it does:** When a new fighter makes their UFC debut, their Elo rating needs an initial value. Without a seed, debutants start at the flat 1500 default, which over-weights debut surprises in the META-V22 stacker. The v2.5 Phase 43 backfill used Sherdog career-profile data (captured before the Content-Signal policy declaration) to compute a one-time per-fighter seed based on pre-UFC career signal.

**Status:** One-time backfill. Active in the canonical corpus shipped via `data/seed/ufc_corpus_v30.dump`.

**Root cause for the "one-time" framing:** The Sherdog data ingest path is now blocked (see above). Net-new debutants since the Content-Signal policy declaration do NOT receive the Sherdog-derived seed; they fall back to the flat 1500 default. This is an accepted limitation, not a regression.

**Suggested workaround for downstream users:** None required for the existing seeds — they ship in the corpus dump and are immutable. For net-new debutants, accept the flat 1500 default until an alt-source is identified or operator policy on debutant seeding changes.

**Milestone history:** v2.5 Phase 43 → one-time Sherdog backfill shipped → seeds frozen into `fighters` / `elo_snapshots` tables → carried verbatim through every subsequent milestone.

> **Downstream TL;DR:** The shipped corpus has the seeds you need. New debutants from this point forward start at 1500.

## Prediction degradation timeline

The model itself (META-V22 + xgb_v2) is canonical and byte-frozen — `xgb_v2.joblib`
SHA `6e7641…0099` and `meta_v2.joblib` SHA `77076d3b…9196` have been unchanged
end-to-end through 5 milestones of evidence. The accuracy you get from a fresh
clone depends entirely on how recently the underlying corpus was refreshed.

**0–3 months from a stale-corpus install.** Predictions remain stable for the
established roster (fighters with ≥5 UFC fights), because their Elo + striking
+ grappling signals are well-anchored by the historical data. Debutants drift
immediately: a fighter who debuts the day after your install has no Elo
trajectory in your DB at all, and the predictor falls back to flat-default
behavior. This is the regime most downstream operators will run in.

**3–6 months from a stale-corpus install.** Established-roster predictions
begin drifting noticeably as new fights happen that the model hasn't seen —
title changes, finish trends, and rookie breakthroughs that adjust the per-
division Elo distribution. Numbers will still be in the right ballpark but
should be treated as informed reference points rather than authoritative.

**6+ months from a stale-corpus install.** Predictions are materially stale.
At this point, either refresh the underlying data (via your own ingest
pipeline; UFCStats is blocked, so this is a real undertaking) or accept
significantly reduced accuracy. The handoff package was not designed to be
indefinitely useful without data refresh — that is explicitly the downstream
operator's responsibility per PROJECT.md scope.

The 3-month and 6-month boundaries are operator-stated PROJECT.md guidance
calibrated against cumulative milestone evidence (Phases 40, 67, 79, 81, 82).
Your mileage will vary with how active the divisions you care about are.

## v3.0 Known Regressions (RESOLVED in v3.0.1)

Both regressions found at v3.0 close were fixed in the v3.0.1 patch (2026-06-14).
They are retained here as audit trail.

### REG-V30-01: Predictor default-version regression — **FIXED in v3.0.1**

The shipped `ModelPredictor(model_dir="models")` constructor resolved "latest"
to `xgb_v3.joblib` — a candidate that was trained, evaluated, and *never
promoted* during v2.5–v2.7. The canonical pairing required by the shipped
`meta_v2_candidate.joblib` (META-V22) is `xgb_v2.joblib` (SHA `6e7641…0099`).
The version mismatch caused both `ufc predict matchup` (without `--version`)
and `POST /api/v1/predict` to fail.

**Fix:** The default model version is now explicitly pinned to `"v2"` at the
two entry points that previously relied on `get_latest_version()`:

- `src/ufc_prediction/api/v1/predict.py` — `_get_predictor()` now passes
  `version="v2"` to `ModelPredictor`
- `src/ufc_prediction/cli/predict.py` — `predict_matchup`'s `--version`
  option defaults to `"v2"` instead of `None`

Both `ufc predict matchup "A" vs "B"` and `POST /api/v1/predict` now succeed
out of the box. The fix does not touch any AUDIT-01-protected files.

**Surfaced in:** Phase 87 Plan 87-01 live-CLI capture (2026-06-08).
**Fixed in:** v3.0.1 patch (2026-06-14).

### REG-V30-02: Integration test port-binding quirk — **FIXED in v3.0.1**

`tests/integration/test_db_seed.py` allocated a disposable `postgres:16-alpine`
container on host port 55555 to exercise the `ufc db seed` round-trip. On some
macOS Docker Desktop installs (Docker 29.4.1 confirmed on the originating host),
host port 55555 failed to bind the userland proxy.

**Fix:** The test now requests an ephemeral port from the OS via
`socket.bind(("127.0.0.1", 0))` and uses whatever the OS allocates. This is
reliable across Docker Desktop, colima, and Linux.

**Surfaced in:** Phase 88 Plan 88-03 integration test execution (2026-06-10).
**Fixed in:** v3.0.1 patch (2026-06-14).

## v3.1 Remediation Queue

| ID | Description | Severity | Status |
|---|---|---|---|
| OPS-V30-01 | Pre-commit hook missing on some clones | Low (operator-side) | **Resolved** (v3.1, unreleased) |

OPS-V30-01 was surfaced at Phase 88 Plan 88-03 close: `.git/hooks/pre-commit`
was not present on the originating clone. The byte-identity diff is the binding
AUDIT-01 contract, so this is belt-and-suspenders rather than a behavioral
regression — the CI `pre-commit` job re-runs every hook on each PR regardless of
local hook state.

**Resolution (v3.1):** git cannot auto-install hooks on clone (it never runs
arbitrary code on checkout), so the fix removes the *friction* that made the
step easy to skip. `pre-commit` now ships in the dev dependency group
(`pyproject.toml`), so a plain `uv sync` puts it in the project venv and a
fresh clone's setup is exactly two documented steps:

```bash
uv sync --frozen
uv run pre-commit install
```

Both are now the first entries in the CONTRIBUTING.md "First-time setup" block
(previously only `uv sync` was listed, which is why the hook was missed). No
separate `uv tool install pre-commit` is required anymore.

## Model performance clarification — META-V22 provides no lift over base xgb_v2 (2026-07-01)

**TL;DR:** The headline "~75–78% accuracy / ~0.15 Brier" for META-V22 was a
**deduplicated-Phase-26-substrate artifact**. On the current corpus the stacker
performs **≈ the base `xgb_v2` model (~70% accuracy / ~0.20 Brier)** — it adds no
real lift. Treat the base model as the honest performance ceiling. Do **not**
attempt to "enable" META-V22 at inference; doing so regresses predictions.

**Two facts operators should know:**

1. **META-V22 has been dormant at inference the whole time.** Its 13-feature
   Level-1 vector requires `days_since_last_fight_diff`, which is not populated in
   the live inference path, so `predict()` sets `meta_skipped_reason=
   "nan_meta_input"` and returns the **base `xgb_v2`** probability. Every
   prediction the CLI/API has ever served is the base model's — which is fine,
   because (see #2) the base is as good as the meta anyway.

2. **The stacker does not beat the base — verified three ways (2026-07-01):**
   - Promoted `meta_v2`'s own metrics block reports **Brier 0.213 / acc 0.70**
     (12mo) — the "~0.15/78%" numbers appear only in the `meta_v2_candidate` /
     `meta_v2_dedup` sidecars, measured on the deduplicated Phase-26 substrate.
   - Frozen `meta_v2` run on the current substrate → **Brier 0.42, 72.8% of
     outputs saturated at 0/1**: the documented baseline-scaler-OOD substrate-drift
     confound (its `StandardScaler` was fit on the Phase-26 substrate).
   - A clean **refit** of the same architecture on the current substrate →
     **Brier 0.207 / acc 0.71**, statistically indistinguishable from base
     **(0.193 / 0.70)**. So even the ideal fix yields no lift.

**Why (mechanistic):** 10 of META-V22's 13 Level-1 inputs are already base
`xgb_v2` features; the base (gradient-boosted trees) already models their
interactions and is already probability-calibrated. The only genuinely-new
input is `division_finish_rate_shrunk` (a single weak division-level rate);
`elo_prob` largely duplicates the base's Elo diffs. There is almost no
incremental signal for the stacker to exploit.

**Consequences / guidance:**
- **Canonical predictor = base `xgb_v2` (~70% / ~0.20 Brier).** Where other docs
  (`BUSINESS_HANDOFF.md`, `TECHNICAL_HANDOFF.md`, `METHODOLOGY_CLIENT.md`) cite
  ~75%, read it as **~70%** on the current corpus.
- The prior "re-measure META-V22 on v2.3 widened slices" item
  (`docs/QUICKSTART-PARTNER.md`) is **closed as no-lift** — not worth pursuing.
- Frozen artifacts are unchanged and were never touched by this investigation
  (`xgb_v2.joblib` / `meta_v2.joblib` SHAs match the AUDIT-01 baseline).
- Reproduce: the head-to-head + refit are derivable from
  `scripts/remeasure_meta_v22_v23.py` plus a same-architecture refit on the
  post-cutoff (out-of-fold) rows; no frozen-model or model-weight change is
  required or advisable.

## How to Report New Issues

Open a GitHub issue at the project repository. One-line title format helps
triage — for example:

> `REG: ufc predict matchup --version v2 segfaults on Python 3.14.2`

Include reproduction steps (`uv run ...` invocation + observed output) and
your environment baseline (output of `uv run python --version` + OS +
`docker --version` if relevant). Reference `docs/INSTALL.md` for the canonical
environment baseline the project was tested against. If you can attach the
contents of `data/seed/PROVENANCE.md` for context, that helps confirm you are
running against the shipped corpus.

---

*Last updated 2026-06-10; reflects state at v3.0 milestone close. Source-of-truth*
*for scraper status, prediction-degradation timeline, and v3.0 known regressions.*
