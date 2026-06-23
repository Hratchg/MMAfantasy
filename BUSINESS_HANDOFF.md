# UFC Fight Prediction — Business Handoff

**Latest release:** v3.0.1 (2026-06-14). Ready to deliver to a new owner.
**Audience:** business stakeholders, prospective owners, partners evaluating the project.

---

## 1. What this is, in one sentence

A predictive analytics service that estimates the win probability of any UFC matchup, with the explanation behind each number — designed for fantasy MMA platforms and decision-support tools.

You give it two fighter names. It returns:

- A calibrated probability (e.g. "Adesanya 44% / Strickland 56%")
- A skill rating for each fighter
- The top features driving the prediction (so users can see *why*)

It runs as a Python service with two access methods: a command-line tool for analysts and a hosted HTTP API for downstream products.

---

## 2. Current state of the product

**Ready to ship.** All four version 3.0 milestones closed cleanly. The product has been worked through 13 release tags (v1.0 through v3.0.1) over roughly two months of structured development. This is the most thoroughly validated point in the project's history.

| Capability | Status |
|---|---|
| Predict any matchup from a 16,000-fight historical corpus | Working |
| Command-line interface for analysts | Working |
| HTTP API ready for downstream product integration | Working |
| Containerized deployment (Docker; runs on any container host) | Working |
| Versioned partner contracts (schemas v1.0, v1.1, v1.2, v1.3) with semver guarantees | Working |
| Authentication, rate limiting, observability hooks | Working |
| Portable database snapshot for self-hosted users | Working |
| Legal hygiene (license, disclaimer, source attribution) | Working |

The corpus is frozen at v3.0 close. Every fight up to early June 2026 is in there. The model files are byte-frozen and audit-protected — they can't drift accidentally.

The accuracy is about 75% on recent fights. The closing betting market itself hits ~70–75%. UFC has high inherent variance — a single punch flips an 8-second outcome — so 75% is roughly the ceiling on this sport. The product's edge isn't beating the bookmakers; it's the per-fighter explainability layered on top.

---

## 3. What ships with the project

A new owner receives:

- **The trained model itself** — the canonical production artifacts, audit-locked since release v2.1
- **The complete training corpus** — a 10 MB portable database snapshot covering 16,902 fights, 6,820 fighters, 25,632 betting-line records, 90,000 historical skill snapshots, and 12 supporting tables
- **Loadable in one command** — `ufc db seed` loads the snapshot into a fresh local database
- **Working CLI + API** — both production-ready
- **Tested install walkthrough** — 5-minute setup from clone to first prediction
- **Five external publishable PDFs** — technical handoff, methodology docs (internal + client-facing), data strategy, feature reference
- **Complete release history** — every release tag, retrospective, and architecture decision documented

---

## 4. Honest limitations — things to know before signing on

### Data refresh is the central operational question

The corpus is frozen as of June 2026. Predictions stay accurate for established fighters for roughly 3–6 months; new fighters drift faster because the system falls back to a default rating until it has seen their UFC fights. Beyond ~6 months without fresh data flowing in, predictions are materially stale.

The current data pipeline includes scrapers for every major public source — and **all of them were used to build the corpus you're receiving.** The historical data, the per-round statistics, the fighter career profiles, the betting lines — all of it was ingested by this project's own pipeline. The closing-odds source (BestFightOdds) is still refreshing weekly via an automated workflow.

A new owner's first strategic question is how to keep data flowing. The options are familiar to anyone who's run a sports data product:

- Continue running the existing scrapers (technical investment, ongoing maintenance)
- License a commercial data feed from a major sports data provider
- Negotiate a direct arrangement with one of the underlying sites
- Treat the existing corpus as a backtest / replay dataset and don't refresh

Each comes with different cost, reputational, and engineering trade-offs. None are blocked off; it's a decision about how much investment and what kind of relationships the new owner wants.

### Accuracy ceiling is real

UFC is high-variance. Any predictor claiming >70% should be examined skeptically; the closing betting line itself sits in that range. This product's market position is *explainability + a calibrated number*, not *beating the bookmakers*. Anyone presenting it should frame it that way.

---

## Reference

The full technical state is in [`TECHNICAL_HANDOFF.md`](TECHNICAL_HANDOFF.md). Per-release history is in [`CHANGELOG.md`](CHANGELOG.md). The historical bug log and per-source scraper status are in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md). The repo lives at the GitHub URL the previous owner shared.

*This document was generated 2026-06-14 reflecting v3.0.1.*
