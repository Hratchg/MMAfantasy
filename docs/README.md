# `docs/` — Project Documentation

This directory holds project documentation. The files below are grouped by who they're for and what they cover.

---

## For new users / installers

**Start here if you want to run the project locally.**

| File | What it covers |
|---|---|
| [`INSTALL.md`](INSTALL.md) | One-page tested walkthrough: `git clone` → install Python deps → start Postgres → load corpus → predict a matchup → (optional) serve the API |

You may also want [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) — per-scraper status + the v3.0 regressions + workarounds.

---

## For partners integrating against the API

**Start here if you're consuming the predictor over HTTP.**

| File | What it covers |
|---|---|
| [`QUICKSTART-PARTNER.md`](QUICKSTART-PARTNER.md) | Minimal "make a successful API call" walkthrough — auth, request shape, response shape, common errors |
| [`PARTNER-CONTRACT.md`](PARTNER-CONTRACT.md) | What the API promises: schemas, semver guarantees, additive-only rule, breaking-change policy |
| [`PARTNER-DEPRECATION-POLICY.md`](PARTNER-DEPRECATION-POLICY.md) | Notice periods, schema retirement, support windows |
| [`../src/ufc_prediction/contracts/`](../src/ufc_prediction/contracts/) | Canonical JSON schemas + OpenAPI specs (source of truth) |

---

## For new contributors / code owners

**Start here if you're going to modify or extend the codebase.**

| File | What it covers |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System architecture: how the three model layers fit together, data flow, key abstractions |
| [`GLOSSARY.md`](GLOSSARY.md) | Defines every internal term (META-V22, AUDIT-01, "widened eval slices", "confound_block", etc.) |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | How to contribute: pre-commit hooks, protected files, commit message convention, PR checklist |
| [`../TECHNICAL_HANDOFF.md`](../TECHNICAL_HANDOFF.md) | Practical handoff doc — what works, what's broken, what to do next |

---

## PDF builds

| File | What it covers |
|---|---|
| [`pdfs/`](pdfs/) | Generated PDF builds of selected markdown documents (see [`pdfs/README.md`](pdfs/README.md)) |

---

## Where things live outside `docs/`

Not all documentation is here — some sits at the repo root:

| Location | What's there |
|---|---|
| [`../README.md`](../README.md) | First-impression overview + quickstart + index |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Per-tag release notes |
| [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) | What's broken right now + workarounds |
| [`../TECHNICAL_HANDOFF.md`](../TECHNICAL_HANDOFF.md) | Practical handoff to a new technical owner |
| [`../BUSINESS_HANDOFF.md`](../BUSINESS_HANDOFF.md) | Handoff doc for business-side readers |
| [`../METHODOLOGY_CLIENT.md`](../METHODOLOGY_CLIENT.md) | ML methodology distilled for partners |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Contribution guide + protected-file policy |
