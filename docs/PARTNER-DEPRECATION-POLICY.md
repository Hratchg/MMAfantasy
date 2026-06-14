# Partner Schema Deprecation Policy

**Version:** 1.0
**Last updated:** 2026-05-27
**Audience:** Integration partners consuming the `/api/v1/predict` API.

This document formalises the deprecation and sunset commitments referenced by
the project invariant **D-22** (`/.planning/PROJECT.md`). It is the canonical
source for the migration timelines that partners can rely on when planning
upgrades between Partner schema versions.

---

## Scope

This policy applies to the **Partner schema versions** emitted by the
`/api/v1/predict` endpoint and any downstream artefacts versioned alongside
them — namely:

- The Pydantic envelope `PredictorOutputV{X.Y.Z}` and its embedded
  `schema_version` field.
- The standalone JSON Schema files
  (`src/ufc_prediction/contracts/predictor.schema.v{X.Y.Z}.json`).
- The OpenAPI 3.1 spec files
  (`src/ufc_prediction/contracts/openapi.v{X.Y.Z}.json`).

It does **not** apply to internal model artefacts, ML feature column hashes,
or any other surface not consumed by integration partners. Internal-only
changes that keep the Partner schema byte-stable are out of scope.

A canonical commitment to the 12-month window also lives at **D-22** in
`/.planning/PROJECT.md`; the present document is the partner-readable
reformulation of that invariant.

---

## Deprecation window

Once a new Partner schema version `v(N+1)` is released:

- Partners on `v(N)` have **at least 12 months** to migrate before responses
  on `v(N)` begin emitting deprecation warnings.
- Partners on `v(N)` have **at least 18 months** from the release of `v(N+1)`
  before `v(N)` may be sunset (i.e. responses may begin returning
  `410 Gone`, or field defaults may change in ways that previously-conforming
  clients would observe).

In practice, this means a partner targeting `v(N)` always has a contiguous
12-month window of fully-supported, no-warning behaviour after the release
of the successor schema, and a further 6 months of operational warning before
the schema may be retired.

These windows are minimums. The project may extend them on a case-by-case
basis but will not shorten them without partner-by-partner negotiation.

The 12-month commitment is also recorded in the project invariants at D-22
(`/.planning/PROJECT.md`) and remains the load-bearing contract behind this
document.

---

## Version matrix

| Schema Version | Released   | Status | Deprecation Date | Sunset Date |
|----------------|------------|--------|------------------|-------------|
| v1.0.0         | 2026-05-17 | Active | —                | —           |
| v1.1.0         | 2026-05-22 | Active | —                | —           |
| v1.2.0         | 2026-05-27 | Active | —                | —           |

**Notes on the version matrix:**

- v1.0.0 was first committed to the repository on 2026-05-17 as part of the
  Phase 25 forward-compat lock (the same date the schema bytes were frozen).
  Partners with releases predating this commit should treat 2026-05-17 as the
  canonical reference date for migration timelines.
- v1.1.0 ships with the v2.3 milestone release (2026-05-22) and is an
  additive minor; clients written against v1.0.0 continue to parse v1.1.0
  responses by ignoring unknown fields.
- v1.2.0 ships with Phase 35 (2026-05-27) and adds the additive
  `prediction_metadata` block. Same forward-compatibility guarantee.

All three versions are currently **Active** — no deprecation or sunset clocks
are running as of the date at the top of this document. When a future schema
version (v1.3.0, v2.0.0, etc.) is released, this matrix will be updated in
the same change that publishes the successor, and the deprecation /
sunset dates for the immediately-previous minor will be populated.

---

## Notice channels

Deprecation and sunset signalling is communicated through the following
channels, in roughly increasing order of urgency:

- **Repository `CHANGELOG.md`** — every schema release is recorded with its
  release date and any deprecation notes for prior versions.
- **GitHub Releases page** — every schema version has an accompanying
  tagged release at `releases/v{X.Y.Z}` with human-readable migration notes.
- **Repository pinning of the schema files** — the JSON Schema and OpenAPI
  files committed under `src/ufc_prediction/contracts/` are the immutable
  source-of-truth. Partners are encouraged to mirror or pin the specific
  hash of the version they consume.
- **Inline `Deprecation:` HTTP response header** — *forward intent*. During
  the 6 months prior to sunset, deprecated-schema responses are intended to
  carry the standard `Deprecation:` and `Sunset:` HTTP response headers (per
  RFC 8594 / draft-ietf-httpapi-deprecation-header). **As of this document
  version, automatic emission of those headers is not yet implemented.** The
  policy is set; the header automation is on the roadmap (currently targeted
  at the v2.5 milestone). Until then, partners should rely on the
  `CHANGELOG.md` and GitHub Releases channels for advance notice.

In all cases the project will not silently sunset a Partner schema: at least
one of the channels above will carry the deprecation announcement at least
12 months before any change to wire behaviour.

---

## Breaking change policy

The project follows **semantic-versioning-aligned** rules for the Partner
schema:

- **Additive changes within a major version** are non-breaking and ship as
  minor releases (`v1.0.0` → `v1.1.0` → `v1.2.0`). New fields are tolerated
  by clients that ignore unknown JSON fields, and existing field names,
  types, and semantics never change inside a major. This commitment is the
  Phase 25 forward-compat lock and is mechanically enforced by the
  fuzz-test / property-test suite that runs against every promoted model
  candidate.
- **Breaking changes bump the major version** (`v1.x.y` → `v2.0.0`).
  Examples include: renaming or removing a required field, narrowing a
  value range previously honoured, changing the semantic meaning of a
  numeric field, or changing the discriminator strategy. Any such change
  starts a fresh deprecation window for the prior major.
- **Partners on the prior major** keep the full 12 / 18-month migration
  window described above, and the project commits to running the prior
  major in parallel during that window.

If a security or correctness fix forces a behaviour change that cannot wait
for a normal deprecation window, the project will treat it as a separate
"emergency" event and communicate the change through all notice channels
simultaneously, while documenting the exception in `CHANGELOG.md`.

---

## Questions or migration support

For migration questions, deprecation timing clarifications, or to be added
to a partner-notification list, please contact
`partner-support@example.com` (placeholder — replace with the operator's
real channel before public publication of this document). General feedback
on this policy is welcome through the project's issue tracker.
