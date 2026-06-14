# `@ufc-fight-prediction/sdk-ts`

TypeScript SDK for the UFC Fight Prediction PARTNER v1.x API. Auto-generated from the OpenAPI 3.1 schema committed at [`src/ufc_prediction/contracts/openapi.v1.3.0.json`](../../src/ufc_prediction/contracts/openapi.v1.3.0.json).

Phase 53 (API-V26-03) shipped the scaffold + codegen pipeline. Phase 70 (API-V261-03/04) bumped the codegen source to v1.3.0 and added the CI drift-detection workflow at `.github/workflows/sdk-codegen.yml`.

## Status

| Schema version | Status |
|---|---|
| v1.0.0 | Byte-frozen contract (forward-compat lock); type-universe inherits from v1.3.0 codegen |
| v1.1.0 | Byte-frozen contract (forward-compat lock); type-universe inherits from v1.3.0 codegen |
| v1.2.0 | Byte-frozen contract (forward-compat lock); type-universe inherits from v1.3.0 codegen |
| v1.3.0 | **Current SDK source** (Phase 70 — codegen against `openapi.v1.3.0.json`) |

The v1.x success-path bytes are byte-identical across all four versions (v2.6.1 invariant #3, locked by `test_forward_compat_v1_3_0.py`). The v1.3.0 OpenAPI artifact captures the broadest live surface: ProblemDetails opt-in error wrapper, full `accept_schema_version` enum, and all FastAPI routes (`/health`, `/ready`, `/api/v1/predict`, `/api/v1/fighters/{name}`, `/api/v1/rankings`, `/api/v1/matchup/*`, `/api/v1/history/*`).

## Install

This SDK is currently **private** — `package.json` carries `"private": true` and the package is not published to the npm registry. Partner integrations consume the source artifact directly:

```bash
# Option 1: clone the repo and link locally
git clone https://github.com/anthropic/ufc-fight-prediction.git
cd ufc-fight-prediction/clients/typescript
npm install

# Option 2: download the committed predict.ts directly and vendor it
# https://github.com/anthropic/ufc-fight-prediction/raw/master/clients/typescript/src/v1/predict.ts
```

When the npm package goes public (v2.7+ decision), this section will be replaced with:

```bash
npm install @ufc-fight-prediction/sdk-ts
```

## Use

The generated `src/v1/predict.ts` exports `paths` / `components` / `operations` TypeScript namespaces — no runtime client. Wrap with your fetch client of choice:

```typescript
import type { paths, components } from "@ufc-fight-prediction/sdk-ts";

type PredictRequest = paths["/api/v1/predict"]["post"]["requestBody"]["content"]["application/json"];
type PredictResponse = paths["/api/v1/predict"]["post"]["responses"]["200"]["content"]["application/json"];

// Or pull the schema directly:
type PredictorOutputV1 = components["schemas"]["PredictorOutputV1"];
type ProblemDetailsV13 = components["schemas"]["ProblemDetailsV13"];

async function predict(req: PredictRequest, apiKey: string): Promise<PredictResponse> {
  const r = await fetch("https://api.ufc-fight-prediction.example/api/v1/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": apiKey,
      // Phase 52 (API-V26-01): opt in to RFC 7807 application/problem+json
      // error responses; omit or use Accept: application/json for the
      // legacy { detail: ... } shape.
      "Accept": "application/problem+json",
    },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    // When Accept: application/problem+json is set, error bodies follow
    // RFC 7807 ({ type, title, status, detail, instance }).
    const problem = (await r.json()) as ProblemDetailsV13;
    throw new Error(`Predict failed: ${problem.title} (${problem.status})`);
  }
  return r.json() as Promise<PredictResponse>;
}
```

## Regenerate

```bash
cd clients/typescript
npm ci          # or `npm install` if updating deps
npm run codegen
```

The script reads from `src/ufc_prediction/contracts/openapi.v1.3.0.json` and writes `src/v1/predict.ts`.

`npm run codegen:check` regenerates + asserts no drift via `git diff --exit-code` — used by the CI workflow below.

## CI gate

`.github/workflows/sdk-codegen.yml` (Phase 70 — API-V261-04):

1. Triggers: PRs and pushes to `master` that touch `clients/typescript/**` or `src/ufc_prediction/contracts/openapi*.json`; also `workflow_dispatch`.
2. Sets up Node 20.
3. `cd clients/typescript && npm ci`.
4. Runs `npm run codegen:check` → fails the PR if `git diff --exit-code src/v1/predict.ts` is non-empty (drift detected).

This catches drift between the committed Pydantic source-of-truth and the committed TypeScript types on every PR.

## Files

| Path | Purpose |
|---|---|
| `package.json` | npm metadata + script entry points |
| `package-lock.json` | npm lockfile for reproducible `npm ci` |
| `tsconfig.json` | strict TypeScript compile config |
| `codegen.sh` | codegen driver (invokes `openapi-typescript`) |
| `src/v1/predict.ts` | generated types (against v1.3.0) |
| `README.md` | this file |

## Lineage

- **Phase 25** (CONTRACT-V23-01) — PARTNER v1.0.0 schema lock; first OpenAPI artifact emit
- **Phase 32** (CONTRACT-V23-02) — PARTNER v1.1.0 additive bump
- **Phase 35** (CONTRACT-V24-02) — PARTNER v1.2.0 additive `prediction_metadata` block
- **Phase 52** (API-V26-01/02) — PARTNER v1.3.0 ProblemDetails opt-in error wrapper
- **Phase 53** (API-V26-03) — SDK scaffold (placeholder `predict.ts`, codegen pipeline)
- **Phase 69** (API-V261-01/02) — emit `openapi.v1.3.0.json` sibling; forward-compat v1.3.0 tests
- **Phase 70** (API-V261-03/04) — replace placeholder with codegen output against v1.3.0; ship CI drift workflow

See [`docs/PARTNER-CONTRACT.md`](../../docs/PARTNER-CONTRACT.md) for the canonical contract policy.
