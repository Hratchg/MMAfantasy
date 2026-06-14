# UFC Fight Prediction — Partner Contract (v1.0.0)

This document describes the partner-facing HTTP contract for the
UFC Fight Prediction API. It covers versioning, deprecation policy,
client codegen, authentication, and the sibling artifact discipline.

**Schema version:** 1.0.0
**API surface:** `POST /api/v1/predict`
**Schema source-of-truth:** `src/ufc_prediction/contracts/predictor.schema.v1.0.0.json`
**OpenAPI 3.1 spec:** `src/ufc_prediction/contracts/openapi.v1.0.0.json`
**Sibling contract artifact:** `models/xgb_v2-contract.json`

## Versioning Policy

The API uses **URL-path major versioning + envelope minor versioning**:

- **Major version** (breaking changes): URL changes (`/api/v1` → `/api/v2`)
- **Minor version** (additive changes): URL stable; response `schema_version`
  field advances (`1.0.0` → `1.1.0`)

Examples:
- Adding an Optional field to the response → minor bump (clients
  using `schema_version` discrimination continue to parse)
- Renaming a required field, changing a field's type, removing a field
  → major bump (URL path changes; old version stays live during deprecation)

## Deprecation Window

The previous major version stays live for **12 months** after a new
major release. During the deprecation window, every response from the
deprecated version includes RFC 8594 headers:

```
HTTP/1.1 200 OK
Deprecation: Sun, 11 May 2026 23:59:59 GMT
Sunset: Tue, 11 May 2027 23:59:59 GMT
Link: <https://api.example.com/api/v2/predict>; rel="successor-version"
Content-Type: application/json
```

Where:
- **Deprecation:** the HTTP-date when the version was marked deprecated
- **Sunset:** the HTTP-date when the version will be removed (Deprecation + 12 months)
- **Link rel="successor-version":** the new URL to migrate to

(Header middleware lands in a future phase; this phase only documents
the policy. Until then, partners can rely on the `schema_version`
envelope field to detect minor migrations.)

## Sibling Contract Artifact (xgb_v2-contract.json)

The model artifact (`models/xgb_v2.joblib`) ships with TWO sibling JSON files:

- `xgb_v2_meta.json` — internal training metadata (feature columns, hyperparameters,
  training metrics). Partners do NOT depend on this file; it may change
  between minor versions.
- `xgb_v2-contract.json` — **partner-facing** contract metadata. SHA-checkable
  independently of meta.json. Contains:
  - `schema_version` (matches the API envelope schema_version)
  - `gate_contract_ref` (path to the calibration gate this model passed)
  - `feature_columns_hash` (deterministic hash of the v2.2 feature column set)
  - `min_partner_version_supported`
  - `deprecation_policy: "N >= 2 minor versions"` (we support the two most recent
    minor versions; older clients receive Deprecation headers)
  - `model_artifact_sha256` (byte-identity check for the joblib)
  - `created_at` (ISO date of the contract emission)

Partners SHOULD checksum `xgb_v2-contract.json` on every refresh — its
hash is a stable detector of model-shape changes (whereas `meta.json`
changes on every retrain even if the contract surface is identical).

## Client Codegen Recipes

The OpenAPI 3.1 spec (`openapi.v1.0.0.json`) and the standalone
JSON Schema (`predictor.schema.v1.0.0.json`) drive client codegen.
Below are the recommended (2026-current) tools per language:

### TypeScript

```bash
# Recommended: @hey-api/openapi-ts (modern; OpenAPI 3.1 native)
npm install --save-dev @hey-api/openapi-ts
npx @hey-api/openapi-ts \
  --input src/ufc_prediction/contracts/openapi.v1.0.0.json \
  --output ./src/api/v1 \
  --client fetch
```

Generates typed request/response interfaces + a fetch-based client.

### Python

```bash
# Recommended: datamodel-code-generator (Pydantic v2 output)
pip install 'datamodel-code-generator[http]'
datamodel-codegen \
  --input src/ufc_prediction/contracts/openapi.v1.0.0.json \
  --input-file-type openapi \
  --output ./api_v1.py \
  --output-model-type pydantic_v2.BaseModel
```

Generates Pydantic v2 BaseModels that exactly mirror the API surface.

### Go

```bash
# Recommended: oapi-codegen v2
go install github.com/deepmap/oapi-codegen/v2/cmd/oapi-codegen@latest
oapi-codegen \
  -package api \
  -generate types,client \
  src/ufc_prediction/contracts/openapi.v1.0.0.json > api.gen.go
```

**Caveat:** `oapi-codegen` has limited OpenAPI 3.1 support as of 2025-2026.
If you hit `unsupported OpenAPI version` errors, run a 3.1→3.0
down-conversion via `redocly bundle`:

```bash
npx @redocly/cli bundle \
  src/ufc_prediction/contracts/openapi.v1.0.0.json \
  --output openapi.v1.0.0.3.0.json \
  --ext json
```

Then point `oapi-codegen` at the 3.0 bundle.

## Authentication

**Out of scope for v1.0.0.** Authentication / authorization / rate-limiting
are forthcoming in a future phase (target: v2.3+). For now, partners
integrating against `/api/v1/predict` should treat the endpoint as
open access and wire their own gateway if necessary.

When auth lands, it will be additive (header-based; not URL-path
breaking) and will be announced via the standard deprecation header
pattern documented above.

## Schema Validation

Partners can validate any received response against the standalone
JSON Schema:

```python
import json
from jsonschema import Draft202012Validator

schema = json.loads(open("predictor.schema.v1.0.0.json").read())
validator = Draft202012Validator(schema)
validator.validate(response_payload)
```

The schema is JSON Schema Draft 2020-12 (matches OpenAPI 3.1.0's
schema-object semantics).

## Forward Compatibility

The `PredictorOutputV1` schema includes the following Optional fields
that may become non-null in future minor versions:

- `base_prob` — raw XGBoost output before meta-learner wrapping
- `meta_prob` — meta-learner output (currently always null; populated
  when a calibrated meta-learner is promoted)
- `meta_learner_version` — meta-learner version tag (currently null)
- `meta_skipped_reason` — reason meta was skipped (currently
  `"no_meta_artifact"`; will become null when meta is active)

Clients SHOULD treat these fields as Optional and handle null values
gracefully. The schema is forward-compatible by construction:
lock-time fuzz testing (Phase 25 D-09) validates a mocked
meta-learner-active response against the v1.0.0 schema, guaranteeing
that future activations do not break v1.0.0 clients.
