# Partner Quickstart

5-minute onboarding for a partner integrating against the UFC Fight Prediction API.

For background on what the model does and how it was built, see [`PARTNER-RELEASE-v2.3.md`](PARTNER-RELEASE-v2.3.md). For deeper architecture, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. Run the API locally

```bash
git clone <repo>
cd ufc-fight-prediction
uv sync
uv run uvicorn ufc_prediction.api.app:app --reload
```

Now visit:

- **http://localhost:8000/docs** — interactive Swagger UI; try requests directly in the browser
- **http://localhost:8000/openapi.json** — full OpenAPI 3.x spec (codegen-friendly)
- **http://localhost:8000/redoc** — alternative ReDoc-style API browser

## 2. Predict a matchup

The primary endpoint:

```http
POST /api/v1/predict
Content-Type: application/json

{
  "fighter_a": "Israel Adesanya",
  "fighter_b": "Sean Strickland",
  "fight_date": "2026-06-15"
}
```

**Response (v1.1.0 schema):**

```json
{
  "schema_version": "v1.1.0",
  "fighter_a": "Israel Adesanya",
  "fighter_b": "Sean Strickland",
  "prob_a_wins": 0.6234,
  "prob_b_wins": 0.3766,
  "elo_breakdown": {
    "fighter_a": {
      "overall": 1612,
      "striking": 1645,
      "grappling": 1488
    },
    "fighter_b": {
      "overall": 1521,
      "striking": 1534,
      "grappling": 1492
    },
    "diffs": {
      "overall": 91,
      "striking": 111,
      "grappling": -4
    }
  },
  "model": "META-V22",
  "gate_contract_ref": "v2.3",
  "model_candidates": [
    {"name": "xgb_v2", "sha256": "6e7641...", "phase": "v2.2"},
    {"name": "META-V22", "sha256": "98120a6...", "phase": "v2.2"}
  ],
  "phase_chain_audit_sha": "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
}
```

> **Note:** Response shape above is illustrative. Authoritative field list is in `src/ufc_prediction/contracts/predictor.schema.v1.1.0.json`. The 3 new v1.1.0 fields (`gate_contract_ref`, `model_candidates`, `phase_chain_audit_sha`) are optional and default to `null` — v1.0.0 deserializers see the same response shape they always did.

## 3. Use the Elo breakdown

The Elo breakdown is the partner-facing differentiator. The same closing-odds-derived probability is available from any odds aggregator; the per-domain Elo split is unique to this product.

Rendering ideas:

- **Headline pick + confidence:** `prob_a_wins=0.62` → "Adesanya, moderate confidence"
- **Per-domain bars:** Striking +111 (Adesanya), Grappling -4 (basically even), Overall +91
- **Style narrative:** Striking advantage + neutral grappling → "Stand-up matchup; favors striker"

## 4. Calibration expectations

Two measurement scopes — read both.

### 4a. xgb_v2 base model on v2.3 widened eval slices

Numbers below come from the v2.3 10-seed bootstrap noise-floor spike. Source: [`.planning/gate_contract_v2.3.json`](../.planning/gate_contract_v2.3.json), per-slice fields.

| Eval slice | Median accuracy (xgb_v2) | Median Brier (xgb_v2) | Gate-floor `accuracy_min` (median + σ) | Gate-floor `brier_max` (median − σ) |
|------------|--------------------------|------------------------|----------------------------------------|--------------------------------------|
| Most-recent 12 months | **75.9%** | 0.161 | 78.1% | 0.151 |
| Most-recent 24 months | **74.7%** | 0.165 | 76.3% | 0.158 |
| Random 15% holdout | **74.7%** | 0.154 | 78.1% | 0.137 |

The **median** column is the actual measured accuracy of the base model on each slice. The **gate-floor** columns are derived as `median + 1σ` (for accuracy) or `median − 1σ` (for Brier) per the locked formula in the gate contract — these are the bars that a *candidate* model must beat to be promoted, not the production model's measured performance.

### 4b. META-V22 meta-learner at v2.2 training time

The production meta-learner `models/meta/meta_v2.joblib` was trained on v2.2-era slices. Its per-slice metrics from training are recorded in [`models/meta/meta_v2_meta.json`](../models/meta/meta_v2_meta.json):

| Eval slice | Median accuracy (META-V22) | Median Brier (META-V22) | AUC-ROC |
|------------|----------------------------|--------------------------|---------|
| Most-recent 12 months | 70.1% | 0.213 | 0.732 |
| Most-recent 24 months | 70.1% | 0.213 | 0.732 |
| Random 15% holdout | 78.6% | 0.187 | 0.786 |

Note: 12mo == 24mo because the v2.2 eval slices collapsed to identical row sets after symmetric NaN-drop on `closing_prob_diff` (this artifact is what v2.3 Phase 29 fixed by widening to per-feature NaN handling — but the production META-V22 weights weren't re-trained on the widened slices). Re-measuring META-V22 on v2.3 widened slices is open as v2.4+ scope.

### Important caveats

1. **Closing odds dependency.** Full accuracy assumes BFO closing odds are available at prediction time (typically day-of-fight). Predictions made early in the week (opening odds only) have not been validated — accuracy is expected to degrade because closing odds incorporate sharp-money corrections that opening odds don't.
2. **Strict temporal split.** Eval uses fights after 2023-01-01 only (no train-on-test). See `cutoff_date` field in [`models/xgb_v2_meta.json`](../models/xgb_v2_meta.json).
3. **Brier > accuracy for sports models.** Brier score rewards calibration (well-calibrated 60% predictions matter even when argmax flips). Accuracy alone is misleading because closing odds itself already sits at ~70–75% in UFC; the partner-facing question is what the model *adds* over closing odds, which is best measured in Brier delta.
4. **Sport-specific accuracy ceiling.** Per [`DATA_STRATEGY.md`](../DATA_STRATEGY.md): "Any model that claims >70% accuracy on UFC fights is either overfit, leaked, or lying." The 75–78% numbers above are on held-out slices with strict temporal separation, but partners should treat any single-card prediction at face value — UFC has high inherent variance (a single punch can flip an 8-second outcome).

## 5. Version migration (v1.0.0 → v1.1.0)

No code changes required. v1.1.0 is additive-only.

If you want to consume the new fields:
- `gate_contract_ref` — useful for "this prediction was made against gate vN"
- `model_candidates` — useful for "this prediction came from META-V22; here's its SHA so you can audit"
- `phase_chain_audit_sha` — useful for "verify the production model file hasn't been tampered with"

If you don't want them: ignore them. Your v1.0.0 deserializer continues to work.

## 6. Where to ask questions

- **API behavior:** check the [OpenAPI spec](../src/ufc_prediction/contracts/openapi.v1.1.0.json) and the [JSON schema](../src/ufc_prediction/contracts/predictor.schema.v1.1.0.json)
- **Model behavior:** [`PARTNER-RELEASE-v2.3.md`](PARTNER-RELEASE-v2.3.md) + [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Domain terminology:** [`GLOSSARY.md`](GLOSSARY.md)
- **Edge cases not in docs:** open an issue with a minimal reproduction (fighter names + expected vs actual)

## 7. Roadmap awareness

v2.3 is the **first** public partner release. v2.4+ items already on the backlog that may affect partner integrations:

- **TRAVEL feature primitives** — venue lat/lon/tz/distance-from-home features (currently NaN in the feature matrix despite populated substrate). May improve away-fighter prediction accuracy in a future minor bump.
- **Early-line variant** — a no-closing-odds model variant for partners who want predictions before closing odds form. Will ship as `META-V22-NO-ODDS` (or similar) with its own accuracy floor.
- **`predictor.py` cols-dispatch generalization** — internal change; no partner impact unless `meta_v3` promotes (would add a new entry to `model_candidates`).

None of these will require schema changes (additive-only contract).
