# Model artifacts

This directory holds the trained model files. **Most of them are research
variants — you almost certainly only care about the canonical pair.**

## 👉 The canonical model (use this)

The production predictor is **META-V22**, a two-stage stack:

| Role | File | Notes |
|---|---|---|
| Base model | `xgb_v2.joblib` | XGBoost, 72 features (incl. closing-odds). AUDIT-01 byte-locked. |
| Stacker | `meta/meta_v2.joblib` | Logistic meta-learner over the base output + odds. AUDIT-01 byte-locked. |

This is what the API (`api/v1/predict.py`) and the `ufc predict` CLI load by
default — `ModelPredictor(model_dir="models", version="v2")`. Nothing else in
this directory needs to be present for predictions to work, **except** the
no-odds fallback below.

### Automatic fallback

| File | Notes |
|---|---|
| `xgb_v2_no_odds.joblib` | 67-feature variant the predictor lazy-loads when closing-odds features are missing/stale at predict time. |

## File-naming convention

Each model ships as a triplet:

- `xgb_<ver>.joblib` / `meta/meta_<ver>.joblib` — the serialized model
- `<name>_meta.json` — training metadata (feature count, train window, timestamp)
- `<name>-contract.json` — the partner/gate I/O contract (where present)

## Research variants (not loaded in production)

Kept for provenance and methodology reproducibility. Safe to ignore unless you
are retraining or studying the model history.

| File | What it is |
|---|---|
| `xgb_v1.joblib` | First-generation base model. Superseded by v2. |
| `xgb_v3.joblib` | Later 72-feature base candidate (not promoted; v2 remains canonical). |
| `xgb_v2_netd.joblib` | 92-feature net-difference / time-decayed experiment. |
| `xgb_v2_refv2.joblib` | 92-feature refit on the reference-v2 substrate. |
| `meta/meta_v22_travel.joblib` | Stacker with travel-distance features. |
| `meta/meta_v2_netd.joblib` | Stacker paired with the `xgb_v2_netd` base. |
| `meta/meta_v2_refv2.joblib` | Stacker paired with the `xgb_v2_refv2` base. |
| `meta/meta_v2_refit_v2.6.joblib` | Refit baseline (AUDIT-01 soft-protected). |
| `meta/meta_v3_candidate.joblib` | v3 stacker candidate (not promoted). |

> **AUDIT-01:** `xgb_v2.joblib`, `meta/meta_v2.joblib` (and `meta_v2_dedup`) are
> byte-identity locked — a pre-commit guard blocks edits without an explicit
> `AUDIT01_OVERRIDE=1`. See `CONTRIBUTING.md` § AUDIT-01.
