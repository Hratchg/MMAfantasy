# scripts/

One-off and operational scripts. **None of this is part of the importable
`ufc_prediction` package** — the product lives in `src/`. These are tools you
run by hand (`uv run python scripts/<name>.py`), plus a large amount of
research/training provenance kept so the model history is reproducible.

If you just want to **use** the predictor, you can ignore this whole directory —
see the repo `README.md`. The scripts matter only for operating the data
pipeline or rebuilding/studying the models.

> ⚠️ Many training/substrate/gate scripts depend on intermediate artifacts that
> are **not shipped** (`data/intermediate/*.parquet`, `.planning/` archives).
> They will error with a "missing parquet/contract" message unless those are
> regenerated or restored. This is expected.

## Wired into automation (do not delete)

| Script | Used by |
|---|---|
| `check_audit01_protected_files.py` | pre-commit hook — blocks edits to AUDIT-01 byte-locked files |
| `build_travel_substrate_v261.py` | referenced from `pyproject.toml` |

## Operational utilities (run by hand)

| Script | Purpose |
|---|---|
| `dump_db.sh` / `restore_db.sh` | dump / restore the Postgres corpus |
| `bfo_backfill.py` | refresh BestFightOdds closing odds |
| `scrape_referees_full.py` | scrape referee data |
| `backfill_fighter_aliases_from_dedup_recon.py`, `backfill_venue_geocodes.py`, `refresh_fighters_names_v26.py`, `recon_dedup.py`, `ingest_pre_ufc_records_v25.py` | data backfills / ingest |
| `emit_partner_contracts.py` | regenerate partner JSON contracts |
| `generate_handoff_pdf.py`, `generate_business_handoff_pdf.py`, `generate_client_pdf.py` | render the handoff docs to PDF |
| `load_to_supabase.sh`, `prune_stale_worktrees.sh` | misc ops/dev helpers |

## Model reproduction (training pipeline)

Grouped by naming convention; tied to the model history in `../models/`. Run
order is roughly: build substrate → train base → train/compose meta → verify gate.

- `build_*_substrate_v*.py` — assemble the feature substrate
- `train_xgb_*.py`, `retrain_xgb_*.py` — train/retrain the XGBoost base models
- `train_meta_*.py`, `refit_meta_*.py`, `compose_*_meta*.py` — train the meta stackers
- `run_travel_gate_v26.py`, `verify_*_gate_v25.py`, `verify_travel_oof_v25.py` — promotion-gate checks

## Research / analysis (archived)

Exploratory work kept for provenance — not part of any live workflow:

- `spike_*.py` — design spikes (noise floor, pagerank, net-v2)
- `audit_*.py` / `audit_*.sh` — data/feature audits (camp, referees, physical features, BFO reachability, xgb_v2 SHA)
- `baselines_v24.py`, `calibration_v24.py`, `feature_importance_v24.py`, `corpus_v25_delta.py`, `remeasure_meta_v22_v23.py`, `rerun_v22_meta_spike_on_deduplicated_corpus.py`, `_emit_30_variance_fixtures.py` — measurement / diagnostics

> **AUDIT-01:** `spike_noise_floor_v22.py`, `spike_noise_floor_v23.py`, and
> `train_meta_v22.py` are byte-identity locked as methodology provenance — do
> not edit or remove without `AUDIT01_OVERRIDE=1`. See `CONTRIBUTING.md`.
