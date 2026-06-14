"""``ufc_prediction.ml`` package surface.

Re-exports public symbols from submodules for ergonomic import.
"""

from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

__all__ = ["load_substrate_snapshot"]
