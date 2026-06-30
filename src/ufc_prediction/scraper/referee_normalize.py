"""Referee name normalization shared by audit_referees.py and ingest.py.

Pure stdlib (RESEARCH Finding 7): NFKD Unicode fold + ASCII strip + whitespace
collapse + alias-map override. NO python-slugify dependency (would add transitive
text-unidecode/Unidecode + a runtime dep for ~10 lines of work).

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""

from __future__ import annotations

import re
import unicodedata

REFEREE_ALIASES: dict[str, str] = {
    "herbert-dean": "herb-dean",
    # Additional aliases populated post-Task-4 audit (REF_00_AUDIT.json
    # per_referee_top30 alias_variations field surfaces real-world drift).
}


def normalize_referee_name(raw: str | None) -> str | None:
    """Normalize a raw referee name to canonical lowercase-hyphen form.

    'Herb Dean'    -> 'herb-dean'
    'Herbert Dean' -> 'herb-dean'  (via REFEREE_ALIASES)
    'Mike Beltrán' -> 'mike-beltran'  (NFKD + ASCII fold)
    None / ''      -> None
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    folded = unicodedata.normalize("NFKD", stripped).encode("ascii", "ignore").decode("ascii")
    folded = folded.lower().strip()
    folded = re.sub(r"\s+", "-", folded)
    folded = re.sub(r"[^a-z0-9-]", "", folded)
    if not folded:
        return None
    return REFEREE_ALIASES.get(folded, folded)
