"""Objective mesh checks, so a broken 3D asset is rejected without a human looking.

Thresholds here are calibrated against 21 real meshes produced by this pipeline
(see the table in the commit that added this file), not guessed. The rule from
validate.py applies: only an OBJECTIVELY broken signal may fail a task, because
a false rejection costs a full regeneration -- tens of minutes per mesh.
"""
import json
from pathlib import Path

# TRELLIS intermittently returns a flat plane instead of a solid: measured
# min/max extents of 0.003/1.00 -- a 3 mm sheet -- while still writing an 8 MB,
# 40k-face GLB that every byte-count check happily passes. Roughly a fifth of
# the meshes on the box were these. A genuinely flat PROP (a door, a rug) still
# measures ~0.13 at its thinnest, so 0.05 separates broken from legitimately
# thin with a wide margin on both sides.
MIN_RELATIVE_THICKNESS = 0.05


def metrics_for(mesh: Path) -> dict | None:
    """Metrics written beside the mesh by templates/blender_preview.py."""
    p = Path(str(mesh) + ".metrics.json")
    try:
        return json.loads(p.read_text())
    except Exception:      # no preview ran, or it failed -- never a rejection
        return None


def verdict(mesh: Path) -> tuple[bool, str]:
    m = metrics_for(mesh)
    if not m:
        return True, "no metrics"          # absence of evidence is not a failure
    bbox = m.get("bbox") or []
    if len(bbox) == 3 and max(bbox) > 0:
        thin = min(bbox) / max(bbox)
        if thin < MIN_RELATIVE_THICKNESS:
            return False, (f"{Path(mesh).name}: degenerate mesh — thinnest extent is "
                           f"{thin:.1%} of the longest (bbox {bbox}). TRELLIS returned "
                           f"a flat plane, not a solid.")
    return True, "mesh ok"
