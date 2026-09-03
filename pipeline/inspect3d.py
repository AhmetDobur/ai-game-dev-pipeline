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


def gltf_doc(mesh: Path) -> dict | None:
    """The JSON chunk of a .glb, or None when it cannot be read."""
    import json as _json
    import struct
    try:
        d = mesh.read_bytes()
        if d[:4] != b"glTF":
            return None            # .gltf/.obj etc -- not our business
        n = struct.unpack_from("<I", d, 12)[0]
        return _json.loads(d[20:20 + n])
    except Exception:
        return None                # unreadable -- never a rejection


def clips(mesh: Path) -> list[dict]:
    """Every animation in a .glb: name, duration in seconds, bones it moves.

    Read from the file rather than from the task spec, because the spec records
    what was ASKED FOR and this records what exists. They diverge constantly --
    a requested clip whose mocap lookup missed still lands in the file, just
    driven by a synthetic cycle instead.
    """
    doc = gltf_doc(mesh)
    if not doc:
        return []
    acc = doc.get("accessors", [])
    out = []
    for a in doc.get("animations", []):
        # duration = the largest input-accessor max across the clip's samplers
        end = 0.0
        for smp in a.get("samplers", []):
            mx = acc[smp["input"]].get("max") if smp.get("input") is not None else None
            if mx:
                end = max(end, float(mx[0]))
        targets = {ch["target"].get("node") for ch in a.get("channels", [])
                   if ch.get("target")}
        out.append({"name": a.get("name", "?"), "seconds": round(end, 2),
                    "bones": len(targets)})
    return out


def has_skin(mesh: Path) -> bool | None:
    """Does this .glb bind its mesh to a skeleton? None when it cannot be read.

    A rigged character with no skin is the pipeline's quietest failure: the glb
    is the right size, carries JOINTS_0/WEIGHTS_0 and every named clip, and
    imports without a single error -- but the bones animate nothing and the
    character slides about frozen in its rest pose. Blender emits exactly this
    when bone-heat weighting fails on non-manifold geometry.
    """
    doc = gltf_doc(mesh)
    if doc is None:
        return None                # unreadable -- never a rejection
    return bool(doc.get("skins")) and any(
        "skin" in node for node in doc.get("nodes", []))


def rig_verdict(mesh: Path) -> tuple[bool, str]:
    """Reject an animated model whose mesh is not bound to its skeleton."""
    if has_skin(mesh) is False:
        return False, (f"{mesh.name}: rigged model has no skin -- the mesh is not "
                       "bound to its skeleton, so every clip animates nothing")
    return True, ""
