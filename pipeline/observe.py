"""What a finished artifact ACTUALLY contains, measured from its own files.

A task's spec records what was ASKED FOR. That is what the delta decomposer used
to be shown, and the two diverge constantly: a rig whose mocap lookup missed
still lists "idle, walk, run" in its spec while the file holds three synthetic
cycles, and a mesh that came back a flat plane still reads "a bookshelf".

Asked to "make the animation more realistic", a router shown only specs has to
guess which artifact is at fault. Shown these facts it can see that the clips are
procedural, and target the rig rather than the concept art. Verify, don't guess.

Every reader here is best-effort: an artifact that cannot be measured contributes
nothing rather than a guess, and never raises.
"""
import json
from pathlib import Path

from .inspect3d import clips, has_skin, metrics_for

MAX_FACTS = 240      # keep the manifest readable for a small router


def _motion_sidecar(mesh: Path) -> dict:
    try:
        return json.loads(Path(str(mesh) + ".motion.json").read_text())
    except Exception:
        return {}


def _rig_facts(mesh: Path) -> list[str]:
    found = clips(mesh)
    prov = _motion_sidecar(mesh).get("clips", {})
    if found:
        facts = [", ".join(
            f"{c['name']} ({prov.get(c['name'], 'source unrecorded')}, "
            f"{c['seconds']}s, {c['bones']} bones)" for c in found)]
    else:
        facts = ["NO animation clips in the file"]
    if has_skin(mesh) is False:
        facts.append("NOT SKINNED: the mesh is not bound to the skeleton, so no "
                     "clip can move it")
    return facts


def _mesh_facts(mesh: Path) -> list[str]:
    m = metrics_for(mesh)
    if not m:
        return []
    bb = m.get("bbox") or []
    facts = []
    if len(bb) == 3:
        facts.append("size " + "x".join(str(round(float(v), 2)) for v in bb))
    if m.get("faces"):
        facts.append(f"{m['faces']} faces")
    if m.get("aspect") and float(m["aspect"]) > 20:
        facts.append("FLAT PLANE, not a solid")
    return facts


def facts_for(kind: str, output_path: str) -> str:
    """One short factual line about what this task produced, or "" if unknown."""
    paths = [Path(p) for p in str(output_path or "").split(",") if p.strip()]
    models = [p for p in paths if p.suffix.lower() in (".glb", ".gltf") and p.exists()]
    facts: list[str] = []
    try:
        if kind == "rig_animate" and models:
            facts = _rig_facts(models[0])
        elif kind == "design_3d" and models:
            facts = _mesh_facts(models[0])
        elif kind == "design_2d":
            imgs = [p for p in paths
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p.exists()]
            if imgs:
                from PIL import Image
                w, h = Image.open(imgs[0]).size
                facts = [f"{w}x{h} image"]
        elif kind == "code":
            files = [p for p in paths if p.exists()]
            if files:
                n = files[0].read_text(encoding="utf-8", errors="replace").count("\n") + 1
                facts = [f"{n} lines"]
        elif kind == "assemble":
            builds = [p for p in paths if p.exists()]
            if builds:
                facts = [f"build {builds[0].stat().st_size // 1_000_000} MB"]
    except Exception:
        return ""            # a fact we cannot measure is not a fact to report
    return "; ".join(f for f in facts if f)[:MAX_FACTS]
