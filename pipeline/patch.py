"""Patching: keep working on a finished game instead of regenerating it.

A patch is a new run linked to its parent (`parent_id`, `revision`). The parent's
whole workspace (game + artifacts) is copied in as this revision's starting point,
and its task graph is copied in as `done`. The delta decomposition then marks a
subset stale (MODIFY an artifact, or ADD a new one); the invalidation walk cascades
staleness to every transitive dependent; the existing scheduler re-runs exactly the
stale tasks and reuses every `done` task's already-produced files.

The graph transform is a pure function (`build_patch_graph`) so the cascade logic is
tested without a GPU, a DB or Blender. The whole graph is inserted in one atomic
transaction, so a crash mid-build leaves zero tasks (a clean re-run), never half.
"""
import re
import shutil
from collections import defaultdict
from pathlib import Path

from . import db


def start_patch(cfg, conn, parent_id: str, instruction_path, reference_images=None) -> str:
    """Create the patch run. It chains off `parent_id`, so v1 -> v2 -> v3 stacks."""
    parent = db.get_run(conn, parent_id)
    if parent is None:
        raise ValueError(f"unknown parent run {parent_id!r}")
    return db.create_run(conn, str(instruction_path), reference_images or [],
                         parent_id=parent_id, revision=parent["revision"] + 1)


def _run_dir(cfg, run_id: str) -> Path:
    return Path(cfg["paths"]["workspace"]) / "runs" / run_id


def prepare_workspace(cfg, run_id: str, parent_id: str) -> None:
    """Snapshot the parent revision's files into this run's workspace. Idempotent
    (dirs_exist_ok), so a resumed patch re-runs it harmlessly."""
    src, dst = _run_dir(cfg, parent_id), _run_dir(cfg, run_id)
    if src.exists():
        # ponytail: full copy per revision — simplest correct snapshot; disk is cheap here
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.mkdir(parents=True, exist_ok=True)


_RUN_PREFIX = re.compile(r"^[0-9a-f]{12}-")


def short_id(full: str) -> str:
    """`349edb5bf375-hero_anim` -> `hero_anim`.

    The router is shown these. Run-scoped ids are an internal concern, and a 7B
    asked to copy twenty 25-character ids exactly will not: in a live run it
    wrote the bare id anyway, because that is the form every example it has
    seen uses. Showing the short form removes the whole class of mistake, and
    repair_patch_list still accepts the long one.
    """
    return _RUN_PREFIX.sub("", full or "")


def manifest(parent_rows: list[dict]) -> list[dict]:
    """Compact view of what the game already contains, for the delta decomposer.

    Carries BOTH what each task was asked to make (`summary`, read from its spec)
    and what it actually produced (`observed`, measured from its output files).
    The two disagree often enough that a router shown only the spec has to guess
    which artifact an instruction like "make the animation more realistic" is
    even about.
    """
    from .observe import facts_for
    out = []
    for t in parent_rows:
        s = t["spec"]
        summary = (s.get("file") or s.get("prompt") or s.get("text")
                   or ",".join(s.get("animations", [])) or s.get("export_preset") or "")
        out.append({"id": t["id"], "type": t["type"], "summary": str(summary)[:80],
                    "observed": facts_for(t["type"], t.get("output_path", "")),
                    # carried for the deterministic repairs, never rendered into
                    # the prompt -- the router sees only the four fields above
                    "spec": s if isinstance(s, dict) else {}})
    return out


def _reprefix(value, old: str, new: str):
    """Re-scope a task id (and any id embedded in a dep list or spec) from the parent
    run's prefix to this run's, leaving unrelated strings untouched."""
    if isinstance(value, str):
        return new + value[len(old):] if value.startswith(old + "-") else value
    if isinstance(value, list):
        return [_reprefix(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _reprefix(v, old, new) for k, v in value.items()}
    return value


def build_patch_graph(parent_rows: list[dict], patch_tasks: list[dict],
                      parent_id: str, run_id: str) -> tuple[list[dict], set[str]]:
    """Pure transform: parent graph + delta ops -> the patch run's full task rows.

    Copies parent tasks (re-scoped, output paths repointed), applies MODIFY/ADD ops,
    then marks every stale task and its transitive dependents `pending` with attempts
    reset. The single assemble task is rewired to depend on everything so the build
    is always regenerated last. Returns (rows, stale_ids)."""
    src_frag, dst_frag = f"runs/{parent_id}", f"runs/{run_id}"
    rows: dict[str, dict] = {}
    for t in parent_rows:
        nid = _reprefix(t["id"], parent_id, run_id)
        rows[nid] = {
            "id": nid, "type": t["type"],
            "spec": _reprefix(t["spec"], parent_id, run_id),
            "depends_on": [_reprefix(d, parent_id, run_id) for d in t["depends_on"]],
            "status": t["status"], "attempts": t["attempts"],
            # normalize separators first: on Windows the stored path is
            # backslashed and would never match the forward-slash fragment
            "output_path": t["output_path"].replace("\\", "/").replace(src_frag, dst_frag),
            "error": t["error"],
        }

    # The delta decomposer references artifacts by their PARENT-scoped ids (that is
    # what the manifest showed it). Every such ref is reprefixed parent->child to
    # match `rows`; new ADD ids are scoped to this run and mapped on top.
    id_map = {pt["id"]: f"{run_id}-{pt['id']}" for pt in patch_tasks if "target" not in pt}

    def _scalar(s):
        s = run_id + s[len(parent_id):] if s.startswith(parent_id + "-") else s
        return id_map.get(s, s)

    def remap(ref):
        if isinstance(ref, str):
            return _scalar(ref)
        if isinstance(ref, list):
            return [remap(r) for r in ref]
        if isinstance(ref, dict):
            return {k: remap(v) for k, v in ref.items()}
        return ref

    seeds: set[str] = set()
    for pt in patch_tasks:
        if "target" in pt:
            tgt = _scalar(pt["target"])
            if tgt not in rows:
                raise ValueError(f"patch target {pt['target']!r} is not an artifact in this game")
            rows[tgt]["spec"] = remap(pt["spec"])
            seeds.add(tgt)
        else:
            nid = id_map[pt["id"]]
            # an ADD id that collides with a reused parent task would silently
            # overwrite it (both land on the same run-scoped key) — refuse it
            if nid in rows:
                raise ValueError(f"patch ADD id {pt['id']!r} collides with an existing "
                                 f"artifact ({nid!r}); the delta must use a fresh id")
            rows[nid] = {
                "id": nid, "type": pt["type"], "spec": remap(pt.get("spec", {})),
                "depends_on": [remap(d) for d in pt.get("depends_on", [])],
                "status": "pending", "attempts": 0, "output_path": "", "error": "",
            }
            seeds.add(nid)

    # assemble always runs last and re-exports the whole game after any change;
    # strip any dependency ON assemble (an ADD op may name it) or that edge plus
    # this rewiring would be a cycle
    assemble_ids = {rid for rid, r in rows.items() if r["type"] == "assemble"}
    others = [rid for rid in rows if rid not in assemble_ids]
    for r in rows.values():
        if r["type"] == "assemble":
            r["depends_on"] = list(others)
            seeds.add(r["id"])
        else:
            r["depends_on"] = [d for d in r["depends_on"] if d not in assemble_ids]

    # patching a parent that ended in failure must repair it: any copied task that
    # is not already 'done' (a failed/blocked/interrupted task) is re-run, or the
    # rewired assemble would stay blocked on it forever and the revision can't finish
    for rid, r in rows.items():
        if r["status"] != "done":
            seeds.add(rid)

    stale = _cascade(rows, seeds)
    for rid in stale:
        rows[rid].update(status="pending", attempts=0, output_path="", error="")
    return list(rows.values()), stale


def _cascade(rows: dict[str, dict], seeds: set[str]) -> set[str]:
    """Transitive closure of dependents over the depends_on edges."""
    children = defaultdict(list)
    for r in rows.values():
        for d in r["depends_on"]:
            children[d].append(r["id"])
    stale, queue = set(), list(seeds)
    while queue:
        rid = queue.pop()
        if rid in stale:
            continue
        stale.add(rid)
        queue.extend(children[rid])
    return stale


def validate_patch_list(patch_tasks, manifest_ids: set[str]) -> None:
    if not isinstance(patch_tasks, list) or not patch_tasks:
        raise ValueError("patch is not a non-empty JSON array")
    add_ids = {pt["id"] for pt in patch_tasks if "target" not in pt and "id" in pt}
    known = manifest_ids | add_ids
    for pt in patch_tasks:
        if "target" in pt:
            if pt["target"] not in manifest_ids:
                raise ValueError(f"patch target {pt['target']!r} not in game")
            if not isinstance(pt.get("spec"), dict):
                raise ValueError("MODIFY op needs a spec object")
        else:
            if not pt.get("id") or pt.get("type") not in db.TASK_TYPES or pt["type"] == "assemble":
                raise ValueError(f"bad ADD op {pt.get('id')!r}: need id + non-assemble type")
            if not isinstance(pt.get("spec"), dict):
                raise ValueError(f"ADD op {pt['id']!r} needs a spec object")
            for d in pt.get("depends_on", []):
                if d not in known:
                    raise ValueError(f"ADD op {pt['id']!r}: unknown dependency {d!r}")


# What a plainly-worded instruction is about. Deliberately coarse: this exists to
# catch the MAIN mistake -- a router that was asked about the animation and went
# and rewrote the concept art -- not to adjudicate subtle wording.
_DOMAIN_WORDS = {
    "rig_animate": ("animation", "animate", "anim ", "motion", "walk", "walking",
                    "run ", "running", "idle", "rig", "skeleton", "movement",
                    "moves", "moving"),
    "design_2d": ("concept art", "artwork", "texture", "palette", "colour",
                  "color", "art style"),
    "design_3d": ("mesh", "model", "geometry", "sculpt", "silhouette",
                  "proportions", "topology"),
    "audio": ("sound", "audio", "voice", "music", "speech", "spoken"),
    "code": ("script", "code", "controller", "gameplay", "combat", "input",
             "camera", "logic"),
}


def instruction_domains(instruction: str) -> set[str]:
    text = " " + (instruction or "").lower() + " "
    return {kind for kind, words in _DOMAIN_WORDS.items()
            if any(w in text for w in words)}


def check_patch_grounding(patch_tasks, manifest_rows, instruction) -> None:
    """Refuse a patch that ignores what the instruction is plainly about.

    The router is shown each artifact's observed facts precisely so it can pick
    the right one; this is the check that it actually did. It fires only when the
    instruction names EXACTLY ONE domain and the game contains artifacts of that
    domain and the patch touches none of them -- an unambiguous miss, not a
    judgement call. The message names the candidates and their observed facts, so
    the retry has everything it needs to correct itself.
    """
    domains = instruction_domains(instruction)
    if len(domains) != 1:
        return                      # ambiguous or silent -- do not second-guess
    want = domains.pop()
    candidates = [m for m in manifest_rows if m["type"] == want]
    if not candidates:
        return                      # nothing of that kind to target
    by_id = {m["id"]: m for m in manifest_rows}
    touched = {by_id[pt["target"]]["type"] for pt in patch_tasks
               if "target" in pt and pt["target"] in by_id}
    touched |= {pt.get("type") for pt in patch_tasks if "target" not in pt}
    if want in touched:
        return
    if len(candidates) == 1:
        # one obvious answer: hand it over whole. The value of this check is that
        # it is grounded in measured facts, not that a 7B rediscovers the target
        # unaided on the third try -- which, in a live run, it did not.
        c = candidates[0]
        import json as _json
        raise ValueError(
            f"the instruction is about {want}, but this patch changes none. "
            f"There is exactly one {want} artifact and its observed state is: "
            f"{c.get('observed') or 'not measured'}. "
            f'Reply with exactly: [{{"target": "{short_id(c["id"])}", "spec": '
            f"{_json.dumps(c.get('spec', {}))}}}] "
            f"with that spec edited to make the change.")
    listing = "; ".join(
        f"{short_id(m['id'])} (observed: {m.get('observed') or 'not measured'})"
        for m in candidates[:4])
    raise ValueError(
        f"the instruction is about {want}, but this patch changes none. "
        f"The {want} artifacts in this game are: {listing}. "
        f"MODIFY the one whose observed facts show the problem — its target is "
        f"its id from that list.")


# Required spec keys per task type, for patch ops. The fresh decomposer gets these
# from its prompt's worked example; a patch op arrives with no such scaffolding and
# a small router invents plausible-looking fields ("skin_to_rig": true) instead.
_REQUIRED_SPEC = {
    "design_2d": ("prompt",),
    "design_3d": ("prompt",),
    "rig_animate": ("mesh_from", "body_plan", "animations"),
    "code": ("file", "description"),
    "audio": ("text",),
}


def repair_patch_list(patch_tasks, manifest_rows) -> None:
    """Resolve the ids a router actually writes onto the ids that exist.

    The manifest shows run-scoped ids ("349edb5bf375-hero_mesh") and the router
    writes the bare decomposer id ("hero_mesh") -- which is how it was originally
    told to name things, and how every example it has ever seen looks. Rejecting
    that threw away an otherwise correct patch three times in a row. A bare id is
    accepted whenever it resolves to exactly ONE manifest id; an ambiguous one is
    left alone for the validator to report.
    """
    if not isinstance(patch_tasks, list):
        return
    known = {m["id"] for m in manifest_rows}
    new_ids = {pt["id"] for pt in patch_tasks
               if isinstance(pt, dict) and "target" not in pt and pt.get("id")}

    def resolve(value):
        if not isinstance(value, str) or value in known or value in new_ids:
            return value
        hits = [k for k in known if short_id(k) == value] \
            or [k for k in known if k.endswith("-" + value)]
        return hits[0] if len(hits) == 1 else value

    def walk(v):
        if isinstance(v, str):
            return resolve(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    for pt in patch_tasks:
        if not isinstance(pt, dict):
            continue
        if "target" in pt:
            pt["target"] = resolve(pt["target"])
        pt["depends_on"] = [resolve(d) for d in pt.get("depends_on", [])]
        if isinstance(pt.get("spec"), dict):
            pt["spec"] = walk(pt["spec"])


def collapse_duplicate_adds(patch_tasks, manifest_rows, parent_specs) -> None:
    """An ADD that rebuilds something the game already has is a MODIFY of it.

    Told the rig is unskinned, a router adds a NEW rig_animate for the same mesh
    rather than fixing the existing one -- which would ship the character twice.
    Only an exact same-type, same-source match is collapsed.
    """
    if not isinstance(patch_tasks, list):
        return
    by_type_src = {}
    for m in manifest_rows:
        spec = parent_specs.get(m["id"], {})
        src = spec.get("mesh_from") or spec.get("concept_from") or spec.get("file")
        if src:
            by_type_src[(m["type"], src)] = m["id"]
    for pt in patch_tasks:
        if not isinstance(pt, dict) or "target" in pt:
            continue
        spec = pt.get("spec") or {}
        src = spec.get("mesh_from") or spec.get("concept_from") or spec.get("file")
        existing = by_type_src.get((pt.get("type"), src))
        if existing:
            merged = dict(parent_specs.get(existing, {}))
            merged.update(spec)          # keep the parent shape, apply the change
            pt.clear()
            pt.update({"target": existing, "spec": merged})


def validate_patch_specs(patch_tasks) -> None:
    """Every op's spec must have the keys its executor reads."""
    for pt in patch_tasks:
        kind = pt.get("type")
        spec = pt.get("spec") or {}
        if "target" in pt and not kind:
            continue                     # MODIFY type is fixed by the target row
        missing = [k for k in _REQUIRED_SPEC.get(kind, ()) if k not in spec]
        if missing:
            raise ValueError(
                f"{kind} op {pt.get('id') or pt.get('target')!r} is missing spec "
                f"key(s) {missing}; a {kind} spec needs "
                f"{list(_REQUIRED_SPEC[kind])} and nothing else is read.")
