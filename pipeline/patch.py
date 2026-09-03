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


def manifest(parent_rows: list[dict]) -> list[dict]:
    """Compact view of what the game already contains, for the delta decomposer."""
    out = []
    for t in parent_rows:
        s = t["spec"]
        summary = (s.get("file") or s.get("prompt") or s.get("text")
                   or ",".join(s.get("animations", [])) or s.get("export_preset") or "")
        out.append({"id": t["id"], "type": t["type"], "summary": str(summary)[:80]})
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
