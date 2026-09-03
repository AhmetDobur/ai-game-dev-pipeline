"""Stage 1: resident router LLM turns instruction.md into a typed, dependency-aware
task list, which is inserted into the SQLite queue."""
import json
import re

from . import db
from .adapters.llm import LlamaServer, extract_json

DECOMPOSE_PROMPT = """You are the task decomposer of an automated game-development pipeline.

Read the game description below and break it into a dependency-aware task list.

Allowed task types and what their spec must contain:
- "design_2d": {{"prompt": <SDXL prompt>, "purpose": <what this art is for>}}
- "design_3d": {{"prompt": <text spec>, "concept_from": <design_2d task id>}}
- "rig_animate": {{"mesh_from": <design_3d task id>,
                   "body_plan": "humanoid" | "nonhumanoid",
                   "animations": [<clip names, e.g. "idle","walk","attack">],
                   "extras": [<non-skeletal moving parts: "tail","jaw","wings","cloak">]}}
- "code": {{"file": <relative path in the Godot project>, "description": <what to implement>,
            "frame_data": <optional — see below>}}
- "audio": {{"text": <line to speak>, "voice": <voice name>}}
- "assemble": {{"export_preset": <Godot export preset name>}}

Rules:
- Every task: {{"id": <short unique string>, "type": <type>, "depends_on": [<ids>], "spec": {{...}}}}
- design_3d depends on the design_2d it is conditioned on. rig_animate depends on its design_3d.
- rig_animate body_plan is "humanoid" ONLY for roughly human-shaped bipeds (they get
  real mocap/generated motion); every other creature is "nonhumanoid" (procedural
  motion). List any non-skeletal moving parts (a tail, an oversized jaw, wings) in
  "extras" — they get procedural secondary motion regardless of body_plan.
- A design_2d prompt that feeds a design_3d MUST depict exactly ONE subject: full body,
  centered, plain background, "solo" in the prompt. Never a character sheet, turnaround,
  or multiple poses — the 3D stage reconstructs everything in frame, so three poses
  become three meshes.
- A design_3d is always ONE object or ONE character — a bookshelf, a table, a statue.
  NEVER a whole room, interior, hall, landscape or scene: the 3D generator cannot
  build spaces, only things. For an environment, emit SEVERAL prop-sized design_3d
  tasks (2-4 of them, each with its own single-object design_2d); the engine
  arranges them into the room automatically.
- code tasks for the Godot project; exactly one final "assemble" task depending on everything.
- When the game description contains combat timing tables (frame data), copy them
  VERBATIM into one code task with "file": "scripts/combat_sim.gd" as
  "frame_data": {{<move>: {{"startup": n, "active": n, "hitstun": n,
  "knockback": [x, y], "tolerance": n}}}}. Frame counts are at 60fps. That task
  will be graded by a headless simulation against these exact numbers.
- Reference images uploaded by the user: {ref_note}
- When a reference image clearly depicts a task's subject, set "ref_image": "<one
  of the paths above>" inside that design_2d or design_3d spec. A character
  reference on a design_3d makes the mesh match the reference directly; an
  environment/mood reference on a design_2d guides its composition. Never invent
  paths not in the list.

Example (a different, minimal game — copy the STRUCTURE, not the content):
[
 {{"id": "hero_art", "type": "design_2d", "depends_on": [],
   "spec": {{"prompt": "a knight, solo, full body, centered, plain background",
             "purpose": "concept for the hero mesh",
             "ref_image": "<the knight reference path, IF the user uploaded one>"}}}},
 {{"id": "hero_mesh", "type": "design_3d", "depends_on": ["hero_art"],
   "spec": {{"prompt": "the knight as a game-ready mesh", "concept_from": "hero_art",
             "ref_image": "<same reference path — the mesh is built from it directly>"}}}},
 {{"id": "hero_anim", "type": "rig_animate", "depends_on": ["hero_mesh"],
   "spec": {{"mesh_from": "hero_mesh", "body_plan": "humanoid",
             "animations": ["idle", "walk"], "extras": []}}}},
 {{"id": "player", "type": "code", "depends_on": ["hero_anim"],
   "spec": {{"file": "scripts/player.gd",
             "description": "CharacterBody3D controller: WASD moves, camera follows"}}}},
 {{"id": "level", "type": "code", "depends_on": [],
   "spec": {{"file": "scenes/main.tscn",
             "description": "main scene: ground plane, light, spawns the player"}}}},
 {{"id": "build", "type": "assemble", "depends_on": ["hero_art", "hero_mesh",
   "hero_anim", "player", "level"], "spec": {{"export_preset": "Windows Desktop"}}}}
]

Game description:
---
{instruction}
---

Reply with ONLY a JSON array of tasks."""


def decompose(router: LlamaServer, instruction: str, reference_images: list[str],
              temperature: float = 0.6, max_tokens: int = 4096, on_token=None) -> list[dict]:
    ref_note = ", ".join(reference_images) if reference_images else "none"
    prompt = DECOMPOSE_PROMPT.format(instruction=instruction, ref_note=ref_note)
    messages = [{"role": "user", "content": prompt}]
    last_err = None
    for _ in range(3):  # a small router gets structure wrong; feed the error back
        reply = router.chat(messages, temperature=temperature,
                            max_tokens=max_tokens, on_token=on_token)
        try:
            tasks = extract_json(reply)
            repair_task_list(tasks, reference_images)
            validate_task_list(tasks)
            return tasks
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            last_err = e
            messages = messages[:1] + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That task list is invalid: {e}. "
                 "Reply with ONLY the corrected full JSON array."}]
    raise ValueError(f"decomposition failed after 3 attempts: {last_err}")


PATCH_PROMPT = """You are patching an EXISTING game. Do NOT rebuild what the change
doesn't touch. Here is what the game already contains (id — type — summary):

{manifest}

Apply ONLY this change:
---
{instruction}
---

Reply with ONLY a JSON array of patch operations. Each item is exactly one of:
- MODIFY an existing artifact (re-runs it and everything downstream):
  {{"target": "<id from the list above>", "spec": {{<the FULL new spec for that task>}}}}
- ADD a new artifact:
  {{"id": "<new short id>", "type": "<design_2d|design_3d|rig_animate|code|audio>",
    "depends_on": [<ids, existing from the list or other new ids>], "spec": {{...}}}}
Spec shapes are the same as a fresh decomposition. Do NOT emit an "assemble" task —
the build is regenerated automatically. Touch the fewest artifacts necessary.
Reference images uploaded by the user: {ref_note}"""


def decompose_patch(router, manifest_rows: list[dict], instruction: str,
                    reference_images: list[str], temperature: float = 0.6,
                    max_tokens: int = 4096, on_token=None) -> list[dict]:
    from .patch import validate_patch_list
    lines = "\n".join(f"{m['id']} — {m['type']} — {m['summary']}" for m in manifest_rows)
    ref_note = ", ".join(reference_images) if reference_images else "none"
    prompt = PATCH_PROMPT.format(manifest=lines, instruction=instruction, ref_note=ref_note)
    reply = router.chat([{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens, on_token=on_token)
    patch_tasks = extract_json(reply)
    validate_patch_list(patch_tasks, {m["id"] for m in manifest_rows})
    return patch_tasks


def _repair_ref_image(spec: dict, reference_images: list[str]) -> None:
    """The router mangles paths; keep ref_image only if it resolves to a real ref
    (exact, or unique basename-substring match), else drop it silently."""
    ref = spec.get("ref_image")
    if not ref or not isinstance(ref, str):
        spec.pop("ref_image", None)
        return
    if ref in reference_images:
        return
    from pathlib import PurePath
    frag = PurePath(ref.replace("\\", "/")).name.lower()
    hits = [r for r in reference_images
            if frag and frag in PurePath(r.replace("\\", "/")).name.lower()]
    if len(hits) == 1:
        spec["ref_image"] = hits[0]
    else:
        spec.pop("ref_image", None)


def repair_task_list(tasks, reference_images: list[str] | None = None) -> None:
    """Deterministic fixes for mistakes every small router makes: spec references
    (concept_from/mesh_from) imply dependencies, and assemble depends on everything."""
    if not isinstance(tasks, list):
        return
    ids = {t.get("id") for t in tasks if isinstance(t, dict)}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        if not isinstance(t.get("spec"), dict):
            t["spec"] = {}          # routers omit assemble's empty spec constantly
        deps = set(t.get("depends_on") or [])
        for key in ("concept_from", "mesh_from"):
            v = t["spec"].get(key)
            if isinstance(v, str) and v in ids:
                deps.add(v)
        if t.get("type") == "assemble":
            deps = ids - {t.get("id")}
        else:
            # nothing may depend on assemble — with assemble depending on
            # everything, that edge would be a cycle
            deps -= {x.get("id") for x in tasks
                     if isinstance(x, dict) and x.get("type") == "assemble"}
        if t.get("type") in ("design_2d", "design_3d"):
            _repair_ref_image(t["spec"], reference_images or [])
        if t.get("type") == "design_3d" and not t["spec"].get("concept_from"):
            # the 3D stage is image-conditioned; auto-link the design_2d this task
            # already depends on. Done even when ref_image is set: the ref takes
            # precedence at execution, the concept is the fallback if it vanishes.
            by_id = {x.get("id"): x for x in tasks if isinstance(x, dict)}
            for d in sorted(deps):  # sets iterate nondeterministically
                if by_id.get(d, {}).get("type") == "design_2d":
                    t["spec"]["concept_from"] = d
                    break
        if t.get("type") == "code":
            fd = t["spec"].get("frame_data")
            # frame_data is ONLY the combat-timing contract; routers stuff generic
            # config in it (lighting, camera...) which would trigger the combat-sim
            # grader on an unrelated task. Strip junk KEYS, keep real move entries.
            if isinstance(fd, dict):
                moves = {k: v for k, v in fd.items()
                         if isinstance(v, dict) and "startup" in v}
                if moves:
                    t["spec"]["frame_data"] = moves
                else:
                    del t["spec"]["frame_data"]
            if not t["spec"].get("file"):
                # a consistent invented path beats a rejected plan; the coder
                # writes whatever file it is told to
                slug = re.sub(r"[^a-z0-9_]+", "_", str(t.get("id", "task")).lower())
                t["spec"]["file"] = f"scripts/{slug}.gd"
        t["depends_on"] = sorted(deps)
    # non-ASCII ids (a Qwen router drifts into Chinese) slug to the same file —
    # de-collide deterministically. frame_data tasks claim their name FIRST: the
    # combat grader hard-loads res://scripts/combat_sim.gd, so the graded task
    # must never be the one renamed away from it.
    code_tasks = [t for t in tasks if isinstance(t, dict) and t.get("type") == "code"]
    seen: set[str] = set()
    for t in sorted(code_tasks, key=lambda t: not t["spec"].get("frame_data")):
        f = t["spec"]["file"]
        n = 0
        new = f
        while new in seen:
            n += 1
            stem, dot, ext = f.rpartition(".")
            new = f"{stem}_{n}{dot}{ext}" if dot else f"{f}_{n}"
        t["spec"]["file"] = new
        seen.add(new)


def validate_task_list(tasks) -> None:
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("decomposition is not a non-empty JSON array")
    if any(not isinstance(t, dict) for t in tasks):
        raise ValueError("every task must be a JSON object")
    ids = [t.get("id") for t in tasks]
    if len(ids) != len(set(ids)) or any(not i for i in ids):
        raise ValueError("task ids must be unique and non-empty")
    known = set(ids)
    by_id = {t.get("id"): t for t in tasks}
    for t in tasks:
        if t.get("type") not in db.TASK_TYPES:
            raise ValueError(f"task {t.get('id')}: unknown type {t.get('type')!r}")
        if not isinstance(t.get("spec"), dict):
            raise ValueError(f"task {t.get('id')}: spec must be an object")
        for dep in t.get("depends_on", []):
            if dep not in known:
                raise ValueError(f"task {t.get('id')}: unknown dependency {dep!r}")
    # structural rules — a graph that violates these cannot build a game
    kinds = [t["type"] for t in tasks]
    if kinds.count("assemble") != 1:
        raise ValueError("exactly one assemble task is required")
    if "code" not in kinds:
        raise ValueError('at least one "code" task is required — the game has no'
                         " scripts, scenes or player controller without them")
    for t in tasks:
        spec = t["spec"]
        if t["type"] == "design_2d" and not spec.get("prompt"):
            raise ValueError(f'design_2d {t["id"]}: spec needs a "prompt"')
        if t["type"] == "design_3d":
            if not spec.get("prompt"):
                raise ValueError(f'design_3d {t["id"]}: spec needs a "prompt"')
            c = spec.get("concept_from")
            if not spec.get("ref_image") and by_id.get(c, {}).get("type") != "design_2d":
                raise ValueError(f'design_3d {t["id"]}: the 3D generator is image-'
                                 'conditioned — set "concept_from" to a design_2d task'
                                 ' id (or "ref_image" to a provided reference image)')
        if t["type"] == "rig_animate":
            m = spec.get("mesh_from")
            if by_id.get(m, {}).get("type") != "design_3d":
                raise ValueError(f'rig_animate {t["id"]}: mesh_from must name a'
                                 " design_3d task id (the mesh to rig)")
        if t["type"] == "code" and not spec.get("file"):
            raise ValueError(f'code {t["id"]}: spec needs a "file" path')
    # cycle check (Kahn): a cyclic graph would deadlock the scheduler forever
    indeg = {t["id"]: len(t.get("depends_on", [])) for t in tasks}
    dependents: dict[str, list[str]] = {}
    for t in tasks:
        for d in t.get("depends_on", []):
            dependents.setdefault(d, []).append(t["id"])
    queue = [i for i, n in indeg.items() if n == 0]
    seen = 0
    while queue:
        seen += 1
        for nxt in dependents.get(queue.pop(), []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if seen != len(tasks):
        raise ValueError("the dependency graph contains a cycle")


def insert_tasks(conn, run_id: str, tasks: list[dict]) -> None:
    """Insert with run-scoped ids so decomposer ids never collide across runs.
    Spec values that reference another task (concept_from, mesh_from, ...) are
    remapped too, so executors can resolve them via dep_outputs."""
    id_map = {t["id"]: f"{run_id}-{t['id']}" for t in tasks}

    def remap(value):
        if isinstance(value, str):
            return id_map.get(value, value)
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        return value

    # one transaction: resume treats "has tasks" as "has ALL tasks", so a crash
    # mid-insert must leave zero rows, never a partial graph
    db.add_tasks(conn, run_id,
                 [(id_map[t["id"]], t["type"], remap(t["spec"]),
                   [id_map[d] for d in t.get("depends_on", [])]) for t in tasks])
    print(json.dumps({"run": run_id, "tasks": len(tasks)}))
