"""Stage 1: resident router LLM turns instruction.md into a typed, dependency-aware
task list, which is inserted into the SQLite queue."""
import json

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
- code tasks for the Godot project; exactly one final "assemble" task depending on everything.
- When the game description contains combat timing tables (frame data), copy them
  VERBATIM into one code task with "file": "scripts/combat_sim.gd" as
  "frame_data": {{<move>: {{"startup": n, "active": n, "hitstun": n,
  "knockback": [x, y], "tolerance": n}}}}. Frame counts are at 60fps. That task
  will be graded by a headless simulation against these exact numbers.
- Reference images uploaded by the user: {ref_note}

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
            repair_task_list(tasks)
            validate_task_list(tasks)
            return tasks
        except (ValueError, KeyError, TypeError) as e:
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


def repair_task_list(tasks) -> None:
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
        t["depends_on"] = sorted(deps)


def validate_task_list(tasks) -> None:
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("decomposition is not a non-empty JSON array")
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
            if c and by_id.get(c, {}).get("type") != "design_2d":
                raise ValueError(f'design_3d {t["id"]}: concept_from must name a'
                                 " design_2d task id")
        if t["type"] == "rig_animate":
            m = spec.get("mesh_from")
            if by_id.get(m, {}).get("type") != "design_3d":
                raise ValueError(f'rig_animate {t["id"]}: mesh_from must name a'
                                 " design_3d task id (the mesh to rig)")
        if t["type"] == "code" and not spec.get("file"):
            raise ValueError(f'code {t["id"]}: spec needs a "file" path')


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

    for t in tasks:
        db.add_task(conn, run_id, t["type"], remap(t["spec"]),
                    depends_on=[id_map[d] for d in t.get("depends_on", [])],
                    task_id=id_map[t["id"]])
    print(json.dumps({"run": run_id, "tasks": len(tasks)}))
