"""Stage 1: resident router LLM turns instruction.md into a typed, dependency-aware
task list, which is inserted into the SQLite queue."""
import json
import re
from pathlib import PurePath

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
 {{"id": "crate_art", "type": "design_2d", "depends_on": [],
   "spec": {{"prompt": "a wooden crate, single isolated object, centered, plain background",
             "purpose": "concept for a level prop"}}}},
 {{"id": "crate_mesh", "type": "design_3d", "depends_on": ["crate_art"],
   "spec": {{"prompt": "the wooden crate as a game-ready prop mesh",
             "concept_from": "crate_art"}}}},
 {{"id": "player", "type": "code", "depends_on": ["hero_anim"],
   "spec": {{"file": "scripts/player.gd",
             "description": "CharacterBody3D controller: WASD moves, camera follows"}}}},
 {{"id": "level", "type": "code", "depends_on": [],
   "spec": {{"file": "scenes/main.tscn",
             "description": "main scene: ground plane, light, spawns the player"}}}},
 {{"id": "build", "type": "assemble", "depends_on": ["hero_art", "hero_mesh",
   "hero_anim", "crate_art", "crate_mesh", "player", "level"],
   "spec": {{"export_preset": "Windows Desktop"}}}}
]

Game description:
---
{instruction}
---

Reply with ONLY a JSON array of tasks."""


_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)
_SCENE_WORDS = ("hall", "room", "interior", "environment", "scene",
                "landscape", "backdrop", "mural", "skybox", "vista", "level")

# An optional "Style: ..." line in instruction.md is the run's art direction. It
# is appended VERBATIM and UNCONDITIONALLY to every design_2d prompt, which is
# what makes sibling props converge: the seed and negative prompt are already
# shared, so with a byte-identical style tail the only thing left differing
# between two props is the subject phrase itself.
_STYLE_RE = re.compile(r"^[ \t]*style[ \t]*:[ \t]*(.+?)[ \t]*$", re.I | re.M)

# Every concept that feeds TRELLIS needs these, exactly once. The wording is
# deliberately emphatic: the mild version ("single isolated object, centered,
# plain white background") lost to the style tail and came back as a seated
# portrait on a throne. "studio cutout" and the explicit no-* terms are what
# hold. Verified by A/B render on the box.
_ISO_COMMON = ("isolated on a plain flat white background", "studio cutout",
               "no scenery", "no furniture", "no architecture")
# A character must be framed head-to-toe or the mesh arrives cropped at the
# waist; a bookshelf is not "standing, head to toe", so props get their own.
_ISO_CHARACTER = ("full body", "head to toe", "standing", "arms away from body")
_ISO_PROP = ("single isolated object", "centered")


def _iso_clauses(is_character: bool) -> tuple:
    return (_ISO_CHARACTER if is_character else _ISO_PROP) + _ISO_COMMON

# Applied even when the instruction names no style. RENDERING TREATMENT ONLY --
# and deliberately FANTASY, not photoreal: RealVisXL's prior is photographic, so
# without "stylized realism / heroic exaggerated proportions" it renders an
# ordinary man in costume rather than a game character. A/B-verified on the box.
# every word here must describe how a surface is rendered, never what is around
# it. Scene nouns ("candle-lit", "weathered stone") reliably beat "plain white
# background" in the sampler and put the subject back in a room, which is
# exactly what TRELLIS cannot reconstruct. Verified by A/B render on the box.
# ponytail: text-level style lock. Upgrade path is IP-Adapter conditioning in
# sdxl.json if props still diverge in palette after this ships.
_QUALITY_TAIL = ("dark fantasy video game character concept art, heroic "
                 "exaggerated proportions, stylized realism, cinematic key "
                 "lighting, highly detailed, intricate surface detail, "
                 "physically based materials, sharp focus")


def _drop_unused_concepts(tasks: list) -> None:
    """Remove concept art that a referenced mesh will never look at."""
    by_id = {x.get("id"): x for x in tasks if isinstance(x, dict)}
    for t in list(tasks):
        if not (isinstance(t, dict) and t.get("type") == "design_3d"
                and t["spec"].get("ref_image")):
            continue
        cid = t["spec"].get("concept_from")
        concept = by_id.get(cid)
        if not (isinstance(concept, dict) and concept.get("type") == "design_2d"):
            continue
        # keep it if anything OTHER than this mesh still needs it. "assemble"
        # never counts: a later repair makes it depend on every task in the
        # graph, so it would veto every drop.
        others = [o for o in tasks if isinstance(o, dict) and o is not t
                  and o.get("type") != "assemble"
                  and (o.get("spec", {}).get("concept_from") == cid
                       or cid in (o.get("depends_on") or []))]
        if others:
            continue
        t["spec"].pop("concept_from", None)
        t["depends_on"] = [d for d in (t.get("depends_on") or []) if d != cid]
        tasks.remove(concept)
        for o in tasks:
            if isinstance(o, dict):
                o["depends_on"] = [d for d in (o.get("depends_on") or []) if d != cid]


def _apply_style(tasks, instruction: str) -> None:
    """Append the run's art direction to every design_2d prompt.

    Concept art that feeds a mesh gets the TREATMENT TAIL ONLY. The operator's
    "Style:" line names a place ("dark candle-lit baroque hall"), and a place
    beats "plain white background" in the sampler -- the subject comes back
    sitting in a room and TRELLIS then models the room. Those concepts exist to
    be clean reconstruction input, not to look like the game; the game's mood
    comes from the scaffold's lighting. Backdrop art, which no mesh is built
    from, gets the full style line.
    """
    m = _STYLE_RE.search(instruction or "")
    styled = f"{m.group(1).rstrip('.')}, {_QUALITY_TAIL}" if m else _QUALITY_TAIL
    feeds_a_mesh = {t["spec"].get("concept_from") for t in tasks
                    if isinstance(t, dict) and t.get("type") == "design_3d"}
    for t in tasks:
        if not (isinstance(t, dict) and t.get("type") == "design_2d"):
            continue
        p = (t.get("spec", {}).get("prompt") or "").rstrip().rstrip(",")
        # unconditional within each group: a gate here is how two props in one
        # run end up with different tails, the divergence this exists to remove
        tail = _QUALITY_TAIL if t.get("id") in feeds_a_mesh else styled
        t["spec"]["prompt"] = f"{p}, {tail}" if p else tail


def _coerce_task_list(parsed):
    """R1-style routers wrap the array ({"tasks": [...]}), emit one bare task,
    or key the whole graph by task id ({"build": {"type": "assemble", ...}})."""
    if isinstance(parsed, dict):
        lists = [v for v in parsed.values()
                 if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
        if len(lists) == 1:
            return lists[0]
        if "id" in parsed and "type" in parsed:
            return [parsed]
        if parsed and all(isinstance(v, dict) and ("type" in v or "id" in v)
                          for v in parsed.values()):
            return [dict(v, id=v.get("id", k)) for k, v in parsed.items()]
    return parsed


def decompose(router: LlamaServer, instruction: str, reference_images: list[str],
              temperature: float = 0.6, max_tokens: int = 4096, on_token=None) -> list[dict]:
    ref_note = ", ".join(reference_images) if reference_images else "none"
    prompt = DECOMPOSE_PROMPT.format(instruction=instruction, ref_note=ref_note)
    messages = [{"role": "user", "content": prompt}]
    last_err, reply = None, ""
    for _ in range(3):  # a small router gets structure wrong; feed the error back
        raw = router.chat(messages, temperature=temperature,
                          max_tokens=max_tokens, on_token=on_token)
        # an R1 distill precedes its answer with a <think> block whose example
        # snippets parse as JSON and hijack extraction — strip before parsing,
        # and never echo the block back on retry (it only breeds more thinking)
        reply = _THINK_RE.sub("", raw).strip() or raw
        try:
            tasks = _coerce_task_list(extract_json(reply))
            repair_task_list(tasks, reference_images, instruction)
            validate_task_list(tasks)
            return tasks
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            last_err = e
            messages = messages[:1] + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That task list is invalid: {e}. "
                 "Reply with ONLY the corrected full JSON array."}]
    raise ValueError(f"decomposition failed after 3 attempts: {last_err}; "
                     f"last reply tail: {reply[-500:]!r}")


PATCH_PROMPT = """You are patching an EXISTING game. Do NOT rebuild what the change
doesn't touch.

Here is what the game already contains. Each line is:
  id — type — asked for: <what this task was told to make>
                observed: <what it ACTUALLY produced, measured from its files>

{manifest}

"observed" is ground truth and "asked for" is only a request. They disagree
often: a rig whose motion lookup missed still lists "idle, walk, run" in what it
asked for, while what it observed is three procedural cycles.

CHECK BEFORE YOU CHANGE. Work out which artifacts the instruction is about, read
their observed facts, and target the one whose observed facts show the problem:
- "make the animation more realistic" is about the rig_animate whose clips are
  observed as procedural — not the mesh and not the concept art.
- "the character does not move" is about a rig observed as NOT SKINNED.
- "the shelf looks wrong" is about a mesh observed as a FLAT PLANE.
If an observed line is empty, that artifact was never measured — do not invent
what it contains. If the instruction names something no line mentions, ADD it.
Never assume an artifact holds something its observed facts do not show.

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
    from .patch import check_patch_grounding, validate_patch_list
    lines = "\n".join(
        f"{m['id']} — {m['type']} — asked for: {m['summary']}\n"
        f"{' ' * 4}observed: {m.get('observed') or '(not measured)'}"
        for m in manifest_rows)
    ref_note = ", ".join(reference_images) if reference_images else "none"
    prompt = PATCH_PROMPT.format(manifest=lines, instruction=instruction, ref_note=ref_note)
    messages = [{"role": "user", "content": prompt}]
    ids = {m["id"] for m in manifest_rows}
    last_err, reply = None, ""
    # same three-strike loop the fresh decomposition has had all along: a small
    # router gets this wrong first time, and the grounding error is written to be
    # read BY the router, so feeding it back is what makes the check useful
    for _ in range(3):
        raw = router.chat(messages, temperature=temperature,
                          max_tokens=max_tokens, on_token=on_token)
        reply = _THINK_RE.sub("", raw).strip() or raw
        try:
            patch_tasks = extract_json(reply)
            validate_patch_list(patch_tasks, ids)
            check_patch_grounding(patch_tasks, manifest_rows, instruction)
            return patch_tasks
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            last_err = e
            messages = messages[:1] + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That patch is wrong: {e} "
                 "Reply with ONLY the corrected full JSON array."}]
    raise ValueError(f"patch decomposition failed after 3 attempts: {last_err}; "
                     f"last reply tail: {reply[-500:]!r}")


def _repair_ref_image(spec: dict, reference_images: list[str]) -> None:
    """The router mangles paths; keep ref_image only if it resolves to a real ref
    (exact, or unique basename-substring match), else drop it silently."""
    ref = spec.get("ref_image")
    if not ref or not isinstance(ref, str):
        spec.pop("ref_image", None)
        return
    if ref in reference_images:
        return
    frag = PurePath(ref.replace("\\", "/")).name.lower()
    hits = [r for r in reference_images
            if frag and frag in PurePath(r.replace("\\", "/")).name.lower()]
    if len(hits) == 1:
        spec["ref_image"] = hits[0]
    else:
        spec.pop("ref_image", None)


def repair_task_list(tasks, reference_images: list[str] | None = None,
                     instruction: str = "") -> None:
    """Deterministic fixes for mistakes every small router makes: spec references
    (concept_from/mesh_from) imply dependencies, and assemble depends on everything."""
    if not isinstance(tasks, list):
        return
    ids = {t.get("id") for t in tasks if isinstance(t, dict)}
    for t in tasks:                 # normalize early: the passes below read spec
        if isinstance(t, dict) and not isinstance(t.get("spec"), dict):
            t["spec"] = {}          # routers omit assemble's empty spec constantly
    # the router regularly forgets that prop concepts exist to BECOME meshes: a
    # design_2d nothing builds from, whose subject is a thing (not a space),
    # gets a synthesized design_3d. Spaces stay 2D — TRELLIS cannot build rooms,
    # so scene-headed prompts ("a vast library hall") are left as backdrop art;
    # only the head before a location preposition decides ("a bookshelf IN a
    # library hall" is still a bookshelf).
    consumed = {t["spec"].get("concept_from") for t in tasks
                if isinstance(t, dict) and t.get("type") == "design_3d"}
    synthesized = []
    for t in tasks:
        if not (isinstance(t, dict) and t.get("type") == "design_2d"):
            continue
        if not t.get("id") or t["id"] in consumed:
            continue
        p = str(t["spec"].get("prompt", ""))
        head = re.split(r"\b(?:in|inside|within|at|among)\b", p.lower())[0]
        if not p or any(w in head for w in _SCENE_WORDS):
            continue
        mid = f"{t['id']}_mesh"
        if mid in ids:
            continue
        synthesized.append({"id": mid, "type": "design_3d", "depends_on": [t["id"]],
                            "spec": {"prompt": p + ", game-ready prop mesh",
                                     "concept_from": t["id"]}})
        ids.add(mid)
    tasks.extend(synthesized)
    # a router often rigs the ART, not the mesh: retarget mesh_from through the
    # design_3d built from that design_2d (synthesized above if it was missing)
    by_id = {t.get("id"): t for t in tasks if isinstance(t, dict)}
    for t in tasks:
        if not (isinstance(t, dict) and t.get("type") == "rig_animate"):
            continue
        m = t["spec"].get("mesh_from")
        if isinstance(m, str) and by_id.get(m, {}).get("type") == "design_2d":
            mesh = next((x["id"] for x in tasks if isinstance(x, dict)
                         and x.get("type") == "design_3d"
                         and x["spec"].get("concept_from") == m), None)
            if mesh:
                t["spec"]["mesh_from"] = mesh
    for t in tasks:
        if not isinstance(t, dict):
            continue
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
            if fd is not None and not isinstance(fd, dict):
                del t["spec"]["frame_data"]   # string/list junk ("frame_data")
            if isinstance(fd, dict):
                moves = {k: v for k, v in fd.items()
                         if isinstance(v, dict) and "startup" in v}
                # the grader hard-loads scripts/combat_sim.gd, so frame_data on
                # any other file is either hallucinated timing (a router once
                # gave "idle" a startup) or unfulfillable grading — drop both
                if not str(t["spec"].get("file", "")).endswith("combat_sim.gd"):
                    moves = {}
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
    # user references must actually condition something — the router drops them
    # nondeterministically, and a dropped character ref silently costs identity.
    # File names carry intent (the drop convention names refs *_char / *_env).
    refs = reference_images or []
    def _named(words):
        return [r for r in refs
                if any(w in PurePath(r.replace("\\", "/")).name.lower() for w in words)]
    d3 = [t for t in tasks if isinstance(t, dict) and t.get("type") == "design_3d"]
    # per-task, not graph-wide: the old `not any(... for t in d3)` guard meant one
    # prop mesh carrying a ref_image starved EVERY character mesh of its reference
    char_refs = _named(("char", "hero", "player", "figure", "portrait"))
    rig_meshes = {t["spec"].get("mesh_from") for t in tasks
                  if isinstance(t, dict) and t.get("type") == "rig_animate"}
    # each character mesh takes a DIFFERENT figure: one sheet per character when
    # the operator dropped several, otherwise successive figures off the shared
    # sheet. Handing every hero char_refs[0] built the same character twice.
    char_meshes = [t for t in d3 if t.get("id") in rig_meshes
                   and not t["spec"].get("ref_image")]
    for i, t in enumerate(char_meshes):
        if not char_refs:
            break
        if i < len(char_refs):
            t["spec"]["ref_image"] = char_refs[i]
        else:
            t["spec"]["ref_image"] = char_refs[0]
            t["spec"]["ref_subject"] = i
    d2_backdrops = [t for t in tasks if isinstance(t, dict)
                    and t.get("type") == "design_2d" and t.get("id") not in
                    {x["spec"].get("concept_from") for x in d3}]
    if d2_backdrops and not any(t["spec"].get("ref_image") for t in d2_backdrops):
        env_refs = _named(("env", "background", "backdrop", "scene"))
        for t in d2_backdrops:
            if env_refs:
                t["spec"]["ref_image"] = env_refs[0]
    # concept art that feeds TRELLIS must be ONE isolated object: img2img from a
    # scene reference forces whole-room composition into the concept, and TRELLIS
    # turns a room image into polygon noise. Style lives in the prompt text; the
    # scene reference stays only on design_2d art that no mesh is built from.
    by_id = {t.get("id"): t for t in tasks if isinstance(t, dict)}
    for t in tasks:
        if not (isinstance(t, dict) and t.get("type") == "design_3d"):
            continue
        c = by_id.get(t.get("spec", {}).get("concept_from"))
        if isinstance(c, dict) and c.get("type") == "design_2d":
            c["spec"].pop("ref_image", None)
            # "plain white background" is load-bearing: concept art reaches
            # TRELLIS UNCROPPED (executors.design_3d only crops a ref_image), so
            # the matte has to come from the render itself. Append per-clause:
            # a blanket `if "isolated" not in p` gate let a router prompt that
            # merely said "single isolated object" skip the background clause
            # too, leaving that one prop with no guaranteed matte.
            # "product render" used to ride along here and was the pipeline's
            # only unconditional style instruction to SDXL -- a clean-studio-
            # catalogue token, applied to characters as well. Removed: the look
            # now comes from _apply_style, which the operator controls.
            p = c["spec"].get("prompt", "")
            missing = [cl for cl in _iso_clauses(t.get("id") in rig_meshes)
                       if cl not in p.lower()]
            if missing:
                c["spec"]["prompt"] = (p.rstrip().rstrip(",") + ", "
                                       + ", ".join(missing))
    # A design_3d with a ref_image NEVER opens its concept art: executors.design_3d
    # takes the cropped reference and returns before touching concept_from. So the
    # concept task is pure waste -- SDXL time spent on an image nothing consumes,
    # and a drawer full of bad regenerations of a character the user already drew.
    # Drop it, unless something else in the graph still depends on it.
    _drop_unused_concepts(tasks)
    # style goes on LAST so every design_2d prompt ends with the same tail, no
    # matter which earlier repair added words to it
    _apply_style(tasks, instruction)
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
