"""Headless Blender rig + animate stage. Run by pipeline/adapters/motion.py as:

    blender --background --python templates/blender_motion.py -- '<json args>'

args (JSON): mesh, body_plan("humanoid"|"nonhumanoid"), animations[list],
extras[list], out_dir, cmu_dir, unirig, kimodo_url.

Design guarantee: this ALWAYS produces one animated <clip>.glb per requested clip
using nothing but Blender itself (the procedural floor). The optional local tools
are folded in when their paths/urls are configured, and any failure in them falls
back to procedural — so an arbitrary creature ("dragon + amoeba") still ships a
moving model even with zero AI tools installed. Nothing here touches the network
except an explicit, configured Kimodo localhost call.

This file needs bpy; it only runs inside Blender, so it is exercised on the target
box, not by the pipeline's unit tests (which mock the Blender subprocess).
"""
import json
import math
import os
import re
import sys

import bpy  # provided by Blender's own Python; absent in the pipeline venv
import mathutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import punch_mining  # noqa: E402  -- sibling module, path fixed just above

ARGS = json.loads(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else {}
FPS = 30


# --- scene / import --------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        (getattr(bpy.ops.wm, "obj_import", None) or bpy.ops.import_scene.obj)(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise RuntimeError(f"unsupported mesh format: {ext}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh object after import")
    # join to a single object so one armature drives the whole creature
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    # normalize: feet on the ground at origin, character-sized (the game scene
    # assumes ~1.8m; a raw TRELLIS mesh arrives at arbitrary scale/offset)
    bb = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    height = max(v.z for v in bb) - min(v.z for v in bb)
    if height > 0:
        s = 1.8 / height
        obj.scale = (obj.scale[0] * s, obj.scale[1] * s, obj.scale[2] * s)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bb = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    obj.location.x -= (max(v.x for v in bb) + min(v.x for v in bb)) / 2
    obj.location.y -= (max(v.y for v in bb) + min(v.y for v in bb)) / 2
    obj.location.z -= min(v.z for v in bb)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    bpy.context.view_layer.update()  # refresh matrix_world before rigging reads it
    return obj


# --- rigging ---------------------------------------------------------------

def try_unirig(mesh_obj, unirig, out_dir):
    """Auto-rig any topology with UniRig if configured. Returns the armature or None.
    Kept behind a broad except: a missing/failing UniRig must not sink the stage."""
    if not unirig or not os.path.isdir(unirig):
        return None
    try:
        import subprocess
        rigged = os.path.join(out_dir, "_unirig.glb")
        export_glb(rigged)  # hand UniRig the imported mesh
        # UniRig's own CLI writes a rigged glb; convention: run.py <in> <out>
        # sys.executable inside Blender is Blender's BUNDLED python, which has no
        # UniRig deps — use the system python (override with UNIRIG_PYTHON)
        py = os.environ.get("UNIRIG_PYTHON", "python")
        subprocess.run([py, os.path.join(unirig, "run.py"), rigged, rigged],
                       check=True, cwd=unirig, timeout=1200)
        reset_scene()
        bpy.ops.import_scene.gltf(filepath=rigged)
        arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
        return arms[0] if arms else None
    except Exception as e:  # noqa: BLE001 — fall back to procedural rig
        print(f"[motion] UniRig unavailable, procedural rig instead: {e}")
        return None


def procedural_rig(mesh_obj, body_plan):
    """Fit a minimal armature to the mesh bounding box. Humanoid gets a biped
    template; anything else gets a head->tail spine chain that suits fish, blobs,
    dragons, quadruped torsos — whatever shape arrived."""
    bb = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
    xs = [p.x for p in bb]; ys = [p.y for p in bb]; zs = [p.z for p in bb]
    lo = (min(xs), min(ys), min(zs)); hi = (max(xs), max(ys), max(zs))
    cx = (lo[0] + hi[0]) / 2; cy = (lo[1] + hi[1]) / 2; cz = (lo[2] + hi[2]) / 2
    h = hi[2] - lo[2]

    arm_data = bpy.data.armatures.new("rig")
    arm = bpy.data.objects.new("rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")

    def bone(name, head, tail, parent=None):
        b = arm_data.edit_bones.new(name)
        b.head = head; b.tail = tail
        if parent:
            b.parent = parent; b.use_connect = False
        return b

    if body_plan == "humanoid":
        # CMU/Mixamo bone names, deliberately. retarget_onto() matches mocap to
        # rig by name, and the old names (pelvis/spine/arm.L/leg.L) hit exactly
        # ONE CMU bone ("head"), below the 4-bone thres_hold -- so every clip
        # silently fell through to procedural_clip's sine wave, which is why the
        # animation looked the way it did. These names match CMU directly.
        #
        # Separate upper/fore arm and thigh/shin, too: with one shoulder-to-hand
        # bone there is no elbow or knee to bend, so any arm motion swung the
        # whole limb rigidly from the shoulder.
        def z(f):
            return lo[2] + h * f
        hips = bone("Hips", (cx, cy, z(0.50)), (cx, cy, z(0.56)))
        spine = bone("Spine", hips.tail, (cx, cy, z(0.68)), hips)
        chest = bone("Spine1", spine.tail, (cx, cy, z(0.80)), spine)
        neck = bone("Neck", chest.tail, (cx, cy, z(0.86)), chest)
        bone("Head", neck.tail, (cx, cy, hi[2]), neck)
        for side, sx in (("Left", -1), ("Right", 1)):
            sh = bone(f"{side}Shoulder", (cx, cy, z(0.82)),
                      (cx + sx * h * 0.10, cy, z(0.82)), chest)
            up = bone(f"{side}Arm", sh.tail, (cx + sx * h * 0.22, cy, z(0.72)), sh)
            fore = bone(f"{side}ForeArm", up.tail,
                        (cx + sx * h * 0.30, cy, z(0.60)), up)
            bone(f"{side}Hand", fore.tail, (cx + sx * h * 0.34, cy, z(0.55)), fore)
            thigh = bone(f"{side}UpLeg", (cx + sx * h * 0.08, cy, z(0.50)),
                         (cx + sx * h * 0.09, cy, z(0.28)), hips)
            shin = bone(f"{side}Leg", thigh.tail,
                        (cx + sx * h * 0.09, cy, z(0.05)), thigh)
            bone(f"{side}Foot", shin.tail,
                 (cx + sx * h * 0.09, cy - h * 0.06, lo[2]), shin)
    else:
        # generic articulated spine along the longest horizontal axis
        n = 5
        prev = None
        span = max(hi[0] - lo[0], hi[1] - lo[1], 1e-4)
        along_x = (hi[0] - lo[0]) >= (hi[1] - lo[1])
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            # run the spine through the mesh's vertical center (cz), not cy —
            # otherwise the rig lands off an upright creature and auto-weights fail
            if along_x:
                head = (lo[0] + span * t0, cy, cz); tail = (lo[0] + span * t1, cy, cz)
            else:
                head = (cx, lo[1] + span * t0, cz); tail = (cx, lo[1] + span * t1, cz)
            prev = bone(f"seg.{i}", head, tail, prev)

    bpy.ops.object.mode_set(mode="OBJECT")
    # skin: automatic weights binds arbitrary geometry to whatever bones we made
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True); arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    if not weight_total(mesh_obj):
        # Bone heat needs clean manifold geometry and TRELLIS output is neither.
        # It fails with "failed to find solution for one or more bones", still
        # creates the vertex groups, and leaves EVERY vertex at zero weight --
        # on the reference character, 0 of 36994. The glb then exports with
        # JOINTS_0/WEIGHTS_0 but no skin at all, so Godot imports a static mesh
        # and the character slides through the level frozen in its rest pose.
        print("[motion] bone heat produced no weights — binding by distance")
        bind_by_distance(mesh_obj, arm)
    return arm


def weight_total(mesh_obj):
    """Sum of every vertex-group weight. Zero means nothing is actually bound,
    however many groups exist."""
    return sum(g.weight for v in mesh_obj.data.vertices for g in v.groups)


def bind_by_distance(mesh_obj, arm, blend=4, falloff=4, cutoff=2.0):
    """Weight each vertex to its `blend` nearest bone segments, with a steep
    inverse-`falloff` power and a hard `cutoff` on how much farther than the
    nearest bone an influence may be.

    The cutoff is what makes a joint local. Without it a vertex takes weight from
    its two nearest bones however far away the second one is, so a chest vertex
    was part-owned by an arm and bending one joint dragged the whole figure.
    Anything more than `cutoff` x the nearest distance now contributes nothing.

    Purely geometric, so unlike bone heat it cannot fail on self-intersecting or
    non-manifold meshes -- which is every mesh this pipeline generates. Quality is
    below a solved heat map, but a crude bind that animates beats a perfect one
    that does not exist.
    """
    from mathutils import Vector

    to_local = mesh_obj.matrix_world.inverted() @ arm.matrix_world
    bones = [(b.name, to_local @ b.head_local, to_local @ b.tail_local)
             for b in arm.data.bones if b.use_deform]
    if not bones:
        return
    groups = {name: (mesh_obj.vertex_groups.get(name)
                     or mesh_obj.vertex_groups.new(name=name))
              for name, _h, _t in bones}

    def seg_dist(p, a, b):
        ab = b - a
        d2 = ab.length_squared
        t = 0.0 if d2 == 0.0 else max(0.0, min(1.0, (p - a).dot(ab) / d2))
        return (p - (a + ab * t)).length

    for v in mesh_obj.data.vertices:
        near = sorted(((seg_dist(v.co, h, t), name) for name, h, t in bones))[:blend]
        limit = near[0][0] * cutoff + 1e-4
        near = [(d, name) for d, name in near if d <= limit]
        # +1e-4 keeps a vertex sitting exactly on a bone from dividing by zero
        raw = [(1.0 / (d + 1e-4) ** falloff, name) for d, name in near]
        total = sum(w for w, _ in raw) or 1.0
        for w, name in raw:
            groups[name].add([v.index], w / total, "REPLACE")


# --- animation -------------------------------------------------------------

_CMU_INDEX_CACHE = {}
_TRIAL_RE = re.compile(r"^(\d+_\d+)\s+(.*\S)\s*$")

# Words the index uses for motions we ask for under a different name. Kept small
# and literal on purpose -- this is a lookup aid, not a synonym engine.
_CLIP_SYNONYMS = {
    "run": ("run", "jog"),
    "walk": ("walk",),
    "idle": ("idle", "stand"),
    "attack": ("punch", "strike", "boxing"),
    "punch": ("punch", "strike", "boxing"),
    "kick": ("kick",),
    "sword": ("sword", "swordplay"),
    "jump": ("jump", "leap", "hop"),
    "dodge": ("sidestep", "duck", "dodge", "evade"),
    "duck": ("duck", "crouch"),
    "roll": ("roll",),
    "spin": ("spin", "twirl"),
    "block": ("block", "guard"),
    "turn": ("turn",),
    "stagger": ("stagger", "limp"),
    "getup": ("get up", "getup"),
    "fall": ("fall",),
}

# "punch_2" is the second DISTINCT punch clip, not a second copy of the first.
# The moveset needs followups that never look identical, and the library has 18
# punches / 28 kicks / 18 sidesteps to draw them from.
_VARIANT_RE = re.compile(r"^(.*?)[._](\d+)$")


def cmu_index(cmu_dir):
    """trial id -> description, parsed from the database's own index file.

    The BVH conversions are named by trial (01_01.bvh), so a clip called "walk"
    matched no filename and the mocap path could never fire -- every clip fell
    through to the procedural sine wave. The shipped index is what names them.
    """
    if cmu_dir in _CMU_INDEX_CACHE:
        return _CMU_INDEX_CACHE[cmu_dir]
    entries = {}
    # Search the PARENT too. The distribution puts the BVH files in a data/
    # subdirectory and the index beside it, not inside it, so pointing cmu_dir
    # at data/ -- which is what you must do for the clips to be found -- hid the
    # index completely: every walk, run and idle in this project has been a
    # procedural sine wave, not mocap, for exactly this reason.
    roots = [cmu_dir, os.path.dirname(os.path.abspath(cmu_dir or "."))]
    for root in roots:
      for base, _dirs, files in os.walk(root):
        for f in files:
            if "index" not in f.lower() or not f.lower().endswith(".txt"):
                continue
            with open(os.path.join(base, f), encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = _TRIAL_RE.match(line.replace("\t", " ").strip()
                                        if "\t" in line else line.rstrip("\n"))
                    if m:
                        entries.setdefault(m.group(1), m.group(2).lower())
      if entries:
          break        # the nearer index wins; do not let the parent shadow it
    _CMU_INDEX_CACHE[cmu_dir] = entries
    return entries


def find_cmu_clip(cmu_dir, clip):
    """A BVH whose CMU index description mentions `clip`, else a name match.

    CMU's own advice is to prefer higher-numbered subjects ("the lower numbers
    contain some of our earliest motion capture sessions"), so candidates are
    ranked that way and the best one is returned.
    """
    if not cmu_dir or not os.path.isdir(cmu_dir):
        return None
    paths = {}
    for base, _dirs, files in os.walk(cmu_dir):
        for f in files:
            if f.lower().endswith(".bvh"):
                paths.setdefault(os.path.splitext(f)[0], os.path.join(base, f))
    # a curated/renamed file still wins, exactly as before
    for stem, path in paths.items():
        if clip.lower() in stem.lower() and not _TRIAL_RE.match(stem + " x"):
            return path
    base, variant = clip.lower(), 0
    m = _VARIANT_RE.match(base)
    if m and m.group(1) in _CLIP_SYNONYMS:
        base, variant = m.group(1), int(m.group(2))
    words = _CLIP_SYNONYMS.get(base, (base,))
    hits = [trial for trial, desc in cmu_index(cmu_dir).items()
            if trial in paths and any(re.search(rf"\b{re.escape(w)}", desc) for w in words)]
    if not hits:
        return None
    hits.sort(key=lambda t: (-int(t.split("_")[0]), int(t.split("_")[1])))
    # spread variants across SUBJECTS first: two clips from one performer in one
    # session look like the same move twice, which is the thing to avoid here
    by_subject = {}
    for t in hits:
        by_subject.setdefault(t.split("_")[0], []).append(t)
    spread = [t for i in range(max(len(v) for v in by_subject.values()))
              for v in by_subject.values() if i < len(v) for t in (v[i],)]
    return paths[spread[variant % len(spread)]]


PROVENANCE = {}      # clip -> how it was actually produced, written beside the glb
# clip -> (start, end) in SECONDS. Seconds, not frames: the punch library is
# mined from 120 fps CMU data, while retarget_onto imports with use_fps_scale so
# the sampled track is at the scene's 30 -- a 4841-frame trial arrives as 1210.
# Frame indices from the survey would land a quarter of the way into the clip.
CLIP_WINDOWS = {}

# CMU trials indexed as boxing. Every one is continuous shadowboxing, so the
# named punches are cut out of them by punch_mining rather than looked up.
PUNCH_TRIALS = ("13_17", "13_18", "14_01", "14_02", "14_03", "15_13")
_PUNCH_TRACK_BONES = ("LeftHand", "RightHand", "LeftArm", "RightArm", "Hips")
_PUNCH_CACHE = {}
_PUNCH_PAD_S = 0.06     # guard held either side of the throw


def _bvh_paths(cmu_dir):
    out = {}
    for base, _dirs, files in os.walk(cmu_dir or ""):
        for f in files:
            if f.lower().endswith(".bvh"):
                out.setdefault(os.path.splitext(f)[0], os.path.join(base, f))
    return out


def bvh_fps(bvh_path, default=FPS):
    """Frames per second from the BVH's own header.

    CMU ships at 120 fps. Mining at the module's 30 shrinks every duration
    bound by a factor of four: the windup cap becomes 0.075 s, so a punch is cut
    to a 20-frame stub and the classifier reads whatever fragment that leaves.
    """
    try:
        with open(bvh_path, errors="replace") as fh:
            for line in fh:
                if line.strip().lower().startswith("frame time:"):
                    dt = float(line.split(":", 1)[1])
                    return round(1.0 / dt) if dt > 0 else default
    except Exception:  # noqa: BLE001
        pass
    return default


def _sample_tracks(bvh_path):
    """World positions per frame for the joints the punch miner needs.

    The scene's frame range is Blender's default 1..250 and has nothing to do
    with the clip; reading it instead of the action's own range truncated every
    forty-second boxing trial to two seconds, which is why the first pass over
    real data found four punches in a session containing fifty.
    """
    before = set(bpy.context.scene.objects)
    try:
        bpy.ops.import_anim.bvh(filepath=bvh_path, target="ARMATURE",
                                rotate_mode="NATIVE", update_scene_fps=False)
    except Exception as e:  # noqa: BLE001
        print(f"[motion] punch source {os.path.basename(bvh_path)} failed: {e}")
        return None, 0, 0
    src = next((o for o in bpy.context.scene.objects
                if o not in before and o.type == "ARMATURE"), None)
    if src is None:
        return None, 0, 0
    try:
        if any(b not in src.pose.bones for b in _PUNCH_TRACK_BONES):
            return None, 0, 0
        act = src.animation_data.action if src.animation_data else None
        f0, f1 = (int(act.frame_range[0]), int(act.frame_range[1])) if act else (0, 0)
        tracks = {b: [] for b in _PUNCH_TRACK_BONES}
        for f in range(f0, f1 + 1):
            bpy.context.scene.frame_set(f)
            for b in tracks:
                tracks[b].append(tuple(src.matrix_world @ src.pose.bones[b].head))
        return tracks, f0, f1
    finally:
        act = src.animation_data.action if src.animation_data else None
        bpy.data.objects.remove(src, do_unlink=True)
        if act and act.users == 0:
            bpy.data.actions.remove(act)


def punch_library(cmu_dir, out_dir=""):
    """Every punch in every boxing trial, mined once and cached.

    Sampling six forty-second trials means thirty thousand frame_set calls, so
    the result is written to disk: a rig job asks for six named punches and must
    not pay for the survey six times, nor again on the next character.
    """
    if not cmu_dir or not os.path.isdir(cmu_dir):
        return []
    if cmu_dir in _PUNCH_CACHE:
        return _PUNCH_CACHE[cmu_dir]
    cache = os.path.join(out_dir or cmu_dir, "punch_library.json")
    try:
        with open(cache) as fh:
            found = json.load(fh)
        if found:
            _PUNCH_CACHE[cmu_dir] = found
            print(f"[motion] punch library: {len(found)} punches (cached)")
            return found
    except Exception:  # noqa: BLE001  -- absent or unreadable: mine it
        pass
    paths = _bvh_paths(cmu_dir)
    found = []
    for trial in PUNCH_TRIALS:
        if trial not in paths:
            continue
        tracks, f0, _f1 = _sample_tracks(paths[trial])
        if not tracks:
            continue
        fps = bvh_fps(paths[trial])
        for p in punch_mining.mine(tracks, fps):
            p.update(trial=trial, path=paths[trial], offset=f0, fps=fps)
            found.append(p)
        print(f"[motion] punch library: {trial} -> {len(found)} so far")
    _PUNCH_CACHE[cmu_dir] = found
    try:
        with open(cache, "w") as fh:
            json.dump(found, fh)
    except OSError:
        pass          # a read-only mocap directory is not a reason to fail
    return found


def punch_clip(clip, cmu_dir, out_dir=""):
    """(bvh_path, (first, last)) for a named punch, or None.

    Checked before the description search, because every boxing trial matches
    the word "boxing" and the description search would hand back forty unlabelled
    seconds of it for any of these names.
    """
    base, variant = clip.lower(), 0
    m = _VARIANT_RE.match(base)
    if m and m.group(1) in punch_mining.MOVESET:
        base, variant = m.group(1), int(m.group(2))
    if base not in punch_mining.MOVESET:
        return None
    hit = punch_mining.select(punch_library(cmu_dir, out_dir), base, variant)
    if not hit:
        return None
    fps = hit.get("fps") or bvh_fps(hit["path"])
    # a couple of frames of guard either side: starting exactly on the chamber
    # makes the punch begin mid-motion, which reads as a snap rather than a throw
    pad = _PUNCH_PAD_S
    return hit["path"], (max(0.0, hit["start"] / fps - pad),
                         (hit["end"] / fps) + pad), hit


def bvh_for(clip, body_plan, cmu_dir, kimodo_url, out_dir):
    """Path to a BVH for this humanoid clip (CMU exact match, else Kimodo), or None."""
    if body_plan != "humanoid":
        return None
    punch = punch_clip(clip, cmu_dir, out_dir)
    if punch:
        path, window, hit = punch
        CLIP_WINDOWS[clip] = window
        print(f"[motion] {clip}: mined from {hit['trial']} frames "
              f"{hit['start']}-{hit['impact']}-{hit['end']} ({hit['kind']}/{hit['target']})")
        PROVENANCE[clip] = (f"mocap:{hit['trial']} punch {hit['start']}-{hit['end']} "
                            f"{hit['hand']}/{hit['kind']}/{hit['target']}")
        return path
    bvh = find_cmu_clip(cmu_dir, clip)
    if bvh:
        print(f"[motion] {clip}: CMU clip {os.path.basename(bvh)}")
        PROVENANCE[clip] = "mocap:" + os.path.basename(bvh)
        return bvh
    if kimodo_url:
        try:
            import urllib.request
            req = urllib.request.Request(
                kimodo_url.rstrip("/") + "/generate",
                data=json.dumps({"prompt": clip, "fps": FPS}).encode(),
                headers={"Content-Type": "application/json"})
            out = os.path.join(out_dir, f"_{clip}.bvh")
            with urllib.request.urlopen(req, timeout=600) as resp:
                open(out, "wb").write(resp.read())
            print(f"[motion] {clip}: Kimodo generated")
            PROVENANCE[clip] = "generated:kimodo"
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[motion] {clip}: Kimodo failed, procedural instead: {e}")
    return None


# --- clip conditioning ------------------------------------------------------
# Raw retargeted mocap is not a game clip. It is a long trial at an arbitrary
# phase, it carries per-frame jitter from fitting one skeleton onto another, and
# its ends do not meet, so a looping clip visibly pops every cycle. These four
# passes fix each of those in turn: align, trim, smooth, close the loop.

_CYCLIC = ("idle", "walk", "run", "jog")
_ONESHOT_SECONDS = 2.5      # a punch is not twelve seconds long
_SMOOTH_RADIUS = 2          # +/- 2 frames: kills jitter, keeps the snap
_BLEND_FRACTION = 0.25      # tail cross-faded back into the head


def is_cyclic(clip_name):
    base = str(clip_name).lower().split("_")[0]
    return base in _CYCLIC


def align_signs(track):
    """q and -q are the same rotation; a sign flip between frames is not.

    Blender hands back whichever representative it likes, so a raw sample stream
    contains flips that every later pass -- averaging, blending, interpolation --
    reads as a full rotation. Make each frame agree with the one before it.
    """
    for quats in track.values():
        for i in range(1, len(quats)):
            if quats[i].dot(quats[i - 1]) < 0:
                quats[i] = -quats[i]


def _motion_energy(track, i, j):
    """How much the body actually moves across [i, j) -- summed per-frame change."""
    total = 0.0
    for quats in track.values():
        for k in range(i + 1, min(j, len(quats))):
            total += 1.0 - abs(quats[k].dot(quats[k - 1]))
    return total


def _gait_period(track, n):
    """Frames per locomotion cycle, by autocorrelation on the legs.

    A walk clip should be exactly one stride, so it can loop. Compare the pose
    stream against itself at every plausible lag and take the best match.
    """
    legs = [q for name, q in track.items() if "leg" in name.lower()]
    if not legs or n < 2 * FPS // 3:
        return None
    best, best_score = None, -1.0
    for lag in range(FPS // 3, min(2 * FPS, n // 2) + 1):   # 0.33s .. 2.0s strides
        score, count = 0.0, 0
        for quats in legs:
            for i in range(0, len(quats) - lag, 2):
                score += abs(quats[i].dot(quats[i + lag]))
                count += 1
        if count and score / count > best_score:
            best, best_score = lag, score / count
    # a weak best match means this is not really cyclic; do not force a stride
    return best if best_score > 0.995 else None


def clip_window(track, clip_name, n):
    """The [a, b] slice of the trial worth keeping, inclusive.

    Cyclic clips become exactly one stride, starting where the motion is
    strongest so the stride is not half a stand-still. One-shot clips keep the
    busiest few seconds, which is where the punch actually is.
    """
    if n <= 1:
        return 0, max(0, n - 1)
    if is_cyclic(clip_name):
        period = _gait_period(track, n)
        if period:
            best_a, best_e = 0, -1.0
            for a in range(0, n - period, max(1, period // 4)):
                e = _motion_energy(track, a, a + period)
                if e > best_e:
                    best_a, best_e = a, e
            return best_a, best_a + period
    span = min(n - 1, int(_ONESHOT_SECONDS * FPS))
    best_a, best_e = 0, -1.0
    for a in range(0, n - span, max(1, span // 8)):
        e = _motion_energy(track, a, a + span)
        if e > best_e:
            best_a, best_e = a, e
    return best_a, best_a + span


def smooth_quats(quats, radius=_SMOOTH_RADIUS):
    """Moving average over a window, renormalised.

    Retargeting one skeleton's proportions onto another leaves a per-frame
    tremor that reads as buzzing on the final mesh. Averaging a couple of frames
    either side removes it without visibly softening the impact of a strike.
    """
    if len(quats) < 2 * radius + 1:
        return list(quats)
    out = []
    for i in range(len(quats)):
        lo, hi = max(0, i - radius), min(len(quats), i + radius + 1)
        acc = mathutils.Quaternion((0.0, 0.0, 0.0, 0.0))
        for k in range(lo, hi):
            q = quats[k]
            if q.dot(quats[i]) < 0:
                q = -q
            acc.w += q.w; acc.x += q.x; acc.y += q.y; acc.z += q.z
        acc.normalize()
        out.append(acc)
    return out


def loop_blend(track, fraction=_BLEND_FRACTION):
    """Cross-fade the tail back toward the head so the clip closes on itself.

    Even one clean stride starts and ends on slightly different poses, and the
    mismatch shows as a jerk at every loop point -- the single most visible
    artefact in a walk or idle that plays continuously.
    """
    for quats in track.values():
        n = len(quats)
        span = max(1, int(n * fraction))
        if n < 3:
            continue
        first = quats[0]
        for i in range(n - span, n):
            t = (i - (n - span) + 1) / span          # 0 -> 1 across the tail
            quats[i] = quats[i].slerp(first, t)
        quats[-1] = first.copy()


def retarget_onto(arm, bvh_path, clip_name=""):
    """Sample a BVH and copy its per-bone local rotation onto `arm` (the armature the
    mesh is actually skinned to) by matching bone names. Returns the number of bones
    matched — 0 means the caller must fall back to procedural so the mesh still moves.
    This is why a real clip animates the CHARACTER, not a detached skeleton."""
    before = set(bpy.context.scene.objects)
    try:
        bpy.ops.import_anim.bvh(filepath=bvh_path, target="ARMATURE",
                                use_fps_scale=True, update_scene_fps=False)
    except Exception as e:  # noqa: BLE001
        print(f"[motion] BVH import failed, procedural instead: {e}")
        return 0
    src = next((o for o in bpy.context.scene.objects
                if o not in before and o.type == "ARMATURE"), None)
    if src is None:
        return 0
    try:
        tgt = {b.name.lower(): b.name for b in arm.pose.bones}
        pairs = [(sb.name, tgt[sb.name.lower()]) for sb in src.pose.bones
                 if sb.name.lower() in tgt]
        if len(pairs) < max(4, len(arm.pose.bones) // 4):
            # a couple of coincidental name hits would "succeed" into a mostly
            # static body — demand a real skeleton match or go procedural
            print(f"[motion] only {len(pairs)} bone(s) matched, procedural instead")
            return 0
        act = src.animation_data.action if src.animation_data else None
        f0, f1 = (int(act.frame_range[0]), int(act.frame_range[1])) if act else (0, FPS)

        # Sample the whole trial FIRST, then decide what part of it is the clip.
        # A CMU trial runs ten to sixty seconds and contains the move surrounded
        # by the performer walking up, waiting and walking off; keyframing the
        # raw range gave a twenty-second "walk" that never looped.
        track = {tname: [] for _sname, tname in pairs}
        for f in range(f0, f1 + 1):
            bpy.context.scene.frame_set(f)
            for sname, tname in pairs:
                # matrix_basis captures the source's local pose whatever its rot mode
                track[tname].append(src.pose.bones[sname].matrix_basis.to_quaternion())
        n = f1 - f0 + 1
        align_signs(track)
        # A mined punch already knows its own frames, down to the chamber and
        # the recovery; the energy heuristic would re-pick "the busiest 2.5
        # seconds" of a forty-second trial and throw that precision away.
        win = CLIP_WINDOWS.get(clip_name)
        if win:
            scene_fps = bpy.context.scene.render.fps or FPS
            a = max(0, min(int(win[0] * scene_fps), n - 1))
            b = max(0, min(int(win[1] * scene_fps), n - 1))
            if b - a < 2:
                a, b = clip_window(track, clip_name, n)
        else:
            a, b = clip_window(track, clip_name, n)
        for tname, quats in track.items():
            track[tname] = smooth_quats(quats[a:b + 1])
        if is_cyclic(clip_name):
            loop_blend(track)

        arm.animation_data_create()
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="POSE")
        for i in range(len(next(iter(track.values())))):
            for _sname, tname in pairs:
                tb = arm.pose.bones[tname]
                tb.rotation_mode = "QUATERNION"
                tb.rotation_quaternion = track[tname][i]
                tb.keyframe_insert("rotation_quaternion", frame=i)
                if i == 0:
                    # rotation_mode is unkeyed DNA: a later procedural clip flips
                    # it to XYZ and this clip would replay wrong. Key it per clip.
                    tb.keyframe_insert("rotation_mode", frame=i)
        bpy.ops.object.mode_set(mode="OBJECT")
        print(f"[motion] {clip_name}: trimmed {n} frames -> "
              f"{len(next(iter(track.values())))}"
              f"{' (looped)' if is_cyclic(clip_name) else ''}")
        print(f"[motion] retargeted {len(pairs)} bone(s) from mocap")
        return len(pairs)
    finally:
        src_act = src.animation_data.action if src.animation_data else None
        bpy.data.objects.remove(src, do_unlink=True)
        if src_act and src_act.users == 0:
            # the orphaned BVH action is often named exactly like the clip (file
            # stem) — left around, our renamed action becomes "walk.001" and the
            # exported animation name breaks the player's has_animation lookup
            bpy.data.actions.remove(src_act)


def procedural_clip(arm, clip):
    """Keyframe a short looping base cycle by rule. Works for ANY rig — it just
    oscillates whatever body bones exist (arms/legs/spine/generic segments)."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    dur = FPS  # a 1s loop
    amp = {"idle": 0.06, "walk": 0.5, "attack": 0.9}.get(clip, 0.3)
    for f in range(dur + 1):
        phase = 2 * math.pi * f / dur
        for pb in arm.pose.bones:
            name = pb.name.lower()
            if any(k in name for k in ("tail", "jaw", "wing")):
                continue  # extras are keyframed separately, on every path
            pb.rotation_mode = "XYZ"
            swing = amp * math.sin(phase + (math.pi if "right" in name
                                            or name.endswith(".r") else 0))
            if "forearm" in name or name.endswith("leg"):
                # elbows and knees bend one way, at roughly half the swing, and
                # a knee never hyperextends -- abs() keeps the bend signed
                pb.rotation_euler = (-abs(swing) * 0.6, 0, 0)
            elif any(k in name for k in ("hand", "foot", "shoulder")):
                pb.rotation_euler = (swing * 0.25, 0, 0)
            elif any(k in name for k in ("arm", "leg", "seg")):
                pb.rotation_euler = (swing, 0, 0)
            elif "spine" in name or "pelvis" in name or "hips" in name:
                pb.rotation_euler = (0, 0, amp * 0.3 * math.sin(phase))
            pb.keyframe_insert("rotation_euler", frame=f)
            if f == 0:
                # keyed per clip: a retargeted sibling clip sets QUATERNION and
                # this clip's euler keys would be ignored on replay
                pb.keyframe_insert("rotation_mode", frame=f)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[motion] {clip}: procedural cycle")


def procedural_extras(arm, clip, extras):
    """Secondary motion for non-skeletal parts (tail/jaw/wings). Runs on EVERY clip,
    including the mocap/retarget path — a mocap body still needs its tail to swing."""
    bones = [pb for pb in arm.pose.bones
             if any(k in pb.name.lower() for k in ("tail", "jaw", "wing", "cloak", "cape"))]
    if not bones:
        return
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    dur = FPS
    for f in range(dur + 1):
        phase = 2 * math.pi * f / dur
        for pb in bones:
            name = pb.name.lower()
            pb.rotation_mode = "XYZ"
            if "tail" in name and "tail" in extras:
                pb.rotation_euler = (0, 0, 0.8 * math.sin(phase))
            elif "jaw" in name and "jaw" in extras:
                pb.rotation_euler = (max(0.0, math.sin(phase)) if clip == "attack" else 0, 0, 0)
            elif "wing" in name and "wings" in extras:
                pb.rotation_euler = (0, 0.9 * math.sin(phase), 0)
            elif ("cloak" in name or "cape" in name) and "cloak" in extras:
                pb.rotation_euler = (0.15 * math.sin(phase), 0, 0.1 * math.sin(phase * 0.7))
            else:
                continue
            pb.keyframe_insert("rotation_euler", frame=f)
    bpy.ops.object.mode_set(mode="OBJECT")


# --- export ----------------------------------------------------------------

def export_glb(path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB",
                              use_selection=False, export_animations=True)


def main():
    out_dir = ARGS["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    body_plan = ARGS.get("body_plan", "humanoid")
    extras = [e.lower() for e in ARGS.get("extras", [])]

    reset_scene()
    mesh = import_mesh(ARGS["mesh"])
    arm = try_unirig(mesh, ARGS.get("unirig", ""), out_dir) if ARGS.get("unirig") else None
    if arm is None:
        # try_unirig may have reset the scene on failure; re-import so procedural_rig
        # always gets a live mesh object (never a stale, deleted reference)
        reset_scene()
        mesh = import_mesh(ARGS["mesh"])
        arm = procedural_rig(mesh, body_plan)

    # build every clip as a named action stashed on its own NLA track, then export
    # ONE glb — Godot's AnimationPlayer then has idle/walk/run to switch between,
    # instead of one file per clip of which the game could only ever use one
    for clip in ARGS.get("animations", ["idle"]):
        if arm.animation_data:
            arm.animation_data.action = None    # fresh action per clip
        bvh = bvh_for(clip, body_plan, ARGS.get("cmu_dir", ""), ARGS.get("kimodo_url", ""),
                      out_dir)
        if not (bvh and retarget_onto(arm, bvh, clip)):
            procedural_clip(arm, clip)          # base body motion (never leaves it static)
            PROVENANCE[clip] = "procedural"     # retarget may have rejected the bvh too
        procedural_extras(arm, clip, extras)    # tail/jaw/wings on EVERY path
        act = arm.animation_data.action if arm.animation_data else None
        if act:
            act.name = clip
            track = arm.animation_data.nla_tracks.new()
            track.name = clip
            track.strips.new(clip, int(act.frame_range[0]) + 1, act)
    if arm.animation_data:
        arm.animation_data.action = None        # export from NLA tracks only
    out = os.path.join(out_dir, "character.glb")
    export_glb(out)
    # Provenance the finished .glb cannot carry: whether a clip is real mocap or
    # a synthetic cycle looks identical inside the file, but it is exactly what
    # someone asking to "make the animation more realistic" needs to know.
    mesh_obj = next((o for o in bpy.context.scene.objects if o.type == "MESH"), None)
    write_json(out + ".motion.json", {
        "clips": {c: PROVENANCE.get(c, "procedural")
                  for c in ARGS.get("animations", ["idle"])},
        "body_plan": body_plan,
        "extras": extras,
        "bones": len(arm.data.bones),
        "weighted_verts": sum(1 for v in mesh_obj.data.vertices
                              if any(g.weight > 1e-6 for g in v.groups))
                          if mesh_obj else 0,
        "total_verts": len(mesh_obj.data.vertices) if mesh_obj else 0,
    })


def write_json(path, obj):
    """Best-effort sidecar: a failed write must never fail the motion stage."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
    except Exception as e:      # noqa: BLE001
        print(f"[motion] could not write {os.path.basename(path)}: {e}")


if __name__ == "__main__":
    main()
