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
import sys

import bpy  # provided by Blender's own Python; absent in the pipeline venv
import mathutils

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
        pelvis = bone("pelvis", (cx, cy, lo[2] + h * 0.5), (cx, cy, lo[2] + h * 0.55))
        spine = bone("spine", pelvis.tail, (cx, cy, lo[2] + h * 0.8), pelvis)
        bone("head", spine.tail, (cx, cy, hi[2]), spine)
        for s, sx in (("L", -1), ("R", 1)):
            bone(f"arm.{s}", (cx + sx * h * 0.1, cy, lo[2] + h * 0.78),
                 (cx + sx * h * 0.35, cy, lo[2] + h * 0.6), spine)
            bone(f"leg.{s}", (cx + sx * h * 0.08, cy, lo[2] + h * 0.5),
                 (cx + sx * h * 0.08, cy, lo[2]), pelvis)
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
    return arm


# --- animation -------------------------------------------------------------

def find_cmu_clip(cmu_dir, clip):
    """Recursive name match: a clip named 'walk' hits any */`*walk*.bvh` under
    cmu_dir. Raw CMU files are numbered (01_01.bvh) — rename or symlink the ones
    you curate to descriptive names for them to be found."""
    if not cmu_dir or not os.path.isdir(cmu_dir):
        return None
    for base, _dirs, files in os.walk(cmu_dir):
        for f in files:
            stem = os.path.splitext(f)[0].lower()
            if f.lower().endswith(".bvh") and clip.lower() in stem:
                return os.path.join(base, f)
    return None


def bvh_for(clip, body_plan, cmu_dir, kimodo_url, out_dir):
    """Path to a BVH for this humanoid clip (CMU exact match, else Kimodo), or None."""
    if body_plan != "humanoid":
        return None
    bvh = find_cmu_clip(cmu_dir, clip)
    if bvh:
        print(f"[motion] {clip}: CMU clip {os.path.basename(bvh)}")
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
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[motion] {clip}: Kimodo failed, procedural instead: {e}")
    return None


def retarget_onto(arm, bvh_path):
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
        arm.animation_data_create()
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="POSE")
        for f in range(f0, f1 + 1):
            bpy.context.scene.frame_set(f)
            for sname, tname in pairs:
                tb = arm.pose.bones[tname]
                tb.rotation_mode = "QUATERNION"
                # matrix_basis captures the source's local pose whatever its rot mode
                tb.rotation_quaternion = src.pose.bones[sname].matrix_basis.to_quaternion()
                tb.keyframe_insert("rotation_quaternion", frame=f)
                if f == f0:
                    # rotation_mode is unkeyed DNA: a later procedural clip flips
                    # it to XYZ and this clip would replay wrong. Key it per clip.
                    tb.keyframe_insert("rotation_mode", frame=f)
        bpy.ops.object.mode_set(mode="OBJECT")
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
            swing = amp * math.sin(phase + (math.pi if name.endswith(".r") else 0))
            if any(k in name for k in ("arm", "leg", "seg")):
                pb.rotation_euler = (swing, 0, 0)
            elif "spine" in name or "pelvis" in name:
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
        if not (bvh and retarget_onto(arm, bvh)):
            procedural_clip(arm, clip)          # base body motion (never leaves it static)
        procedural_extras(arm, clip, extras)    # tail/jaw/wings on EVERY path
        act = arm.animation_data.action if arm.animation_data else None
        if act:
            act.name = clip
            track = arm.animation_data.nla_tracks.new()
            track.name = clip
            track.strips.new(clip, int(act.frame_range[0]) + 1, act)
    if arm.animation_data:
        arm.animation_data.action = None        # export from NLA tracks only
    export_glb(os.path.join(out_dir, "character.glb"))


if __name__ == "__main__":
    main()
