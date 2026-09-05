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


# --- fitting the rig to the mesh instead of to a template -------------------
#
# The biped used to be laid out purely as fractions of the bounding box: the
# hand bone sat 0.34 of the character's height out from the centre line. On the
# reference character the mesh is only 0.256 of its height wide at that point,
# so both hands hung a third of their length OUTSIDE the mesh. Bone heat cannot
# solve a bone that is not inside the volume -- it failed, every weight came out
# zero, and the fallback distance bind then glued each hand to whatever happened
# to be nearest, which is the thigh. That is what "the limbs are misaligned"
# looks like from the inside.
#
# So measure the silhouette and put the bones where the body actually is.

_FIT_BANDS = 48


def silhouette(pts, lo_z, hi_z, bands=_FIT_BANDS):
    """Per-height slice of the mesh: how wide it is and where its centre sits."""
    h = hi_z - lo_z or 1e-6
    out = []
    for i in range(bands):
        z0 = lo_z + h * i / bands
        z1 = lo_z + h * (i + 1) / bands
        band = [p for p in pts if z0 <= p.z < z1]
        if not band:
            out.append(None)
            continue
        xs = [p.x for p in band]
        ys = [p.y for p in band]
        out.append({"z": (z0 + z1) / 2,
                    "half_w": (max(xs) - min(xs)) / 2,
                    "xc": (max(xs) + min(xs)) / 2,
                    "yc": (max(ys) + min(ys)) / 2,
                    "half_d": (max(ys) - min(ys)) / 2})
    return out


def slice_at(prof, lo_z, hi_z, frac):
    """The measured slice nearest a height given as a fraction of total height,
    walking outward when that exact slice is empty."""
    n = len(prof)
    i = min(n - 1, max(0, int(frac * n)))
    for d in range(n):
        for j in (i - d, i + d):
            if 0 <= j < n and prof[j]:
                return prof[j]
    return None


def inside(prof, lo_z, hi_z, frac, lateral, keep=0.82):
    """Clamp a lateral offset so the bone stays inside the mesh at that height.

    `keep` leaves a margin off the surface: a bone exactly on the skin weights
    the far side of a limb as strongly as the near side.
    """
    sl = slice_at(prof, lo_z, hi_z, frac)
    if not sl or sl["half_w"] <= 1e-6:
        return lateral
    limit = sl["half_w"] * keep
    return max(-limit, min(limit, lateral))


def shoulder_frac(prof, lo_z, hi_z, default=0.82):
    """Height where the torso stops being torso-wide, as a fraction of height.

    Shoulders are where the silhouette narrows sharply on the way up. Measured
    off the width profile rather than assumed, because a hooded figure, a
    broad-shouldered brawler and a slim one put their shoulders in different
    places and a rest pose that misses the joint bends the arm from the chest.
    """
    torso = slice_at(prof, lo_z, hi_z, 0.72)
    if not torso:
        return default
    thresh = torso["half_w"] * 0.85
    n = len(prof)
    best = None
    for i in range(int(n * 0.60), int(n * 0.95)):
        sl = prof[i]
        if sl and sl["half_w"] >= thresh:
            best = (i + 0.5) / n
    return best if best and 0.68 <= best <= 0.90 else default


def neck_frac(prof, lo_z, hi_z, shoulder, default=0.86):
    """Where the silhouette has narrowed to head width."""
    torso = slice_at(prof, lo_z, hi_z, 0.72)
    if not torso:
        return default
    thresh = torso["half_w"] * 0.45
    n = len(prof)
    for i in range(int(n * 0.95), int(n * 0.60), -1):
        sl = prof[i]
        if sl and sl["half_w"] >= thresh:
            f = (i + 1.5) / n
            return f if shoulder + 0.02 <= f <= 0.95 else default
    return default


def procedural_rig(mesh_obj, body_plan, extras=()):
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
        prof = silhouette(bb, lo[2], hi[2])

        def z(f):
            return lo[2] + h * f

        def core(f):
            """Centre line of the body at this height, measured not assumed."""
            sl = slice_at(prof, lo[2], hi[2], f)
            return (sl["xc"], sl["yc"]) if sl else (cx, cy)

        def side_pt(f, frac_of_half, sx):
            """A point out to the side at height f, kept inside the mesh.

            frac_of_half is how far toward the surface to sit, so an arm rides
            near the outside of the silhouette and a leg nearer the middle,
            without either ever leaving the volume.
            """
            sl = slice_at(prof, lo[2], hi[2], f)
            if not sl:
                return (cx + sx * h * frac_of_half * 0.3, cy, z(f))
            return (sl["xc"] + sx * sl["half_w"] * frac_of_half, sl["yc"], z(f))

        sh_f = shoulder_frac(prof, lo[2], hi[2])
        neck_f = neck_frac(prof, lo[2], hi[2], sh_f)

        hx, hy = core(0.50)
        hips = bone("Hips", (hx, hy, z(0.50)), (hx, hy, z(0.56)))
        spine = bone("Spine", hips.tail, core(0.68) + (z(0.68),), hips)
        chest = bone("Spine1", spine.tail, core(sh_f) + (z(sh_f),), spine)
        neck = bone("Neck", chest.tail, core(neck_f) + (z(neck_f),), chest)
        bone("Head", neck.tail, core(0.97) + (hi[2],), neck)
        # Arms hang at the sides inside a robe rather than swinging out on a
        # diagonal: elbow and wrist stay near the silhouette edge and descend,
        # which is also the rest pose CMU's mocap rotations assume.
        arm_f = (sh_f, sh_f - 0.11, sh_f - 0.24, sh_f - 0.31)
        for side, sx in (("Left", -1), ("Right", 1)):
            sh = bone(f"{side}Shoulder", core(sh_f) + (z(sh_f),),
                      side_pt(arm_f[0], 0.42, sx), chest)
            up = bone(f"{side}Arm", sh.tail, side_pt(arm_f[1], 0.80, sx), sh)
            fore = bone(f"{side}ForeArm", up.tail, side_pt(arm_f[2], 0.82, sx), up)
            bone(f"{side}Hand", fore.tail, side_pt(arm_f[3], 0.80, sx), fore)
            thigh = bone(f"{side}UpLeg", side_pt(0.50, 0.34, sx),
                         side_pt(0.28, 0.40, sx), hips)
            shin = bone(f"{side}Leg", thigh.tail, side_pt(0.06, 0.42, sx), thigh)
            fx, fy, fz = side_pt(0.02, 0.42, sx)
            bone(f"{side}Foot", shin.tail, (fx, fy - h * 0.06, lo[2]), shin)

        # A cloak is not skin: it hangs off the shoulders and swings a beat
        # behind the body. Give it its own chain down the back BEFORE binding,
        # so the weighting pass below picks the back-hanging geometry up on
        # these bones instead of gluing it to the spine. Godot then drives the
        # chain with a verlet solver (scripts/cloak.gd) rather than a canned
        # wiggle -- the cloth reacts to how the character actually moved.
        if any("cloak" in e or "cape" in e for e in extras):
            prev, top = None, z(sh_f + 0.02)
            for i in range(CLOAK_SEGMENTS):
                a = top + (lo[2] - top) * (i / CLOAK_SEGMENTS)
                b = top + (lo[2] - top) * ((i + 1) / CLOAK_SEGMENTS)
                # follow the measured back surface rather than a fixed offset:
                # a fixed one leaves the chain outside a slim figure and buried
                # in the chest of a broad one
                fa = (a - lo[2]) / (h or 1e-6)
                fb = (b - lo[2]) / (h or 1e-6)
                sa = slice_at(prof, lo[2], hi[2], fa)
                sb = slice_at(prof, lo[2], hi[2], fb)
                ya = sa["yc"] + sa["half_d"] * 0.55 if sa else cy
                yb = sb["yc"] + sb["half_d"] * 0.55 if sb else cy
                xa = sa["xc"] if sa else cx
                xb = sb["xc"] if sb else cx
                prev = bone(f"Cloak.{i}", (xa, ya, a), (xb, yb, b), prev or chest)

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
    print(f"[motion] rig: {len(arm.data.bones)} bones "
          f"({sum(1 for b in arm.data.bones if b.name.startswith('Cloak'))} cloak)"
          f" extras={list(extras)}")
    # skin: automatic weights binds arbitrary geometry to whatever bones we made
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True); arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    weld(mesh_obj)
    # Bone heat is solved for the BODY only. A cloak bone tracks the back
    # surface, so it sits at the very edge of the volume and is the likeliest
    # bone to have no solution -- and bone heat is all-or-nothing: one
    # unsolvable bone returns zero weights for every bone, which is how a whole
    # character ended up bound by raw distance with its hands on its thighs.
    cloaks = [b for b in arm.data.bones if b.name.startswith("Cloak")]
    for b in cloaks:
        b.use_deform = False
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    for b in cloaks:
        b.use_deform = True

    if not weight_total(mesh_obj):
        # Bone heat needs clean manifold geometry and TRELLIS output is neither.
        # It fails with "failed to find solution for one or more bones", still
        # creates the vertex groups, and leaves EVERY vertex at zero weight --
        # on the reference character, 0 of 36994. The glb then exports with
        # JOINTS_0/WEIGHTS_0 but no skin at all, so Godot imports a static mesh
        # and the character slides through the level frozen in its rest pose.
        print("[motion] bone heat produced no weights — binding by distance")
        bind_by_distance(mesh_obj, arm)
    else:
        n = sum(1 for v in mesh_obj.data.vertices
                if any(g.weight > 1e-6 for g in v.groups))
        print(f"[motion] bone heat bound {n}/{len(mesh_obj.data.vertices)} verts")
    if cloaks:
        # order matters: take the coat off the arms first, then hand the
        # back-hanging geometry down the chain
        trim_sleeves(mesh_obj, arm)
        paint_cloak(mesh_obj, arm)
    return arm


def trim_sleeves(mesh_obj, arm, sleeve=0.38):
    """Keep each arm to its own sleeve and give the rest of the coat to the cloak.

    Bone heat is solved on the garment's outer shell, so wherever an arm hangs
    inside a coat the nearest bone to a whole panel of cloth is that arm. At
    rest it looks perfect -- the panel is exactly where it should be -- and then
    the character throws a punch and the entire coat goes with his fist. On this
    character the hand bone alone owned 8,560 vertices spanning hip height, and
    the arms between them held half the mesh.

    A vertex belongs to an arm only if it lies within a sleeve's radius of that
    arm's own bone. Anything beyond that is cloth hanging off the torso and goes
    to the cloak chain, where the cloth solver drives it instead. The radius is
    stated in upper-arm lengths so it scales with the rig rather than the scene,
    and it is deliberately generous: a heavy coat sleeve is much thicker than
    the arm inside it, and stripping a real sleeve off the arm would be a worse
    failure than leaving a little cloth on it.
    """
    to_local = mesh_obj.matrix_world.inverted() @ arm.matrix_world
    limbs = []
    for side in ("Left", "Right"):
        upper = arm.data.bones.get(f"{side}Arm")
        if upper is None:
            continue
        radius = sleeve * (to_local @ upper.tail_local
                           - to_local @ upper.head_local).length
        for part in ("Arm", "ForeArm", "Hand"):
            b = arm.data.bones.get(f"{side}{part}")
            if b is not None and mesh_obj.vertex_groups.get(b.name):
                limbs.append((b.name, to_local @ b.head_local,
                              to_local @ b.tail_local, radius))
    if not limbs:
        return

    chain = sorted((b for b in arm.data.bones if b.name.startswith("Cloak")),
                   key=lambda b: -(to_local @ b.head_local).z)
    if chain:
        spans = [((to_local @ b.head_local).z, (to_local @ b.tail_local).z, b.name)
                 for b in chain]
    else:                                   # no cloak rig: park it on the chest
        fallback = next((n for n in ("Spine1", "Spine", "Hips")
                         if arm.data.bones.get(n)), None)
        if fallback is None:
            return
        spans = [(0.0, 0.0, fallback)]
    groups = {name: (mesh_obj.vertex_groups.get(name)
                     or mesh_obj.vertex_groups.new(name=name))
              for _, _, name in spans}

    moved = 0
    for v in mesh_obj.data.vertices:
        shed = 0.0
        for name, head, tail, radius in limbs:
            g = next((g for g in v.groups
                      if mesh_obj.vertex_groups[g.group].name == name), None)
            if g is None or g.weight <= 1e-3:
                continue
            if _seg_distance(v.co, head, tail) <= radius:
                continue                    # inside the sleeve: genuinely arm
            shed += g.weight
            mesh_obj.vertex_groups[name].add([v.index], 0.0, "REPLACE")
        if shed <= 1e-3:
            continue
        seg = min(spans, key=lambda s: abs((s[0] + s[1]) / 2 - v.co.z))[2]
        have = next((g.weight for g in v.groups
                     if mesh_obj.vertex_groups[g.group].name == seg), 0.0)
        groups[seg].add([v.index], min(1.0, have + shed), "REPLACE")
        moved += 1
    print(f"[motion] sleeves: {moved} coat verts taken off the arms")


def _seg_distance(p, head, tail):
    d = tail - head
    length_sq = d.dot(d)
    if length_sq < 1e-12:
        return (p - head).length
    t = max(0.0, min(1.0, (p - head).dot(d) / length_sq))
    return (p - (head + d * t)).length


def paint_cloak(mesh_obj, arm, reach=0.55, strength=0.9):
    """Hand the back-hanging geometry to the cloak chain.

    Bone heat never sees these bones (see above), and letting it own the cloak
    would be wrong anyway: a cape is not skin, and weighting it to the spine
    makes it a rigid shell that turns with the chest instead of swinging behind
    it. Weight is taken FROM the body groups rather than added alongside them,
    so the total per vertex stays 1 and the cloth does not fight the torso.
    """
    to_local = mesh_obj.matrix_world.inverted() @ arm.matrix_world
    chain = sorted((b for b in arm.data.bones if b.name.startswith("Cloak")),
                   key=lambda b: -(to_local @ b.head_local).z)
    if not chain:
        return
    groups = {b.name: (mesh_obj.vertex_groups.get(b.name)
                       or mesh_obj.vertex_groups.new(name=b.name)) for b in chain}
    spans = [((to_local @ b.head_local).z, (to_local @ b.tail_local).z, b.name)
             for b in chain]
    top = max(s[0] for s in spans)

    ys = [v.co.y for v in mesh_obj.data.vertices]
    y_mid = (max(ys) + min(ys)) / 2
    y_back = max(ys)
    if y_back - y_mid < 1e-6:
        return

    painted = 0
    for v in mesh_obj.data.vertices:
        if v.co.z > top:
            continue                       # above the collar: shoulders, not cape
        # how far toward the back surface this vertex sits, 0 at the spine
        depth = (v.co.y - y_mid) / (y_back - y_mid)
        if depth < reach:
            continue
        w = strength * min(1.0, (depth - reach) / max(1e-6, 1.0 - reach))
        if w <= 1e-3:
            continue
        seg = min(spans, key=lambda s: abs((s[0] + s[1]) / 2 - v.co.z))[2]
        for g in list(v.groups):
            name = mesh_obj.vertex_groups[g.group].name
            if not name.startswith("Cloak"):
                mesh_obj.vertex_groups[g.group].add([v.index], g.weight * (1.0 - w),
                                                    "REPLACE")
        groups[seg].add([v.index], w, "REPLACE")
        painted += 1
    print(f"[motion] cloak: {painted} verts moved onto {len(chain)} segments")


def weld(mesh_obj, threshold=0.0002):
    """Merge the duplicate vertices glTF import creates, before any binding.

    A .glb stores one vertex per (position, normal, uv) combination, so every
    UV seam and every hard edge splits the vertex. The reference character
    imported as 36994 vertices in 2032 disconnected islands -- topologically
    shattered, even though it looks like one solid body. Bone heat solves a
    diffusion across the surface, and a diffusion cannot cross a gap, so it
    failed on every bone and the character fell back to a distance bind with its
    hands weighted to its thighs.

    Welding takes it to 16572 vertices in 2 components and costs nothing that
    matters: UVs live on loops rather than vertices, so the texture is intact,
    and the face count drops by one.
    """
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=threshold)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh_obj.data)
    bm.free()
    mesh_obj.data.update()
    print(f"[motion] welded {before} -> {len(mesh_obj.data.vertices)} verts "
          f"(glTF seam duplicates; bone heat needs connected geometry)")


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
# Trials measured to actually do what their description says, checked before the
# index search. CMU's own advice to prefer higher-numbered subjects is about
# capture quality, not content, and following it alone picked 140_06 for "idle"
# -- a trial whose performer spends it resting in a deep wide crouch. Retargeted
# faithfully that is a character squatting like a sumo. Scoring every candidate's
# knee bend across 25 trials put these at the top; 111_28 stands with its knees
# straight (0 degrees) and its feet 0.145 of body height apart.
_CLIP_PREFERRED = {
    "idle": ("111_28", "113_21", "77_02"),
}

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
    for trial in _CLIP_PREFERRED.get(base, ())[variant:]:
        if trial in paths:
            return paths[trial]
    words = _CLIP_SYNONYMS.get(base, (base,))
    hits = [trial for trial, desc in cmu_index(cmu_dir).items()
            if trial in paths and any(re.search(rf"\b{re.escape(w)}", desc) for w in words)]
    if not hits:
        return None
    # A cyclic clip needs a source long enough to contain a whole stride with
    # room to pick the best one. CMU's run/jog matches include 0.8-second
    # fragments, and the highest-subject-first rule happily returned one of
    # those as "run", leaving a 25-frame loop.
    if is_cyclic(clip):
        long_enough = [t for t in hits
                       if bvh_frames(paths[t]) >= _MIN_CYCLIC_SECONDS
                       * bvh_fps(paths[t])]
        hits = long_enough or hits
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


def bvh_frames(bvh_path):
    """Frame count from the BVH header, without importing the file."""
    try:
        with open(bvh_path, errors="replace") as fh:
            for line in fh:
                if line.strip().lower().startswith("frames:"):
                    return int(line.split(":", 1)[1])
    except Exception:  # noqa: BLE001
        pass
    return 0


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
    # beside the mocap, not beside the run: the survey describes the LIBRARY, so
    # every character reuses one copy instead of re-mining per output directory
    caches = [os.path.join(d, "punch_library.json") for d in (cmu_dir, out_dir) if d]
    for cache in caches:
        try:
            with open(cache) as fh:
                found = json.load(fh)
        except Exception:  # noqa: BLE001  -- absent or unreadable: try the next
            continue
        if found and all("fps" in p for p in found):
            _PUNCH_CACHE[cmu_dir] = found
            print(f"[motion] punch library: {len(found)} punches (cached)")
            return found
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
    for cache in caches:
        try:
            with open(cache, "w") as fh:
                json.dump(found, fh)
            break
        except OSError:
            continue  # a read-only mocap directory is not a reason to fail
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
_MIN_CYCLIC_SECONDS = 3.0   # a walk/run source shorter than this is a fragment
_BREATH_SECONDS = 4.3       # ~14 breaths a minute, resting
_BREATH_INHALE = 0.4        # in is quicker than out, which is what reads as alive
CLOAK_SEGMENTS = 6          # enough links to bend twice; more costs runtime for
                            # detail the silhouette will not show


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
    if str(clip_name).lower().split("_")[0] == "idle":
        # An idle is not a gait. The period detector finds a half-second twitch
        # in a standing trial and loops that, which is the fidget the character
        # was doing rather than the character standing. Take one BREATH instead,
        # and let breathe() carry it.
        #
        # Calm is not enough on its own: the calmest stretch of a trial can be
        # the performer resting in a deep wide crouch, and retargeting that
        # faithfully gave a character squatting like a sumo with his coat draped
        # over his knees. The knee's local rotation IS its bend, because the rig
        # rests straight-legged, so score uprightness alongside stillness.
        span = min(n - 1, int(_BREATH_SECONDS * FPS))
        knees = [track[k] for k in ("LeftLeg", "RightLeg") if k in track]

        def crouch(a, b):
            if not knees:
                return 0.0
            angs = [2.0 * math.acos(min(1.0, abs(q.w)))
                    for k in knees for q in k[a:b]]
            return sum(angs) / max(1, len(angs))

        best_a, best_score = 0, float("inf")
        for a in range(0, max(1, n - span), max(1, span // 8)):
            # crouch dominates: a still squat is not an idle, a breathing stand is
            score = _motion_energy(track, a, a + span) + 4.0 * crouch(a, a + span)
            if score < best_score:
                best_a, best_score = a, score
        return best_a, best_a + span
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


def breath_phase(u):
    """0 at rest, 1 at full inhale, back to 0, over one normalised cycle.

    Asymmetric on purpose: a resting breath draws in over about four tenths of
    the cycle and lets out over the rest. A symmetric sine reads as a pump.
    Both halves are raised cosines, so the ends meet and the clip still loops.
    """
    u = u % 1.0
    if u < _BREATH_INHALE:
        return 0.5 - 0.5 * math.cos(math.pi * u / _BREATH_INHALE)
    return 0.5 + 0.5 * math.cos(math.pi * (u - _BREATH_INHALE) / (1 - _BREATH_INHALE))


# Which bones carry a breath, and how much. The chest leads, the neck and head
# counter-rotate so the gaze stays level instead of nodding along with the
# ribcage, and the shoulders ride a fraction of the chest.
_BREATH_BONES = {"Spine": 1.0, "Spine1": 0.75, "Neck": -0.45, "Head": -0.25,
                 "LeftShoulder": 0.3, "RightShoulder": 0.3}


def breathe(track, amp=0.05):
    """Lay a breathing cycle over a standing clip, in place.

    Mocap of somebody standing still already contains their breathing, but the
    clip is a few seconds cut out of a longer trial and smoothed, and what
    survives is too small and too irregular to read on a game character. This
    adds one clean, loopable breath on top of whatever the performer was doing.
    """
    n = len(next(iter(track.values()), []))
    if n < 2:
        return
    for name, weight in _BREATH_BONES.items():
        quats = track.get(name)
        if not quats:
            continue
        for i in range(n):
            a = amp * weight * breath_phase(i / n)
            cls = type(quats[i])
            quats[i] = quats[i] @ cls((math.cos(a / 2), math.sin(a / 2), 0.0, 0.0))


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


# Arms out along X: the one pose both a mocap skeleton and a rig fitted to
# hanging-armed concept art can be put into, so their bones can be compared.
# Everything not listed keeps its own rest direction, which for a spine, a leg
# or a head already agrees between any two humanoid rigs.
T_POSE_REF = {
    "leftarm": (1.0, 0.0, 0.0), "leftforearm": (1.0, 0.0, 0.0),
    "lefthand": (1.0, 0.0, 0.0),
    "rightarm": (-1.0, 0.0, 0.0), "rightforearm": (-1.0, 0.0, 0.0),
    "righthand": (-1.0, 0.0, 0.0),
}
# The clavicles are deliberately NOT in that table. A clavicle points outward
# and forward at rest in BOTH rigs -- the difference between them is anatomy,
# not a pose difference to normalise away -- and swinging one onto a pure X
# axis leaves its roll arbitrary, which then propagates down the whole arm.
# Doing that pulled the shoulder joints inward by a fixed amount per clip, up
# to 21% on the run, and the arms came with them: "the arms and shoulders is
# really in each other".

def basis_toward(want):
    """A full orientation for a bone pointing along `want`.

    A direction alone does not pin a bone down -- the roll about its own axis is
    still free, and roll on an upper arm decides which way the elbow bends. So
    build the whole basis: the bone runs along `want` and its local Z leans
    toward world up. Shared by the retarget and by authored poses, so both mean
    the same thing by a direction.
    """
    y = mathutils.Vector(want).normalized()
    up = mathutils.Vector((0.0, 0.0, 1.0))
    if abs(y.dot(up)) > 0.99:
        up = mathutils.Vector((0.0, 1.0, 0.0))
    z = (up - y * up.dot(y)).normalized()
    return mathutils.Matrix((y.cross(z), y, z)).transposed().to_quaternion()


def aim_world(rest_q, want):
    """World rotation that points a bone's rest direction at `want`.

    A direction does not determine a rotation -- the roll about the bone axis
    is still free -- so the obvious construction (build a basis from `want` and
    some reference up-vector) silently invents a roll. On a torso that reads as
    the character spinning on the spot, and on an upper arm it turns the plane
    the elbow bends in. The minimal rotation from rest to `want` adds no twist
    at all, so every bone keeps the roll its rest pose gave it.
    """
    d = rest_q @ mathutils.Vector((0.0, 1.0, 0.0))
    return d.rotation_difference(mathutils.Vector(want).normalized()) @ rest_q


def solve_local(arm, world_rots):
    """Local rotations that put each named bone at the given world orientation.

    Blender composes a posed bone as W = W_parent . Rl . b, so b falls out as
    (W_parent . Rl)^-1 . W once the parents are known. Bones left unnamed stay
    at rest, which is what makes an authored pose safe on this mesh: the skin
    was bound at rest, and every vertex the pose does not mention stays exactly
    where the binding put it.
    """
    rest = {b.name: b.bone.matrix_local.to_quaternion() for b in arm.pose.bones}
    parent = {b.name: (b.parent.name if b.parent else None) for b in arm.pose.bones}
    depth = {b.name: len(b.parent_recursive) for b in arm.pose.bones}
    world, out = {}, {}
    for name in sorted(rest, key=lambda n: depth[n]):
        par = parent[name]
        wp = world[par] if par else mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        rl = (rest[par].inverted() @ rest[name]) if par else rest[name]
        rest_world = wp @ rl
        world[name] = world_rots.get(name, rest_world)
        if name in world_rots:
            out[name] = rest_world.inverted() @ world[name]
    return out


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
        # Put each of our bones where the source's bone actually is, composing
        # down the hierarchy, having first put both rigs in a common pose.
        #
        # The two rigs do not share a rest pose. CMU's is a T-pose; the rig
        # fitted to this mesh has the arms hanging, because that is how the
        # character was drawn. Copying matrix_basis across bent him double at
        # the waist. Matching each rig's delta from its own rest stood him up
        # but applied CMU's T-pose-to-hanging arm drop a second time to arms
        # that already hang, so they flared out. Matching the source's absolute
        # orientation fixes the arms and breaks the spine instead, because the
        # hips bone does not point the same way in the two skeletons and the
        # difference lands on the whole body as a forward pitch.
        #
        # Both failures are the same missing piece: a shared frame of
        # reference. So normalise each rig to a T-pose reference first --
        # identical for both, arms out along X -- and take the fixed offset
        # between the two references as the correction. A source bone sitting
        # at its own rest then puts our bone at OUR rest, whatever either rest
        # happens to be, which is the property both earlier attempts lacked.
        def ref_rot(bone, want, into):
            """The bone's reference orientation: its own rest, or a canonical
            basis pointing along `want` when it is overridden.

            Building the basis outright rather than swinging the rest pose onto
            `want` also fixes the roll, which a minimal swing leaves wherever it
            happened to be -- and roll on an upper arm rotates the plane its
            elbow bends in. For an overridden bone this makes the correction
            exactly identity, which is the honest statement of what the arms
            want: put ours where theirs is. Clearance is a separate problem,
            handled below."""
            if want is None:
                return (into @ bone.matrix_local).to_quaternion()
            return basis_toward(want)

        rest_t = {b.name: b.bone.matrix_local.to_quaternion() for b in arm.pose.bones}
        parent_of = {b.name: (b.parent.name if b.parent else None)
                     for b in arm.pose.bones}
        pair_map = {tname: sname for sname, tname in pairs}
        depth = {b.name: len(b.parent_recursive) for b in arm.pose.bones}
        order = sorted(pair_map, key=lambda n: depth[n])
        to_arm = arm.matrix_world.inverted()
        src_to_arm = to_arm @ src.matrix_world
        ident = mathutils.Matrix.Identity(4)

        corr, rel_rest = {}, {}
        for tname in order:
            want = T_POSE_REF.get(tname.lower())
            rs = ref_rot(src.pose.bones[pair_map[tname]].bone, want, src_to_arm)
            rt = ref_rot(arm.pose.bones[tname].bone, want, ident)
            corr[tname] = rs.inverted() @ rt
            par = parent_of[tname]
            rel_rest[tname] = (rest_t[par].inverted() @ rest_t[tname]) if par \
                else rest_t[tname]

        # Blender composes a posed bone as  W = W_parent . Rl . b, so the local
        # rotation that lands it on the orientation we want is
        # (W_parent . Rl)^-1 . W. Parents have to be solved first: that is what
        # the depth sort above is for.
        # A CMU performer is an ordinary build. This character is a broad man in
        # a heavy coat, and his rig is fitted to that silhouette -- his hands
        # rest 0.255 of his height out from the centre line because that is
        # where the art puts them. Copying arm orientations straight across
        # walked his hands in to 0.067, which is inside his own chest.
        #
        # The rest pose is the measurement of where the body actually is, so
        # take it as the floor: an arm hanging at the side may not sit closer in
        # than its own rest, less a small margin so it can still brush the body.
        down = mathutils.Vector((0.0, 0.0, -1.0))
        fwd = mathutils.Vector((0.0, 1.0, 0.0))
        world_names = set(order)
        abduction = {}
        for tname in order:
            if tname.lower() not in ("leftarm", "rightarm"):
                continue
            d = rest_t[tname] @ mathutils.Vector((0.0, 1.0, 0.0))
            if d.z < -0.1:
                # signed, so an arm that has swung PAST the centre line is the
                # furthest from its floor rather than exempt from it
                abduction[tname] = math.atan2(d.x, -d.z) * 0.9

        # The correction has to swing the whole arm about the shoulder. Applied
        # to the upper arm alone it opened the shoulder and left the forearm on
        # the source's absolute orientation, so the hand simply folded back in
        # and only the elbow angle changed.
        below = {}
        for tname in abduction:
            side = "Left" if tname.lower().startswith("left") else "Right"
            below[tname] = [b for b in (f"{side}ForeArm", f"{side}Hand")
                            if b in world_names]

        def arm_clear_fix(q, limit):
            """The rotation that opens a hanging arm back out to its rest
            clearance, or None if it is already clear or is not hanging."""
            d = q @ mathutils.Vector((0.0, 1.0, 0.0))
            if d.z > -0.3:                  # raised or thrown forward: leave it
                return None
            delta = limit - math.atan2(d.x, -d.z)
            if (limit >= 0.0) == (delta <= 0.0):
                return None                 # already out at least this far
            # about +Y a positive angle swings the bone's tip toward -x, so the
            # correction that opens the arm outward is the negated deficit
            return mathutils.Quaternion(fwd, -delta)

        track = {tname: [] for _sname, tname in pairs}
        for f in range(f0, f1 + 1):
            bpy.context.scene.frame_set(f)
            # pose_bone.matrix is evaluated, not driven straight by the animation
            # system the way matrix_basis is, so it reads stale without this --
            # which produced a sumo squat: feet 0.55 of body height apart and
            # knees bent 58 degrees, from a source frame standing straight.
            bpy.context.view_layer.update()
            world = {}
            for tname in order:
                sb = src.pose.bones[pair_map[tname]]
                world[tname] = (src_to_arm @ sb.matrix).to_quaternion() @ corr[tname]
            for tname, limit in abduction.items():
                fix = arm_clear_fix(world[tname], limit)
                if fix is None:
                    continue
                world[tname] = fix @ world[tname]
                for child in below[tname]:
                    world[child] = fix @ world[child]
            for tname in order:
                par = parent_of[tname]
                # an unmatched parent (a cloak root, say) stays at its rest
                wp = world.get(par, rest_t[par]) if par else \
                    mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
                track[tname].append((wp @ rel_rest[tname]).inverted() @ world[tname])

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
        if str(clip_name).lower().split("_")[0] == "idle":
            breathe(track)
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


# A standing idle is the one clip that must NOT come from mocap.
#
# The mesh is generated from the concept sheet, and the rig is fitted to that
# mesh, so the rest pose already IS the pose the art defines -- arms out at
# 0.255 of body height with the elbows at 31 degrees, for this character.
# Retargeting somebody else's "standing still" trial replaces that with the
# performer's own habits: CMU 111_28 stands with his arms folded, which put
# the hands on the body's centre line (0.016) and bent the elbows to 96
# degrees. The character stopped looking like his own concept art.
#
# So build the idle from rest instead, and add only what a person standing
# still actually does: breathe, shift their weight, and drift their head. The
# periods are set from the clip length so every one of them closes on the loop,
# and they are deliberately different lengths so the cycle never visibly
# repeats as a single beat.
_IDLE_SECONDS = 8.0


def idle_from_rest(arm):
    """Keyframe a breathing, weight-shifting stand on top of the rig's rest pose."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    n = max(2, int(_IDLE_SECONDS * FPS))
    have = {pb.name: pb for pb in arm.pose.bones}

    def key(name, f, x=0.0, y=0.0, z=0.0):
        pb = have.get(name)
        if pb is None:
            return
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (x, y, z)
        pb.keyframe_insert("rotation_euler", frame=f)
        if f == 0:
            # keyed per clip: a retargeted sibling sets QUATERNION and these
            # euler keys would be ignored on replay
            pb.keyframe_insert("rotation_mode", frame=f)

    for f in range(n + 1):
        u = f / n
        breath = breath_phase(u * 2.0)                 # two breaths per loop
        sway = math.sin(2 * math.pi * u)               # one weight shift per loop
        drift = math.sin(2 * math.pi * u * 3.0)        # three slow head drifts
        # the arms hang off the ribcage, so they answer the breath a beat late
        lag = breath_phase(u * 2.0 - 0.12)

        for name, w in _BREATH_BONES.items():
            key(name, f, x=0.090 * w * breath)
        # weight shift: the hips roll and the far knee softens to take it
        key("Hips", f, y=0.045 * sway, z=0.028 * sway)
        key("LeftLeg", f, x=-0.045 * max(0.0, sway))
        key("RightLeg", f, x=-0.045 * max(0.0, -sway))
        # arms drift out and back with the chest instead of hanging rigid
        key("LeftArm", f, z=-0.075 * lag - 0.030 * sway)
        key("RightArm", f, z=0.075 * lag - 0.030 * sway)
        key("LeftForeArm", f, x=-0.055 * lag)
        key("RightForeArm", f, x=-0.055 * lag)
        key("Neck", f, y=0.040 * drift, z=0.024 * sway)

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[motion] idle: built from the rig's rest pose "
          f"({n} frames, {_IDLE_SECONDS:.0f}s loop)")


# --- authored movesets ------------------------------------------------------
#
# Mined mocap tears this character. A CMU shadowboxing trial lunges deep and
# twists hard, and on a 2.80 m body with a heavy coat the skin -- bound at rest
# -- stretches into streaks. The rest-pose idle proved the other direction
# works: poses near the bind pose hold together.
#
# So strikes are authored rather than retargeted, against a named reference used
# as a specification. Pious Force is built to the grappler archetype: planted
# feet, slow committed swings, damage in the body rather than in the hands. The
# legs barely leave rest on purpose -- that is both what a grappler looks like
# and what keeps the mesh intact.
#
# Directions are world-space in the rig's own frame: the character faces -Y,
# up is +Z, his left is -X and his right is +X. A bone not named in a keyframe
# stays at rest, so every vertex the pose does not speak for stays exactly where
# the binding put it.
_F, _B = (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)
_U, _D = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
_L, _R = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)


def _dir(*parts):
    """Blend named directions by weight: _dir((_F, 2), (_D, 1)) is forward-down.

    Plain arithmetic rather than mathutils, because the moveset table below is
    built at import time and this module is imported outside Blender by the
    tests.
    """
    x = sum(d[0] * w for d, w in parts)
    y = sum(d[1] * w for d, w in parts)
    z = sum(d[2] * w for d, w in parts)
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return (x / n, y / n, z / n)


_ARM = ("Arm", "ForeArm", "Hand")
_LEG = ("UpLeg", "Leg", "Foot")


def _fist(sx, fwd, out, up, elbow=(0.0, -0.3, -1.0)):
    """Put one fist at a target, measured in arm-lengths from its own shoulder.

    Strikes used to be authored as a direction per bone, which made the fist's
    position an emergent property of three separate guesses. It went wrong in
    the way that is hard to see and impossible to tune: the neutral guard
    already held the hand two thirds of the way into a punch, so a jab had
    almost no visible travel and had to wind up half a metre behind the
    shoulder to read as a strike at all -- a windmill.

    A fist target is what a fight animator actually specifies, and stated in
    arm-lengths it is rig-independent: 1.0 forward is full extension for a
    2.8 m giant and for a 1.6 m one. `out` is away from the body on that side,
    so a negative value crosses the centre line -- which is what a hook does.
    """
    return _limb(sx, _ARM, fwd, out, up, elbow)


def _limb(sx, chain, fwd, out, up, pole):
    side = "Left" if sx < 0 else "Right"
    target = (fwd, out * (1.0 if sx > 0 else -1.0), up)
    # keyed per limb, not per side, so a move can pose an arm and a leg at once
    return {"_ik": {side + chain[0]: (side, chain, target, pole)}}


def _fists(fwd, out, up, elbow=(0.0, -0.3, -1.0)):
    """Both fists, mirrored."""
    return _merge(_fist(-1, fwd, out, up, elbow), _fist(1, fwd, out, up, elbow))


def _foot(sx, fwd, out, up, knee=(1.0, 0.0, 0.0)):
    """Put a foot at a target, in leg-lengths from that hip.

    Same idea as _fist and the same reason: a kick is specified by where the
    foot ends up. The knee pole points forward by default because that is the
    only way a knee bends, and getting it wrong is not a subtle error.
    """
    return _limb(sx, _LEG, fwd, out, up, knee)


def _stand():
    """Both feet under the hips. Neutral for anything that is not a kick."""
    return _merge(_foot(-1, 0.02, 0.12, -0.97), _foot(1, 0.02, 0.12, -0.97))


def _guard():
    """A grappler's guard: hands open, low, in front of the belly, elbows out.

    A boxer hides behind his gloves. This man is waiting to grab you, so the
    hands sit where they can close on something rather than where they protect
    the head -- and low enough that a punch from here has somewhere to travel.
    """
    return _fists(0.34, 0.30, -0.36)


def solve_limb(arm, side, chain, target, pole, rest_w):
    """Two-bone IK from an end-effector target: the joint angles that reach it.

    Returns world rotations for the three arm bones, or {} if the rig has no
    such chain. The elbow is placed by a pole vector rather than left free,
    because the plane the elbow bends in is what separates a punch from a
    chicken wing, and a direction alone does not determine it.
    """
    names = [f"{side}{part}" for part in chain]
    if any(n not in arm.pose.bones for n in names):
        return {}
    upper, lower, tip = (arm.pose.bones[n].bone for n in names)
    root = upper.head_local
    l1 = (lower.head_local - upper.head_local).length
    l2 = (tip.head_local - lower.head_local).length
    if l1 <= 0.0 or l2 <= 0.0:
        return {}

    v = mathutils.Vector((target[1], -target[0], target[2]))   # out, fwd, up
    if v.length > 0.98:
        v = v.normalized() * 0.98      # never fully lock the elbow straight
    goal = root + v * (l1 + l2)

    reach = goal - root
    d = max(1e-5, min(reach.length, l1 + l2 - 1e-4))
    u = reach.normalized()
    cos_a = max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)))
    a = math.acos(cos_a)

    pole_v = mathutils.Vector((pole[1], -pole[0], pole[2]))
    axis = u.cross(pole_v)
    if axis.length < 1e-6:
        axis = u.cross(mathutils.Vector((0.0, 0.0, 1.0)))
    if axis.length < 1e-6:
        return {}
    axis.normalize()

    # Both rotation signs reach the target; the one that swings the elbow
    # toward the pole is the one that looks like an arm.
    best = None
    for sign in (1.0, -1.0):
        cand = u.copy()
        cand.rotate(mathutils.Quaternion(axis, sign * a))
        score = (cand * l1).dot(pole_v)
        if best is None or score > best[0]:
            best = (score, cand)
    upper_dir = best[1]

    joint = root + upper_dir * l1
    fore_dir = (goal - joint)
    if fore_dir.length < 1e-6:
        fore_dir = upper_dir.copy()
    fore_dir.normalize()

    # The check that matters, run on the real rig on every build: the solved
    # chain has to actually end where the move said the fist goes. Reaching the
    # target is guaranteed by the law of cosines only if the axis and sign are
    # right, and those are exactly what a silent geometry bug gets wrong.
    landed = joint + fore_dir * l2
    miss = (landed - goal).length / (l1 + l2)
    if miss > 0.02:
        print(f"[motion] WARNING {names[0]} missed its target by "
              f"{miss * 100:.0f}% of arm reach")

    return {names[0]: aim_world(rest_w[names[0]], upper_dir),
            names[1]: aim_world(rest_w[names[1]], fore_dir),
            names[2]: aim_world(rest_w[names[2]], fore_dir)}


def _bend(pitch, twist):
    """A tilt away from whatever this rig's rest direction happens to be.

    Returned as a function of the rest direction rather than a fixed vector,
    because the torso must be posed RELATIVE to the bind pose. Aiming the spine
    at absolute vertical looks right in the abstract and is wrong here: the
    concept art this character was built from stands with a slight backward
    lean, so an absolute aim straightened him on frame one of every strike and
    snapped the silhouette off the idle that was approved.
    """
    def tilt(rest):
        v = (rest + mathutils.Vector(_F) * pitch + mathutils.Vector(_R) * twist)
        return tuple(v.normalized())
    return tilt


def _lean(pitch=0.0, twist=0.0):
    """Torso attitude: pitch forward/back, twist toward one side.

    Neutral keys nothing at all, so an unleaned frame is the rest pose exactly.
    """
    if not pitch and not twist:
        return {}
    return {
        "Hips": _bend(pitch * 0.08, twist * 0.06),
        "Spine": _bend(pitch * 0.20, twist * 0.10),
        "Spine1": _bend(pitch * 0.27, twist * 0.16),
    }


def _merge(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            if k == "_ik":
                out.setdefault("_ik", {}).update(v)   # one entry per side
            else:
                out[k] = v
    return out


GRAPPLER = {
    # The light button. A grappler's fast punch is still a shovel, but it has
    # next to no windup -- the fist starts low and goes, and the weight is in
    # the step behind it rather than in a backswing.
    "jab": dict(seconds=0.45, keys=[
        (0.00, _merge(_guard(), _lean())),
        (0.28, _merge(_guard(), _fist(1, 0.18, 0.26, -0.28),
                      _lean(pitch=-0.15, twist=0.35))),
        (0.52, _merge(_fist(-1, 0.30, 0.32, -0.40), _fist(1, 0.88, 0.14, -0.12),
                      _lean(pitch=0.45, twist=-0.35))),
        (1.00, _merge(_guard(), _lean())),
    ]),
    # The heavy hook. The fist chambers beside the ribs, never behind the back:
    # the power in this punch is the torso turning through it, and a hand that
    # travels behind its own shoulder reads as a windmill rather than a strike.
    "cross": dict(seconds=0.75, keys=[
        (0.00, _merge(_guard(), _lean())),
        (0.30, _merge(_fist(-1, 0.34, 0.30, -0.30), _fist(1, 0.04, 0.46, -0.16),
                      _lean(pitch=-0.25, twist=0.85))),
        (0.58, _merge(_fist(-1, 0.22, 0.36, -0.42), _fist(1, 0.78, -0.24, 0.02),
                      _lean(pitch=0.50, twist=-0.85))),
        (1.00, _merge(_guard(), _lean())),
    ]),
    # Headbutt. Hands stay in guard the whole way -- this one is all spine, and
    # the hit is the skull arriving after the lean has already committed.
    "overhand": dict(seconds=0.85, keys=[
        (0.00, _merge(_guard(), _lean())),
        (0.34, _merge(_guard(), _lean(pitch=-0.7),
                      {"Neck": _dir((_U, 2.0), (_B, 1.0)),
                       "Head": _dir((_U, 2.0), (_B, 1.2))})),
        (0.56, _merge(_fists(0.52, 0.34, -0.30), _lean(pitch=1.3),
                      {"Neck": _dir((_F, 1.6), (_U, 0.5)),
                       "Head": _dir((_F, 2.0), (_D, 0.3))})),
        (1.00, _merge(_guard(), _lean())),
    ]),
    # Spinning lariat: both arms locked straight out sideways and the body
    # carried round by them. The arms lead the turn and then hold while the
    # torso catches up, which is what makes it read as spinning rather than
    # as two separate arm swings.
    "left_uppercut": dict(seconds=1.05, keys=[
        (0.00, _merge(_guard(), _lean())),
        (0.22, _merge(_fist(-1, -0.20, 0.90, 0.10), _fist(1, 0.30, 0.86, 0.10),
                      _lean(twist=-1.0))),
        (0.62, _merge(_fists(0.0, 0.94, 0.06), _lean(pitch=0.3, twist=1.0))),
        (1.00, _merge(_guard(), _lean())),
    ]),
    # Command grab: both hands out, clamp shut, then haul the whole weight up
    # off the ground. The lean reverses on the lift -- he sits back under it.
    "right_uppercut": dict(seconds=1.10, keys=[
        (0.00, _merge(_guard(), _lean())),
        (0.30, _merge(_fists(0.80, 0.26, -0.14), _lean(pitch=0.7))),
        (0.52, _merge(_fists(0.72, 0.08, 0.02), _lean(pitch=0.9))),
        (0.78, _merge(_fists(0.34, 0.24, 0.78), _lean(pitch=-0.6))),
        (1.00, _merge(_guard(), _lean())),
    ]),
    # Body splash: everything overhead, then everything down at once.
    "left_bodyshot": dict(seconds=0.80, keys=[
        (0.00, _merge(_guard(), _lean())),
        (0.36, _merge(_fists(0.10, 0.42, 0.86), _lean(pitch=-0.5))),
        (0.58, _merge(_fists(0.66, 0.30, -0.48), _lean(pitch=1.4))),
        (1.00, _merge(_guard(), _lean())),
    ]),
}

def _tight():
    """A striker's guard: hands high and tight, elbows in, bladed to the front.

    The opposite of the grappler's open hands. She is not waiting to catch
    anything -- she is covering her own head so she can throw from behind it.
    """
    return _merge(_fists(0.30, 0.15, 0.24, elbow=(0.0, -0.15, -1.0)), _stand())


# Veiled Shadow is built to the striker archetype, drawn from the same source
# the grappler was: a named fighter used as a specification rather than copied.
# Where the grappler plants and swings, she is bladed, quick and leg-led -- the
# heavy buttons are all kicks, the startup is short, and nothing commits the
# whole body the way a lariat does.
STRIKER = {
    # Straight jab off the front hand. Almost no travel and back instantly;
    # this is the button she uses to make room, not to hurt anyone.
    "jab": dict(seconds=0.28, keys=[
        (0.00, _merge(_tight(), _lean())),
        (0.45, _merge(_tight(), _fist(1, 0.92, 0.06, 0.16),
                      _lean(pitch=0.35, twist=-0.30))),
        (1.00, _merge(_tight(), _lean())),
    ]),
    # Spinning backfist: the turn is the attack. The fist stays close to its own
    # shoulder and the torso whips it round, which is why it beats a hook to the
    # same spot despite travelling less.
    "cross": dict(seconds=0.42, keys=[
        (0.00, _merge(_tight(), _lean())),
        (0.30, _merge(_tight(), _fist(1, 0.10, 0.40, 0.24),
                      _lean(twist=1.1))),
        (0.62, _merge(_tight(), _fist(1, 0.72, -0.34, 0.18),
                      _lean(pitch=0.30, twist=-1.1))),
        (1.00, _merge(_tight(), _lean())),
    ]),
    # Rising knee. The foot tucks under while the thigh drives up, so the knee
    # is the leading edge -- the pole vector is what makes that read.
    "overhand": dict(seconds=0.45, keys=[
        (0.00, _merge(_tight(), _lean())),
        (0.34, _merge(_tight(), _foot(1, 0.30, 0.12, -0.62,
                                      knee=(1.0, 0.0, 0.5)),
                      _lean(pitch=0.30))),
        (0.60, _merge(_tight(), _foot(1, 0.34, 0.10, -0.30,
                                      knee=(1.0, 0.0, 0.9)),
                      _lean(pitch=-0.45))),
        (1.00, _merge(_tight(), _lean())),
    ]),
    # Rising anti-air kick: one long line from the planted foot to the toe,
    # thrown almost vertically. Everything else gets out of its way.
    "left_uppercut": dict(seconds=0.60, keys=[
        (0.00, _merge(_tight(), _lean())),
        (0.26, _merge(_tight(), _foot(1, 0.22, 0.12, -0.72),
                      _lean(pitch=0.35))),
        (0.56, _merge(_fist(-1, 0.20, 0.30, 0.30), _fist(1, 0.10, 0.22, -0.20),
                      _stand(), _foot(1, 0.50, 0.10, 0.70),
                      _lean(pitch=-0.85))),
        (1.00, _merge(_tight(), _lean())),
    ]),
    # Diving spiral thrust along the ground. Without root motion the travel has
    # to be sold by the shape: she folds down over a fully committed front leg.
    "left_bodyshot": dict(seconds=0.70, keys=[
        (0.00, _merge(_tight(), _lean())),
        (0.28, _merge(_tight(), _foot(1, 0.34, 0.12, -0.66),
                      _lean(pitch=-0.35))),
        (0.58, _merge(_fists(0.42, 0.20, -0.10), _stand(),
                      _foot(1, 0.94, 0.08, -0.24), _lean(pitch=1.5))),
        (1.00, _merge(_tight(), _lean())),
    ]),
    # High roundhouse: chambered tight, then whipped out flat at head height.
    "right_uppercut": dict(seconds=0.55, keys=[
        (0.00, _merge(_tight(), _lean())),
        (0.30, _merge(_tight(), _foot(1, 0.28, 0.40, -0.60,
                                      knee=(0.6, 0.8, 0.2)),
                      _lean(twist=0.7))),
        (0.58, _merge(_fist(-1, 0.24, 0.34, 0.20), _fist(1, 0.06, 0.26, -0.24),
                      _stand(), _foot(1, 0.66, 0.34, 0.52,
                                      knee=(0.6, 0.5, 0.3)),
                      _lean(pitch=-0.55, twist=-0.7))),
        (1.00, _merge(_tight(), _lean())),
    ]),
}

MOVESETS = {"grappler": GRAPPLER, "striker": STRIKER}


def authored_clip(arm, clip, moveset):
    """Keyframe a move from its authored poses. Returns True if it exists here."""
    spec = MOVESETS.get(moveset, {}).get(clip.lower())
    if not spec:
        return False
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    n = max(2, int(spec["seconds"] * FPS))
    rest_w = {b.name: b.bone.matrix_local.to_quaternion() for b in arm.pose.bones}
    ident = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    frames, track = [], {}
    for t, dirs in spec["keys"]:
        frames.append(int(round(t * n)))
        want = {}
        for b, d in dirs.items():
            if b == "_ik" or b not in arm.pose.bones:
                continue
            rq = rest_w[b]
            if callable(d):     # a tilt relative to this rig's own rest pose
                d = d(rq @ mathutils.Vector((0.0, 1.0, 0.0)))
            want[b] = aim_world(rq, d)
        for side, chain, target, pole in dirs.get("_ik", {}).values():
            want.update(solve_limb(arm, side, chain, target, pole, rest_w))
        local = solve_local(arm, want)
        for pb in arm.pose.bones:
            if any(k in pb.name.lower()
                   for k in ("tail", "jaw", "wing", "cloak", "cape")):
                continue        # procedural_extras owns these, in euler
            track.setdefault(pb.name, []).append(local.get(pb.name, ident))

    # Blender interpolates a quaternion's four channels as four independent
    # F-curves rather than slerping them. Two consecutive keys in opposite
    # hemispheres describe the same two poses but blend the long way round, and
    # the character visibly tumbles flat between them -- which is exactly what
    # the first render of these moves did on every contact frame.
    align_signs(track)

    for name, quats in track.items():
        pb = arm.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        for f, q in zip(frames, quats):
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=f)
        pb.keyframe_insert("rotation_mode", frame=0)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[motion] {clip}: authored {moveset} move "
          f"({n} frames, {spec['seconds']:.2f}s)")
    return True


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
    # Span whatever the body clip already occupies. A fixed 30-frame cycle here
    # extended every action to a round 1.00 s, so a 17-frame mined jab shipped
    # with 13 frames of nothing on the end and combat.gd, scaling playback to
    # match its frame data, stretched the punch to cover the padding.
    act = arm.animation_data.action if arm.animation_data else None
    if act and act.fcurves:
        f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    else:
        f0, f1 = 0, FPS
    dur = max(1, f1 - f0)

    # These parts are BONE CHAINS, and a bone's rotation is relative to its
    # parent, so an amplitude written to every bone in a chain compounds. Six
    # cloak segments at 0.15 rad each put the hem 65 degrees off vertical and
    # the cloak stood out around the character like a ballgown. Divide by the
    # chain's length so the number below means the swing at the TIP, whatever
    # the rig's segment count happens to be.
    def kind_of(pb):
        n = pb.name.lower()
        for k in ("tail", "jaw", "wing"):
            if k in n:
                return k
        return "cloak" if ("cloak" in n or "cape" in n) else ""

    chain = {}
    for pb in bones:
        chain.setdefault(kind_of(pb), []).append(pb)
    index = {pb.name: i for group in chain.values()
             for i, pb in enumerate(sorted(group, key=lambda b: b.name))}

    for f in range(f0, f1 + 1):
        phase = 2 * math.pi * (f - f0) / dur
        for pb in bones:
            kind = kind_of(pb)
            if not kind:
                continue
            n = len(chain[kind])
            # each segment lags the one above it, so the chain travels as a wave
            # instead of swinging rigid -- that lag is what reads as cloth
            lag = phase - 0.45 * index[pb.name]
            pb.rotation_mode = "XYZ"
            if kind == "tail" and "tail" in extras:
                pb.rotation_euler = (0, 0, 0.8 / n * math.sin(lag))
            elif kind == "jaw" and "jaw" in extras:
                pb.rotation_euler = (max(0.0, math.sin(phase)) if clip == "attack" else 0, 0, 0)
            elif kind == "wing" and "wings" in extras:
                pb.rotation_euler = (0, 0.9 / n * math.sin(lag), 0)
            elif kind == "cloak" and "cloak" in extras:
                pb.rotation_euler = (0.30 / n * math.sin(lag), 0,
                                     0.20 / n * math.sin(lag * 0.7))
            else:
                continue
            pb.keyframe_insert("rotation_euler", frame=f)
    bpy.ops.object.mode_set(mode="OBJECT")


# --- export ----------------------------------------------------------------

def export_glb(path, arm=None, mesh=None):
    """Export the character and its rig -- and nothing else that happens to be
    in the scene.

    This used to export the whole scene, and a stray 42-vertex Icosphere
    spanning the full [-1,1] unit cube rode along inside character.glb. In the
    game its upper hemisphere sat around the character's legs and clipped at
    the floor, which reads exactly like a wide bell-shaped skirt -- so the
    cloak looked like it was flaring even after the rig and the cloth solver
    were both correct, and every fix aimed at those was aimed at the wrong
    thing."""
    keep = {o for o in (arm, mesh) if o is not None}
    if not keep:
        keep = {o for o in bpy.context.scene.objects
                if o.type == "ARMATURE" or o.type == "MESH"}
    keep |= {o for o in bpy.context.scene.objects if o.parent in keep}
    dropped = [o.name for o in bpy.context.scene.objects if o not in keep]
    if dropped:
        print(f"[motion] not exporting {len(dropped)} stray object(s): "
              f"{', '.join(sorted(dropped)[:6])}")
    bpy.ops.object.select_all(action="DESELECT")
    for o in keep:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB",
                              use_selection=True, export_animations=True)


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
        arm = procedural_rig(mesh, body_plan, extras)

    # build every clip as a named action stashed on its own NLA track, then export
    # ONE glb — Godot's AnimationPlayer then has idle/walk/run to switch between,
    # instead of one file per clip of which the game could only ever use one
    for clip in ARGS.get("animations", ["idle"]):
        if arm.animation_data:
            arm.animation_data.action = None    # fresh action per clip
        moveset = ARGS.get("moveset", "")
        if clip.lower().split("_")[0] == "idle":
            idle_from_rest(arm)                 # the rest pose IS the concept art
            PROVENANCE[clip] = "rest-pose idle"
        elif authored_clip(arm, clip, moveset):
            PROVENANCE[clip] = f"authored:{moveset}"
        else:
            bvh = bvh_for(clip, body_plan, ARGS.get("cmu_dir", ""),
                          ARGS.get("kimodo_url", ""), out_dir)
            if not (bvh and retarget_onto(arm, bvh, clip)):
                procedural_clip(arm, clip)      # never leaves the body static
                PROVENANCE[clip] = "procedural" # retarget may have rejected it too
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
    body = next((o for o in bpy.context.scene.objects
                 if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)),
                None)
    export_glb(out, arm, body)
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
