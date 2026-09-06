"""Graft a separately-generated part mesh onto a body mesh.

A single frontal image gives an image-to-3D model very little to say about the
back of a head, so a head generated as part of a whole figure comes out coarse
-- it is a small fraction of the voxel grid and gets the detail that implies.
Generating the head on its own, from the concept sheet's own close-up panel,
spends the entire grid on it and produces a face worth keeping. This joins that
head onto the body.

The neck is found the same way the rig finds it, by reusing blender_motion's
silhouette measurements, so the cut lands where the rig thinks the neck is
rather than at a guessed height. The part is scaled to the width of the region
it replaces and dropped in with a deliberate overlap: a seam that is slightly
buried is invisible, and a seam with a gap is a hole in the character.

    blender -b --python blender_graft.py -- '{"body": "...", "part": "...",
                                              "out": "...", "overlap": 0.25}'
"""
import json
import os
import sys

import bpy
import mathutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_motion as bm            # silhouette / neck_frac / shoulder_frac


def scene_meshes():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def bounds(objs):
    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    lo = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts),
                           min(p.z for p in pts)))
    hi = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts),
                           max(p.z for p in pts)))
    return lo, hi


def import_glb(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    added = [o for o in bpy.context.scene.objects if o not in before]
    for o in added:                    # bake the importer's transform in
        if o.type == "MESH":
            o.data.transform(o.matrix_world)
            o.matrix_world = mathutils.Matrix.Identity(4)
    return [o for o in added if o.type == "MESH"]


def neck_height(body):
    """Where the rig would put the neck, in world z."""
    lo, hi = bounds(body)
    pts = [o.matrix_world @ v.co for o in body for v in o.data.vertices]
    prof = bm.silhouette(pts, lo.z, hi.z)
    sh = bm.shoulder_frac(prof, lo.z, hi.z)
    return lo.z + (hi.z - lo.z) * bm.neck_frac(prof, lo.z, hi.z, sh), lo, hi


def cut_above(objs, z):
    """Delete every vertex above z. Returns how many went."""
    gone = 0
    for o in objs:
        me = o.data
        doomed = [v.index for v in me.vertices if (o.matrix_world @ v.co).z > z]
        if not doomed:
            continue
        gone += len(doomed)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for i in doomed:
            me.vertices[i].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")
    return gone


def fit_part(part, target_lo, target_hi, cut_z, overlap):
    """Scale and place the part to fill the region that was cut away.

    Width drives the scale rather than height: a generated head close-up
    includes however much neck and shoulder the framing happened to catch, so
    its height is not a reliable measure of the head, but the width across the
    skull is.
    """
    lo, hi = bounds(part)
    span = hi - lo
    want = target_hi - target_lo
    if span.x <= 1e-6 or want.x <= 1e-6:
        return 0.0
    scale = want.x / span.x
    # sit the part so its base is buried below the cut by `overlap` of its height
    base = cut_z - span.z * scale * overlap
    centre = mathutils.Vector(((target_lo.x + target_hi.x) / 2,
                               (target_lo.y + target_hi.y) / 2, 0.0))
    src_centre = mathutils.Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, 0.0))
    for o in part:
        o.data.transform(
            mathutils.Matrix.Translation(centre - src_centre * scale
                                         + mathutils.Vector((0, 0, base - lo.z * scale)))
            @ mathutils.Matrix.Scale(scale, 4))
    return scale


def main():
    args = json.loads(sys.argv[sys.argv.index("--") + 1])
    bpy.ops.wm.read_factory_settings(use_empty=True)

    body = import_glb(args["body"])
    if not body:
        raise SystemExit("[graft] body has no mesh")
    cut_z, _lo, _hi = neck_height(body)

    # measure the region about to be removed BEFORE removing it: that is what
    # the replacement has to fill
    pts = [o.matrix_world @ v.co for o in body for v in o.data.vertices
           if (o.matrix_world @ v.co).z > cut_z]
    if not pts:
        raise SystemExit("[graft] nothing above the neck to replace")
    head_lo = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts),
                                min(p.z for p in pts)))
    head_hi = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts),
                                max(p.z for p in pts)))

    part = import_glb(args["part"])
    if not part:
        raise SystemExit("[graft] part has no mesh")
    scale = fit_part(part, head_lo, head_hi, cut_z,
                     float(args.get("overlap", 0.25)))
    gone = cut_above(body, cut_z)
    print(f"[graft] neck at z={cut_z:.3f}, removed {gone} verts, "
          f"part scaled x{scale:.3f}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in body + part:
        o.select_set(True)
    bpy.context.view_layer.objects.active = body[0]
    if len(body + part) > 1:
        bpy.ops.object.join()
    bpy.ops.export_scene.gltf(filepath=args["out"], export_format="GLB",
                              use_selection=True)
    print(f"[graft] wrote {args['out']}")


if __name__ == "__main__":
    main()
