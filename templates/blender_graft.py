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


def robust_span(pts, lo_z, hi_z, q=0.90):
    """Width, depth and centreline of a mesh, immune to a stray shard.

    Image-to-3D output routinely carries a shard alongside the subject. The head
    close-up this was written for came with a paper-thin plane wider than the
    whole bust: the bust measures 0.33 across and the bounding box read 1.00, so
    a part fitted by its bounding box came out at a seventh of size. Nothing in
    a render shows the shard, and it survives every cheap filter -- it is not a
    loose island (these meshes arrive split into thousands of those, so "keep
    the biggest piece" throws the subject away), and it is not a thin enough
    tail for a percentile over vertices to clip.

    What does isolate it is that it occupies ONE height slice. Measuring width
    per slice and taking a high percentile across slices means a single bad
    slice cannot set the scale, however many vertices it holds.
    """
    prof = [s for s in bm.silhouette(pts, lo_z, hi_z) if s]
    if not prof:
        return 0.0, 0.0, 0.0, 0.0

    def pct(vals):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * q))]

    def mid(vals):
        vals = sorted(vals)
        return vals[len(vals) // 2]

    return (pct(s["half_w"] * 2 for s in prof), pct(s["half_d"] * 2 for s in prof),
            mid(s["xc"] for s in prof), mid(s["yc"] for s in prof))


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


def delete_where(objs, doomed):
    """Delete every vertex for which `doomed(world_co)` is true."""
    gone = 0
    for o in objs:
        me = o.data
        hit = [v.index for v in me.vertices if doomed(o.matrix_world @ v.co)]
        if not hit:
            continue
        gone += len(hit)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for i in hit:
            me.vertices[i].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")
    return gone


def fit_part(part, target_pts, cut_z, overlap):
    """Scale and place the part to fill the region that was cut away.

    Width drives the scale rather than height: a generated head close-up
    includes however much neck and shoulder the framing happened to catch, so
    its height is not a reliable measure of the head, but the width across the
    skull is. Both meshes are measured by robust_span, so the comparison is
    like for like.
    """
    pts = [o.matrix_world @ v.co for o in part for v in o.data.vertices]
    lo, hi = bounds(part)
    pw, pd, pxc, pyc = robust_span(pts, lo.z, hi.z)

    # the shard has to go as well as be ignored: scaled up with everything else
    # it would cross the finished character as a visible plate
    tossed = delete_where(part, lambda p: abs(p.x - pxc) > pw or abs(p.y - pyc) > pd)
    if tossed:
        print(f"[graft] trimmed {tossed} verts outside the part's silhouette")
        lo, hi = bounds(part)

    t_lo = mathutils.Vector((min(p.x for p in target_pts), min(p.y for p in target_pts),
                             min(p.z for p in target_pts)))
    t_hi = mathutils.Vector((max(p.x for p in target_pts), max(p.y for p in target_pts),
                             max(p.z for p in target_pts)))
    tw, _td, txc, tyc = robust_span(target_pts, t_lo.z, t_hi.z)
    if pw <= 1e-6 or tw <= 1e-6:
        return 0.0
    scale = tw / pw

    # sit the part so its base is buried below the cut by `overlap` of its height
    base = cut_z - (hi.z - lo.z) * scale * overlap
    offset = mathutils.Vector((txc - pxc * scale, tyc - pyc * scale,
                               base - lo.z * scale))
    for o in part:
        o.data.transform(mathutils.Matrix.Translation(offset)
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
    part = import_glb(args["part"])
    if not part:
        raise SystemExit("[graft] part has no mesh")
    scale = fit_part(part, pts, cut_z, float(args.get("overlap", 0.25)))
    gone = delete_where(body, lambda p: p.z > cut_z)
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
