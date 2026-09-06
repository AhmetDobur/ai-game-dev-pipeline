"""Objective mesh-health report for a rigged .glb, run inside Blender.

    blender --background --python mesh_report.py -- <mesh.glb> [...]

Everything here is a count, not an opinion: how many separate shells the mesh is
in, how much of it is paper-thin shards, how many edges bound nothing. These are
the numbers that decide whether a character needs a repair pass or a rebuild.

Import note: the glTF importer's default bone display creates a 42-vertex
"Icosphere" spanning a 2 m cube for every armature it reads. It is an artifact
of the IMPORT, not something in the file -- the .glb itself contains one mesh --
but it lands in bpy.data, inflates any bounding box computed over "all meshes",
and reads exactly like a stray object shipping inside the character. Passing
bone_heuristic="TEMPERANCE" stops it being created.
"""
import sys
from collections import defaultdict

import bmesh
import bpy
import mathutils


def report(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path, bone_heuristic="TEMPERANCE")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    print(f"\n=== {path}")
    print(f"  objects: {len(meshes)} mesh, {len(arms)} armature")
    print(f"  clips: {len(bpy.data.actions)} -> "
          f"{sorted(a.name for a in bpy.data.actions)[:12]}")

    for ob in meshes:
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        raw = len(bm.verts)
        # glTF stores one vertex per (position, normal, UV) tuple and Blender's
        # importer keeps that split (merge_vertices defaults off), so every UV
        # seam and hard edge reads as a topological cut. Counting shells on the
        # unwelded mesh measures the FILE FORMAT, not the model. Weld on
        # position first -- the rig and the motion stage both already do
        # (blender_motion.py welds at 2e-4 before parenting).
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-4)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # connected shells, by flood fill over edges
        seen = set()
        shells = []
        for v in bm.verts:
            if v.index in seen:
                continue
            stack, comp = [v], []
            seen.add(v.index)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for e in cur.link_edges:
                    other = e.other_vert(cur)
                    if other.index not in seen:
                        seen.add(other.index)
                        stack.append(other)
            shells.append(comp)
        shells.sort(key=len, reverse=True)
        total = len(bm.verts)
        biggest = len(shells[0]) if shells else 0

        nonmanifold = sum(1 for e in bm.edges if not e.is_manifold)
        boundary = sum(1 for e in bm.edges if e.is_boundary)
        tiny = sum(1 for f in bm.faces if f.calc_area() < 1e-8)

        co = [ob.matrix_world @ v.co for v in bm.verts]
        lo = mathutils.Vector([min(c[k] for c in co) for k in range(3)])
        hi = mathutils.Vector([max(c[k] for c in co) for k in range(3)])
        dims = hi - lo

        print(f"  {ob.name}: {raw} verts as stored, {total} after welding, "
              f"{len(bm.faces)} faces")
        print(f"    shells       {len(shells)}  (largest holds "
              f"{biggest / max(total, 1):.1%} of verts)")
        print(f"    fragments    {sum(1 for s in shells if len(s) < 50)} shells "
              f"under 50 verts, {sum(len(s) for s in shells if len(s) < 50)} verts")
        print(f"    non-manifold {nonmanifold} edges, {boundary} boundary edges "
              f"({boundary / max(len(bm.edges), 1):.1%})")
        print(f"    degenerate   {tiny} faces")
        print(f"    bbox         {dims.x:.2f} x {dims.y:.2f} x {dims.z:.2f} "
              f"(height {dims.z:.2f})")

        # how much of the mesh lives out at the extremities -- a hand fused into
        # a sleeve reads as one shell where two are expected
        groups = defaultdict(int)
        names = [g.name for g in ob.vertex_groups]
        for v in ob.data.vertices:
            best = max(v.groups, key=lambda g: g.weight, default=None)
            if best is not None:
                groups[names[best.group]] += 1
        for want in ("LeftHand", "RightHand", "LeftForeArm", "RightForeArm"):
            print(f"    owned by {want:<13} {groups.get(want, 0)} verts")
        bm.free()


if __name__ == "__main__":
    for arg in sys.argv[sys.argv.index("--") + 1:]:
        report(arg)
