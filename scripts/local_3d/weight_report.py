"""Per-bone skin weight report for a rigged .glb, run inside Blender.

    blender --background --python scripts/local_3d/weight_report.py -- <mesh.glb>

Two numbers decide whether a set of weights is usable, and neither of them is
visible in a screenshot of the rest pose:

  share  -- how much of the model's total weight mass a bone owns. An arm that
            owns 5% of a coated figure is holding a coat panel it should not be
            holding, and that panel lifts with the arm on every punch.
  reach  -- the vertical span holding the middle 90% of a bone's weight MASS,
            as a fraction of the model's height. A forearm whose mass spans 30%
            of the figure is bound to something the length of a leg.

Reach is measured on mass rather than on the outer extent of the vertices a
bone touches at all, because the two rank weighting schemes in opposite
directions. A smooth solution spreads a trace of influence a long way and looks
terrible by extent while behaving correctly; bone heat cuts hard edges and looks
tidy by extent while handing a whole coat panel to one arm. Where the weight
actually sits is the thing that moves the mesh.

Both are wrong in the same direction and for the same reason -- bone heat gives
a hanging panel wholly to whichever limb happens to sit inside it -- so they are
reported together, alongside the count of vertices no bone claims at all, which
are the ones that stay behind when the character moves.
"""
import sys

import bpy


def report(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path, bone_heuristic="TEMPERANCE")
    mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    names = [g.name for g in mesh.vertex_groups]
    zs = [(mesh.matrix_world @ v.co).z for v in mesh.data.vertices]
    height = (max(zs) - min(zs)) or 1.0

    mass = {n: 0.0 for n in names}
    spread = {n: [] for n in names}     # (z, weight) pairs, for the mass span
    unweighted = 0
    for v in mesh.data.vertices:
        groups = [g for g in v.groups if g.weight > 0.01]
        if not groups:
            unweighted += 1
            continue
        z = zs[v.index]
        for g in groups:
            n = names[g.group]
            mass[n] += g.weight
            spread[n].append((z, g.weight))
    total = sum(mass.values()) or 1.0

    def reach_of(n):
        pairs = sorted(spread[n])
        if not pairs:
            return 0.0
        want = sum(w for _, w in pairs)
        seen = 0.0
        lo = hi = pairs[0][0]
        for z, w in pairs:
            seen += w
            if seen < want * 0.05:
                lo = z
            if seen <= want * 0.95:
                hi = z
        return (hi - lo) / height

    print(f"\n{path}")
    print(f"  {len(mesh.data.vertices)} verts, {len(names)} bones, "
          f"{unweighted} unweighted")
    print(f"  {'bone':<20}{'share':>8}{'reach':>8}")
    rows = sorted(mass.items(), key=lambda kv: -kv[1])
    for n, m in rows:
        print(f"  {n:<20}{m / total * 100:7.1f}%{reach_of(n) * 100:7.1f}%")
    worst = max((reach_of(n) for n in names), default=0.0)
    print(f"  worst reach {worst * 100:.1f}% of height, "
          f"{unweighted} unweighted verts")


if __name__ == "__main__":
    for arg in sys.argv[sys.argv.index("--") + 1:]:
        report(arg)
