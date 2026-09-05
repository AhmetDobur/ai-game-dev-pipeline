# Headless GLB preview render + objective mesh metrics:
#   blender --background --python blender_preview.py -- <glb> <out.png>
# Writes <out.png> and <glb>.metrics.json. The metrics exist so the pipeline can
# reject a mesh WITHOUT a human looking at it: the failures this catches are the
# ones that actually shipped -- a character reconstructed as a featureless block,
# and a texture baked at cfg 1.0 that came out near-uniform.
# One 3/4 view with neutral studio lighting — enough to see whether a generated
# mesh is a character or a rectangle. Never used in the game itself.
import math
import sys

import bpy
import mathutils

glb, out_png = sys.argv[sys.argv.index("--") + 1:][:2]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh in glb")

pts = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
lo = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
center = (lo + hi) / 2
size = max((hi - lo).length, 0.01)

cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.scene.collection.objects.link(cam)
offset = mathutils.Vector((1, -1, 0.6)).normalized() * size * 1.6
cam.location = center + offset
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_data.clip_end = size * 10
bpy.context.scene.camera = cam

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
bpy.context.scene.collection.objects.link(sun)
fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "SUN"))
fill.data.energy = 1
fill.rotation_euler = (math.radians(-40), 0, math.radians(-120))
bpy.context.scene.collection.objects.link(fill)

world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.2, 0.2, 0.22, 1)
bpy.context.scene.world = world

sc = bpy.context.scene
try:
    sc.render.engine = "BLENDER_EEVEE_NEXT"   # Blender 4.2+
except TypeError:
    sc.render.engine = "BLENDER_EEVEE"        # older Blender
sc.render.resolution_x = sc.render.resolution_y = 512
sc.render.filepath = out_png
bpy.ops.render.render(write_still=True)


# ------------------------------------------------------------------ metrics
import json

metrics = {"bbox": [round(v, 4) for v in (hi - lo)],
           "verts": sum(len(o.data.vertices) for o in meshes),
           "faces": sum(len(o.data.polygons) for o in meshes),
           "objects": len(meshes)}

dx, dy, dz = (hi - lo)
longest = max(dx, dy, dz) or 1.0
shortest = min(dx, dy, dz) or 1.0
metrics["aspect"] = round(longest / shortest, 4)

# top-slice width: how wide the mesh still is in its top 8%, relative to its
# widest point. A whole standing figure tapers to a head there (~0.29 measured
# on a good character); one the remesher cut off at the chest or waist is still
# nearly full width (0.97 and 0.69 measured). This is the truncation detector --
# height/width cannot do the job, because a robed character measures 1.91x
# taller than wide and a decapitated one 1.85x.
try:
    pts = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
    top = hi.z - (hi.z - lo.z) * 0.08
    band = [p_ for p_ in pts if p_.z >= top]
    widest = max(dx, dy) or 1.0
    if band:
        bw = max(max(p_.x for p_ in band) - min(p_.x for p_ in band),
                 max(p_.y for p_ in band) - min(p_.y for p_ in band))
        metrics["top_width_ratio"] = round(bw / widest, 4)
    else:
        metrics["top_width_ratio"] = None
except Exception:
    metrics["top_width_ratio"] = None

# fill ratio: a figure with limbs leaves most of its bounding box empty; a
# solid block fills nearly all of it. This is the "it came out a rectangle"
# detector, and it needs no notion of what the subject was supposed to be.
try:
    import bmesh
    vol = 0.0
    for o in meshes:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.transform(o.matrix_world)
        vol += abs(bm.calc_volume(signed=True))
        bm.free()
    bbox_vol = float(dx * dy * dz) or 1.0
    metrics["fill_ratio"] = round(min(vol / bbox_vol, 1.0), 4)
except Exception:
    metrics["fill_ratio"] = None

# texture spread: the mean per-channel standard deviation of the baked base
# colour. A washed-out / unguided bake collapses toward one colour.
try:
    spreads = []
    for img in bpy.data.images:
        if not img.has_data or img.size[0] < 8:
            continue
        px = list(img.pixels)
        step = max(1, (len(px) // 4) // 4000) * 4      # ~4k samples, RGBA stride
        for ch in range(3):
            vals = px[ch::step] if step > 4 else px[ch::4]
            if len(vals) < 16:
                continue
            m = sum(vals) / len(vals)
            spreads.append((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)
    metrics["texture_spread"] = round(sum(spreads) / len(spreads), 4) if spreads else None
except Exception:
    metrics["texture_spread"] = None

with open(glb + ".metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=1)
print("METRICS " + json.dumps(metrics))
