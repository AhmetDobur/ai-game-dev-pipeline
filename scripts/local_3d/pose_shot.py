"""Render a rigged .glb at one clip's contact frame, from the side.

    blender --background --python scripts/local_3d/pose_shot.py -- \\
        <mesh.glb> <out.png> [clip] [phase]

Skin weights are argued about in tables and decided in pictures. The rest pose
looks correct under any weighting -- that is what makes bad weights expensive --
so the frame worth looking at is the one where a limb is furthest from where it
started. `phase` is where in the clip to sample, and defaults to the contact
fraction combat.gd uses for the cross, so the picture is the same instant the
hitbox opens.

Import note: the glTF importer's default bone display creates a 42-vertex
"Icosphere" spanning a 2 m cube for every armature it reads. It is an artifact
of the IMPORT, not something in the file -- the .glb itself contains one mesh --
but it lands in bpy.data, inflates any bounding box computed over "all meshes",
and reads exactly like a stray object shipping inside the character. Passing
bone_heuristic="TEMPERANCE" stops it being created.
"""
import sys

import bpy
import mathutils

args = sys.argv[sys.argv.index("--") + 1:]
glb, out_png = args[:2]
clip = args[2] if len(args) > 2 else "cross"
phase = float(args[3]) if len(args) > 3 else 0.58

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb, bone_heuristic="TEMPERANCE")

arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
action = next((a for a in bpy.data.actions if clip.lower() in a.name.lower()), None)
if arm and action:
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = action
    lo, hi = action.frame_range
    bpy.context.scene.frame_set(int(lo + (hi - lo) * phase))
    print(f"[pose] {action.name} frames {lo:.0f}-{hi:.0f} at "
          f"{bpy.context.scene.frame_current}")
else:
    print(f"[pose] no '{clip}' action; rendering rest pose")

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
lo3 = mathutils.Vector([min(p[k] for p in pts) for k in range(3)])
hi3 = mathutils.Vector([max(p[k] for p in pts) for k in range(3)])
center = (lo3 + hi3) / 2
size = max((hi3 - lo3).length, 0.01)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
bpy.context.scene.collection.objects.link(cam)
# straight from the side: a punch travels along the character's forward axis,
# and a 3/4 view foreshortens exactly the motion being judged
cam.location = center + mathutils.Vector((1.0, 0.0, 0.05)).normalized() * size * 1.5
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.clip_end = size * 10
bpy.context.scene.camera = cam

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 4
sun.rotation_euler = (0.9, 0.2, 0.7)
bpy.context.scene.collection.objects.link(sun)

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = scene.render.resolution_y = 900
scene.render.film_transparent = False
scene.render.filepath = out_png
bpy.ops.render.render(write_still=True)
print(f"[pose] {out_png}")
