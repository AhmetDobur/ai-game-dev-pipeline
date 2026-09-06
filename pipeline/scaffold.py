"""Deterministic Godot project scaffolding for the assemble step.

The coder LLM writes gameplay scripts, but the project skeleton — project.godot,
the input map, the export preset and a world scene wiring the generated assets —
is written HERE, deterministically. A generated scene file is a wildcard (the
coder has produced GDScript inside a .tscn); a templated one always parses.
"""
import math
import shutil
import sys
from pathlib import Path

# physical keycodes. Seat 1 gets WASD (+ the mouse), seat 2 gets the arrows to
# move and IJKL to look -- split-screen has to be playable on one keyboard when
# only one gamepad is plugged in, and the arrows used to be mere aliases of WASD
# so they drove seat 1 and left seat 2 with no binding at all.
_KEYS = {"move_left": 65, "move_right": 68, "move_forward": 87, "move_back": 83,
         "p2_move_left": 4194319, "p2_move_right": 4194321,
         "p2_move_forward": 4194320, "p2_move_back": 4194322,
         "p2_look_left": 74, "p2_look_right": 76,
         "p2_look_up": 73, "p2_look_down": 75,
         # Q/E strike with the left/right hand. Seat 2 gets N/M, which sit under
         # the same hand as its IJKL look keys.
         "attack_left": 81, "attack_right": 69,
         "p2_attack_left": 78, "p2_attack_right": 77}

PROJECT_GODOT = """config_version=5

[application]

config/name="{title}"
run/main_scene="res://scenes/menu.tscn"

[autoload]

Settings="*res://scripts/settings.gd"
DemoDrive="*res://scripts/demo_drive.gd"

[editor]

movie_writer/fps=60
movie_writer/disable_vsync=true

[input]

{input_map}
[display]

window/size/viewport_width=1600
window/size/viewport_height=900

[rendering]

renderer/rendering_method="forward_plus"
"""

_ACTION = """{name}={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"physical_keycode":{a},"pressed":false,"echo":false,"script":null)
]
}}
"""

EXPORT_PRESETS = """[preset.0]

name="Windows Desktop"
platform="Windows Desktop"
runnable=true
advanced_options=false
dedicated_server=false
custom_features=""
export_filter="all_resources"
include_filter=""
exclude_filter=""
patches=PackedStringArray()
encryption_include_filters=""
encryption_exclude_filters=""
seed=0
encrypt_pck=false
encrypt_directory=false
script_export_mode=2

[preset.0.options]

custom_template/debug=""
custom_template/release=""
debug/export_console_wrapper=1
binary_format/embed_pck=true
texture_format/s3tc_bptc=true
texture_format/etc2_astc=false
binary_format/architecture="x86_64"
"""


def _prop_ring(env_glbs: list[str], per_glb: int = 4, radius: float = 11.0):
    """Deterministic prop layout: each environment mesh repeated around a circle,
    facing the center (a library IS the same bookshelf many times). Returns
    (glb_index, x, z, yaw) tuples."""
    total = len(env_glbs) * per_glb
    out = []
    for k in range(total):
        ang = 2 * math.pi * k / total
        x, z = radius * math.sin(ang), radius * math.cos(ang)
        out.append((k % len(env_glbs), round(x, 2), round(z, 2),
                    round(ang + math.pi, 4)))  # +pi: face inward
    return out


# ---------------------------------------------------------------- room shell
# A single-image 3D reconstructor cannot build a space you stand inside (see
# workflows/trellis.json: one image -> one bounded voxel volume, camera outside
# it by construction). So the HALL is authored here as parametric geometry and
# the generated meshes are placed INTO it as furniture. Dimensions follow the
# reference: a long nave, a two-storey shelf wall down each side, a colonnade,
# and a raised apse at the far end.
HALL_L, HALL_W, HALL_H = 44.0, 18.0, 13.0
_WALL_T = 0.6


def _mix(a, b, k):
    """k=0 keeps a, k=1 becomes b. Used to darken palette swatches toward the
    reference's near-black ground without losing its hue."""
    return tuple(round(a[i] * (1 - k) + b[i] * k, 3) for i in range(3))


def _c(rgb, a=1.0):
    r, g, b = rgb
    return f"Color({r}, {g}, {b}, {a})"


def _box_body(name: str, size, pos, mat_id: str, sub: list) -> str:
    """A StaticBody3D box with matching collision -- walls, steps, carpet."""
    sx, sy, sz = size
    x, y, z = pos
    sub.append(f'[sub_resource type="BoxMesh" id="{name}_m"]\nsize = Vector3({sx}, {sy}, {sz})')
    sub.append(f'[sub_resource type="BoxShape3D" id="{name}_s"]\nsize = Vector3({sx}, {sy}, {sz})')
    return (f'[node name="{name}" type="StaticBody3D" parent="."]\n'
            f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {x}, {y}, {z})\n\n'
            f'[node name="{name}Mesh" type="MeshInstance3D" parent="{name}"]\n'
            f'mesh = SubResource("{name}_m")\n'
            f'material_override = SubResource("{mat_id}")\n\n'
            f'[node name="{name}Col" type="CollisionShape3D" parent="{name}"]\n'
            f'shape = SubResource("{name}_s")')


def _column(name: str, x: float, z: float, mat_id: str, sub: list) -> str:
    h = HALL_H - 2.0
    sub.append(f'[sub_resource type="CylinderMesh" id="{name}_m"]\n'
               f'top_radius = 0.55\nbottom_radius = 0.62\nheight = {h}')
    sub.append(f'[sub_resource type="CylinderShape3D" id="{name}_s"]\n'
               f'radius = 0.62\nheight = {h}')
    return (f'[node name="{name}" type="StaticBody3D" parent="."]\n'
            f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {x}, {h / 2}, {z})\n\n'
            f'[node name="{name}Mesh" type="MeshInstance3D" parent="{name}"]\n'
            f'mesh = SubResource("{name}_m")\n'
            f'material_override = SubResource("{mat_id}")\n\n'
            f'[node name="{name}Col" type="CollisionShape3D" parent="{name}"]\n'
            f'shape = SubResource("{name}_s")')


def _ramp(name: str, width: float, z0: float, z1: float, top: float, sub) -> str:
    """Collision-only slope from floor level at z0 up to `top` at z1.

    No mesh: this exists purely so a body can walk up geometry that is drawn as
    steps. Thin, and sunk by its own half-thickness so its upper face meets the
    step noses rather than floating above them.
    """
    run = z1 - z0
    length = math.hypot(run, top)
    angle = -math.atan2(top, run)        # -X rotation lifts the +Z end
    c, sn = round(math.cos(angle), 6), round(math.sin(angle), 6)
    t = 0.2
    cz, cy = (z0 + z1) / 2, top / 2 - t / 2
    sub.append(f'[sub_resource type="BoxShape3D" id="{name}_s"]\n'
               f"size = Vector3({round(width, 3)}, {t}, {round(length, 3)})")
    return (f'[node name="{name}" type="StaticBody3D" parent="."]\n'
            f"transform = Transform3D(1, 0, 0, 0, {c}, {sn}, 0, {-sn}, {c}, "
            f"0, {round(cy, 3)}, {round(cz, 3)})\n\n"
            f'[node name="{name}Col" type="CollisionShape3D" parent="{name}"]\n'
            f'shape = SubResource("{name}_s")')


def _hall_geometry(pal: dict, sub: list) -> list:
    """Floor, carpet, four walls, ceiling, colonnade, apse steps, rose window."""
    hw, hl = HALL_W / 2, HALL_L / 2
    for mid, rgb, rough, metal in (
            ("mat_floor", _mix(pal["stone"], pal["shadow"], 0.45), 0.18, 0.25),
            ("mat_wood", _mix(pal["wood"], pal["shadow"], 0.6), 0.75, 0.0),
            ("mat_accent", pal["accent"], 0.3, 0.15),
            ("mat_dark", _mix(pal["shadow"], (0, 0, 0), 0.5), 0.9, 0.0)):
        sub.append(f'[sub_resource type="StandardMaterial3D" id="{mid}"]\n'
                   f'albedo_color = {_c(rgb)}\nroughness = {rough}\nmetallic = {metal}')
    # the rose window is the hall's far light source, so it emits rather than reflects
    sub.append(f'[sub_resource type="StandardMaterial3D" id="mat_glass"]\n'
               f'albedo_color = {_c(pal["gold"])}\nemission_enabled = true\n'
               f'emission = {_c(pal["gold"])}\nemission_energy_multiplier = 3.0')

    n = [
        _box_body("Floor", (HALL_W, 0.4, HALL_L), (0, -0.2, 0), "mat_floor", sub),
        _box_body("Carpet", (5.5, 0.06, HALL_L - 6), (0, 0.03, 1.5), "mat_accent", sub),
        _box_body("WallL", (_WALL_T, HALL_H, HALL_L), (-hw, HALL_H / 2, 0), "mat_wood", sub),
        _box_body("WallR", (_WALL_T, HALL_H, HALL_L), (hw, HALL_H / 2, 0), "mat_wood", sub),
        _box_body("WallBack", (HALL_W, HALL_H, _WALL_T), (0, HALL_H / 2, hl), "mat_wood", sub),
        _box_body("WallFront", (HALL_W, HALL_H, _WALL_T), (0, HALL_H / 2, -hl), "mat_wood", sub),
        _box_body("Ceiling", (HALL_W, 0.4, HALL_L), (0, HALL_H, 0), "mat_dark", sub),
    ]
    # raised apse at the far end: three steps up to a platform. The run ENDS at
    # the platform's front face -- it used to start 5.5 m out and march into the
    # apse, so the middle step was half swallowed and the top step sat entirely
    # inside the block, leaving a single 0.7 m wall where a stair was drawn.
    apse_d, apse_h, step_d, step_h = 4.0, 1.05, 1.2, 0.35
    apse_front = hl - 2.4 - apse_d / 2
    stair_front = apse_front - 3 * step_d
    for i in range(3):
        n.append(_box_body(f"Step{i}", (HALL_W - 4, step_h, step_d),
                           (0, step_h / 2 + i * step_h,
                            stair_front + step_d * (i + 0.5)), "mat_floor", sub))
    n.append(_box_body("Apse", (HALL_W - 4, apse_h, apse_d),
                       (0, apse_h / 2, hl - 2.4), "mat_floor", sub))
    # Godot 4's CharacterBody3D has no step-up: a 0.35 m riser reads as a wall
    # and the whole apse end -- the rose window, the part worth walking to -- was
    # unreachable. An invisible ramp over the stair makes it walkable without
    # touching how the stair looks.
    n.append(_ramp("StairRamp", HALL_W - 4, stair_front, apse_front, apse_h, sub))
    n.append(_box_body("RoseWindow", (5.0, 5.0, 0.2), (0, 7.5, hl - 0.4), "mat_glass", sub))
    # colonnade down both sides
    for i in range(6):
        z = -hl + 5.0 + i * ((HALL_L - 10) / 5)
        n.append(_column(f"ColL{i}", -hw + 2.6, round(z, 2), "mat_accent", sub))
        n.append(_column(f"ColR{i}", hw - 2.6, round(z, 2), "mat_accent", sub))
    return n


def _wall_props(env_glbs: list, per_glb: int = 3):
    """Generated meshes line the side walls facing in -- a shelf belongs against
    a wall, not on a circle in the middle of the floor."""
    hw, hl = HALL_W / 2, HALL_L / 2
    slots, total = [], max(1, len(env_glbs) * per_glb)
    for k in range(total):
        side = -1 if k % 2 == 0 else 1
        row = k // 2
        z = -hl + 6.0 + row * ((HALL_L - 12) / max(1, (total // 2)))
        # face the centre line: -90deg on the left wall, +90deg on the right
        yaw = math.pi / 2 if side < 0 else -math.pi / 2
        slots.append((k % len(env_glbs), round(side * (hw - 4.2), 2),
                      round(z, 2), round(yaw, 4)))
    return slots


MAX_PLAYERS = 2          # split-screen halves; a third would need a 2x2 grid
SPAWN_Z = -16.0          # near the hall's entrance, facing the apse


def _players(char_glbs: list[str], script_id: str | None, ext_res) -> str:
    """Player bodies + the split-screen viewports their cameras render into.

    With one character this is the plain single-camera setup. With two, each
    player gets half the window: the cameras must live INSIDE the SubViewports to
    render there, so they are not children of the players and player.gd carries
    each pivot's transform onto its camera every frame. The SubViewports keep
    `own_world_3d` at its default false, so they inherit the root viewport's
    World3D and every player sees the same hall.
    """
    n = min(len(char_glbs), MAX_PLAYERS) or 1
    split = n > 1
    out = []
    for i in range(n):
        name = f"Player{i + 1}" if split else "Player"
        x = (i - (n - 1) / 2) * 3.0
        cam = f'NodePath("../Split/Half{i}/View/Camera")' if split else None
        script = ""
        if script_id:
            script = f'script = ExtResource("{script_id}")\ndevice = {i}\n'
            if cam:
                script += f'camera_path = {cam}\n'
        mesh = ""
        if i < len(char_glbs):
            mid = ext_res("PackedScene", char_glbs[i])
            # yaw 180: TRELLIS reconstructs a front-view reference so the model
            # looks down its own +Z, while a Godot body moves along -Z. Left
            # unrotated the character walks backwards, facing its own camera.
            mesh = (f'\n[node name="Mesh" parent="{name}" '
                    f'instance=ExtResource("{mid}")]\n'
                    'transform = Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0)')
        # CamPivot is emitted BEFORE the mesh instance on purpose: a .glb that
        # fails to load raises a parse error that drops every node after it in
        # the file, and losing the pivot means the player cannot look or aim.
        # Movement must survive a missing asset.
        # yaw 180 deg: spawns are at the hall's near end and Godot bodies face
        # -Z, so an unrotated player starts nose-to-wall with the apse, the rose
        # window and every candle behind them
        out.append(f"""[node name="{name}" type="CharacterBody3D" parent="."]
transform = Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, {round(x, 2)}, 1.2, {SPAWN_Z})
{script}
[node name="Collision" type="CollisionShape3D" parent="{name}"]
position = Vector3(0, 0.9, 0)
shape = SubResource("player_shape")

[node name="CamPivot" type="Node3D" parent="{name}"]
position = Vector3(0, 1.6, 0)

[node name="SpringArm3D" type="SpringArm3D" parent="{name}/CamPivot"]
position = Vector3(0, 0.4, 0)
spring_length = 3.5
margin = 0.2

[node name="CamMount" type="Marker3D" parent="{name}/CamPivot/SpringArm3D"]
{mesh}""")
        if not split:
            out.append(f"""[node name="Camera3D" type="Camera3D" parent="{name}/CamPivot/SpringArm3D/CamMount"]""")
    if not split:
        return "\n\n".join(out)

    out.append("""[node name="Split" type="HBoxContainer" parent="."]
anchor_right = 1.0
anchor_bottom = 1.0
grow_horizontal = 2
grow_vertical = 2
theme_override_constants/separation = 2""")
    for i in range(n):
        out.append(f"""[node name="Half{i}" type="SubViewportContainer" parent="Split"]
stretch = true
layout_mode = 2
size_flags_horizontal = 3
size_flags_vertical = 3
mouse_filter = 2

[node name="View" type="SubViewport" parent="Split/Half{i}"]
handle_input_locally = false
render_target_update_mode = 4

[node name="Camera" type="Camera3D" parent="Split/Half{i}/View"]
current = true""")
    return "\n\n".join(out)


def _world_tscn(player_script: str | None, char_glbs: list[str],
                env_glbs: list[str], env_ref: str | None = None) -> str:
    from .palette import roles
    pal = roles(env_ref)
    ext, nodes, sub = [], [], []
    rid = 1

    def ext_res(kind: str, path: str) -> str:
        nonlocal rid
        rid += 1
        ext.append(f'[ext_resource type="{kind}" path="{path}" id="{rid}"]')
        return str(rid)

    shot_id = ext_res("Script", "res://scripts/godot_shot.gd")
    nodes.extend(_hall_geometry(pal, sub))
    env_ids = [ext_res("PackedScene", g) for g in env_glbs]
    for i, (gi, x, z, yaw) in enumerate(_wall_props(env_glbs) if env_glbs else []):
        c, s = round(math.cos(yaw), 4), round(math.sin(yaw), 4)
        nodes.append(f"""[node name="Prop{i}" type="StaticBody3D" parent="."]
transform = Transform3D({c}, 0, {s}, 0, 1, 0, {-s}, 0, {c}, {x}, 0, {z})

[node name="PropShape{i}" type="CollisionShape3D" parent="Prop{i}"]
position = Vector3(0, 2, 0)
shape = SubResource("prop_shape")

[node name="PropMesh{i}" parent="Prop{i}" instance=ExtResource("{env_ids[gi]}")]""")

    script_id = ext_res("Script", player_script) if player_script else None
    players = _players(char_glbs, script_id, ext_res)

    # candle-warm point lights around the room + a weak cool moonlight key
    candle_slots = [(x, round(-HALL_L / 2 + 5.0 + i * ((HALL_L - 10) / 5), 2))
                    for i in range(6)
                    for x in (-HALL_W / 2 + 2.6, HALL_W / 2 - 2.6)]
    candles = "\n\n".join(
        f"""[node name="Candle{i}" type="OmniLight3D" parent="."]
position = Vector3({x}, 3.2, {z})
light_color = Color(1, 0.72, 0.42, 1)
light_energy = 2.4
omni_range = 8.5
shadow_enabled = {"true" if i % 3 == 0 else "false"}"""
        for i, (x, z) in enumerate(candle_slots))

    return f"""[gd_scene load_steps={rid + len(sub) + 6} format=3]

{chr(10).join(ext)}

{chr(10).join(sub)}

[sub_resource type="BoxShape3D" id="prop_shape"]
size = Vector3(2.4, 4, 2.4)

[sub_resource type="CapsuleShape3D" id="player_shape"]
height = 1.8

[sub_resource type="Environment" id="world_env"]
background_mode = 1
background_color = Color(0.02, 0.015, 0.01, 1)
ambient_light_source = 2
; near-neutral, low energy: a saturated ambient tint multiplies into EVERY
; material and re-colours generated textures on their way to the screen, which
; hides whatever the art stage actually produced. Mood belongs to the lights
; (the candles below), not to a global wash over every asset.
ambient_light_color = Color(0.30, 0.29, 0.28, 1)
ambient_light_energy = 0.10
tonemap_mode = 3
glow_enabled = true
glow_intensity = 0.6
glow_bloom = 0.15
volumetric_fog_enabled = true
volumetric_fog_density = 0.004
volumetric_fog_albedo = Color(0.55, 0.50, 0.45, 1)
volumetric_fog_emission = Color(0.04, 0.032, 0.022, 1)

[node name="World" type="Node3D"]
script = ExtResource("{shot_id}")

[node name="WorldEnvironment" type="WorldEnvironment" parent="."]
environment = SubResource("world_env")

[node name="Moon" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.707, -0.5, 0.5, 0, 0.707, 0.707, -0.707, -0.5, 0.5, 0, 10, 0)
light_color = Color(0.65, 0.7, 0.9, 1)
light_energy = 0.25
shadow_enabled = true

{candles}

{chr(10).join(nodes)}

{players}
"""


MENU_TSCN = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/menu.gd" id="1"]

[node name="Menu" type="Control"]
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
script = ExtResource("1")
shadow = Color({shadow})
stone = Color({stone})
accent = Color({accent})
gold = Color({gold})
title_text = "{title}"
background = "{background}"
world_scene = "res://scenes/world.tscn"
"""


def _menu_tscn(title: str, env_ref: str | None, background: str = "") -> str:
    """The title screen, in the same palette the hall is keyed to.

    The menu samples the run's own reference image rather than carrying a
    hardcoded scheme, so a project generated from different concept art gets a
    menu that belongs to it instead of one in somebody else's colours.
    """
    from .palette import roles
    pal = roles(env_ref)

    def c(role, floor=0.0):
        r, g, b = pal[role]          # palette.roles yields 0..1 triples already
        # Text has to stay legible whatever the art happens to be keyed to. The
        # room's mid-tone is chosen for stone, not for type, and on this scheme
        # it lands at 0.23/0.17/0.12 -- barely off the background. Lifting the
        # swatch to a luminance floor keeps its hue and makes it readable.
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if floor and lum < floor:
            k = floor / max(lum, 1e-3)
            r, g, b = (min(1.0, v * k) for v in (r, g, b))
        return f"{r:.3f}, {g:.3f}, {b:.3f}, 1"

    return MENU_TSCN.format(shadow=c("shadow"), stone=c("stone", 0.62),
                            accent=c("accent"), gold=c("gold", 0.45),
                            title=title.replace('"', ""), background=background)


def scaffold(game_dir: Path, title: str, dep_outputs: dict[str, list[str]],
             dep_types: dict[str, str], dep_specs: dict[str, dict]) -> None:
    """Copy generated assets into the project and write every file Godot needs to
    run and export it. Idempotent; a coder-written file never blocks it (the
    skeleton files are ours, only ours, and always overwritten)."""
    game_dir.mkdir(parents=True, exist_ok=True)

    # classify deps: rig_animate output is the playable character; any design_3d
    # not consumed as a rig's source mesh is environment
    consumed = {s.get("mesh_from") for s in dep_specs.values() if isinstance(s, dict)}
    char_glbs, env_glbs = [], []
    for dep, paths in dep_outputs.items():
        files = [Path(p) for p in paths if p]
        if not files:
            continue
        # ids are "<12-hex-run-id>-<decomposer-id>"; the decomposer id itself may
        # contain hyphens, so split once from the left, not from the right
        slug = dep.split("-", 1)[1] if "-" in dep else dep
        copied = []
        for f in files:
            if f.suffix.lower() in (".glb", ".gltf", ".png", ".jpg", ".webp") and f.exists():
                dest = game_dir / "assets" / slug / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                copied.append(f"res://assets/{slug}/{f.name}")
        kind = dep_types.get(dep, "")
        if kind == "rig_animate" and copied:
            # character.glb carries ALL clips as named animations; per-clip files
            # are the pre-0.7 layout, kept as fallback for old runs
            best = ([c for c in copied if c.endswith("character.glb")]
                    or [c for c in copied if "idle" in c.lower()] or copied)
            char_glbs.append(best[0])
        elif kind == "design_3d" and dep not in consumed and copied:
            env_glbs += [c for c in copied if c.endswith((".glb", ".gltf"))]

    # the environment reference the run was given -- the scaffold reads its
    # palette so the authored hall sits in the same key as the generated art
    env_ref = None
    for s in dep_specs.values():
        if isinstance(s, dict) and s.get("ref_image"):
            name = str(s["ref_image"]).replace("\\", "/").rsplit("/", 1)[-1].lower()
            if any(w in name for w in ("env", "background", "backdrop", "scene",
                                       "library", "level")):
                cand = Path(str(s["ref_image"]).replace("\\", "/"))
                if cand.exists():
                    env_ref = str(cand)
                    break

    # the coder LLM's player.gd wins when it is usable; otherwise ship ours, so a
    # run never ends in a world nobody can walk around. Split-screen needs the
    # camera_path/device exports, which only our template has.
    p = game_dir / "scripts" / "player.gd"
    usable = p.exists() and "CharacterBody3D" in p.read_text(
        encoding="utf-8", errors="replace")
    if len(char_glbs) > 1 or not usable:
        src = Path(__file__).resolve().parent.parent / "templates" / "player.gd"
        if src.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, p)
            usable = True
    player_script = "res://scripts/player.gd" if usable else None

    input_map = "".join(_ACTION.format(name=n, a=a) for n, a in _KEYS.items())
    (game_dir / "scenes").mkdir(exist_ok=True)
    (game_dir / "project.godot").write_text(
        PROJECT_GODOT.format(title=title.replace('"', ""), input_map=input_map),
        encoding="utf-8")
    (game_dir / "export_presets.cfg").write_text(EXPORT_PRESETS, encoding="utf-8")
    # shipped with every project so the assemble step can photograph the world it
    # just built; harmless when unused, and it never steals an existing camera
    for name in ("combat.gd", "cloak.gd", "demo_drive.gd", "menu.gd", "settings.gd"):
        src = Path(__file__).resolve().parent.parent / "templates" / name
        if src.exists():
            shutil.copy2(src, game_dir / "scripts" / name)
    # Combat audio is synthesised rather than shipped as assets: it is a few
    # hundred lines of stdlib DSP, it costs nothing, and it means the sounds are
    # generated to match this project's own strike table instead of being a
    # folder of wavs somebody has to keep in sync.
    try:
        from templates.make_sfx import generate as _make_sfx
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            from templates.make_sfx import generate as _make_sfx
        except ImportError:
            _make_sfx = None
    if _make_sfx is not None:
        (game_dir / "audio").mkdir(exist_ok=True)
        _make_sfx(str(game_dir / "audio"))

    shot_src = Path(__file__).resolve().parent.parent / "templates" / "godot_shot.gd"
    if shot_src.exists():
        (game_dir / "scripts").mkdir(exist_ok=True)
        shutil.copy2(shot_src, game_dir / "scripts" / "godot_shot.gd")
    (game_dir / "scenes" / "world.tscn").write_text(
        _world_tscn(player_script, char_glbs, env_glbs, env_ref), encoding="utf-8")
    # the menu's backdrop is the run's own environment reference, copied in so
    # the exported game carries it
    menu_bg = ""
    if env_ref and Path(env_ref).exists():
        dest = game_dir / "assets" / ("menu_bg" + Path(env_ref).suffix.lower())
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_ref, dest)
        menu_bg = f"res://assets/{dest.name}"
    (game_dir / "scenes" / "menu.tscn").write_text(
        _menu_tscn(title, env_ref, menu_bg), encoding="utf-8")
