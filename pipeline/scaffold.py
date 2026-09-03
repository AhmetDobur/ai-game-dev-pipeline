"""Deterministic Godot project scaffolding for the assemble step.

The coder LLM writes gameplay scripts, but the project skeleton — project.godot,
the input map, the export preset and a world scene wiring the generated assets —
is written HERE, deterministically. A generated scene file is a wildcard (the
coder has produced GDScript inside a .tscn); a templated one always parses.
"""
import math
import shutil
from pathlib import Path

# physical keycodes: WASD + arrows
_KEYS = {"move_left": (65, 4194319), "move_right": (68, 4194321),
         "move_forward": (87, 4194320), "move_back": (83, 4194322)}

PROJECT_GODOT = """config_version=5

[application]

config/name="{title}"
run/main_scene="res://scenes/world.tscn"

[input]

{input_map}
[rendering]

renderer/rendering_method="forward_plus"
"""

_ACTION = """{name}={{
"deadzone": 0.5,
"events": [Object(InputEventKey,"physical_keycode":{a},"pressed":false,"echo":false,"script":null)
, Object(InputEventKey,"physical_keycode":{b},"pressed":false,"echo":false,"script":null)
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


def _hall_geometry(pal: dict, sub: list) -> list:
    """Floor, carpet, four walls, ceiling, colonnade, apse steps, rose window."""
    hw, hl = HALL_W / 2, HALL_L / 2
    for mid, rgb, rough, metal in (
            ("mat_floor", pal["stone"], 0.25, 0.15),
            ("mat_wood", pal["wood"], 0.7, 0.05),
            ("mat_accent", pal["accent"], 0.35, 0.2),
            ("mat_dark", pal["shadow"], 0.8, 0.0)):
        sub.append(f'[sub_resource type="StandardMaterial3D" id="{mid}"]\n'
                   f'albedo_color = {_c(rgb)}\nroughness = {rough}\nmetallic = {metal}')
    # the rose window is the hall's far light source, so it emits rather than reflects
    sub.append(f'[sub_resource type="StandardMaterial3D" id="mat_glass"]\n'
               f'albedo_color = {_c(pal["gold"])}\nemission_enabled = true\n'
               f'emission = {_c(pal["gold"])}\nemission_energy_multiplier = 6.0')

    n = [
        _box_body("Floor", (HALL_W, 0.4, HALL_L), (0, -0.2, 0), "mat_floor", sub),
        _box_body("Carpet", (5.5, 0.06, HALL_L - 6), (0, 0.03, 1.5), "mat_accent", sub),
        _box_body("WallL", (_WALL_T, HALL_H, HALL_L), (-hw, HALL_H / 2, 0), "mat_wood", sub),
        _box_body("WallR", (_WALL_T, HALL_H, HALL_L), (hw, HALL_H / 2, 0), "mat_wood", sub),
        _box_body("WallBack", (HALL_W, HALL_H, _WALL_T), (0, HALL_H / 2, hl), "mat_wood", sub),
        _box_body("WallFront", (HALL_W, HALL_H, _WALL_T), (0, HALL_H / 2, -hl), "mat_wood", sub),
        _box_body("Ceiling", (HALL_W, 0.4, HALL_L), (0, HALL_H, 0), "mat_dark", sub),
    ]
    # raised apse at the far end: three steps up to a platform
    for i in range(3):
        n.append(_box_body(f"Step{i}", (HALL_W - 4, 0.35, 1.2),
                           (0, 0.17 + i * 0.35, hl - 5.5 + i * 1.2), "mat_floor", sub))
    n.append(_box_body("Apse", (HALL_W - 4, 1.05, 4.0), (0, 0.52, hl - 2.4), "mat_floor", sub))
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


def _world_tscn(player_script: str | None, char_glb: str | None,
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

    player_type = "CharacterBody3D"
    script_line = ""
    if player_script:
        sid = ext_res("Script", player_script)
        script_line = f'script = ExtResource("{sid}")\n'
    char_line = ""
    if char_glb:
        cid = ext_res("PackedScene", char_glb)
        char_line = f'\n[node name="Mesh" parent="Player" instance=ExtResource("{cid}")]'

    # candle-warm point lights around the room + a weak cool moonlight key
    candle_slots = [(x, round(-HALL_L / 2 + 5.0 + i * ((HALL_L - 10) / 5), 2))
                    for i in range(6)
                    for x in (-HALL_W / 2 + 2.6, HALL_W / 2 - 2.6)]
    candles = "\n\n".join(
        f"""[node name="Candle{i}" type="OmniLight3D" parent="."]
position = Vector3({x}, 3.2, {z})
light_color = Color(1, 0.72, 0.42, 1)
light_energy = 4.0
omni_range = 13.0
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
ambient_light_energy = 0.25
tonemap_mode = 3
glow_enabled = true
glow_intensity = 0.6
glow_bloom = 0.15
volumetric_fog_enabled = true
volumetric_fog_density = 0.008
volumetric_fog_albedo = Color(0.55, 0.50, 0.45, 1)
volumetric_fog_emission = Color(0.04, 0.032, 0.022, 1)

[node name="World" type="Node3D"]

[node name="WorldEnvironment" type="WorldEnvironment" parent="."]
environment = SubResource("world_env")

[node name="Moon" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.707, -0.5, 0.5, 0, 0.707, 0.707, -0.707, -0.5, 0.5, 0, 10, 0)
light_color = Color(0.65, 0.7, 0.9, 1)
light_energy = 0.25
shadow_enabled = true

{candles}

{chr(10).join(nodes)}

[node name="Player" type="{player_type}" parent="."]
position = Vector3(0, 1.2, -16)
{script_line}
[node name="Collision" type="CollisionShape3D" parent="Player"]
position = Vector3(0, 0.9, 0)
shape = SubResource("player_shape")
{char_line}

[node name="CamPivot" type="Node3D" parent="Player"]
position = Vector3(0, 1.6, 0)

[node name="Camera3D" type="Camera3D" parent="Player/CamPivot"]
position = Vector3(0, 0.4, 3.5)
"""


def scaffold(game_dir: Path, title: str, dep_outputs: dict[str, list[str]],
             dep_types: dict[str, str], dep_specs: dict[str, dict]) -> None:
    """Copy generated assets into the project and write every file Godot needs to
    run and export it. Idempotent; a coder-written file never blocks it (the
    skeleton files are ours, only ours, and always overwritten)."""
    game_dir.mkdir(parents=True, exist_ok=True)

    # classify deps: rig_animate output is the playable character; any design_3d
    # not consumed as a rig's source mesh is environment
    consumed = {s.get("mesh_from") for s in dep_specs.values() if isinstance(s, dict)}
    char_glb, env_glbs = None, []
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
            char_glb = best[0]
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

    player_script = None
    p = game_dir / "scripts" / "player.gd"
    if p.exists() and "CharacterBody3D" in p.read_text(encoding="utf-8", errors="replace"):
        player_script = "res://scripts/player.gd"

    input_map = "".join(_ACTION.format(name=n, a=a, b=b) for n, (a, b) in _KEYS.items())
    (game_dir / "scenes").mkdir(exist_ok=True)
    (game_dir / "project.godot").write_text(
        PROJECT_GODOT.format(title=title.replace('"', ""), input_map=input_map),
        encoding="utf-8")
    (game_dir / "export_presets.cfg").write_text(EXPORT_PRESETS, encoding="utf-8")
    (game_dir / "scenes" / "world.tscn").write_text(
        _world_tscn(player_script, char_glb, env_glbs, env_ref), encoding="utf-8")
