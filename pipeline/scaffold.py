"""Deterministic Godot project scaffolding for the assemble step.

The coder LLM writes gameplay scripts, but the project skeleton — project.godot,
the input map, the export preset and a world scene wiring the generated assets —
is written HERE, deterministically. A generated scene file is a wildcard (the
coder has produced GDScript inside a .tscn); a templated one always parses.
"""
import shutil
from pathlib import Path

# physical keycodes: WASD + arrows
_KEYS = {"move_left": (65, 4194319), "move_right": (68, 4194321),
         "move_forward": (87, 4194320), "move_back": (83, 4194322)}

PROJECT_GODOT = """config_version=5

[application]

config/name="{title}"
run/main_scene="res://scenes/world.tscn"
config/features=PackedStringArray("4.4")

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


def _world_tscn(player_script: str | None, char_glb: str | None,
                env_glbs: list[str]) -> str:
    ext, nodes = [], []
    rid = 1

    def ext_res(kind: str, path: str) -> str:
        nonlocal rid
        rid += 1
        ext.append(f'[ext_resource type="{kind}" path="{path}" id="{rid}"]')
        return str(rid)

    for i, g in enumerate(env_glbs):
        eid = ext_res("PackedScene", g)
        nodes.append(f'[node name="Env{i}" parent="." instance=ExtResource("{eid}")]')

    player_type = "CharacterBody3D"
    script_line = ""
    if player_script:
        sid = ext_res("Script", player_script)
        script_line = f'script = ExtResource("{sid}")\n'
    char_line = ""
    if char_glb:
        cid = ext_res("PackedScene", char_glb)
        char_line = f'\n[node name="Mesh" parent="Player" instance=ExtResource("{cid}")]'

    return f"""[gd_scene load_steps={rid + 4} format=3]

{chr(10).join(ext)}

[sub_resource type="BoxShape3D" id="floor_shape"]
size = Vector3(400, 1, 400)

[sub_resource type="BoxMesh" id="floor_mesh"]
size = Vector3(400, 1, 400)

[sub_resource type="CapsuleShape3D" id="player_shape"]
height = 1.8

[sub_resource type="Environment" id="world_env"]
ambient_light_source = 2
ambient_light_color = Color(1, 0.93, 0.82, 1)
ambient_light_energy = 1.2

[node name="World" type="Node3D"]

[node name="WorldEnvironment" type="WorldEnvironment" parent="."]
environment = SubResource("world_env")

[node name="Sun" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.707, -0.5, 0.5, 0, 0.707, 0.707, -0.707, -0.5, 0.5, 0, 10, 0)
light_energy = 1.2

[node name="Floor" type="StaticBody3D" parent="."]

[node name="FloorShape" type="CollisionShape3D" parent="Floor"]
position = Vector3(0, -0.5, 0)
shape = SubResource("floor_shape")

[node name="FloorMesh" type="MeshInstance3D" parent="Floor"]
position = Vector3(0, -0.5, 0)
mesh = SubResource("floor_mesh")

{chr(10).join(nodes)}

[node name="Player" type="{player_type}" parent="."]
position = Vector3(0, 1.2, 6)
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
        slug = dep.split("-")[-1]
        copied = []
        for f in files:
            if f.suffix.lower() in (".glb", ".gltf", ".png", ".jpg", ".webp") and f.exists():
                dest = game_dir / "assets" / slug / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                copied.append(f"res://assets/{slug}/{f.name}")
        kind = dep_types.get(dep, "")
        if kind == "rig_animate" and copied:
            idle = [c for c in copied if "idle" in c.lower()]
            char_glb = (idle or copied)[0]
        elif kind == "design_3d" and dep not in consumed and copied:
            env_glbs += [c for c in copied if c.endswith((".glb", ".gltf"))]

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
        _world_tscn(player_script, char_glb, env_glbs), encoding="utf-8")
