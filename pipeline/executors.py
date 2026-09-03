"""Production executors: one callable per task type, built from config.
Each takes (task, out_dir) and returns the list of produced files."""
import json
import subprocess
from pathlib import Path

from .adapters.comfy import ComfyClient
from .adapters.llm import LlamaServer
from .adapters.motion import MotionStage
from .adapters.tts import TTSClient

CODE_PROMPT = """You are the coder of an automated Godot 4 game pipeline.

Write the complete content of `{file}` for a Godot 4 project.

What it must do:
{description}

Godot 4 GDScript ONLY — never Godot 3 API. The renames you MUST respect:
deg2rad->deg_to_rad, rad2deg->rad_to_deg, BUTTON_*->MOUSE_BUTTON_*,
File->FileAccess, Directory->DirAccess, JSON.parse(s)->JSON.parse_string(s),
KinematicBody->CharacterBody3D (velocity is built in; move_and_slide() takes
no arguments), Spatial->Node3D, onready var->@onready var, export->@export,
yield->await, .instance()->.instantiate(), OS.get_ticks_msec()->Time.get_ticks_msec(),
rand_range(a,b)->randf_range(a,b), Input singleton constants unchanged.

Reference — a CORRECT Godot 4 third-person controller (copy these idioms exactly:
velocity is inherited, never redeclared; mouse look uses _unhandled_input, there
is no Input.get_mouse_delta):

    extends CharacterBody3D
    @export var speed := 4.0
    @export var run_speed := 8.0
    @onready var cam_pivot: Node3D = $CamPivot
    func _unhandled_input(event: InputEvent) -> void:
        if event is InputEventMouseMotion:
            rotate_y(-event.relative.x * 0.003)
            cam_pivot.rotate_x(-event.relative.y * 0.003)
            cam_pivot.rotation.x = clampf(cam_pivot.rotation.x, -1.2, 1.2)
    func _physics_process(delta: float) -> void:
        var input := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
        var dir := (transform.basis * Vector3(input.x, 0, input.y)).normalized()
        var s := run_speed if Input.is_key_pressed(KEY_SHIFT) else speed
        velocity.x = dir.x * s
        velocity.z = dir.z * s
        if not is_on_floor():
            velocity.y -= 20.0 * delta
        move_and_slide()

{frame_data_note}{fix_note}Reply with ONLY the file content in a single fenced code block."""

# The CombatSim contract lets the pipeline grade timing with its own static
# GDScript test (templates/frame_data_test.gd) instead of trusting the model.
COMBAT_SIM_CONTRACT = """This file is graded by a headless frame-data simulation. It MUST be a
GDScript class implementing exactly this API (60fps fixed steps, no scene tree,
no rendering, pure simulation):

    setup(move: String) -> void   # place two fighters at the move's range
    press(move: String) -> void   # buffer the input for the next frame
    step() -> void                # advance exactly one frame
    hitbox_active() -> bool       # attacker's hitbox live this frame
    opponent_in_hitstun() -> bool
    opponent_offset() -> Vector2  # opponent displacement since setup()

It must read its numbers from res://frame_data.json and honour them exactly:
{frame_data}
"""


def _extract_block(reply: str) -> str:
    import re
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", reply, re.S)
    return max(blocks, key=len) if blocks else reply


def _resolve_dep(task: dict, ref: str) -> Path:
    """Turn a spec reference to another task (e.g. concept_from) into that task's
    first output file. The scheduler injects dep_outputs per dependency id."""
    outputs = task.get("dep_outputs", {}).get(ref, [])
    if not outputs:
        raise RuntimeError(f"task {task['id']}: reference {ref!r} has no resolvable "
                           f"output (dep_outputs keys: {list(task.get('dep_outputs', {}))})")
    return Path(outputs[0])


def build_executors(cfg: dict, workspace: Path,
                    coder: LlamaServer) -> dict:
    comfy = ComfyClient(cfg["comfy"]["url"], cfg["comfy"]["timeout_s"])
    tts = TTSClient(cfg["tts"]["url"], cfg["tts"]["timeout_s"])
    motion = MotionStage(cfg["motion"]["blender"], cfg["motion"]["script"],
                         cfg["motion"]["cmu_dir"], cfg["motion"]["unirig"],
                         cfg["motion"]["kimodo_url"], cfg["motion"]["timeout_s"])
    game_dir = workspace / "game"

    def code(task: dict, out_dir: Path) -> list[Path]:
        spec = task["spec"]
        fix_note = ""
        if task.get("last_error"):
            fix_note = f"Your previous attempt failed validation:\n{task['last_error']}\n\n"
        frame_data_note = ""
        produced = []
        if spec.get("frame_data"):
            # materialize the table + the pipeline's own grader into the game dir
            game_dir.mkdir(parents=True, exist_ok=True)
            fd_path = game_dir / "frame_data.json"
            fd_path.write_text(json.dumps(spec["frame_data"], indent=1), encoding="utf-8")
            test_dst = game_dir / "tests" / "frame_data_test.gd"
            test_dst.parent.mkdir(parents=True, exist_ok=True)
            test_dst.write_text((Path(__file__).parent.parent / "templates"
                                 / "frame_data_test.gd").read_text(encoding="utf-8"),
                                encoding="utf-8")
            produced = [fd_path, test_dst]
            frame_data_note = COMBAT_SIM_CONTRACT.format(
                frame_data=json.dumps(spec["frame_data"], indent=1)) + "\n"
        from . import livelog
        livelog.start(task["run_id"], f"coding {spec['file']}"
                      + (f" (retry {task['attempts']})" if task.get("last_error") else ""))
        reply = coder.chat(
            [{"role": "user", "content": CODE_PROMPT.format(
                file=spec["file"], description=spec["description"],
                frame_data_note=frame_data_note, fix_note=fix_note)}],
            temperature=cfg["llm"]["temperature"], max_tokens=cfg["llm"]["max_tokens"],
            timeout_s=cfg["llm"]["request_timeout_s"],
            on_token=livelog.token_sink(task["run_id"]))
        path = game_dir / spec["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_extract_block(reply), encoding="utf-8")
        return [path, *produced]

    def design_2d(task: dict, out_dir: Path) -> list[Path]:
        return comfy.run_workflow(cfg["comfy"]["sdxl_workflow"],
                                  {"prompt": task["spec"]["prompt"]}, out_dir)

    def design_3d(task: dict, out_dir: Path) -> list[Path]:
        # conditioned on the matching design_2d output when the decomposer linked one
        subs = {"prompt": task["spec"].get("prompt", "")}
        concept = task["spec"].get("concept_from", "")
        if concept:
            subs["image"] = str(_resolve_dep(task, concept))
        return comfy.run_workflow(cfg["comfy"]["trellis_workflow"], subs, out_dir)

    def rig_animate(task: dict, out_dir: Path) -> list[Path]:
        spec = task["spec"]
        mesh_path = _resolve_dep(task, spec["mesh_from"])
        # local Blender: humanoid -> mocap/generated motion; anything else -> procedural.
        # extras (tail/jaw/wings) always get procedural secondary motion on top.
        return motion.build(mesh_path, spec.get("body_plan", "humanoid"),
                            spec.get("animations", ["idle"]), spec.get("extras", []),
                            out_dir)

    def audio(task: dict, out_dir: Path) -> list[Path]:
        out = out_dir / "line.wav"
        return [tts.speak(task["spec"]["text"], task["spec"].get("voice", "leo"), out)]

    def assemble(task: dict, out_dir: Path) -> list[Path]:
        # stable, predictable location so users find the build without a task id
        dist = workspace / "dist"
        dist.mkdir(parents=True, exist_ok=True)
        preset = task["spec"].get("export_preset", "Windows Desktop")
        target = dist / ("game.exe" if "windows" in preset.lower() else "game.zip")
        godot = cfg["paths"]["godot"]
        subprocess.run([godot, "--headless", "--path", str(game_dir), "--import"],
                       check=False, capture_output=True, timeout=600)
        r = subprocess.run([godot, "--headless", "--path", str(game_dir),
                            "--export-release", preset, str(target)],
                           capture_output=True, encoding="utf-8", errors="replace",
                           timeout=1800)
        if r.returncode != 0 or not target.exists():
            raise RuntimeError(f"godot export failed:\n{(r.stderr or r.stdout)[-2000:]}")
        return [target]

    return {"code": code, "design_2d": design_2d, "design_3d": design_3d,
            "rig_animate": rig_animate, "audio": audio, "assemble": assemble}
