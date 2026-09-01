"""Production executors: one callable per task type, built from config.
Each takes (task, out_dir) and returns the list of produced files."""
import subprocess
from pathlib import Path

from .adapters.comfy import ComfyClient
from .adapters.llm import LlamaServer
from .adapters.meshy import MeshyClient
from .adapters.tts import TTSClient

CODE_PROMPT = """You are the coder of an automated Godot 4 game pipeline.

Write the complete content of `{file}` for a Godot 4 project.

What it must do:
{description}

{fix_note}Reply with ONLY the file content in a single fenced code block."""


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
    game_dir = workspace / "game"

    def code(task: dict, out_dir: Path) -> list[Path]:
        spec = task["spec"]
        fix_note = ""
        if task.get("last_error"):
            fix_note = f"Your previous attempt failed validation:\n{task['last_error']}\n\n"
        reply = coder.chat(
            [{"role": "user", "content": CODE_PROMPT.format(
                file=spec["file"], description=spec["description"], fix_note=fix_note)}],
            temperature=cfg["llm"]["temperature"], max_tokens=cfg["llm"]["max_tokens"],
            timeout_s=cfg["llm"]["request_timeout_s"])
        path = game_dir / spec["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_extract_block(reply), encoding="utf-8")
        return [path]

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
        import base64
        mesh_path = _resolve_dep(task, task["spec"]["mesh_from"])
        # Meshy wants a URL; a base64 data URI keeps local meshes local
        model_url = ("data:model/gltf-binary;base64,"
                     + base64.b64encode(mesh_path.read_bytes()).decode())
        meshy = MeshyClient(cfg["meshy"]["url"], cfg["meshy"]["api_key_env"],
                            cfg["meshy"]["poll_interval_s"], cfg["meshy"]["timeout_s"])
        return meshy.rig_and_animate(model_url,
                                     task["spec"].get("animations", ["idle"]), out_dir)

    def audio(task: dict, out_dir: Path) -> list[Path]:
        out = out_dir / "line.wav"
        return [tts.speak(task["spec"]["text"], task["spec"].get("voice", "leo"), out)]

    def assemble(task: dict, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        preset = task["spec"].get("export_preset", "Windows Desktop")
        target = out_dir / ("game.exe" if "windows" in preset.lower() else "game.zip")
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
