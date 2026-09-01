"""Per-branch output validation. Objective checks only — a failed check re-queues the
task in-wave; an LLM opinion never rejects an artifact (only broken signals do)."""
import subprocess
import wave
from pathlib import Path

MIN_IMAGE_BYTES = 20_000       # a real SDXL render is never smaller
MIN_MESH_BYTES = 50_000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MESH_EXTS = {".glb", ".gltf", ".obj", ".fbx"}


def validate(task: dict, output_paths: list[Path], godot_binary: str = "godot",
             project_dir: Path | None = None) -> tuple[bool, str]:
    kind = task["type"]
    if kind == "code":
        return _validate_code(output_paths, godot_binary, project_dir)
    if kind == "design_2d":
        return _validate_files(output_paths, IMAGE_EXTS, MIN_IMAGE_BYTES, "image")
    if kind == "design_3d":
        return _validate_files(output_paths, MESH_EXTS, MIN_MESH_BYTES, "mesh")
    if kind == "rig_animate":
        return _validate_files(output_paths, {".fbx", ".glb"}, MIN_MESH_BYTES, "rigged model")
    if kind == "audio":
        return _validate_audio(output_paths)
    if kind == "assemble":
        return _validate_files(output_paths, {".exe", ".pck", ".zip", ".x86_64", ".app"},
                               1_000_000, "game build")
    return False, f"no validator for task type {kind!r}"


def _validate_files(paths: list[Path], exts: set[str], min_bytes: int,
                    label: str) -> tuple[bool, str]:
    if not paths:
        return False, f"no {label} produced"
    for p in paths:
        if not p.exists():
            return False, f"{p} does not exist"
        if p.suffix.lower() not in exts:
            return False, f"{p.name}: unexpected extension for a {label}"
        if p.stat().st_size < min_bytes:
            return False, f"{p.name}: {p.stat().st_size} bytes — too small for a real {label}"
    return True, f"{len(paths)} {label} file(s) ok"


def _validate_audio(paths: list[Path]) -> tuple[bool, str]:
    if not paths:
        return False, "no audio produced"
    for p in paths:
        try:
            with wave.open(str(p)) as w:
                duration = w.getnframes() / w.getframerate()
        except (wave.Error, EOFError, FileNotFoundError) as e:
            return False, f"{p.name}: not a readable WAV ({e})"
        if duration < 0.2:
            return False, f"{p.name}: {duration:.2f}s — effectively silent"
    return True, f"{len(paths)} audio file(s) ok"


def _validate_code(paths: list[Path], godot_binary: str,
                   project_dir: Path | None) -> tuple[bool, str]:
    """GDScript parses under godot --check-only; other files just need to exist."""
    if not paths:
        return False, "no code file produced"
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            return False, f"{p} missing or empty"
        if p.suffix == ".gd" and project_dir is not None:
            try:
                r = subprocess.run(
                    [godot_binary, "--headless", "--check-only", "--script", str(p),
                     "--path", str(project_dir)],
                    capture_output=True, encoding="utf-8", errors="replace", timeout=120)
            except FileNotFoundError:
                return True, "godot binary not found — syntax check skipped"
            except subprocess.TimeoutExpired:
                return False, f"{p.name}: godot --check-only timed out"
            if r.returncode != 0:
                return False, f"{p.name}: script check failed:\n{(r.stderr or r.stdout)[-2000:]}"
    return True, f"{len(paths)} code file(s) ok"
