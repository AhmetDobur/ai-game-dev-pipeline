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
        ok, detail = _validate_code(output_paths, godot_binary, project_dir)
        if ok and task["spec"].get("frame_data") and project_dir is not None:
            return _validate_frame_data(godot_binary, project_dir)
        return ok, detail
    if kind == "design_2d":
        return _validate_files(output_paths, IMAGE_EXTS, MIN_IMAGE_BYTES, "image")
    if kind == "design_3d":
        ok, detail = _validate_files(output_paths, MESH_EXTS, MIN_MESH_BYTES, "mesh")
        if not ok:
            return ok, detail
        ok, detail = _validate_geometry(output_paths)
        return (ok, detail) if not ok else _validate_rigs(output_paths)
    if kind == "rig_animate":
        ok, detail = _validate_files(output_paths, {".glb", ".gltf"}, MIN_MESH_BYTES,
                                     "animated model")
        if not ok:
            return ok, detail
        ok, detail = _validate_geometry(output_paths)
        return (ok, detail) if not ok else _validate_rigs(output_paths)
    if kind == "audio":
        return _validate_audio(output_paths)
    if kind == "assemble":
        # assemble also returns a world screenshot; judge the BUILD, and never
        # let a 200 KB PNG look like a failed 1 MB export
        builds = [p for p in output_paths if p.suffix.lower() not in IMAGE_EXTS]
        return _validate_files(builds, {".exe", ".pck", ".zip", ".x86_64", ".app"},
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


def _validate_rigs(paths) -> tuple[bool, str]:
    """A character glb must actually bind its mesh to its skeleton."""
    from .inspect3d import rig_verdict
    for p in paths:
        if p.suffix.lower() in (".glb", ".gltf"):
            ok, why = rig_verdict(p)
            if not ok:
                return False, why
    return True, ""


def _validate_geometry(paths: list[Path]) -> tuple[bool, str]:
    """Byte count says nothing about shape: a flat plane still weighs 8 MB."""
    from .inspect3d import verdict
    for p in paths:
        if p.suffix.lower() in (".glb", ".gltf"):
            ok, detail = verdict(p)
            if not ok:
                return False, detail
    return True, "geometry ok"


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


def _validate_frame_data(godot_binary: str, project_dir: Path) -> tuple[bool, str]:
    """Run the pipeline's own headless simulation grader against frame_data.json.
    The grader is templates/frame_data_test.gd, copied into the game by the code
    executor — timing is judged by hard numbers, never by an LLM's opinion."""
    test = project_dir / "tests" / "frame_data_test.gd"
    if not test.exists():
        return False, "frame_data specced but tests/frame_data_test.gd missing from game dir"
    try:
        r = subprocess.run(
            [godot_binary, "--headless", "--path", str(project_dir),
             "--script", "res://tests/frame_data_test.gd"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=300)
    except FileNotFoundError:
        return True, "godot binary not found — frame-data simulation skipped"
    except subprocess.TimeoutExpired:
        return False, "frame-data simulation timed out (combat loop may not terminate)"
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0 and "FRAME_DATA_OK" in out:
        return True, "frame-data simulation passed"
    return False, f"frame-data simulation failed:\n{out[-2000:]}"


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
                # godot resolves a relative --script against res:// — hand it the
                # project-relative path, falling back to absolute outside the project
                try:
                    script = "res://" + p.resolve().relative_to(
                        project_dir.resolve()).as_posix()
                except ValueError:
                    script = str(p.resolve())
                r = subprocess.run(
                    [godot_binary, "--headless", "--check-only", "--script", script,
                     "--path", str(project_dir)],
                    capture_output=True, encoding="utf-8", errors="replace", timeout=120)
            except FileNotFoundError:
                return True, "godot binary not found — syntax check skipped"
            except subprocess.TimeoutExpired:
                return False, f"{p.name}: godot --check-only timed out"
            if r.returncode != 0:
                return False, f"{p.name}: script check failed:\n{(r.stderr or r.stdout)[-2000:]}"
    return True, f"{len(paths)} code file(s) ok"
