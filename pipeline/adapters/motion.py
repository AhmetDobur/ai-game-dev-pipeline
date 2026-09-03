"""Local motion stage — replaces the old Meshy cloud API.

Blender runs headless on a bundled script (templates/blender_motion.py) that rigs
the mesh and writes one animated glTF (.glb) per clip. The script ALWAYS produces
a moving .glb from Blender alone (the procedural floor), and folds in the optional
local tools when configured:

  - UniRig   -> auto-rig arbitrary meshes (humanoid or not)
  - Kimodo   -> text->motion for humanoid clips
  - CMU BVH  -> exact-match mocap clips, reused verbatim

Nothing leaves the machine: no API key, no network dependency. The GPU-bound tools
mean this runs as its own scheduler wave, not a background lane.
"""
import json
import subprocess
from pathlib import Path


class MotionStage:
    def __init__(self, blender: str = "blender",
                 script: str = "templates/blender_motion.py",
                 cmu_dir: str = "", unirig: str = "", kimodo_url: str = "",
                 timeout_s: int = 1800):
        self.blender = blender
        self.script = script
        self.cmu_dir = cmu_dir
        self.unirig = unirig
        self.kimodo_url = kimodo_url
        self.timeout_s = timeout_s

    def render_preview(self, glb: Path) -> Path | None:
        """Best-effort 512px PNG next to the glb so a human (or a review stage) can
        see what was generated. Failure never blocks the pipeline."""
        out = glb.with_suffix(".preview.png")
        script = str(Path(self.script).parent / "blender_preview.py")
        try:
            r = subprocess.run(
                [self.blender, "--background", "--python", script, "--",
                 str(glb), str(out)],
                capture_output=True, encoding="utf-8", errors="replace", timeout=300)
            if r.returncode == 0 and out.exists():
                return out
            failure = f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
        except Exception as e:  # cosmetic side-channel: NOTHING here may block a run
            failure = repr(e)
        try:  # leave the cause on disk — silent preview loss is undiagnosable
            glb.with_suffix(".preview.log").write_text(failure, encoding="utf-8")
        except OSError:
            pass
        return None

    def build(self, mesh_path: Path, body_plan: str, animations: list[str],
              extras: list[str], out_dir: Path) -> list[Path]:
        """Rig `mesh_path` and animate it; return the produced .glb files."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        args = {
            "mesh": str(mesh_path),
            "body_plan": body_plan,
            "animations": list(animations) or ["idle"],
            "extras": list(extras),
            "out_dir": str(out_dir),
            "cmu_dir": self.cmu_dir,
            "unirig": self.unirig,
            "kimodo_url": self.kimodo_url,
        }
        r = subprocess.run(
            [self.blender, "--background", "--python", self.script, "--", json.dumps(args)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=self.timeout_s)
        # "_"-prefixed files are the blender script's temps (e.g. _unirig.glb),
        # not animation clips — returning one would corrupt the character pick
        produced = sorted(p for p in out_dir.glob("*.glb")
                          if not p.name.startswith("_"))
        for p in produced:
            self.render_preview(p)  # best-effort PNG so humans can SEE the mesh
        if r.returncode != 0 or not produced:
            raise RuntimeError(
                f"blender motion stage failed (rc={r.returncode}):\n"
                f"{(r.stderr or r.stdout or '')[-2000:]}")
        return produced
