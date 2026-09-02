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
        produced = sorted(out_dir.glob("*.glb"))
        if r.returncode != 0 or not produced:
            raise RuntimeError(
                f"blender motion stage failed (rc={r.returncode}):\n"
                f"{(r.stderr or r.stdout or '')[-2000:]}")
        return produced
