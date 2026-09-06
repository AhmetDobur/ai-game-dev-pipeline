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
import os
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
        # ABSOLUTE, always: blender resolves a relative output path against its
        # own cwd, not ours. Task output paths are stored relative, so every
        # preview and every .metrics.json was being written to C:\workspace\...
        # and silently lost -- which left _validate_geometry with no metrics to
        # read, and "no metrics" is a pass. A library that came out as a flat
        # 1.3%-thick disc shipped as done because of exactly this.
        glb = Path(glb).resolve()
        out = glb.with_suffix(".preview.png")
        script = str((Path(self.script).parent / "blender_preview.py").resolve())
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

    def _blender(self, args: dict, timeout: int | None = None):
        return subprocess.run(
            [self.blender, "--background", "--python", self.script, "--",
             json.dumps(args)],
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=timeout or self.timeout_s)

    def unirig_skin(self, mesh_path: Path, body_plan: str, extras: list[str],
                    out_dir: Path) -> Path | None:
        """Build our rig, then let UniRig predict its weights. None on failure.

        UniRig will also predict a skeleton, but its bones come out anonymous --
        bone_0 .. bone_27 -- while every authored moveset addresses LeftArm,
        Hand and UpLeg by name, so adopting it would turn every clip into a
        silent no-op. Only the skinning is taken. Bone heat gives a whole coat
        panel to whichever limb hangs inside it, which is right at rest and
        lifts the entire cloak on a punch; predicted weights halve the mesh
        stranded on the arm bones and drop the coat onto the hips.

        Run from HERE rather than from inside Blender. As Blender's child the
        skin pass failed every time with identical figures -- 3 GB requested
        against 20 GB free on an idle card -- and succeeded every time from a
        shell. Environment, PATH, precision, job-object breakaway, scene state
        and host memory were each ruled out by measurement.
        """
        out_dir = Path(out_dir)
        rig_fbx = out_dir / "_unirig_in.fbx"
        skin_fbx = out_dir / "_unirig_skin.fbx"
        target = out_dir / "_unirig_target.glb"
        merged = out_dir / "_unirig_out.glb"

        r = self._blender({"mesh": str(mesh_path), "body_plan": body_plan,
                           "extras": list(extras), "out_dir": str(out_dir),
                           "rig_only": True})
        if r.returncode != 0 or not rig_fbx.exists():
            print(f"[motion] rig export for UniRig failed (rc={r.returncode})",
                  flush=True)
            return None

        py = os.environ.get("UNIRIG_PYTHON", "python")
        env = dict(os.environ)
        # torch 2.6 defaults torch.load to weights_only=True and refuses the Box
        # object UniRig's published checkpoints pickle beside their tensors.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        def unirig(argv):
            return subprocess.run([py] + argv, cwd=self.unirig, env=env,
                                  capture_output=True, encoding="utf-8",
                                  errors="replace", timeout=self.timeout_s)

        # the launchers in launch/inference are three python calls in a trench
        # coat; called directly so this does not need bash on Windows
        stages = [
            ["-m", "src.data.extract",
             "--config=configs/data/quick_inference.yaml",
             "--require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm",
             "--force_override=true", "--num_runs=1", "--id=0",
             "--time=motion", "--faces_target_count=50000",
             f"--input={rig_fbx.as_posix()}", "--output_dir=tmp"],
            ["run.py", "--task=configs/task/quick_inference_unirig_skin.yaml",
             "--seed=12345", f"--input={rig_fbx.as_posix()}",
             f"--output={skin_fbx.as_posix()}", "--npz_dir=tmp",
             "--data_name=raw_data.npz"],
            # require_suffix/num_runs/id are unused on the --source/--target
            # path -- merge.py returns straight out of transfer() -- but
            # argparse marks them required, so they are supplied as dummies
            ["-m", "src.inference.merge", "--require_suffix=fbx",
             "--num_runs=1", "--id=0", f"--source={skin_fbx.as_posix()}",
             f"--target={target.as_posix()}", f"--output={merged.as_posix()}"],
        ]
        for argv in stages:
            r = unirig(argv)
            if r.returncode != 0:
                print(f"[motion] UniRig {argv[0]} failed (rc={r.returncode}), "
                      f"keeping bone-heat weights:\n"
                      f"{(r.stderr or r.stdout or '')[-800:]}", flush=True)
                return None
        return merged if merged.exists() else None

    def build(self, mesh_path: Path, body_plan: str, animations: list[str],
              extras: list[str], out_dir: Path) -> list[Path]:
        """Rig `mesh_path` and animate it; return the produced .glb files."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # A failed skinning pass is not a failed stage: the procedural rig has
        # already produced usable weights, so the character still animates.
        rigged = self.unirig_skin(mesh_path, body_plan, extras, out_dir) \
            if self.unirig else None
        args = {
            "mesh": str(rigged or mesh_path),
            "prerigged": rigged is not None,
            "body_plan": body_plan,
            "animations": list(animations) or ["idle"],
            "extras": list(extras),
            "out_dir": str(out_dir),
            "cmu_dir": self.cmu_dir,
            "kimodo_url": self.kimodo_url,
        }
        r = self._blender(args)
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
