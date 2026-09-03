"""Local motion adapter: builds the right Blender argv, collects out_dir/*.glb,
fails loudly when Blender produces nothing. The bpy script itself runs on the
target box, so the subprocess boundary is mocked here (as godot is elsewhere)."""
import json
from types import SimpleNamespace

import pytest

from pipeline.adapters import motion


def test_motion_stage_passes_args_and_collects_glb(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kw):
        try:
            args = json.loads(argv[-1])
        except json.JSONDecodeError:
            # render_preview's call (argv ends in a .png path) — not the build
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        captured["argv"] = argv
        from pathlib import Path
        (Path(args["out_dir"]) / "idle.glb").write_bytes(b"g" * 60_000)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(motion.subprocess, "run", fake_run)
    stage = motion.MotionStage(blender="blender", script="s.py")
    out = stage.build(tmp_path / "mesh.glb", "nonhumanoid", ["idle"], ["tail"],
                      tmp_path / "out")

    assert [p.name for p in out] == ["idle.glb"]
    assert captured["argv"][:5] == ["blender", "--background", "--python", "s.py", "--"]
    a = json.loads(captured["argv"][-1])
    assert a["body_plan"] == "nonhumanoid" and a["extras"] == ["tail"]
    assert a["animations"] == ["idle"] and a["mesh"].endswith("mesh.glb")


def test_motion_stage_defaults_empty_animations_to_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(motion.subprocess, "run",
                        lambda argv, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    seen = {}

    def fake_run(argv, **kw):
        seen["a"] = json.loads(argv[-1])
        from pathlib import Path
        (Path(seen["a"]["out_dir"]) / "idle.glb").write_bytes(b"g" * 60_000)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(motion.subprocess, "run", fake_run)
    motion.MotionStage().build(tmp_path / "m.glb", "humanoid", [], [], tmp_path / "o")
    assert seen["a"]["animations"] == ["idle"]


def test_motion_stage_raises_when_blender_makes_no_glb(tmp_path, monkeypatch):
    monkeypatch.setattr(motion.subprocess, "run",
                        lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"))
    with pytest.raises(RuntimeError, match="motion stage failed"):
        motion.MotionStage().build(tmp_path / "m.glb", "humanoid", ["idle"], [],
                                   tmp_path / "o")
