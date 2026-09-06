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


def test_blender_motion_lines_reach_the_log(capsys, monkeypatch, tmp_path):
    """Blender's output is captured so a good run is not 400 lines of glTF
    chatter -- but the script's own [motion] lines are its only report of what
    it decided, and swallowing them hid a stray object that shipped inside
    every character."""
    import subprocess

    from pipeline.adapters.motion import MotionStage

    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(
                            [], 0,
                            "glTF: mumble mumble\n"
                            "[motion] not exporting 1 stray object(s): Icosphere\n"
                            "more chatter\n", ""))
    MotionStage()._blender({"mesh": "x.glb"})
    out = capsys.readouterr().out
    assert "[motion] not exporting 1 stray object(s): Icosphere" in out
    assert "mumble" not in out


def test_motion_stage_carries_height_and_moveset_to_blender(tmp_path, monkeypatch):
    """The two keys that decide who a character IS: how tall, and whose punches.

    Both were readable inside blender_motion.py long before anything wrote them,
    so the authored GRAPPLER/STRIKER tables sat unused and every fighter shipped
    at 1.8m playing the same shared mocap. This test is the rail against that
    regressing quietly -- nothing else in the pipeline fails when they go missing.
    """
    captured = {}

    def fake_run(argv, **kw):
        try:
            args = json.loads(argv[-1])
        except json.JSONDecodeError:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        captured.update(args)
        from pathlib import Path
        (Path(args["out_dir"]) / "character.glb").write_bytes(b"g" * 60_000)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(motion.subprocess, "run", fake_run)
    stage = motion.MotionStage(blender="blender", script="s.py")
    stage.build(tmp_path / "mesh.glb", "humanoid", ["jab"], [], tmp_path / "out",
                height=2.8, moveset="grappler")

    assert captured["height"] == 2.8
    assert captured["moveset"] == "grappler"


def test_motion_stage_defaults_stay_backwards_compatible(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kw):
        try:
            args = json.loads(argv[-1])
        except json.JSONDecodeError:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        captured.update(args)
        from pathlib import Path
        (Path(args["out_dir"]) / "character.glb").write_bytes(b"g" * 60_000)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(motion.subprocess, "run", fake_run)
    motion.MotionStage(blender="blender", script="s.py").build(
        tmp_path / "m.glb", "humanoid", ["idle"], [], tmp_path / "out")

    assert captured["height"] == 1.8 and captured["moveset"] == ""


def test_export_writes_leaf_bone_tips():
    """The rig must carry its leaf bones' lengths out of Blender.

    glTF stores joints as points, so a leaf bone's tail exists nowhere in the
    file and the importer fabricates one from the parent's length. That handed
    the shipped character a foot bone 2.6x too long, whose tip sat below the
    floor and swept the coat skirt into anything measuring "what is near the
    foot" -- cloak_audit first, but Godot IK and hitboxes next. Nothing else in
    the pipeline notices when this keyword goes missing, so this is the rail.
    """
    import importlib.util
    import sys
    import types
    from pathlib import Path

    kwargs = {}
    bpy = types.ModuleType("bpy")
    bpy.ops = SimpleNamespace(
        object=SimpleNamespace(select_all=lambda **k: None),
        export_scene=SimpleNamespace(gltf=lambda **k: kwargs.update(k)))
    bpy.context = SimpleNamespace(scene=SimpleNamespace(objects=[]))

    saved = {k: sys.modules.get(k) for k in ("mathutils", "bpy")}
    sys.modules["mathutils"] = types.ModuleType("mathutils")
    sys.modules["bpy"] = bpy
    try:
        path = Path(__file__).resolve().parents[1] / "templates" / "blender_motion.py"
        spec = importlib.util.spec_from_file_location("blender_motion_export", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rig = type("Obj", (), {"type": "ARMATURE", "parent": None, "name": "rig",
                               "select_set": lambda self, v: None})()
        mod.export_glb("/tmp/x.glb", arm=rig)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    assert kwargs["export_leaf_bone"] is True
    assert kwargs["use_selection"] is True and kwargs["export_animations"] is True
