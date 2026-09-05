"""Mesh-shape checks: only objectively broken geometry may fail a task."""
import json
from pathlib import Path



def _mesh(tmp_path, **metrics):
    mesh = tmp_path / "hero.glb"
    mesh.write_bytes(b"glTF")
    (tmp_path / "hero.glb.metrics.json").write_text(json.dumps(metrics))
    return mesh


def test_truncated_figure_is_rejected_only_for_humanoids(tmp_path):
    """The real failure: TRELLIS returned a robe and one boot, cut flat at the
    chest, at 200k faces. Numbers measured from pipeline3d_00030_.glb."""
    from pipeline.inspect3d import verdict
    mesh = _mesh(tmp_path, bbox=[0.543, 0.354, 1.005], top_width_ratio=0.97)

    ok, detail = verdict(mesh, humanoid=True)
    assert not ok and "truncated figure" in detail

    # a prop or an environment is allowed to be squat
    assert verdict(mesh, humanoid=False)[0]


def test_whole_figure_passes(tmp_path):
    """The Pious Force: a real, complete character measuring 1.91x taller than
    wide. The height/width test this replaced rejected exactly this mesh."""
    from pipeline.inspect3d import verdict
    mesh = _mesh(tmp_path, bbox=[0.524, 0.335, 0.998], top_width_ratio=0.293)
    assert verdict(mesh, humanoid=True)[0]


def test_truncation_check_is_skipped_when_the_metric_is_absent(tmp_path):
    """Older meshes have no top_width_ratio; absence of evidence never fails."""
    from pipeline.inspect3d import verdict
    assert verdict(_mesh(tmp_path, bbox=[0.6, 0.4, 1.0]), humanoid=True)[0]
    assert verdict(_mesh(tmp_path, bbox=[0.6, 0.4, 1.0],
                         top_width_ratio=None), humanoid=True)[0]


def test_render_preview_passes_absolute_paths_to_blender(tmp_path, monkeypatch):
    """Blender resolves a relative output path against ITS cwd, not ours, so a
    relative glb sent every preview and every .metrics.json to C:\\workspace."""
    from pipeline.adapters.motion import MotionStage
    import subprocess

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)
    (tmp_path / "art").mkdir()
    (tmp_path / "art" / "m.glb").write_bytes(b"glTF")

    MotionStage().render_preview(Path("art/m.glb"))       # relative, as stored
    glb_arg, out_arg = seen["cmd"][-2], seen["cmd"][-1]
    assert Path(glb_arg).is_absolute() and Path(out_arg).is_absolute()
    assert Path(out_arg).parent == (tmp_path / "art").resolve()
