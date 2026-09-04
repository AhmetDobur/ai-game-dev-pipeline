"""Mesh-shape checks: only objectively broken geometry may fail a task."""
import json



def test_truncated_figure_is_rejected_only_for_humanoids(tmp_path):
    """The real failure: TRELLIS returned a robe and one boot, cut flat at the
    chest, at 200k faces. bbox is the measured one from pipeline3d_00026_.glb."""
    from pipeline.inspect3d import verdict
    mesh = tmp_path / "hero.glb"
    mesh.write_bytes(b"glTF")
    (tmp_path / "hero.glb.metrics.json").write_text(
        json.dumps({"bbox": [0.543, 0.354, 1.005]}))

    ok, detail = verdict(mesh, humanoid=True)
    assert not ok and "truncated figure" in detail

    # a prop or an environment is allowed to be squat
    assert verdict(mesh, humanoid=False)[0]


def test_whole_figure_passes(tmp_path):
    from pipeline.inspect3d import verdict
    mesh = tmp_path / "hero.glb"
    mesh.write_bytes(b"glTF")
    (tmp_path / "hero.glb.metrics.json").write_text(
        json.dumps({"bbox": [0.33, 0.25, 1.0]}))   # 3.0x taller than wide
    assert verdict(mesh, humanoid=True)[0]
