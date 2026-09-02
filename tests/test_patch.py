"""Patching: the pure graph transform (reuse unchanged, cascade to dependents,
rewire assemble), delta validation, and the atomic full-graph db insert."""
import pytest

from pipeline import db
from pipeline.patch import build_patch_graph, validate_patch_list


def _prow(tid, type_, deps, spec, status="done", out=""):
    return {"id": tid, "run_id": "P", "type": type_, "spec": spec, "depends_on": deps,
            "status": status, "attempts": 1, "output_path": out, "error": ""}


def _parent():
    return [
        _prow("P-art", "design_2d", [], {"prompt": "hero"},
              out="runs/P/artifacts/P-art/a.png"),
        _prow("P-mesh", "design_3d", ["P-art"], {"prompt": "hero", "concept_from": "P-art"},
              out="runs/P/artifacts/P-mesh/m.glb"),
        _prow("P-anim", "rig_animate", ["P-mesh"],
              {"mesh_from": "P-mesh", "body_plan": "humanoid", "animations": ["idle"]},
              out="runs/P/artifacts/P-anim/idle.glb"),
        _prow("P-code", "code", [], {"file": "player.gd", "description": "move"},
              out="runs/P/game/player.gd"),
        _prow("P-build", "assemble", ["P-art", "P-mesh", "P-anim", "P-code"],
              {"export_preset": "Windows Desktop"}, out="runs/P/dist/game.exe"),
    ]


def test_modify_cascades_to_dependents_and_reuses_the_rest():
    patch = [{"target": "P-mesh", "spec": {"prompt": "bigger hero", "concept_from": "P-art"}}]
    rows, stale = build_patch_graph(_parent(), patch, "P", "C")
    by = {r["id"]: r for r in rows}

    assert set(by) == {"C-art", "C-mesh", "C-anim", "C-code", "C-build"}
    # only the changed mesh + everything downstream re-runs
    assert stale == {"C-mesh", "C-anim", "C-build"}
    assert by["C-art"]["status"] == "done" and by["C-code"]["status"] == "done"
    assert by["C-mesh"]["status"] == "pending"
    # spec updated + its ref reprefixed parent->child
    assert by["C-mesh"]["spec"]["prompt"] == "bigger hero"
    assert by["C-mesh"]["spec"]["concept_from"] == "C-art"
    # reused artifact path repointed into this revision's dir; stale outputs cleared
    assert by["C-art"]["output_path"] == "runs/C/artifacts/P-art/a.png"
    assert by["C-mesh"]["output_path"] == "" and by["C-mesh"]["attempts"] == 0
    # assemble always re-runs last, depending on everything
    assert set(by["C-build"]["depends_on"]) == {"C-art", "C-mesh", "C-anim", "C-code"}


def test_add_new_character_wires_into_assemble_and_leaves_rest_done():
    patch = [
        {"id": "c2", "type": "design_2d", "depends_on": [], "spec": {"prompt": "boss"}},
        {"id": "m2", "type": "design_3d", "depends_on": ["c2"],
         "spec": {"prompt": "boss", "concept_from": "c2"}},
    ]
    rows, stale = build_patch_graph(_parent(), patch, "P", "C")
    by = {r["id"]: r for r in rows}

    assert "C-c2" in by and "C-m2" in by
    assert by["C-m2"]["depends_on"] == ["C-c2"] and by["C-m2"]["spec"]["concept_from"] == "C-c2"
    assert by["C-c2"]["status"] == "pending"
    # assemble picks up the new artifacts and re-runs
    assert {"C-c2", "C-m2"} <= set(by["C-build"]["depends_on"])
    assert by["C-build"]["status"] == "pending"
    # the whole original chain is untouched and reused
    for rid in ("C-art", "C-mesh", "C-anim", "C-code"):
        assert by[rid]["status"] == "done"
    assert stale == {"C-c2", "C-m2", "C-build"}


def test_modify_code_only_reuses_all_art():
    patch = [{"target": "P-code", "spec": {"file": "player.gd", "description": "double jump"}}]
    rows, stale = build_patch_graph(_parent(), patch, "P", "C")
    assert stale == {"C-code", "C-build"}  # no image/3d/motion work at all


def test_add_id_colliding_with_reused_parent_is_rejected():
    # an ADD id whose run-scoped form equals a copied parent task would silently
    # overwrite it — build_patch_graph must refuse rather than corrupt the graph
    patch = [{"id": "art", "type": "design_2d", "depends_on": [], "spec": {"prompt": "x"}}]
    with pytest.raises(ValueError, match="collides"):
        build_patch_graph(_parent(), patch, "P", "C")


def test_patching_a_failed_parent_reruns_the_failed_task():
    parent = _parent()
    for t in parent:
        if t["id"] == "P-code":
            t["status"] = "failed"      # parent ended in failure (assemble left blocked)
            t["attempts"] = 3
            t["error"] = "boom"
    # patch something unrelated (the mesh); the failed code task must still be repaired
    patch = [{"target": "P-mesh", "spec": {"prompt": "bigger", "concept_from": "P-art"}}]
    rows, stale = build_patch_graph(parent, patch, "P", "C")
    by = {r["id"]: r for r in rows}
    assert "C-code" in stale and by["C-code"]["status"] == "pending"
    assert by["C-code"]["attempts"] == 0 and by["C-code"]["error"] == ""
    assert by["C-build"]["status"] == "pending"   # assemble not blocked forever


def test_validate_patch_list_accepts_good_and_rejects_bad():
    mids = {"P-art", "P-mesh", "P-anim", "P-code", "P-build"}
    validate_patch_list([{"target": "P-code", "spec": {"file": "f.gd", "description": "d"}}], mids)
    validate_patch_list([{"id": "n", "type": "design_2d", "depends_on": ["P-art"],
                          "spec": {"prompt": "p"}}], mids)
    with pytest.raises(ValueError):
        validate_patch_list([{"target": "ghost", "spec": {}}], mids)
    with pytest.raises(ValueError):
        validate_patch_list([{"id": "b", "type": "assemble", "spec": {}}], mids)
    with pytest.raises(ValueError):
        validate_patch_list([{"id": "n", "type": "code", "depends_on": ["nope"],
                              "spec": {}}], mids)
    with pytest.raises(ValueError):
        validate_patch_list([], mids)


def test_add_tasks_full_is_atomic_and_preserves_status(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    rows = [
        {"id": f"{run}-a", "type": "design_2d", "spec": {"prompt": "p"}, "depends_on": [],
         "status": "done", "attempts": 1, "output_path": "x", "error": ""},
        {"id": f"{run}-b", "type": "assemble", "spec": {}, "depends_on": [f"{run}-a"],
         "status": "pending", "attempts": 0, "output_path": "", "error": ""},
    ]
    db.add_tasks_full(conn, run, rows)
    ts = {t["id"]: t for t in db.list_tasks(conn, run)}
    assert ts[f"{run}-a"]["status"] == "done" and ts[f"{run}-b"]["status"] == "pending"
    # done dep -> assemble is ready
    assert [t["id"] for t in db.ready_tasks(conn, run)] == [f"{run}-b"]


def test_add_tasks_full_rolls_back_on_bad_type(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    rows = [{"id": f"{run}-a", "type": "design_2d", "spec": {}, "depends_on": []},
            {"id": f"{run}-b", "type": "not_a_type", "spec": {}, "depends_on": []}]
    with pytest.raises(ValueError):
        db.add_tasks_full(conn, run, rows)
    assert db.list_tasks(conn, run) == []


def test_update_task_can_set_depends_on(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    tid = db.add_task(conn, run, "assemble", {})
    db.update_task(conn, tid, depends_on=["x", "y"])
    assert db.get_task(conn, tid)["depends_on"] == ["x", "y"]


def test_create_run_records_parent_and_revision(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    parent = db.create_run(conn, "i.md")
    child = db.create_run(conn, "d.md", parent_id=parent, revision=2)
    assert db.get_run(conn, child)["parent_id"] == parent
    assert db.get_run(conn, child)["revision"] == 2
