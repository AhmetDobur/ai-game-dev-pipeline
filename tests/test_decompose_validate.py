import wave
from pathlib import Path

import pytest

from pipeline import db
from pipeline.adapters.llm import extract_json, parse_tool_call
from pipeline.decompose import insert_tasks, repair_task_list, validate_task_list
from pipeline.validate import validate

GOOD = [
    {"id": "art", "type": "design_2d", "depends_on": [], "spec": {"prompt": "p"}},
    {"id": "mesh", "type": "design_3d", "depends_on": ["art"],
     "spec": {"prompt": "p", "concept_from": "art"}},
    {"id": "main", "type": "code", "depends_on": [], "spec": {"file": "main.gd",
                                                              "description": "d"}},
    {"id": "build", "type": "assemble", "depends_on": ["mesh", "main"], "spec": {}},
]


def test_validate_task_list_accepts_good_and_rejects_bad():
    validate_task_list(GOOD)
    with pytest.raises(ValueError):
        validate_task_list([{"id": "a", "type": "nope", "depends_on": [], "spec": {}}])
    with pytest.raises(ValueError):
        validate_task_list([{"id": "a", "type": "code", "depends_on": ["ghost"],
                             "spec": {}}])
    with pytest.raises(ValueError):
        validate_task_list([])


def test_structural_rules_and_repair():
    # the graph the 7B router actually emitted once: no code, empty assemble deps,
    # rig pointing at a design_2d — every one must be caught or repaired
    bad = [
        {"id": "char", "type": "design_2d", "depends_on": [], "spec": {"prompt": "p"}},
        {"id": "rig", "type": "rig_animate", "depends_on": [],
         "spec": {"mesh_from": "char"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    repair_task_list(bad)
    # repair also synthesizes a design_3d for the orphan prop concept "char",
    # and a head close-up mesh for the humanoid the rig animates
    assert set(bad[2]["depends_on"]) == {"char", "char_mesh", "char_mesh_head", "rig"}
    head = next(t for t in bad if t["id"] == "char_mesh_head")
    assert head["type"] == "design_3d" and head["spec"]["detail"] == "head"
    assert "close-up" in head["spec"]["prompt"]
    synth = next(t for t in bad if t["id"] == "char_mesh")
    assert synth["type"] == "design_3d" and synth["spec"]["concept_from"] == "char"
    # rig-on-art is retargeted through the synthesized mesh, which implies the dep
    assert bad[1]["spec"]["mesh_from"] == "char_mesh"
    assert bad[1]["depends_on"] == ["char_mesh"]
    with pytest.raises(ValueError, match="code"):
        validate_task_list(bad)
    good = [dict(t) for t in GOOD]
    good.append({"id": "rig", "type": "rig_animate", "depends_on": ["mesh"],
                 "spec": {"mesh_from": "mesh"}})
    good[3] = dict(good[3], depends_on=["mesh", "main", "rig"])
    validate_task_list(good)
    good[4] = dict(good[4], spec={"mesh_from": "art"})    # rig on a 2d image
    with pytest.raises(ValueError, match="design_3d"):
        validate_task_list(good)


def test_repair_breaks_cycles_and_links_concepts():
    tasks = [
        {"id": "art", "type": "design_2d", "depends_on": [], "spec": {"prompt": "p"}},
        {"id": "mesh", "type": "design_3d", "depends_on": ["art", "build"],
         "spec": {"prompt": "p"}},                       # depends on assemble = cycle
        {"id": "main", "type": "code", "depends_on": [], "spec": {"file": "m.gd"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    repair_task_list(tasks)
    assert "build" not in tasks[1]["depends_on"]         # cycle edge stripped
    assert tasks[1]["spec"]["concept_from"] == "art"     # auto-linked to its 2d dep
    validate_task_list(tasks)
    # an irreparable cycle (two code tasks depending on each other) is rejected
    bad = [
        {"id": "a", "type": "code", "depends_on": ["b"], "spec": {"file": "a.gd"}},
        {"id": "b", "type": "code", "depends_on": ["a"], "spec": {"file": "b.gd"}},
        {"id": "art", "type": "design_2d", "depends_on": [], "spec": {"prompt": "p"}},
        {"id": "build", "type": "assemble", "depends_on": [], "spec": {}},
    ]
    repair_task_list(bad)
    with pytest.raises(ValueError, match="cycle"):
        validate_task_list(bad)


def test_insert_tasks_is_atomic(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "a.md")
    broken = [dict(t) for t in GOOD]
    broken[2] = dict(broken[2], type="nope")             # add_tasks must reject ALL
    with pytest.raises(ValueError):
        insert_tasks(conn, run, broken)
    assert db.list_tasks(conn, run) == []                # zero rows, not a prefix


def test_insert_tasks_scopes_ids_per_run(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    r1 = db.create_run(conn, "a.md")
    r2 = db.create_run(conn, "b.md")
    insert_tasks(conn, r1, GOOD)
    insert_tasks(conn, r2, GOOD)  # same decomposer ids, no collision
    t1 = db.list_tasks(conn, r1)
    assert len(t1) == len(GOOD) and len(db.list_tasks(conn, r2)) == len(GOOD)
    mesh = next(t for t in t1 if t["type"] == "design_3d")
    assert mesh["depends_on"] == [f"{r1}-art"]


def test_insert_tasks_remaps_spec_references(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "a.md")
    tasks = [
        {"id": "art", "type": "design_2d", "depends_on": [], "spec": {"prompt": "p"}},
        {"id": "mesh", "type": "design_3d", "depends_on": ["art"],
         "spec": {"prompt": "p", "concept_from": "art"}},
    ]
    insert_tasks(conn, run, tasks)
    mesh = next(t for t in db.list_tasks(conn, run) if t["type"] == "design_3d")
    assert mesh["spec"]["concept_from"] == f"{run}-art"


def test_extract_json_variants():
    assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json('noise {"a": 1} noise') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_tool_call_dialects():
    assert parse_tool_call('<tools>\n{"name": "f", "arguments": {"x": 1}}\n</tools>')["name"] == "f"
    assert parse_tool_call('```json\n{"name": "f", "arguments": {}}\n```')["name"] == "f"
    assert parse_tool_call("just words") is None


def test_validate_audio(tmp_path):
    p = tmp_path / "line.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 24000)  # 1s of silence is still a valid file
    ok, msg = validate({"type": "audio"}, [p])
    assert ok, msg
    p2 = tmp_path / "bad.wav"
    p2.write_bytes(b"not a wav")
    ok, msg = validate({"type": "audio"}, [p2])
    assert not ok


def test_validate_image_size_floor(tmp_path):
    small = tmp_path / "s.png"; small.write_bytes(b"x" * 100)
    ok, _ = validate({"type": "design_2d"}, [small])
    assert not ok
    big = tmp_path / "b.png"; big.write_bytes(b"x" * 30_000)
    ok, _ = validate({"type": "design_2d"}, [big])
    assert ok


def test_run_workflow_uploads_local_file_substitutions(tmp_path, monkeypatch):
    """A substitution value that is a real file must be uploaded and replaced by
    the uploaded name — LoadImage never accepts absolute paths."""
    from pipeline.adapters.comfy import ComfyClient

    wf = tmp_path / "wf.json"
    wf.write_text('{"1": {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}}}')
    img = tmp_path / "concept.png"
    img.write_bytes(b"\x89PNGfake")

    client = ComfyClient("http://127.0.0.1:9")
    submitted = {}
    monkeypatch.setattr(client, "upload_image", lambda p: f"uploaded-{p.name}")

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"prompt_id": "p1"}
    def fake_post(url, json=None, **kw):
        submitted["workflow"] = json["prompt"]
        raise RuntimeError("stop after submit")  # don't enter the poll loop
    monkeypatch.setattr("pipeline.adapters.comfy.requests.post", fake_post)

    try:
        client.run_workflow(wf, {"image": str(img), "prompt": "a fighter"}, tmp_path)
    except RuntimeError:
        pass
    assert submitted["workflow"]["1"]["inputs"]["image"] == "uploaded-concept.png"
