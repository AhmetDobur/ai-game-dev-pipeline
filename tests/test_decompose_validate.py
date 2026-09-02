import wave
from pathlib import Path

import pytest

from pipeline import db
from pipeline.adapters.llm import extract_json, parse_tool_call
from pipeline.decompose import insert_tasks, validate_task_list
from pipeline.validate import validate

GOOD = [
    {"id": "art", "type": "design_2d", "depends_on": [], "spec": {"prompt": "p"}},
    {"id": "mesh", "type": "design_3d", "depends_on": ["art"], "spec": {"prompt": "p"}},
    {"id": "build", "type": "assemble", "depends_on": ["mesh"], "spec": {}},
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


def test_insert_tasks_scopes_ids_per_run(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    r1 = db.create_run(conn, "a.md")
    r2 = db.create_run(conn, "b.md")
    insert_tasks(conn, r1, GOOD)
    insert_tasks(conn, r2, GOOD)  # same decomposer ids, no collision
    t1 = db.list_tasks(conn, r1)
    assert len(t1) == 3 and len(db.list_tasks(conn, r2)) == 3
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
