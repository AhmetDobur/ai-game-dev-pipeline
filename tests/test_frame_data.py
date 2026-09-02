"""Frame-data contract: executor materializes table + grader, validator runs the
headless simulation and believes only FRAME_DATA_OK + exit 0."""
import json
import os
import stat
from pathlib import Path

from pipeline import validate
from pipeline.config import load
from pipeline.executors import build_executors

FRAME_DATA = {"punch": {"startup": 8, "active": 3, "hitstun": 18,
                        "knockback": [40, 0], "tolerance": 2}}


class FakeCoder:
    def chat(self, messages, **kw):
        self.prompt = messages[0]["content"]
        return "```gdscript\nextends RefCounted\n```"


def _fake_godot(tmp_path, script: str) -> str:
    p = tmp_path / "godot"
    p.write_text(f"#!/bin/sh\n{script}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_code_executor_materializes_table_grader_and_contract(tmp_path):
    coder = FakeCoder()
    execs = build_executors(load(), tmp_path, coder)
    task = {"id": "t1", "run_id": "r1", "spec": {"file": "scripts/combat_sim.gd",
                                 "description": "combat", "frame_data": FRAME_DATA}}
    produced = execs["code"](task, tmp_path / "out")
    game = tmp_path / "game"
    assert json.loads((game / "frame_data.json").read_text()) == FRAME_DATA
    assert "FRAME_DATA_OK" in (game / "tests/frame_data_test.gd").read_text()
    assert "opponent_offset" in coder.prompt          # contract API reached the model
    assert '"startup": 8' in coder.prompt             # exact numbers reached the model
    assert {p.name for p in produced} >= {"combat_sim.gd", "frame_data.json",
                                          "frame_data_test.gd"}


def test_frame_data_validator_pass_and_fail(tmp_path):
    game = tmp_path / "game"
    (game / "tests").mkdir(parents=True)
    (game / "tests/frame_data_test.gd").write_text("x")
    code_file = game / "scripts/combat_sim.gd"
    code_file.parent.mkdir(parents=True)
    code_file.write_text("extends RefCounted")
    task = {"type": "code", "spec": {"frame_data": FRAME_DATA}}

    ok, detail = validate.validate(task, [code_file],
                                   _fake_godot(tmp_path, 'echo FRAME_DATA_OK'), game)
    assert ok and "passed" in detail

    ok, detail = validate.validate(task, [code_file],
                                   _fake_godot(tmp_path, 'echo "FAIL punch startup"; exit 1'),
                                   game)
    assert not ok and "FAIL punch startup" in detail

    # exit 0 without the sentinel must NOT pass (script crashed before grading)
    ok, _ = validate.validate(task, [code_file], _fake_godot(tmp_path, "true"), game)
    assert not ok


def test_missing_grader_fails_loudly(tmp_path):
    game = tmp_path / "game"
    code_file = game / "scripts/combat_sim.gd"
    code_file.parent.mkdir(parents=True)
    code_file.write_text("extends RefCounted")
    task = {"type": "code", "spec": {"frame_data": FRAME_DATA}}
    ok, detail = validate.validate(task, [code_file],
                                   _fake_godot(tmp_path, 'echo FRAME_DATA_OK'), game)
    assert not ok and "missing" in detail
