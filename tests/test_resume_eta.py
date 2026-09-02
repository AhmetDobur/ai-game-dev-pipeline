"""Crash-safe resume + ETA engine."""
import time

from pipeline import db, eta
from pipeline.scheduler import Scheduler


def _png(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "a.png"
    p.write_bytes(b"\x89PNG" + b"0" * 30000)
    return [p]


# --- resume ---------------------------------------------------------------

def test_reclaim_resets_status_but_keeps_attempts(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    tid = db.add_task(conn, run, "design_2d", {"prompt": "x"})
    db.update_task(conn, tid, status="in_progress", attempts=2)
    assert db.reclaim_stale(conn, run) == 1
    t = db.get_task(conn, tid)
    assert t["status"] == "pending" and t["attempts"] == 2


def test_scheduler_resumes_and_respects_persisted_attempts(tmp_path):
    """A task killed mid-flight twice gets exactly its remaining attempts, not a
    fresh max_attempts budget."""
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    tid = db.add_task(conn, run, "design_2d", {"prompt": "x"})
    db.update_task(conn, tid, status="in_progress", attempts=2)  # simulated crash
    calls = []

    def executor(task, out_dir):
        calls.append(1)
        raise RuntimeError("still failing")

    Scheduler(conn, run, executors={"design_2d": executor}, workspace=tmp_path,
              max_attempts=3, wave_order=["sdxl"]).run()
    assert len(calls) == 1  # one remaining attempt, not three
    assert db.get_task(conn, tid)["status"] == "failed"


def test_exhausted_task_fails_immediately_without_executor_call(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    tid = db.add_task(conn, run, "design_2d", {"prompt": "x"})
    db.update_task(conn, tid, status="in_progress", attempts=3)
    calls = []
    Scheduler(conn, run, executors={"design_2d": lambda t, o: calls.append(1)},
              workspace=tmp_path, max_attempts=3, wave_order=["sdxl"]).run()
    assert calls == []
    assert db.get_task(conn, tid)["status"] == "failed"


def test_atomic_bulk_insert_all_or_nothing(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    rows = [("a", "design_2d", {"p": 1}, []),
            ("b", "not_a_type", {}, [])]
    try:
        db.add_tasks(conn, run, rows)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert db.list_tasks(conn, run) == []


def test_incomplete_runs_lists_only_unfinished(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    r1 = db.create_run(conn, "a.md")
    r2 = db.create_run(conn, "b.md")
    r3 = db.create_run(conn, "c.md")
    db.set_run_status(conn, r1, "done")
    db.set_run_status(conn, r2, "in_progress")
    assert db.incomplete_runs(conn) == [r2, r3]


# --- ETA ------------------------------------------------------------------

def test_eta_uses_history_and_wave_breakdown(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    db.add_task(conn, run, "design_2d", {"prompt": "a"})
    db.add_task(conn, run, "design_2d", {"prompt": "b"})
    for _ in range(5):
        db.record_duration(conn, "design_2d", 100.0)
        db.record_duration(conn, "load:sdxl", 20.0)
    e = eta.estimate(conn, run, ["sdxl"])
    assert e["seconds_p50"] == 220  # 20 load + 2 * 100
    assert e["seconds_p90"] >= e["seconds_p50"]
    assert e["confidence"] == "history"
    assert e["breakdown"][0]["wave"] == "sdxl" and e["breakdown"][0]["tasks"] == 2


def test_eta_defaults_when_no_history_and_done_excluded(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    t1 = db.add_task(conn, run, "audio", {"text": "x"})
    db.add_task(conn, run, "audio", {"text": "y"})
    db.update_task(conn, t1, status="done")
    e = eta.estimate(conn, run, ["tts"])
    assert e["confidence"] == "defaults"
    assert e["remaining_tasks"] == 1 and e["done_tasks"] == 1
    assert e["seconds_p50"] == eta.DEFAULT_TASK_S["audio"] + eta.DEFAULT_LOAD_S["tts"]


def test_eta_motion_wave_is_serial(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    db.add_task(conn, run, "rig_animate", {"mesh_from": "m"})  # motion wave: default 600s
    db.add_task(conn, run, "audio", {"text": "x"})             # tts wave: 30 + 0 load
    e = eta.estimate(conn, run, ["motion", "tts"])
    # one GPU -> everything serial: motion (0 load + 600) + tts (0 load + 30)
    assert e["seconds_p50"] == 630


def test_in_progress_task_gets_elapsed_credit(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    tid = db.add_task(conn, run, "audio", {"text": "x"})
    for _ in range(5):
        db.record_duration(conn, "audio", 100.0)
        db.record_duration(conn, "load:tts", 0.0)
    db.update_task(conn, tid, status="in_progress")
    conn.execute("UPDATE tasks SET updated_at=? WHERE id=?", (time.time() - 60, tid))
    conn.commit()
    e = eta.estimate(conn, run, ["tts"])
    assert 30 <= e["seconds_p50"] <= 50  # ~100 - 60 elapsed


def test_scheduler_records_durations(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    db.add_task(conn, run, "design_2d", {"prompt": "x"})
    Scheduler(conn, run, executors={"design_2d": lambda t, o: _png(o)},
              workspace=tmp_path, wave_order=["sdxl"],
              wave_setup={"sdxl": lambda: None}).run()
    assert db.duration_stats(conn, "design_2d")["n"] == 1
    assert db.duration_stats(conn, "load:sdxl")["n"] == 1
