from pathlib import Path

from pipeline import db
from pipeline.scheduler import Scheduler


def _png(out_dir: Path, name="a.png", size=30_000) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_bytes(b"\x89PNG" + b"0" * size)
    return [p]


def test_wave_groups_and_setup_teardown_once_per_wave(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    for i in range(3):
        db.add_task(conn, run, "design_2d", {"prompt": f"p{i}"})
    events = []
    sched = Scheduler(
        conn, run,
        executors={"design_2d": lambda t, d: (events.append("exec"), _png(d))[1]},
        workspace=tmp_path, wave_order=["sdxl"],
        wave_setup={"sdxl": lambda: events.append("setup")},
        wave_teardown={"sdxl": lambda: events.append("teardown")},
    )
    sched.run()
    # one model load serves all three tasks: setup, 3x exec, teardown
    assert events == ["setup", "exec", "exec", "exec", "teardown"]
    assert all(t["status"] == "done" for t in db.list_tasks(conn, run))


def test_in_wave_retry_passes_last_error_and_succeeds(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    db.add_task(conn, run, "design_2d", {"prompt": "p"})
    seen_errors = []

    def flaky(task, out_dir):
        seen_errors.append(task["last_error"])
        if len(seen_errors) < 2:
            return _png(out_dir, size=10)  # too small -> validation fails
        return _png(out_dir)

    sched = Scheduler(conn, run, executors={"design_2d": flaky},
                      workspace=tmp_path, wave_order=["sdxl"], max_attempts=3)
    sched.run()
    t = db.list_tasks(conn, run)[0]
    assert t["status"] == "done"
    assert seen_errors[0] == "" and "too small" in seen_errors[1]


def test_exhausted_retries_fail_task_and_run(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    db.add_task(conn, run, "design_2d", {"prompt": "p"})

    def broken(task, out_dir):
        raise RuntimeError("gpu on fire")

    sched = Scheduler(conn, run, executors={"design_2d": broken},
                      workspace=tmp_path, wave_order=["sdxl"], max_attempts=2)
    sched.run()
    t = db.list_tasks(conn, run)[0]
    assert t["status"] == "failed" and t["attempts"] == 2
    assert "gpu on fire" in t["error"]
    assert db.get_run(conn, run)["status"] == "failed"


def test_unservable_task_type_fails_instead_of_spinning(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    db.add_task(conn, run, "audio", {"text": "hi"})  # tts wave not in wave_order
    sched = Scheduler(conn, run, executors={}, workspace=tmp_path,
                      wave_order=["sdxl"])
    sched.run()  # must terminate
    t = db.list_tasks(conn, run)[0]
    assert t["status"] == "failed" and "no wave serves" in t["error"]


def test_dep_outputs_flow_to_executor(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    a = db.add_task(conn, run, "design_2d", {"prompt": "concept"})
    db.add_task(conn, run, "design_2d", {"prompt": "x", "concept_from": a},
                depends_on=[a])
    seen = {}

    def executor(task, out_dir):
        seen[task["id"]] = task["dep_outputs"]
        return _png(out_dir)

    Scheduler(conn, run, executors={"design_2d": executor},
              workspace=tmp_path, wave_order=["sdxl"]).run()
    dependent = [v for k, v in seen.items() if v][0]
    assert a in dependent and dependent[a][0].endswith("a.png")


def test_same_wave_dependents_run_without_reload(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    run = db.create_run(conn, "i.md")
    a = db.add_task(conn, run, "design_2d", {"prompt": "a"})
    db.add_task(conn, run, "design_2d", {"prompt": "b"}, depends_on=[a])
    loads = []
    sched = Scheduler(conn, run, executors={"design_2d": lambda t, d: _png(d)},
                      workspace=tmp_path, wave_order=["sdxl"],
                      wave_setup={"sdxl": lambda: loads.append(1)})
    sched.run()
    assert len(loads) == 1, "dependent task in the same wave must not cost a reload"
    assert all(t["status"] == "done" for t in db.list_tasks(conn, run))
