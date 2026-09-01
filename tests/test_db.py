from pipeline import db


def make_conn(tmp_path):
    return db.connect(tmp_path / "t.db")


def test_ready_respects_dependencies(tmp_path):
    conn = make_conn(tmp_path)
    run = db.create_run(conn, "i.md")
    a = db.add_task(conn, run, "design_2d", {"prompt": "x"})
    b = db.add_task(conn, run, "design_3d", {"prompt": "y"}, depends_on=[a])
    assert [t["id"] for t in db.ready_tasks(conn, run)] == [a]
    db.update_task(conn, a, status="done")
    assert [t["id"] for t in db.ready_tasks(conn, run)] == [b]


def test_run_finished_blocked_by_failure(tmp_path):
    conn = make_conn(tmp_path)
    run = db.create_run(conn, "i.md")
    a = db.add_task(conn, run, "code", {"file": "f.gd", "description": "d"})
    b = db.add_task(conn, run, "assemble", {}, depends_on=[a])
    assert not db.run_finished(conn, run)
    db.update_task(conn, a, status="failed", error="boom")
    # b can never run -> the run is finished (in a failed shape)
    assert db.run_finished(conn, run)
    assert db.get_task(conn, b)["status"] == "pending"


def test_unknown_task_type_rejected(tmp_path):
    conn = make_conn(tmp_path)
    run = db.create_run(conn, "i.md")
    try:
        db.add_task(conn, run, "nonsense", {})
        assert False, "expected ValueError"
    except ValueError:
        pass
