"""Watched-folder auto-start: claim-by-rename, sibling refs, dedup, reconcile."""
import json

from pipeline import db, watch


def _cfg(tmp_path):
    return {"paths": {"db": str(tmp_path / "t.db"),
                      "workspace": str(tmp_path / "ws")},
            "watch": {"dir": str(tmp_path / "inbox"), "poll_interval_s": 1}}


def _no_execute(monkeypatch):
    """Don't actually run the pipeline (no GPU here) — just record the claim."""
    monkeypatch.setattr(watch, "_execute", lambda cfg, run_id: None)


def test_dropped_md_starts_a_run_and_is_claimed(tmp_path, monkeypatch):
    _no_execute(monkeypatch)
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg["paths"]["db"])
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "game.md").write_text("make a game", encoding="utf-8")

    started = watch.poll_once(cfg, conn)
    assert len(started) == 1
    # the file is claimed out of the inbox (won't be re-processed)
    assert not (inbox / "game.md").exists()
    run = db.get_run(conn, started[0])
    assert run["status"] == "pending"
    assert (tmp_path / "inbox" / "started").exists()


def test_sibling_images_become_references(tmp_path, monkeypatch):
    _no_execute(monkeypatch)
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg["paths"]["db"])
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "hero.md").write_text("x", encoding="utf-8")
    (inbox / "hero.png").write_bytes(b"img")
    (inbox / "hero_ref2.jpg").write_bytes(b"img")
    (inbox / "unrelated.png").write_bytes(b"img")

    run_id = watch.poll_once(cfg, conn)[0]
    refs = json.loads(db.get_run(conn, run_id)["reference_images"])
    assert len(refs) == 2                       # hero.png + hero_ref2.jpg, not unrelated
    assert not (inbox / "hero.png").exists()    # refs claimed too


def test_second_poll_does_not_restart_same_file(tmp_path, monkeypatch):
    _no_execute(monkeypatch)
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg["paths"]["db"])
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "g.md").write_text("x", encoding="utf-8")
    assert len(watch.poll_once(cfg, conn)) == 1
    assert watch.poll_once(cfg, conn) == []     # nothing left to claim


def test_dropped_patch_md_starts_a_patch_run(tmp_path, monkeypatch):
    _no_execute(monkeypatch)
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg["paths"]["db"])
    parent = db.create_run(conn, "orig.md")  # a finished game to patch
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "buff.md").write_text(f"patch: {parent}\nmake the jump higher",
                                   encoding="utf-8")

    run_id = watch.poll_once(cfg, conn)[0]
    child = db.get_run(conn, run_id)
    assert child["parent_id"] == parent and child["revision"] == 2


def test_reconcile_recovers_orphaned_claim(tmp_path, monkeypatch):
    """File moved to started/ but its run was never created (crash mid-claim)."""
    _no_execute(monkeypatch)
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg["paths"]["db"])
    started = tmp_path / "inbox" / "started"
    started.mkdir(parents=True)
    orphan = started / "123-orphan.md"
    orphan.write_text("x", encoding="utf-8")

    assert watch.reconcile(cfg, conn) == 1
    runs = [dict(r) for r in conn.execute("SELECT * FROM runs")]
    assert len(runs) == 1 and runs[0]["instruction_path"] == str(orphan)
    # a referenced started file is left alone on the next reconcile
    assert watch.reconcile(cfg, conn) == 0
