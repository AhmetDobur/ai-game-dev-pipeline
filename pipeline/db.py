"""SQLite task queue. stdlib only, WAL mode.

All access goes through a module-level lock: GUI thread, scheduler thread and
rig_animate lane threads may share one connection, and sqlite3's commit path is
not atomic across threads (empirically raced during review).
"""
# ponytail: one global lock serializes all DB access; per-connection locks if throughput matters
import functools
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

TASK_TYPES = ("code", "design_2d", "design_3d", "rig_animate", "audio", "assemble")

_LOCK = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return wrapper

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    instruction_path TEXT NOT NULL,
    reference_images TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    type TEXT NOT NULL,
    spec TEXT NOT NULL,
    depends_on TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    output_path TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_run_status ON tasks(run_id, status);
CREATE TABLE IF NOT EXISTS durations (
    kind TEXT NOT NULL,
    seconds REAL NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_durations_kind ON durations(kind, ts);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")  # survive power loss, throughput is irrelevant here
    conn.executescript(_SCHEMA)
    return conn


@_locked
def create_run(conn, instruction_path: str, reference_images: list[str] | None = None) -> str:
    run_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO runs (id, instruction_path, reference_images, created_at) VALUES (?,?,?,?)",
        (run_id, instruction_path, json.dumps(reference_images or []), time.time()),
    )
    conn.commit()
    return run_id


@_locked
def set_run_status(conn, run_id: str, status: str, error: str = "") -> None:
    conn.execute("UPDATE runs SET status=?, error=? WHERE id=?", (status, error, run_id))
    conn.commit()


@_locked
def get_run(conn, run_id: str):
    return conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


@_locked
def add_task(conn, run_id: str, type_: str, spec: dict,
             depends_on: list[str] | None = None, task_id: str | None = None) -> str:
    if type_ not in TASK_TYPES:
        raise ValueError(f"unknown task type {type_!r}, expected one of {TASK_TYPES}")
    task_id = task_id or uuid.uuid4().hex[:12]
    now = time.time()
    conn.execute(
        "INSERT INTO tasks (id, run_id, type, spec, depends_on, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (task_id, run_id, type_, json.dumps(spec), json.dumps(depends_on or []), now, now),
    )
    conn.commit()
    return task_id


@_locked
def update_task(conn, task_id: str, **fields) -> None:
    allowed = {"status", "attempts", "output_path", "error", "spec"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"cannot update fields {bad}")
    if "spec" in fields and isinstance(fields["spec"], dict):
        fields["spec"] = json.dumps(fields["spec"])
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(
        f"UPDATE tasks SET {sets}, updated_at=? WHERE id=?",
        (*fields.values(), time.time(), task_id),
    )
    conn.commit()


def _row_to_task(row) -> dict:
    t = dict(row)
    t["spec"] = json.loads(t["spec"])
    t["depends_on"] = json.loads(t["depends_on"])
    return t


@_locked
def get_task(conn, task_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return _row_to_task(row) if row else None


@_locked
def list_tasks(conn, run_id: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM tasks", ()
    if run_id:
        q, args = q + " WHERE run_id=?", (run_id,)
    return [_row_to_task(r) for r in conn.execute(q + " ORDER BY created_at", args)]


@_locked
def ready_tasks(conn, run_id: str) -> list[dict]:
    """Pending tasks whose dependencies are all done."""
    tasks = list_tasks(conn, run_id)
    done = {t["id"] for t in tasks if t["status"] == "done"}
    return [t for t in tasks
            if t["status"] == "pending" and all(d in done for d in t["depends_on"])]


@_locked
def run_finished(conn, run_id: str) -> bool:
    """True when no task can make further progress (all done, or blocked by failures)."""
    tasks = list_tasks(conn, run_id)
    if not tasks:
        return False
    if any(t["status"] == "in_progress" for t in tasks):
        return False
    if ready_tasks(conn, run_id):
        return False
    # remaining pending tasks are blocked by failed deps forever
    return True


@_locked
def add_tasks(conn, run_id: str, rows: list[tuple]) -> None:
    """Atomic bulk insert: (task_id, type, spec_dict, depends_on_list) rows.
    All-or-nothing so a crash mid-decompose never leaves a half-inserted run."""
    for _, type_, _, _ in rows:
        if type_ not in TASK_TYPES:
            raise ValueError(f"unknown task type {type_!r}, expected one of {TASK_TYPES}")
    now = time.time()
    with conn:  # one transaction
        conn.executemany(
            "INSERT INTO tasks (id, run_id, type, spec, depends_on, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            [(tid, run_id, type_, json.dumps(spec), json.dumps(deps), now, now)
             for tid, type_, spec, deps in rows])


@_locked
def reclaim_stale(conn, run_id: str | None = None) -> int:
    """Reset in_progress tasks to pending after a crash/kill. Attempts are kept,
    so a task that keeps killing the process still exhausts max_attempts."""
    q = "UPDATE tasks SET status='pending', updated_at=? WHERE status='in_progress'"
    args: tuple = (time.time(),)
    if run_id:
        q += " AND run_id=?"
        args += (run_id,)
    cur = conn.execute(q, args)
    conn.commit()
    return cur.rowcount


@_locked
def incomplete_runs(conn) -> list[str]:
    return [r["id"] for r in conn.execute(
        "SELECT id FROM runs WHERE status IN ('pending','in_progress') ORDER BY created_at")]


@_locked
def record_duration(conn, kind: str, seconds: float) -> None:
    conn.execute("INSERT INTO durations (kind, seconds, ts) VALUES (?,?,?)",
                 (kind, seconds, time.time()))
    conn.commit()


@_locked
def duration_stats(conn, kind: str, window: int = 50) -> dict | None:
    """Median and p90 of the most recent samples for this kind, or None."""
    rows = [r["seconds"] for r in conn.execute(
        "SELECT seconds FROM durations WHERE kind=? ORDER BY ts DESC LIMIT ?",
        (kind, window))]
    if not rows:
        return None
    rows.sort()
    def q(p):
        i = min(len(rows) - 1, max(0, round(p * (len(rows) - 1))))
        return rows[i]
    return {"n": len(rows), "p50": q(0.5), "p90": q(0.9)}
