"""Watched-folder auto-start.

Drop `mygame.md` into the inbox and a run starts on its own. Any sibling image
with the same stem (`mygame.png`, `mygame_ref.jpg`, ...) is picked up as a
reference. Files are claimed by an atomic rename into `inbox/started/`, so a
crash mid-claim can't start the same game twice, and a startup reconcile
recovers the one-in-a-million file that was moved but whose run never got
created.

The actual execution goes through the same `execute_run` + `_run_lock` path as
the GUI, so watched runs, GUI runs and resumes all serialize onto the one GPU.
"""
import threading
import time
from pathlib import Path

from . import db
from .orchestrate import execute_run, start_run


def _dirs(cfg: dict) -> tuple[Path, Path]:
    inbox = Path(cfg["watch"]["dir"])
    started = inbox / "started"
    started.mkdir(parents=True, exist_ok=True)
    return inbox, started


def _refs_for(md: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in md.parent.glob(md.stem + "*")
            if p.suffix.lower() in exts and p.is_file()]


def _claim_and_start(cfg: dict, conn, md: Path) -> str | None:
    """Atomically claim one instruction file and start its run. Returns run id."""
    _, started = _dirs(cfg)
    refs = _refs_for(md)
    dest = started / f"{int(time.time()*1000)}-{md.name}"
    try:
        md.rename(dest)  # atomic on same filesystem; the claim
    except (FileNotFoundError, OSError):
        return None      # another poll/host already took it
    ref_dests = []
    for r in refs:
        rd = dest.with_name(dest.stem + "__" + r.name)
        try:
            r.rename(rd)
            ref_dests.append(str(rd))
        except OSError:
            pass
    run_id = start_run(cfg, conn, dest, ref_dests)
    print(f"[watch] started run {run_id} from {md.name}", flush=True)
    _execute(cfg, run_id)
    return run_id


def _execute(cfg: dict, run_id: str) -> None:
    # own connection + the shared GPU lock (imported lazily to avoid a cycle)
    from .gui import _run_lock
    c = db.connect(cfg["paths"]["db"])
    with _run_lock:
        try:
            execute_run(cfg, c, run_id)
        except Exception as e:
            print(f"[watch] run {run_id} failed: {e}", flush=True)


def reconcile(cfg: dict, conn) -> int:
    """Recover files claimed into started/ whose run was never created (crash between
    rename and create_run). A started file is 'referenced' iff some run points at it."""
    _, started = _dirs(cfg)
    referenced = {Path(r["instruction_path"]).resolve()
                  for r in conn.execute("SELECT instruction_path FROM runs")}
    recovered = 0
    for f in started.iterdir():
        if not f.is_file() or f.suffix.lower() != ".md":
            continue
        if f.resolve() in referenced:
            continue
        run_id = start_run(cfg, conn, f, [])
        print(f"[watch] reconciled orphaned {f.name} -> run {run_id}", flush=True)
        _execute(cfg, run_id)
        recovered += 1
    return recovered


def poll_once(cfg: dict, conn) -> list[str]:
    """Start every unclaimed *.md currently in the inbox. Returns run ids."""
    inbox, _ = _dirs(cfg)
    started = []
    for md in sorted(inbox.glob("*.md")):
        run_id = _claim_and_start(cfg, conn, md)
        if run_id:
            started.append(run_id)
    return started


def watch_loop(cfg: dict, stop: threading.Event | None = None) -> None:
    """Poll the inbox forever (or until `stop`). Reconciles once at startup."""
    conn = db.connect(cfg["paths"]["db"])
    reconcile(cfg, conn)
    interval = cfg["watch"]["poll_interval_s"]
    inbox, _ = _dirs(cfg)
    print(f"[watch] watching {inbox.resolve()} every {interval}s", flush=True)
    while not (stop and stop.is_set()):
        try:
            poll_once(cfg, conn)
        except Exception as e:
            print(f"[watch] poll error: {e}", flush=True)  # a bad file must not kill the loop
        if stop:
            stop.wait(interval)
        else:
            time.sleep(interval)
