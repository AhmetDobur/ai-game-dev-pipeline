"""Entry point: python run.py gui | run <instruction.md> [--ref img ...] | status [run_id]"""
import argparse
import sys
from pathlib import Path

# under pythonw.exe (autostart scheduled task) there is no console: sys.stdout is
# None and the first print() kills the process. Redirect to a log file instead.
if sys.stdout is None or sys.stderr is None:
    _dir = Path(__file__).parent / "workspace"
    _dir.mkdir(parents=True, exist_ok=True)
    _log = open(_dir / "pipeline.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

from pipeline import __version__, config, db


def main():
    ap = argparse.ArgumentParser(prog="ai-game-dev-pipeline")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gui", help="serve the upload/status GUI")

    p_run = sub.add_parser("run", help="run the pipeline on an instruction file")
    p_run.add_argument("instruction", type=Path)
    p_run.add_argument("--ref", action="append", default=[], help="reference image path")

    p_patch = sub.add_parser("patch", help="apply a delta instruction to a finished game (new revision)")
    p_patch.add_argument("parent_run_id")
    p_patch.add_argument("instruction", type=Path, help="delta instruction .md")
    p_patch.add_argument("--ref", action="append", default=[], help="reference image path")

    p_status = sub.add_parser("status", help="show runs / tasks / ETA")
    p_status.add_argument("run_id", nargs="?")

    sub.add_parser("resume", help="continue every run interrupted by kill/shutdown/power cut")

    sub.add_parser("watch", help="watch the inbox folder and auto-start dropped instruction.md files")

    args = ap.parse_args()
    cfg = config.load()
    conn = db.connect(cfg["paths"]["db"])

    if args.cmd == "gui":
        import uvicorn
        uvicorn.run("pipeline.gui:app", host=cfg["gui"]["host"], port=cfg["gui"]["port"])
    elif args.cmd == "run":
        from pipeline import livelog
        from pipeline.orchestrate import execute_run, start_run
        livelog.tee_stdout = True   # stream the model's output to this terminal
        run_id = start_run(cfg, conn, args.instruction, args.ref)
        print(f"run {run_id}")
        execute_run(cfg, conn, run_id)
        print(f"run {run_id}: {db.get_run(conn, run_id)['status']}")
    elif args.cmd == "patch":
        from pipeline import livelog
        from pipeline.orchestrate import execute_run
        from pipeline.patch import start_patch
        livelog.tee_stdout = True
        run_id = start_patch(cfg, conn, args.parent_run_id, args.instruction, args.ref)
        print(f"patch run {run_id} (revision of {args.parent_run_id})")
        execute_run(cfg, conn, run_id)
        print(f"patch {run_id}: {db.get_run(conn, run_id)['status']}")
    elif args.cmd == "resume":
        from pipeline.orchestrate import resume_incomplete_runs
        resumed = resume_incomplete_runs(cfg, conn)
        print(f"resumed {len(resumed)} run(s)" if resumed else "nothing to resume")
    elif args.cmd == "watch":
        from pipeline.watch import watch_loop
        watch_loop(cfg)
    elif args.cmd == "status":
        from pipeline import eta
        if args.run_id:
            for t in db.list_tasks(conn, args.run_id):
                print(f"{t['id']}  {t['type']:12} {t['status']:12} "
                      f"attempts={t['attempts']} {t['error'][:60]}")
            print(eta.line(conn, args.run_id, cfg["scheduler"]["wave_order"]))
        else:
            for r in conn.execute("SELECT * FROM runs ORDER BY created_at"):
                print(f"{r['id']}  {r['status']:12} {r['instruction_path']}")
                if r["status"] in ("pending", "in_progress"):
                    print("  " + eta.line(conn, r["id"], cfg["scheduler"]["wave_order"]))


if __name__ == "__main__":
    main()
