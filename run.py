"""Entry point: python run.py gui | run <instruction.md> [--ref img ...] | status [run_id]"""
import argparse
from pathlib import Path

from pipeline import __version__, config, db


def main():
    ap = argparse.ArgumentParser(prog="ai-game-dev-pipeline")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gui", help="serve the upload/status GUI")

    p_run = sub.add_parser("run", help="run the pipeline on an instruction file")
    p_run.add_argument("instruction", type=Path)
    p_run.add_argument("--ref", action="append", default=[], help="reference image path")

    p_status = sub.add_parser("status", help="show runs / tasks / ETA")
    p_status.add_argument("run_id", nargs="?")

    sub.add_parser("resume", help="continue every run interrupted by kill/shutdown/power cut")

    args = ap.parse_args()
    cfg = config.load()
    conn = db.connect(cfg["paths"]["db"])

    if args.cmd == "gui":
        import uvicorn
        uvicorn.run("pipeline.gui:app", host=cfg["gui"]["host"], port=cfg["gui"]["port"])
    elif args.cmd == "run":
        from pipeline.orchestrate import execute_run, start_run
        run_id = start_run(cfg, conn, args.instruction, args.ref)
        print(f"run {run_id}")
        execute_run(cfg, conn, run_id)
        print(f"run {run_id}: {db.get_run(conn, run_id)['status']}")
    elif args.cmd == "resume":
        from pipeline.orchestrate import resume_incomplete_runs
        resumed = resume_incomplete_runs(cfg, conn)
        print(f"resumed {len(resumed)} run(s)" if resumed else "nothing to resume")
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
