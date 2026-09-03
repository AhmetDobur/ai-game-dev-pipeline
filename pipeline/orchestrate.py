"""Wire everything together for one run: decompose -> wave-schedule -> report.
This is what both the CLI and the GUI call."""
from pathlib import Path

from . import db
from .adapters.comfy import ComfyClient
from .adapters.llm import LlamaServer
from .decompose import decompose, decompose_patch, insert_tasks
from .executors import build_executors
from .patch import build_patch_graph, manifest, prepare_workspace
from .scheduler import Scheduler


def start_run(cfg: dict, conn, instruction_path: Path,
              reference_images: list[str] | None = None) -> str:
    run_id = db.create_run(conn, str(instruction_path), reference_images or [])
    return run_id


def _plan_patch(cfg, conn, run_id, parent_id, instruction, refs, router, livelog) -> None:
    """Snapshot the parent revision, decompose the delta, materialize the patch
    graph atomically. Selective re-run then falls out of the normal scheduler:
    only the stale tasks are pending, the rest carry their parent's outputs."""
    livelog.start(run_id, f"planning patch of {parent_id}")
    prepare_workspace(cfg, run_id, parent_id)
    parent_rows = db.list_tasks(conn, parent_id)
    delta = decompose_patch(router, manifest(parent_rows), instruction, refs,
                            cfg["llm"]["temperature"], cfg["llm"]["max_tokens"],
                            on_token=livelog.token_sink(run_id))
    rows, stale = build_patch_graph(parent_rows, delta, parent_id, run_id)
    db.add_tasks_full(conn, run_id, rows)
    import json
    print(json.dumps({"patch": run_id, "of": parent_id,
                      "tasks": len(rows), "re_running": len(stale)}))


def execute_run(cfg: dict, conn, run_id: str) -> None:
    router = coder = None
    try:
        run = db.get_run(conn, run_id)
        # mark active before planning, not after — the GUI's live view selects
        # the in_progress run, and planning is exactly what it should show
        db.set_run_status(conn, run_id, "in_progress")
        workspace = Path(cfg["paths"]["workspace"]) / "runs" / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        instruction = Path(run["instruction_path"]).read_text(encoding="utf-8")

        router = LlamaServer(cfg["paths"]["llama_server"], cfg["llm"]["router_gguf"],
                             cfg["llm"]["router_port"], cfg["llm"]["router_ctx_size"],
                             cfg["llm"]["load_timeout_s"],
                             extra_args=cfg["llm"]["router_extra_args"])
        coder = LlamaServer(cfg["paths"]["llama_server"], cfg["llm"]["coder_gguf"],
                            cfg["llm"]["coder_port"], cfg["llm"]["ctx_size"],
                            cfg["llm"]["load_timeout_s"],
                            extra_args=cfg["llm"]["coder_extra_args"])
        comfy = ComfyClient(cfg["comfy"]["url"], cfg["comfy"]["timeout_s"])

        # resume support: a run that already has tasks was interrupted mid-flight —
        # skip planning and let the scheduler reclaim + continue. Both the fresh
        # (insert_tasks -> db.add_tasks) and patch (db.add_tasks_full) inserts are
        # single transactions, so "has tasks" means "has ALL tasks".
        import json
        from . import livelog
        if not db.list_tasks(conn, run_id):
            # the router is only needed for planning; stop it right after so its
            # VRAM is free for the waves — 7B router + 32B coder + KV cache
            # does not fit 24GB together
            router.start()
            refs = json.loads(run["reference_images"])
            if run["parent_id"]:
                _plan_patch(cfg, conn, run_id, run["parent_id"], instruction, refs,
                            router, livelog)
            else:
                livelog.start(run_id, "planning tasks from instruction.md")
                tasks = decompose(router, instruction, refs,
                                  cfg["llm"]["temperature"], cfg["llm"]["max_tokens"],
                                  on_token=livelog.token_sink(run_id))
                insert_tasks(conn, run_id, tasks)
            router.stop()

        scheduler = Scheduler(
            conn, run_id,
            executors=build_executors(cfg, workspace, coder),
            workspace=workspace,
            max_attempts=cfg["scheduler"]["max_attempts"],
            wave_order=cfg["scheduler"]["wave_order"],
            # the wave policy in one place: coder LLM loads/unloads around its wave,
            # ComfyUI frees VRAM after image/mesh waves, TTS is externally managed
            wave_setup={"coder": coder.start},
            wave_teardown={"coder": coder.stop,
                           "sdxl": comfy.free_vram,
                           "trellis": comfy.free_vram},
            godot_binary=cfg["paths"]["godot"],
        )
        scheduler.run()
    except Exception as e:
        db.set_run_status(conn, run_id, "failed", f"{type(e).__name__}: {e}")
        raise
    finally:
        if coder:
            coder.stop()
        if router:
            router.stop()


def resume_incomplete_runs(cfg: dict, conn) -> list[str]:
    """Continue every run interrupted by a kill/shutdown/power cut. Called by the
    GUI on startup and by `python run.py resume`. Returns the resumed run ids."""
    resumed = []
    for run_id in db.incomplete_runs(conn):
        print(f"[resume] continuing run {run_id}")
        try:
            execute_run(cfg, conn, run_id)
        except Exception as e:
            print(f"[resume] run {run_id} failed: {e}")
        resumed.append(run_id)
    return resumed
