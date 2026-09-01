"""Wire everything together for one run: decompose -> wave-schedule -> report.
This is what both the CLI and the GUI call."""
from pathlib import Path

from . import db
from .adapters.comfy import ComfyClient
from .adapters.llm import LlamaServer
from .decompose import decompose, insert_tasks
from .executors import build_executors
from .scheduler import Scheduler


def start_run(cfg: dict, conn, instruction_path: Path,
              reference_images: list[str] | None = None) -> str:
    run_id = db.create_run(conn, str(instruction_path), reference_images or [])
    return run_id


def execute_run(cfg: dict, conn, run_id: str) -> None:
    router = coder = None
    try:
        run = db.get_run(conn, run_id)
        workspace = Path(cfg["paths"]["workspace"]) / "runs" / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        instruction = Path(run["instruction_path"]).read_text(encoding="utf-8")

        router = LlamaServer(cfg["paths"]["llama_server"], cfg["llm"]["router_gguf"],
                             cfg["llm"]["router_port"], cfg["llm"]["ctx_size"],
                             cfg["llm"]["load_timeout_s"])
        coder = LlamaServer(cfg["paths"]["llama_server"], cfg["llm"]["coder_gguf"],
                            cfg["llm"]["coder_port"], cfg["llm"]["ctx_size"],
                            cfg["llm"]["load_timeout_s"])
        comfy = ComfyClient(cfg["comfy"]["url"], cfg["comfy"]["timeout_s"])

        router.start()  # resident for the whole run
        import json
        tasks = decompose(router, instruction, json.loads(run["reference_images"]),
                          cfg["llm"]["temperature"], cfg["llm"]["max_tokens"])
        insert_tasks(conn, run_id, tasks)

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
