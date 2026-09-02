"""Load pipeline.toml. Leaf module: imports nothing from the package."""
import os
import tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(os.environ.get("PIPELINE_CONFIG", "pipeline.toml"))

DEFAULTS: dict = {
    "paths": {
        "workspace": "workspace",           # runs, artifacts, logs live here
        "db": "workspace/pipeline.db",
        "godot": "godot",                    # godot 4.x binary on PATH or absolute
        "llama_server": "llama-server",      # llama.cpp server binary
    },
    "llm": {
        "router_gguf": "",                   # DeepSeek-R1-Distill-Qwen-7B Q4_K_M
        "router_port": 8090,
        "coder_gguf": "",                    # dense Qwen2.5-Coder-32B-Instruct Q4_K_M
        "coder_port": 8091,
        "ctx_size": 32768,
        "temperature": 0.6,
        "max_tokens": 4096,
        "load_timeout_s": 300,               # big GGUF from HDD can be slow
        "request_timeout_s": 1200,
    },
    "comfy": {
        "url": "http://127.0.0.1:8188",
        "sdxl_workflow": "workflows/sdxl.json",
        "trellis_workflow": "workflows/trellis.json",
        "timeout_s": 600,
    },
    "tts": {
        "url": "http://127.0.0.1:5005",      # Orpheus-FastAPI
        "timeout_s": 300,
    },
    "motion": {                              # local rig+animate — Blender headless, no cloud
        "blender": "blender",                # Blender 4.x binary on PATH or absolute
        "script": "templates/blender_motion.py",
        "cmu_dir": "",                       # CMU BVH mocap library (optional; exact-match clips)
        "unirig": "",                        # UniRig checkpoint dir (optional auto-rigger)
        "kimodo_url": "",                    # Kimodo local endpoint for text->motion (optional)
        "timeout_s": 1800,
    },
    "scheduler": {
        "max_attempts": 3,                   # in-wave retries per task
        "wave_order": ["coder", "sdxl", "trellis", "motion", "tts"],
    },
    "watch": {
        "dir": "inbox",                      # drop instruction.md files here to auto-start
        "poll_interval_s": 10,
    },
    "gui": {
        "host": "127.0.0.1",
        "port": 8500,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load(path: Path | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, "rb") as f:
            return _merge(DEFAULTS, tomllib.load(f))
    return _merge(DEFAULTS, {})
