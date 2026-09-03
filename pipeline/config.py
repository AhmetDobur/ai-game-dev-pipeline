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
        "ctx_size": 32768,                   # coder context
        "router_ctx_size": 131072,           # manager context: fits a big instruction.md
                                             # (~0.5MB / ~128k tokens, DeepSeek-R1-7B's real max)
        # q8 KV cache: 32k ctx costs 8.6GB at fp16 but 4.3GB at q8 — the 32B coder
        # (19.5GB weights) only fits 24GB VRAM with this on
        "coder_extra_args": ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        # same for the router: 28 layers x 4 KV heads x 128 dim = 56 KiB/token at
        # f16, so its 131072-token context alone is 7.00 GiB of KV. q8_0 halves
        # that to 3.72 GiB, which is what lets the router load on the GPU while
        # ComfyUI still holds a model
        "router_extra_args": ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "temperature": 0.6,
        "max_tokens": 4096,
        "load_timeout_s": 300,               # big GGUF from HDD can be slow
        "request_timeout_s": 1200,
    },
    "comfy": {
        "url": "http://127.0.0.1:8188",
        "sdxl_workflow": "workflows/sdxl.json",
        "sdxl_img2img_workflow": "workflows/sdxl_img2img.json",
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
