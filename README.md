# AI Game Development Pipeline

A game-agnostic, single-GPU, agent-orchestrated pipeline that turns an uploaded
`instruction.md` (plus optional reference images) into a playable Godot 4 game
build — code, 2D art, 3D assets, rigging/animation, and voice audio all
generated, scheduled, and validated automatically.

The pipeline is the product. The first proof-of-concept input will be a small
original fighting-game demo, but nothing about the game is hardcoded: the game
arrives as an instruction file through the GUI, gets decomposed into a typed
task graph, and is executed in **model waves** sized to one 24GB GPU.

- **Version:** 0.4.0 (semver; git tags `vX.Y.Z`)
- **Target box:** Windows, Titan RTX 24GB, i5-14600K, 32GB RAM
- **Engine:** Godot 4.x (text-first `.tscn`/GDScript — everything the coder LLM
  writes is reviewable plain text; headless import/export)
- **Dev machine note:** the spine is pure HTTP + subprocess and runs/tests
  anywhere (developed on macOS); only real generation needs the CUDA box.

---

## Architecture

```
instruction.md ──▶ GUI (FastAPI, localhost:8500)
      + refs           │ creates run
                       ▼
              decompose.py ─── resident router LLM (7B, always loaded)
                       │  typed, dependency-aware task list
                       ▼
              SQLite queue (workspace/pipeline.db)
                       │
                       ▼
              scheduler.py — WAVE SCHEDULER
      ┌────────────────┼───────────────────────────────┐
      │ GPU waves (one model loaded at a time)         │ async lane (no VRAM)
      │  coder → sdxl → trellis → tts → (repeat)       │  rig_animate → Meshy API
      │  each wave drains ALL ready tasks of its type, │ local lane (CPU)
      │  validates IN-WAVE so retries reuse the        │  assemble → godot --headless
      │  already-loaded model                          │
      └────────────────┼───────────────────────────────┘
                       ▼
              validate.py (objective checks per branch)
                       ▼
              workspace/runs/<id>/game/ ──▶ godot --export-release ──▶ build
```

### The wave scheduler (the core optimization)

Only one large model fits in 24GB, and model loads cost minutes. The scheduler
therefore never walks the queue in plain dependency order. Instead:

1. **Waves**: tasks are grouped by the model that serves them
   (`code`→coder LLM, `design_2d`→SDXL, `design_3d`→TRELLIS, `audio`→TTS).
   A wave loads its model once (`wave_setup`), drains **every** ready task of
   that type — including tasks whose dependencies complete *during* the wave —
   then unloads (`wave_teardown`). Model loads per cycle drop from
   one-per-task to one-per-type.
2. **In-wave validation and retry**: every output is validated immediately,
   while the producing model is still resident. A failed artifact retries up
   to `max_attempts` times *inside the wave* (the executor receives
   `last_error` so the retry prompt says what was wrong) instead of waiting a
   full cycle and paying a reload.
3. **Async lane**: `rig_animate` is a network-bound Meshy API call — it runs
   in background threads *alongside* whatever GPU wave is active.
4. **Local lane**: `assemble` is a CPU-only `godot --headless` subprocess and
   runs between waves.
5. The **router LLM stays resident** the whole run (small footprint); the
   coder LLM starts/stops around its own wave; ComfyUI is asked to free VRAM
   (`POST /free`) after image/mesh waves.

### Task lifecycle

`pending → in_progress → done | failed`, with `attempts` counted per task.
A task whose dependency failed stays `pending` forever; the run finishes as
`failed` listing failed + blocked counts. Everything is inspectable in the
GUI table or `python run.py status <run_id>`.

### Task schema

Stored in SQLite (`tasks` table); produced by the router from your
instruction file:

```json
{
  "id": "meshA",
  "type": "design_3d",
  "depends_on": ["artA"],
  "spec": {"prompt": "...", "concept_from": "artA"}
}
```

| type | spec fields | executed by | validated by |
|---|---|---|---|
| `code` | `file`, `description` | coder LLM → writes into `game/` | `godot --check-only` per `.gd` file |
| `design_2d` | `prompt`, `purpose` | ComfyUI SDXL workflow | file type + size floor |
| `design_3d` | `prompt`, `concept_from` | ComfyUI TRELLIS workflow | mesh file type + size floor |
| `rig_animate` | `mesh_from`, `animations[]` | Meshy REST API | FBX exists + size floor |
| `audio` | `text`, `voice` | Orpheus-FastAPI | WAV parses, duration ≥ 0.2s |
| `assemble` | `export_preset` | `godot --headless --export-release` | build artifact ≥ 1MB |

**Validation policy:** only objective signals reject an artifact (parse
failures, missing/too-small files, dead audio, failed exports). No LLM opinion
ever fails a task — an LLM asked to critique working output will always find
something, and acting on that rewrites good artifacts into bad ones.

### Frame-data contract (combat timing as spec, not vibes)

Fighting-game "feel" is quantifiable: startup/active/recovery frames, hitstun,
knockback vectors, hitstop. When the instruction.md contains a timing table,
the decomposer copies it **verbatim** into the `scripts/combat_sim.gd` code
task as `frame_data`:

```json
{"punch": {"startup": 8, "active": 3, "hitstun": 18,
           "knockback": [40, 0], "tolerance": 2}}
```

Frame counts are at 60fps fixed step. The code executor then:

1. writes the table to `frame_data.json` in the game project,
2. copies the pipeline's **own** grader (`templates/frame_data_test.gd` — a
   static GDScript, never written by the model) into `tests/`,
3. tells the coder model the `CombatSim` API contract it must implement:
   `setup(move)`, `press(move)`, `step()`, `hitbox_active()`,
   `opponent_in_hitstun()`, `opponent_offset()`.

Validation runs `godot --headless --script res://tests/frame_data_test.gd`,
which steps the simulation frame by frame and asserts every number in the
table. Wrong startup frame → hard test failure with the exact numbers → the
in-wave retry feeds it back to the coder. Timing is graded by arithmetic,
never by opinion. Tuning the game's feel afterwards means editing a JSON
table and re-running one wave — not reopening code.

### Repo layout

```
pipeline/
  config.py        pipeline.toml loader (leaf module, imports nothing)
  db.py            SQLite queue: runs, tasks, durations, reclaim, ready-set
  decompose.py     router prompt → validated task list → queue
  scheduler.py     wave scheduler + async/local lanes + in-wave retry
  executors.py     production executors (one per task type)
  orchestrate.py   wires config+adapters+scheduler for one run
  validate.py      objective per-branch validators
  eta.py           learned wave-aware ETA (p50–p90 band from run history)
  watch.py         inbox auto-start: claim-by-rename, sibling refs, reconcile
  gui.py           FastAPI page: upload, live task table, ETA, auto-resume + watch
  adapters/
    llm.py         llama-server lifecycle + chat + Qwen2.5 <tools> shim
    comfy.py       ComfyUI /prompt → poll /history → download outputs
    meshy.py       rig + animate + FBX download
    tts.py         Orpheus /v1/audio/speech
run.py             CLI: gui | run <instruction.md> [--ref img] | status
templates/         frame_data_test.gd — the pipeline's own headless timing grader
workflows/         ComfyUI API-format workflow JSONs ({{prompt}} placeholders)
pipeline.toml.example
requirements.txt   pinned exact versions (core: fastapi, uvicorn, requests, python-multipart)
tests/             spine tests — no GPU, no network, run anywhere
scripts/           live-server integration checks (need a running model)
```

---

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows
cp pipeline.toml.example pipeline.toml               # fill in your paths
set MESHY_API_KEY=msy_xxx                            # or setx for persistence
.venv/Scripts/python run.py gui                      # http://127.0.0.1:8500
```

Upload an `instruction.md` (and optional reference images) in the GUI, press
**Start run**, watch the task table drain. Or headless:

```bash
python run.py run my-game/instruction.md --ref my-game/style.png
python run.py status
```

Run tests (no GPU needed):

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

---

## Installing the tool stack (Windows)

Everything below is external to this repo — the spine only talks to these over
HTTP or subprocess. Install once, point `pipeline.toml` at them.

### 1. NVIDIA driver + CUDA

Recent Game Ready or Studio driver is enough for llama.cpp (cuBLAS builds
bundle what they need) and ComfyUI (PyTorch wheels ship CUDA runtime). Verify:
`nvidia-smi` shows the Titan RTX and 24576MiB.

### 2. llama.cpp (`llama-server`)

Download the latest **cudart** Windows release from
<https://github.com/ggml-org/llama.cpp/releases> (e.g.
`llama-bXXXX-bin-win-cuda-x64.zip`), unzip to `C:\llama.cpp`.
Set `paths.llama_server` in `pipeline.toml`. The pipeline starts/stops it
itself with the right flags (`--jinja`, `--flash-attn`, ports from config) —
do not run it manually during a pipeline run.

### 3. Models (GGUF)

| Role | Model | File | VRAM |
|---|---|---|---|
| Router (resident) | DeepSeek-R1-Distill-Qwen-7B | `Q4_K_M` GGUF | ~4GB |
| Coder (wave) | Qwen3-Coder-30B-A3B-Instruct | `Q4_K_M` GGUF | ~18GB |

Get them from Hugging Face (`bartowski` or `unsloth` GGUF repos). Put paths in
`pipeline.toml` under `[llm]`.

**Coder fallback — Qwen2.5-Coder-32B (dense):** speaks a non-standard
tool-call dialect. Measured 2026-09-01 on llama-server `--jinja`: 0/5 native
structured `tool_calls` (it ignores the hermes `<tool_call>` format its own
chat template declares and invents a `<tools>` wrapper), but 5/5 with the
`<tools>` few-shot system prompt + regex shim that ships in
`pipeline/adapters/llm.py` (`TOOLS_SHIM_SYSTEM`, `parse_tool_call`). Same
model scored 13/14 on a hard concurrency benchmark where the 80B MoE
deadlocked at 0/14 — usable, but only behind the shim. The 30B-A3B MoE needs
no shim and stays primary.

### 4. ComfyUI (SDXL + TRELLIS)

```bash
git clone https://github.com/comfyanonymous/ComfyUI C:\ComfyUI
cd C:\ComfyUI && python -m venv venv && venv\Scripts\pip install -r requirements.txt
```

- **SDXL**: download `sd_xl_base_1.0.safetensors` into
  `ComfyUI\models\checkpoints\`. The bundled `workflows/sdxl.json` works as-is
  (1024×1024, DPM++ 2M Karras, 30 steps). Style-lock via IP-Adapter/LoRA can
  be added to that JSON later without touching pipeline code.
- **TRELLIS**: the roughest install of the stack on native Windows (custom
  CUDA ops). Use a ComfyUI TRELLIS custom-node pack (install via ComfyUI
  Manager), build the image→3D graph in the ComfyUI editor, then
  **Workflow → Export (API)** and save it over `workflows/trellis.json`,
  putting `{{prompt}}` / `{{image}}` where the conditioning goes.
  `workflows/trellis.json` in this repo is a placeholder that fails loudly
  until you do this.
- Run ComfyUI before pipeline runs: `venv\Scripts\python main.py --listen 127.0.0.1`
  (default port 8188 matches `pipeline.toml`). The pipeline calls
  `POST /free` between waves so SDXL/TRELLIS don't fight the coder for VRAM.

### 5. Orpheus TTS

```bash
git clone https://github.com/Lex-au/Orpheus-FastAPI C:\Orpheus-FastAPI
```

Follow its README (needs its own small GGUF + llama.cpp or LM Studio backend).
Serve on port 5005 (or update `[tts] url`). ~4-8GB VRAM — the scheduler gives
it its own wave, so it never co-resides with the big models.

### 6. Godot 4

Download the stable Windows editor binary from <https://godotengine.org/download>
plus **export templates** (Editor → Manager → Export Templates, or the
`.tpz` from the same page — required for `--export-release`). Set
`paths.godot`. In the generated project, the assemble step expects an export
preset named in the task spec (default `"Windows Desktop"`).

### 7. Meshy

Create an API key at <https://www.meshy.ai> → set the `MESHY_API_KEY`
environment variable. Network-bound; runs concurrently with GPU waves.

---

## Crash safety & resume

There is no pause button because none is needed: **the queue is the truth and
every state change is fsynced** (`PRAGMA synchronous=FULL` on a WAL SQLite).
Kill the process, shut the machine down, pull the plug mid-run — on the next
start the pipeline continues where it stopped:

- the GUI auto-resumes every unfinished run on startup; `python run.py resume`
  does the same from the terminal;
- tasks caught `in_progress` by the crash are reclaimed to `pending`; their
  attempt counts survive, so a task that keeps killing the process still
  exhausts `max_attempts` instead of crash-looping forever;
- decomposition inserts are one atomic transaction — a crash mid-decompose
  leaves zero tasks, and resume re-decomposes from scratch;
- completed artifacts live on disk and are never re-generated.

The unit of loss on a hard kill is at most the single task that was executing.

## ETA

Both the GUI (progress bar per run) and the terminal (`[eta] ...` line after
every task, and `python run.py status`) show a live estimate that is
**learned, wave-aware and honest about uncertainty**:

- every task execution and every model load is timed and persisted
  (`durations` table), and estimates use the median + p90 of recent history —
  the more the pipeline runs, the sharper it gets;
- the projection replays the scheduler's own plan: remaining waves in order,
  one model-load cost per wave, network-lane work overlapped (max, not sum)
  with the GPU timeline, in-flight tasks credited for time already spent;
- the answer is a p50–p90 *band* (spreads added in quadrature), never a single
  fake-precise number, and it is labeled `[history]`, `[mixed]` or
  `[defaults]` so you know what it is based on.

## Autostart (start devving at boot, lose nothing)

Register the pipeline as a logon Scheduled Task so it comes up automatically
every time you log in and immediately continues any run interrupted by the last
shutdown — no clicking, no lost time:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
```

It runs windowless (`pythonw`), has no execution time limit, and restarts
itself if it ever dies — service-like, without admin rights. The GUI lands on
http://127.0.0.1:8500. Start it immediately without logging out:

```powershell
Start-ScheduledTask -TaskName AIGameDevPipeline
```

Remove it with `scripts\uninstall_autostart.ps1`.

### Watched folder (drop a file, a game starts)

With autostart on, the pipeline also watches an **inbox** folder. Drop any
`*.md` in there and a run begins on its own — no GUI, no clicking:

```
inbox/
  mygame.md            ← dropped; auto-starts within poll_interval_s (default 10s)
  mygame.png           ← optional; any sibling image with the same stem is used as a reference
  started/             ← claimed files move here so they never restart twice
```

- A file is **claimed by an atomic rename** into `inbox/started/`, so two polls
  (or a crash mid-claim) can never start the same game twice.
- Sibling images sharing the `.md`'s stem (`mygame.png`, `mygame_ref2.jpg`)
  become that run's reference images.
- Watched runs share the one-GPU lock with GUI runs and resumes — they queue,
  never collide.
- On startup a reconcile pass recovers any file that was claimed but whose run
  never got created (the one-in-a-million crash between rename and record).

Run the watcher without the GUI: `python run.py watch`. Change the folder or
cadence under `[watch]` in `pipeline.toml`.

Scope note: autostart **resumes unfinished runs and watches the inbox**. A new
game enters either by GUI upload or by landing in the inbox; once started,
nothing short of deleting the workspace stops it from finishing across reboots.

## Configuration reference

All knobs live in `pipeline.toml` (see `pipeline.toml.example`; defaults in
`pipeline/config.py`):

| Section | Key | Meaning |
|---|---|---|
| `paths` | `workspace` | runs, artifacts, logs, SQLite DB root |
| `paths` | `godot`, `llama_server` | binaries |
| `llm` | `router_gguf`, `coder_gguf`, `*_port` | model files + ports |
| `llm` | `ctx_size`, `temperature`, `max_tokens` | generation params |
| `llm` | `load_timeout_s`, `request_timeout_s` | generous by design — a slow local model must never be cut off mid-thought |
| `comfy` | `url`, `sdxl_workflow`, `trellis_workflow`, `timeout_s` | ComfyUI |
| `tts` | `url`, `timeout_s` | Orpheus endpoint |
| `meshy` | `api_key_env`, `poll_interval_s`, `timeout_s` | Meshy REST |
| `scheduler` | `max_attempts` | in-wave retries per task |
| `scheduler` | `wave_order` | GPU wave sequence per cycle |
| `gui` | `host`, `port` | GUI bind (localhost only by default) |

## Versioning

- Semantic versioning; `pipeline.__version__` is the source of truth, releases
  are git tags `vX.Y.Z`.
- `requirements.txt` pins exact versions of the 4 core runtime deps. Heavy
  optional validators (e.g. CLIP style-match scoring) will ship as a separate
  `requirements-validate.txt` extra so the core spine never drags in torch.

## Honest limitations

- Visual ceiling is set by the **generated assets** (TRELLIS meshes, Meshy
  auto-rigs ≈ solid stylized game props, not AAA hero characters) — not by
  Godot. Expect "good-looking stylized 3D", not photoreal AAA.
- Combat feel is generated, not hand-tuned: rough functional demo first.
- Style consistency across 2D→3D depends on doing a style-lock (IP-Adapter or
  project LoRA) in the SDXL workflow; skipping it produces mismatched assets.
- `workflows/trellis.json` must be exported from your own ComfyUI install
  once — TRELLIS node packs differ too much to ship a universal graph.

## Steam notes (for the eventual demo)

- Steam Direct fee: $100 one-time per game.
- **AI content disclosure** is required for player-facing AI art, models,
  audio, and dialogue (AI-generated *code* is exempt).
- Use Steam Playtest for early feedback before any public release.
- Ship original IP only — genre-inspired, no copied characters or move sets.
