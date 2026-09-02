# AI Game Development Pipeline

A game-agnostic, single-GPU, agent-orchestrated pipeline that turns an uploaded
`instruction.md` (plus optional reference images) into a playable Godot 4 game
build — code, 2D art, 3D assets, rigging/animation, and voice audio all
generated, scheduled, and validated automatically.

The pipeline is the product. The first proof-of-concept input will be a small
original fighting-game demo, but nothing about the game is hardcoded: the game
arrives as an instruction file through the GUI, gets decomposed into a typed
task graph, and is executed in **model waves** sized to one 24GB GPU.

- **Version:** 0.6.0 (semver; git tags `vX.Y.Z`)
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
      │ GPU waves (one model loaded at a time)         │ local lane (CPU)
      │  coder → sdxl → trellis → motion → tts → (…)   │  assemble → godot --headless
      │  each wave drains ALL ready tasks of its type, │
      │  validates IN-WAVE so retries reuse the        │
      │  already-loaded model. motion = Blender rig+   │
      │  animate, all local.                           │
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
   (`code`→coder LLM, `design_2d`→SDXL, `design_3d`→TRELLIS,
   `rig_animate`→Blender motion stage, `audio`→TTS).
   A wave loads its model once (`wave_setup`), drains **every** ready task of
   that type — including tasks whose dependencies complete *during* the wave —
   then unloads (`wave_teardown`). Model loads per cycle drop from
   one-per-task to one-per-type.
2. **In-wave validation and retry**: every output is validated immediately,
   while the producing model is still resident. A failed artifact retries up
   to `max_attempts` times *inside the wave* (the executor receives
   `last_error` so the retry prompt says what was wrong) instead of waiting a
   full cycle and paying a reload.
3. **Everything is serial on one GPU**: the motion stage (Blender + UniRig/
   Kimodo) is GPU-bound like the rest, so it runs as its own wave, not a
   parallel lane. Only `assemble` (a CPU-only `godot --headless` subprocess)
   runs off the GPU timeline, between waves.
4. The **router LLM stays resident** the whole run (small footprint); the
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
| `rig_animate` | `mesh_from`, `body_plan`, `animations[]`, `extras[]` | Blender headless (UniRig/Kimodo/CMU + procedural) | animated `.glb` exists + size floor |
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

### Motion (rig + animate) — fully local, arbitrary creatures

The old cloud rig/animate step (Meshy) is **gone** — nothing about a character
leaves the box. `rig_animate` runs headless Blender on
`templates/blender_motion.py`, which always produces one animated `.glb` per
clip **from Blender alone** (the procedural floor) and folds in optional local
tools when configured:

- **UniRig** (MIT) — auto-rigs *any* topology: humans, animals, a
  dragon-amoeba, inorganic shapes. Falls back to a procedural armature fitted
  to the mesh's bounding box when not installed.
- **Kimodo** (NVIDIA Open Model License, commercial-OK, trained on NVIDIA's own
  mocap — not AMASS) — text→motion for **humanoid** clips.
- **CMU mocap** (free for any use) — exact-match clips reused verbatim, checked
  first; also the placeholder net so nothing stalls waiting on a generated clip.

The split, decided per character by the router's `body_plan`:

| body_plan | motion source |
|---|---|
| `humanoid` | CMU exact match → else Kimodo generates → else procedural |
| `nonhumanoid` | procedural (physics/gait keyframes) — no motion *model* can animate a novel body plan |

`extras[]` (`tail`, `jaw`, `wings`, `cloak`) always get procedural secondary
motion on top — a humanoid body with a tail (Mileena-style) is mocap body +
procedural tail + scripted jaw, exactly how AAA does it. **Worst case, with zero
AI tools installed, every creature still ships a moving model** via the
procedural path.

> The bpy script runs inside Blender on the CUDA box; the pipeline's unit tests
> mock the Blender subprocess (as they do godot/ComfyUI/TTS).

### Patching (keep working on a shipped game)

A finished game isn't frozen — `patch` applies a *delta* instruction to it and
produces a **new revision** (v1 → v2 → v3), reusing everything the change
doesn't touch. Mechanism:

1. The parent revision's whole workspace (game + artifacts) is snapshotted into
   the new revision, and its task graph is copied in as `done`.
2. The router decomposes the delta against a **manifest** of what the game
   already contains, emitting only `MODIFY <artifact>` / `ADD <artifact>` ops.
3. A **dependency-aware invalidation walk** marks the changed tasks and every
   transitive dependent stale; the single `assemble` task is rewired to depend
   on everything so the build is always regenerated last.
4. The normal scheduler then re-runs **exactly** the stale tasks and reuses the
   rest. Change one line of code → nothing else loads. Change a boss mesh →
   only its rig + motion re-run, not the whole game.

The whole patch graph is inserted in one atomic transaction, so a crash
mid-build leaves a clean re-run, and patches inherit resume/ETA for free.

### Repo layout

```
pipeline/
  config.py        pipeline.toml loader (leaf module, imports nothing)
  db.py            SQLite queue: runs, tasks, durations, reclaim, ready-set, revisions
  decompose.py     router prompt → validated task list (fresh + patch delta) → queue
  patch.py         revisions: snapshot + reuse + dependency-aware invalidation walk
  scheduler.py     wave scheduler + in-wave retry
  executors.py     production executors (one per task type)
  orchestrate.py   wires config+adapters+scheduler for one run (fresh or patch)
  validate.py      objective per-branch validators
  eta.py           learned wave-aware ETA (p50–p90 band from run history)
  watch.py         inbox auto-start: claim-by-rename, sibling refs, reconcile, patch marker
  livelog.py       in-memory live output buffer (model tokens, per run)
  gui.py           FastAPI page: live output panel, task table, ETA, resume + watch + patch
  adapters/
    llm.py         llama-server lifecycle + chat + Qwen2.5 <tools> shim
    comfy.py       ComfyUI /prompt → poll /history → download outputs
    motion.py      Blender headless rig+animate → animated .glb (local, no cloud)
    tts.py         Orpheus /v1/audio/speech
run.py             CLI: gui | run <md> [--ref img] | patch <parent> <md> | status | resume | watch
templates/         frame_data_test.gd (headless timing grader), blender_motion.py (rig+animate)
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
.venv/Scripts/python run.py gui                      # http://127.0.0.1:8500
```

Upload an `instruction.md` (and optional reference images) in the GUI, press
**Start run**, watch the task table drain. Or headless:

```bash
python run.py run my-game/instruction.md --ref my-game/style.png
python run.py status
python run.py patch <run_id> tweaks.md            # apply a delta → new revision
```

Run tests (no GPU needed):

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

---

## Using it — end to end

### 1. Write an `instruction.md`

This is the whole input. There is no fixed schema — write it like a brief for a
developer. The router reads it and decides the tasks. The more concrete you are
(exact characters, moves, arena, and any timing tables), the less the models
have to invent. A minimal fighting-game example:

```markdown
# Neon Duel — 1v1 fighter demo

Original IP. Two fighters, one arena, best-of-one round, KO ends the match.

## Characters
- **Ravi** — lean street brawler, red jacket, fast light attacks.
- **Bo** — heavy grappler, blue armor, slow but high knockback.

## Arena
- Single flat stage, neon-city backdrop, invisible walls at the edges.

## Combat
- Moves: light punch, heavy kick, block, jump.
- Health bar per fighter; hit reactions; KO state when health hits 0.

## Frame data (60fps) — grade these exactly
| move        | startup | active | hitstun | knockback |
|-------------|---------|--------|---------|-----------|
| light_punch | 6       | 2      | 12      | [25, 0]   |
| heavy_kick  | 12      | 4      | 20      | [60, 0]   |

## UI
- Two health bars, round timer, "KO!" banner.
```

Any timing table like the one above is copied verbatim into the combat code
task and **graded by simulation** (see Frame-data contract above) — get those
numbers right and the feel is right.

### 2. Start a game — three ways

- **GUI:** `python run.py gui` → open http://127.0.0.1:8500 → upload the `.md`
  (and any reference images) → **Start run**. The task table and a live ETA
  bar show progress.
- **Inbox (hands-off):** copy `neon-duel.md` into the `inbox/` folder (put
  `neon-duel.png` next to it to use as a style reference). It auto-starts within
  ~10s. This is what fires on boot when autostart is installed.
- **CLI:** `python run.py run neon-duel.md --ref style.png`.

### 3. Watch it

- **Live output panel (GUI):** a terminal-style pane at the top streams the
  model's tokens as it happens — you literally watch it plan the task list, then
  write each `.gd` file line by line (the header shows what it's coding, e.g.
  `coding scripts/player.gd`). Updates ~1/sec, auto-scrolls.
- GUI also shows a per-run progress bar + p50–p90 ETA + the task table.
- **Terminal:** `python run.py run ...` tees the same live stream to stdout, so
  you see it coding in the console too. An `[eta]` line prints after each task;
  `python run.py status [run_id]` shows run/task state and ETA.

### 4. Get the build

When the run finishes, the exported game is under
`workspace/runs/<run_id>/dist/` (a `.exe` for the `Windows Desktop` preset).
The full generated Godot project — every script, image, mesh, and audio file —
is in `workspace/runs/<run_id>/game/`, so you can open it in the Godot editor
and keep working by hand.

### 5. Tune the feel — or patch it

If a move feels off, edit its numbers in the instruction's frame-data table and
re-run — the combat wave re-grades against the new values. No code editing
needed for timing.

For anything bigger — "add a second boss", "make the jump higher", "swap the
arena" — write a short delta `.md` and `patch` the game. It becomes a new
revision that reuses everything untouched and re-runs only what changed
(`python run.py patch <run_id> delta.md`, the **patch this game** button in the
GUI, or drop a `.md` whose first line is `patch: <run_id>` into the inbox).
Higher-quality humanoid motion comes from installing Kimodo/CMU (see config);
non-humanoid motion is procedural by design.

### If something fails

A failed task turns red in the GUI with the exact validator error; the run
finishes as `failed` and lists how many tasks failed or were blocked. Fix the
cause (usually a tool not running, a bad path in `pipeline.toml`, or the TRELLIS
workflow placeholder) and re-run — completed artifacts are reused, only the
failed branch re-executes.

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
| Coder (wave) | Qwen2.5-Coder-32B-Instruct (dense) | `Q4_K_M` GGUF | ~19GB |

Get them from Hugging Face (`bartowski` or `unsloth` GGUF repos). Put paths in
`pipeline.toml` under `[llm]`.

**Why the dense 32B needs no shim here:** Qwen2.5-Coder-32B speaks a
non-standard *tool-call* dialect (measured 2026-09-01: 0/5 native `tool_calls`,
5/5 behind the `<tools>` few-shot shim in `pipeline/adapters/llm.py`). But this
pipeline's coder never makes tool calls — it is asked for a single fenced code
block (`CODE_PROMPT` → `_extract_block`), so the tool-call weakness does not
apply to its job. The same model scored 13/14 on a hard concurrency benchmark
where the 80B MoE deadlocked at 0/14. The `<tools>` shim stays in `llm.py` for
any future tool-calling use; it is not on the coder path.

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

### 7. Blender + motion tools (rig + animate, all local)

Install **Blender 4.x** and set `motion.blender` to its binary. That alone is
enough — the procedural floor rigs and animates any mesh with Blender only.
Optional local quality boosters, each with a graceful fallback:

- **UniRig** (MIT) — clone it, point `motion.unirig` at its dir for
  auto-rigging arbitrary shapes. No cloud, no key.
- **Kimodo** (NVIDIA Open Model License) — serve it locally and set
  `motion.kimodo_url` for humanoid text→motion.
- **CMU mocap BVH** — download the library, set `motion.cmu_dir` for
  exact-match clips.

No API key, no account, nothing leaves the machine.

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
| `motion` | `blender`, `script`, `cmu_dir`, `unirig`, `kimodo_url`, `timeout_s` | local rig+animate (all optional except `blender`) |
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

- Visual ceiling is set by the **generated assets** (TRELLIS meshes + auto-rigs
  ≈ solid stylized game props, not AAA hero characters) — not by Godot. Expect
  "good-looking stylized 3D", not photoreal AAA.
- **Motion quality is split by body plan**: humanoids can reach real
  mocap/generated quality (CMU/Kimodo); non-humanoid creatures get procedural
  motion — it moves anything, but it won't match hand-authored AAA polish. No
  model, local or cloud, generates believable motion for a novel body plan.
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
