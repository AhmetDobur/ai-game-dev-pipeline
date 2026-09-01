# AI Game Development Pipeline — Project README

## Goal

Build a fully automated, agent-orchestrated pipeline that generates a playable
game demo from a text description — combining LLM orchestration, 2D/3D asset
generation, rigging/animation, audio, and code generation — with minimal to
zero manual human labor.

**Proof-of-concept target:** a small original fighting-game demo (inspired by
the genre, not a copy of any existing IP) featuring:
- 2 playable characters
- 1 arena/map
- Minimal UI/HUD
- Basic combat system (attack, block, hit reactions, KO state)
- Deliverable as a Steam Playtest build for early feedback

**Author framing:** this project is built by a prompt engineer / AI
integration engineer, not a traditional game developer — the pipeline itself
is the product, and the demo is the proof it works.

---

## Hardware Constraints

| Component | Spec |
|---|---|
| GPU | Titan RTX (24GB VRAM) |
| CPU | i5-14600K |
| RAM | 32GB system RAM |

**Implication:** only one large model can comfortably occupy VRAM at a time.
The architecture is built around **sequential model swapping**, not parallel
multi-model loading. The router/orchestrator model stays resident at all
times (small footprint); larger models (coder, TTI, TT3D) load on-demand and
unload after each batch.

---

## Model Stack

### Orchestrator / Router (always resident, ~4GB VRAM)
- **DeepSeek-R1-Distill-Qwen-7B** (Q4_K_M) — task decomposition, queue
  management, output validation. Chosen for distilled chain-of-thought
  reasoning at a small footprint.

### Coder (loads on-demand, ~18-20GB VRAM)
- **Qwen3-Coder-30B-A3B** (Q4_K_M, MoE) — game logic, combat state machine,
  hitbox/hurtbox definitions, headless engine build scripts. Trained for
  agentic tool calling; only ~3B params active per token, so fast at Q4.
- **Ruled out: Qwen2.5-Coder-32B** (dense) — tested 2026-09-01 via
  llama-server `--jinja`: 0/5 structured tool calls on `tool_choice=auto`;
  emits raw JSON in `content` with an invented `<tools>` wrapper instead of
  the `<tool_call>` format its own template defines, and even
  `tool_choice=required` fails to parse. Usable for plain prompt→code
  generation only, not as an agentic coder.
- Alternatives considered: Qwen3-Coder-Next-80B-A3B (MoE, too large for
  concurrent residency with other stages), Qwen3-8B (lighter fallback).

### Text-to-Image / TTI (loads on-demand, ~12GB VRAM)
- SDXL (default) / Flux.1-dev (stretch goal) — concept art, character
  reference sheets, textures, environment art, UI mockups. Note: Titan RTX
  is Turing (no FP8/BF16), so Flux's usual ~12GB fp8 path doesn't apply —
  Flux needs GGUF/NF4 quants and will be slow; SDXL is the safe default.
- Style consistency enforced via **IP-Adapter** (reference-image
  conditioning) or a project-specific **LoRA** trained once per project.

### Text/Image-to-3D (loads on-demand, ~12-16GB VRAM)
- **TRELLIS** (MIT — primary) or **Hunyuan3D-2.x** (Tencent Hunyuan
  Community License, which **excludes the EU** — not safe for a Steam
  release from NL) — generates meshes +
  PBR textures conditioned on both the TTI concept image and the text spec,
  ensuring visual match between 2D concept and 3D asset.

### Rigging & Animation (cloud API, automation target)
- **Meshy API** — auto-rigs humanoid meshes in under 30 seconds, applies
  preset motion clips (idle, walk, punch, kick, block, hit-react, KO) from
  a 600+ clip library, exports FBX with standard bone hierarchies.
- Fallback for custom moves: text-to-motion generation (prompt-based, e.g.
  "a fighter throws a roundhouse kick") for anything outside the preset
  library.

### Text-to-Speech / TTS (~4-8GB VRAM, can run without unloading router)
- **Orpheus 3B** (Apache 2.0, Llama-3B backbone) — dialogue, announcer
  lines, zero-shot voice cloning, supports emotion tags (`<laugh>`,
  `<sigh>`, etc.), ~200ms streaming latency.

### Speech-to-Text / STT (optional, for interactive voice input)
- Whisper large-v3 / Faster-Whisper — only needed if voice-driven
  interaction is added later; not required for the base demo.

---

## Pipeline Strategy

### Stage 0 — Style Lock (once per project)
Router queues a batch of reference images from the TTI model, builds an
IP-Adapter embedding or trains a LoRA, and saves it as the project's style
reference. Every subsequent TTI task conditions on this reference to keep
theme/style consistent across all assets.

### Stage 1 — Task Decomposition
The resident router (DeepSeek-R1-Distill-Qwen-7B) takes a game/feature
description and breaks it into a typed, dependency-aware task queue:

```
Task {
  id: string
  type: "code" | "design_2d" | "design_3d" | "rig_animate" | "audio" | "validate"
  depends_on: [task_id, ...]
  spec: { ... }
  model_needed: string
  status: "pending" | "in_progress" | "done" | "failed"
  output_path: string | null
}
```

Stored in a local SQLite/JSON queue.

### Stage 2 — Dispatcher Loop (sequential, VRAM-aware)
1. Poll queue for the next task whose dependencies are satisfied.
2. Load only the model required for that task type into VRAM.
3. Run the task (batch multiple same-type tasks together to minimize
   reload overhead).
4. Write output path back to the task record, mark as done.
5. Unload the model, flush VRAM (`torch.cuda.empty_cache()` + process
   teardown), loop.

Single-threaded execution only — no parallel job runs, to avoid VRAM/RAM
contention on this hardware.

### Stage 3 — Generation Branches
- `design_2d` → TTI model, style-locked, produces concept art/textures/UI
- `design_3d` → TT3D model, conditioned on matching `design_2d` output +
  spec text
- `rig_animate` → Meshy API, auto-rig + preset/generated motion clips,
  exported as FBX
- `audio` → Orpheus 3B, dialogue/announcer lines from LLM-written script
- `code` → Qwen3-Coder-30B-A3B, combat state machine, hitbox/hurtbox logic,
  input handling, headless engine build scripts

### Stage 4 — Validation
Router (DeepSeek-R1-Distill-Qwen-7B) checks each output against its
original spec before marking the task complete — flags mismatches for
regeneration (e.g. 3D asset doesn't match concept art tags, dialogue
doesn't match character voice).

### Stage 5 — Engine Assembly
Coder model generates a headless build script (e.g. Unity C# + command-line
batch mode) that imports all assets, wires prefabs, binds animations to the
combat state machine, and produces a runnable build — no manual GUI steps.

---

## Requirements

### Software
- Python 3.11+ (dispatcher/orchestration layer)
- llama.cpp / Ollama / vLLM (LLM inference backends)
- ComfyUI or diffusers (TTI pipeline)
- TRELLIS inference environment (Hunyuan3D-2.x only if license territory issue is resolved)
- Meshy API key (rigging/animation)
- Orpheus-FastAPI or equivalent (TTS serving)
- SQLite (task queue persistence)
- Unity or Unreal Engine (headless build target)
- Git + version control for generated asset tracking

### Hardware
- Titan RTX (24GB VRAM) — confirmed sufficient for sequential single-model
  loading up to 32B-dense-class models at Q4 quantization
- 32GB system RAM — sufficient given no parallel model loading; keep
  background processes minimal during generation runs
- Stable internet connection for Meshy API calls

### Distribution (Steam)
- Steam Direct fee ($100, one-time per game)
- Steamworks account + app page setup
- **Steam AI Content Disclosure** required for player-facing AI content:
  art, 3D models, audio, dialogue (code generation via AI coding tools is
  exempt from disclosure)
- Use **Steam Playtest** feature for limited early access testing before
  any public release
- Legal note: build as an original IP inspired by the fighting-game genre —
  do not reproduce copyrighted characters, names, or move sets

---

## Known Limitations (Honest Assessment)

- Combat "feel" (frame data tuning, hit timing polish) is generated, not
  hand-tuned — expect a rough, functional demo rather than a polished
  fighting game.
- Preset animation clips may not perfectly match custom character
  proportions; text-to-motion fallback helps but isn't guaranteed to be
  seamless.
- Style consistency across TTI/TT3D depends heavily on the Stage 0 style
  lock being done well — skipping this step will produce mismatched assets.
- This pipeline accelerates asset creation and code scaffolding; it does
  not replace game design or playtesting judgment.
