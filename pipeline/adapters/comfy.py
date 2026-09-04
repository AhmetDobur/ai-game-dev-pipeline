"""ComfyUI HTTP API adapter — used for both SDXL (design_2d) and TRELLIS (design_3d).
ComfyUI owns VRAM load/unload; we submit a workflow JSON and wait for outputs."""
import json
import time
import uuid
from pathlib import Path

import requests


class ComfyClient:
    def __init__(self, url: str, timeout_s: int = 600):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def run_workflow(self, workflow_path: str | Path, substitutions: dict[str, str],
                     out_dir: Path, seed_offset: int = 0) -> list[Path]:
        """Load a workflow JSON, substitute {{placeholders}}, run, download outputs.
        A substitution value that is an existing local file (e.g. the concept image
        for TRELLIS) is uploaded first — LoadImage only accepts files in ComfyUI's
        own input dir, never absolute paths.

        `seed_offset` shifts every sampler seed in the graph. The workflows carry
        fixed seeds, so without it a re-run reproduces the previous mesh exactly:
        a retry could recover from a timeout but could never produce a better
        figure, and asking for another attempt at a bad mesh was pure waste.
        Offset 0 leaves the workflow untouched, so a first attempt stays
        reproducible and only genuine retries explore a different sample."""
        text = Path(workflow_path).read_text(encoding="utf-8")
        for key, value in substitutions.items():
            if value and Path(value).is_file():
                value = self.upload_image(Path(value))
            text = text.replace("{{" + key + "}}", json.dumps(value)[1:-1])  # json-escape
        workflow = json.loads(text)
        if seed_offset:
            for node in workflow.values():
                inputs = node.get("inputs", {})
                # relative offset, not a fresh random seed: the stages of a
                # TRELLIS graph are tuned against each other and should move
                # together rather than be independently rerolled
                if isinstance(inputs.get("seed"), int):
                    inputs["seed"] = (inputs["seed"] + seed_offset) % (2 ** 31)

        client_id = uuid.uuid4().hex
        r = requests.post(f"{self.url}/prompt",
                          json={"prompt": workflow, "client_id": client_id}, timeout=30)
        r.raise_for_status()
        prompt_id = r.json()["prompt_id"]

        # IDLE timeout, not a total one. A wall clock sized for the meshes we had
        # measured cut a legitimately slower one off at four hours and re-queued
        # it, so a second copy of a ten-hour job sat behind the copy that was
        # still running and about to finish. What we actually want to detect is a
        # hung prompt, and "hung" means no progress -- so the clock resets every
        # time the server's log advances, and only a genuinely stalled job dies.
        last_progress = time.time()
        seen = None
        while time.time() - last_progress < self.timeout_s:
            h = requests.get(f"{self.url}/history/{prompt_id}", timeout=30).json()
            entry = h.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI workflow error: {json.dumps(status)[:500]}")
                if entry.get("outputs"):
                    return self._download_outputs(entry["outputs"], out_dir)
            mark = self._progress_mark()
            if mark != seen:
                seen, last_progress = mark, time.time()
            time.sleep(3)
        raise TimeoutError(f"ComfyUI workflow {prompt_id} made no progress for "
                           f"{self.timeout_s}s")

    def _progress_mark(self) -> str | None:
        """A cheap fingerprint of server-side progress: the tail of ComfyUI's own
        log. Sampler steps, node transitions and model loads all move it. None on
        any failure, which simply means this poll contributes no evidence either
        way rather than being mistaken for a stall."""
        try:
            r = requests.get(f"{self.url}/internal/logs", timeout=15)
            return r.text[-400:] if r.ok else None
        except Exception:
            return None

    def _download_outputs(self, outputs: dict, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for node_output in outputs.values():
            for kind in ("images", "gltfs", "meshes", "files", "3d"):
                for item in node_output.get(kind, []):
                    r = requests.get(f"{self.url}/view", params={
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    }, timeout=120)
                    r.raise_for_status()
                    path = out_dir / item["filename"]
                    path.write_bytes(r.content)
                    saved.append(path)
        if not saved:
            raise RuntimeError("ComfyUI reported done but produced no downloadable outputs")
        return saved

    def upload_image(self, path: Path) -> str:
        """POST a local image into ComfyUI's input dir; return the name LoadImage wants."""
        with open(path, "rb") as f:
            r = requests.post(f"{self.url}/upload/image",
                              files={"image": (path.name, f)},
                              data={"overwrite": "true"}, timeout=120)
        r.raise_for_status()
        return r.json()["name"]

    def free_vram(self) -> None:
        """Ask ComfyUI to unload models + free VRAM between waves."""
        try:
            requests.post(f"{self.url}/free",
                          json={"unload_models": True, "free_memory": True}, timeout=30)
        except requests.RequestException:
            pass  # older ComfyUI without /free — models unload lazily instead
