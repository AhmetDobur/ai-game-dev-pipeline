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
                     out_dir: Path) -> list[Path]:
        """Load a workflow JSON, substitute {{placeholders}}, run, download outputs.
        A substitution value that is an existing local file (e.g. the concept image
        for TRELLIS) is uploaded first — LoadImage only accepts files in ComfyUI's
        own input dir, never absolute paths."""
        text = Path(workflow_path).read_text(encoding="utf-8")
        for key, value in substitutions.items():
            if value and Path(value).is_file():
                value = self.upload_image(Path(value))
            text = text.replace("{{" + key + "}}", json.dumps(value)[1:-1])  # json-escape
        workflow = json.loads(text)

        client_id = uuid.uuid4().hex
        r = requests.post(f"{self.url}/prompt",
                          json={"prompt": workflow, "client_id": client_id}, timeout=30)
        r.raise_for_status()
        prompt_id = r.json()["prompt_id"]

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            h = requests.get(f"{self.url}/history/{prompt_id}", timeout=30).json()
            entry = h.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI workflow error: {json.dumps(status)[:500]}")
                if entry.get("outputs"):
                    return self._download_outputs(entry["outputs"], out_dir)
            time.sleep(3)
        raise TimeoutError(f"ComfyUI workflow {prompt_id} not done after {self.timeout_s}s")

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
