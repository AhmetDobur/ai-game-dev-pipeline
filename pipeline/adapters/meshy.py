"""Meshy API adapter — auto-rig + animate a humanoid mesh, download FBX.
Network-bound: runs in the async lane alongside GPU waves."""
import os
import time
from pathlib import Path

import requests


class MeshyClient:
    def __init__(self, url: str, api_key_env: str = "MESHY_API_KEY",
                 poll_interval_s: int = 15, timeout_s: int = 1800):
        self.url = url.rstrip("/")
        key = os.environ.get(api_key_env, "")
        if not key:
            raise RuntimeError(f"{api_key_env} is not set")
        self.headers = {"Authorization": f"Bearer {key}"}
        self.poll_interval_s, self.timeout_s = poll_interval_s, timeout_s

    def _wait(self, endpoint: str, task_id: str) -> dict:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            r = requests.get(f"{self.url}{endpoint}/{task_id}",
                             headers=self.headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "SUCCEEDED":
                return data
            if data.get("status") in ("FAILED", "CANCELED"):
                raise RuntimeError(f"Meshy task {task_id} {data.get('status')}: "
                                   f"{data.get('task_error', {})}")
            time.sleep(self.poll_interval_s)
        raise TimeoutError(f"Meshy task {task_id} not done after {self.timeout_s}s")

    def rig_and_animate(self, mesh_url_or_path: str, animations: list[str],
                        out_dir: Path) -> list[Path]:
        """Submit rigging, then one animation task per clip name; download FBX files."""
        payload = {"model_url": mesh_url_or_path}
        r = requests.post(f"{self.url}/openapi/v1/rigging", json=payload,
                          headers=self.headers, timeout=60)
        r.raise_for_status()
        rig = self._wait("/openapi/v1/rigging", r.json()["result"])
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = [self._download(rig["result"]["rigged_model_url"], out_dir / "rigged.fbx")]
        for clip in animations:
            ra = requests.post(f"{self.url}/openapi/v1/animations",
                               json={"rigged_task_id": rig["id"], "action": clip},
                               headers=self.headers, timeout=60)
            ra.raise_for_status()
            anim = self._wait("/openapi/v1/animations", ra.json()["result"])
            saved.append(self._download(anim["result"]["model_url"],
                                        out_dir / f"{clip}.fbx"))
        return saved

    @staticmethod
    def _download(url: str, path: Path) -> Path:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        return path
