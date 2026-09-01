"""Orpheus-FastAPI adapter (OpenAI-compatible /v1/audio/speech)."""
from pathlib import Path

import requests


class TTSClient:
    def __init__(self, url: str, timeout_s: int = 300):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def speak(self, text: str, voice: str, out_path: Path) -> Path:
        r = requests.post(f"{self.url}/v1/audio/speech",
                          json={"model": "orpheus", "input": text, "voice": voice,
                                "response_format": "wav"},
                          timeout=self.timeout_s)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        return out_path

    def is_up(self) -> bool:
        try:
            return requests.get(f"{self.url}/docs", timeout=2).status_code < 500
        except requests.RequestException:
            return False
