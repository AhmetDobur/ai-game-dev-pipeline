"""llama-server lifecycle + OpenAI-compatible chat, including the Qwen2.5-Coder
<tools> dialect shim (tested 5/5 on 2026-09-01 where native tool_calls scored 0/5)."""
import json
import re
import subprocess
import time

import requests


class LlamaServer:
    """Start/stop one llama-server process and talk to it. One instance per model."""

    def __init__(self, binary: str, gguf: str, port: int, ctx_size: int = 32768,
                 load_timeout_s: int = 300, extra_args: list[str] | None = None):
        self.binary, self.gguf, self.port = binary, gguf, port
        self.ctx_size, self.load_timeout_s = ctx_size, load_timeout_s
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self.is_healthy():
            return  # already running (externally managed or previous start)
        args = [self.binary, "--model", self.gguf, "--host", "127.0.0.1",
                "--port", str(self.port), "--n-gpu-layers", "999",
                "--ctx-size", str(self.ctx_size), "--flash-attn", "on",
                "--parallel", "1", "--jinja", *self.extra_args]
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        deadline = time.time() + self.load_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode} for {self.gguf}")
            if self.is_healthy():
                return
            time.sleep(2)
        self.stop()
        raise TimeoutError(f"llama-server not healthy after {self.load_timeout_s}s: {self.gguf}")

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.proc = None

    def is_healthy(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def chat(self, messages: list[dict], temperature: float = 0.6,
             max_tokens: int = 4096, timeout_s: int = 1200,
             on_token=None) -> str:
        """Return the full reply. If on_token is given, stream tokens to it as
        they arrive (for live view) and still return the assembled text."""
        body = {"model": "local", "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if on_token is None:
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=body,
                              timeout=timeout_s)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        body["stream"] = True
        r = requests.post(f"{self.base_url}/v1/chat/completions", json=body,
                          timeout=timeout_s, stream=True)
        r.raise_for_status()
        parts = []
        for piece in iter_sse_content(r.iter_lines()):
            parts.append(piece)
            on_token(piece)
        return "".join(parts)


def iter_sse_content(lines):
    """Yield assistant-content deltas from an OpenAI-style SSE stream.
    `lines` is an iterable of bytes/str lines (requests' iter_lines())."""
    for raw in lines:
        if not raw:
            continue
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0].get("delta", {})
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        content = delta.get("content")
        if content:
            yield content


def extract_json(text: str) -> dict | list:
    """Parse the largest JSON object/array in an LLM reply (fenced or bare)."""
    fences = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    candidates = fences + [text]
    for c in sorted(candidates, key=len, reverse=True):
        c = c.strip()
        # whichever bracket opens first is the outermost structure
        pairs = sorted((("{", "}"), ("[", "]")),
                       key=lambda p: (c.find(p[0]) == -1, c.find(p[0])))
        for start, end in pairs:
            i, j = c.find(start), c.rfind(end)
            if i != -1 and j > i:
                try:
                    return json.loads(c[i:j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no parseable JSON in reply: {text[:200]!r}")


# --- Qwen2.5-Coder <tools> dialect shim ------------------------------------

TOOLS_SHIM_SYSTEM = """You are a coding agent with access to these tools:

{tools_json}

To call a tool, output ONLY a <tools> block containing one JSON object, no backticks:
<tools>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tools>

If no tool is needed, answer normally without any <tools> block."""

_SHIM_PATTERNS = [
    re.compile(r"<tools>\s*(\{.*?\})\s*(?:</tools>|$)", re.S),
    re.compile(r"```json\s*(\{.*?\})\s*```", re.S),
    re.compile(r"^\s*(\{\"name\".*\})\s*$", re.S),
]


def parse_tool_call(content: str) -> dict | None:
    """Extract {'name': ..., 'arguments': {...}} from a <tools>-dialect reply."""
    for pat in _SHIM_PATTERNS:
        m = pat.search(content or "")
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if "name" in obj and isinstance(obj.get("arguments", {}), dict):
            return obj
    return None
