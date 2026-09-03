"""In-memory live output buffer, one capped text ring per run.

The coder (and router) stream their tokens here as they generate, so the GUI can
show the model writing code in real time and the terminal can tee it. Live view
only — not persisted; a restart starts the buffer fresh while the durable task
state lives in SQLite.
"""
import sys
import threading

MAX_CHARS = 20_000          # keep the tail; a whole game's output would be huge
_lock = threading.Lock()
_buffers: dict[str, str] = {}
_headers: dict[str, str] = {}   # run_id -> current activity line
tee_stdout = False              # run.py sets this so CLI users see it live too


def start(run_id: str, header: str) -> None:
    """Mark a new activity (e.g. 'coding scripts/player.gd')."""
    with _lock:
        _headers[run_id] = header
        prefix = f"\n\n=== {header} ===\n"
        _buffers[run_id] = (_buffers.get(run_id, "") + prefix)[-MAX_CHARS:]
    if tee_stdout:
        sys.stdout.write(prefix)
        sys.stdout.flush()


def append(run_id: str, text: str) -> None:
    with _lock:
        _buffers[run_id] = (_buffers.get(run_id, "") + text)[-MAX_CHARS:]
    if tee_stdout:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except UnicodeEncodeError:
            # cp1252 console on Windows can't take every model token; the tee is
            # cosmetic — never let it kill the run
            sys.stdout.write(text.encode("ascii", "replace").decode())
            sys.stdout.flush()


def get(run_id: str) -> dict:
    with _lock:
        return {"header": _headers.get(run_id, ""), "text": _buffers.get(run_id, "")}


def token_sink(run_id: str):
    """A callable to hand to LlamaServer.chat(on_token=...)."""
    return lambda tok: append(run_id, tok)
