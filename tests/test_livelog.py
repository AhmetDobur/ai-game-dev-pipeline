"""Live view: SSE parsing, capped buffer, and the coder streaming into it."""
from pathlib import Path

from pipeline import livelog
from pipeline.adapters.llm import iter_sse_content
from pipeline.config import load
from pipeline.executors import build_executors


def test_iter_sse_content_parses_deltas_and_stops_on_done():
    lines = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}',
        b"",
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',
        b'data: {"choices":[{"delta":{}}]}',        # no content -> skipped
        b"data: [DONE]",
        b'data: {"choices":[{"delta":{"content":"AFTER"}}]}',  # ignored after DONE
    ]
    assert "".join(iter_sse_content(lines)) == "hello"


def test_buffer_is_capped_to_tail(monkeypatch):
    monkeypatch.setattr(livelog, "MAX_CHARS", 10)
    livelog._buffers.clear()
    livelog.append("r1", "abcdefghijː")      # 11 chars
    got = livelog.get("r1")["text"]
    assert len(got) == 10 and got.endswith("ː")


class StreamingCoder:
    def chat(self, messages, on_token=None, **kw):
        for tok in ["ext", "ends ", "Node"]:
            on_token(tok)
        return "extends Node"


def test_code_executor_streams_into_livelog(tmp_path):
    livelog._buffers.clear()
    execs = build_executors(load(), tmp_path, StreamingCoder())
    task = {"id": "t", "run_id": "runX",
            "spec": {"file": "player.gd", "description": "a node"}}
    execs["code"](task, tmp_path / "out")
    live = livelog.get("runX")
    assert "coding player.gd" in live["header"]
    assert "extends Node" in live["text"]
