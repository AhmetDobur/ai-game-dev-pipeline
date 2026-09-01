"""Qwen2.5-Coder <tools>-dialect shim test: few-shot system prompt + regex parser."""
import json, re, urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"

TOOLS = [
    {"name": "read_file", "description": "Read a file from disk and return its contents",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "run_tests", "description": "Run the pytest suite, optionally for one file",
     "parameters": {"type": "object", "properties": {"target": {"type": "string"}}}},
]

SYSTEM = f"""You are a coding agent with access to these tools:

{json.dumps(TOOLS, indent=2)}

To call a tool, output ONLY a <tools> block containing one JSON object, no backticks:
<tools>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tools>

Example:
User: What is in /etc/hosts?
Assistant: <tools>
{{"name": "read_file", "arguments": {{"path": "/etc/hosts"}}}}
</tools>

If no tool is needed, answer normally without any <tools> block."""

# ponytail: shim = 3 regex variants seen in the wild (tools tag, json fence, bare json)
PATTERNS = [
    re.compile(r"<tools>\s*(\{.*?\})\s*(?:</tools>|$)", re.S),
    re.compile(r"```json\s*(\{.*?\})\s*```", re.S),
    re.compile(r"^\s*(\{\"name\".*\})\s*$", re.S),
]

def parse_shim(content):
    for pat in PATTERNS:
        m = pat.search(content or "")
        if m:
            try:
                obj = json.loads(m.group(1))
                if "name" in obj and isinstance(obj.get("arguments", {}), dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None

def chat(messages, temp=0.6):
    body = json.dumps({"model": "q", "messages": messages,
                       "temperature": temp, "max_tokens": 512}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]

# 5 trials: should call read_file
hits = []
for i in range(5):
    m = chat([{"role": "system", "content": SYSTEM},
              {"role": "user", "content": "What is inside /app/config.yaml?"}])
    call = parse_shim(m.get("content"))
    ok = bool(call) and call["name"] == "read_file" and "path" in call["arguments"]
    hits.append(ok)
    if not ok:
        print(f"trial {i+1} MISS, raw: {(m.get('content') or '')[:150]!r}")
print(f"tool-call via shim: {sum(hits)}/5")

# multi-turn: feed tool result back, expect plain answer
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What is inside /app/config.yaml?"},
        {"role": "assistant", "content": '<tools>\n{"name": "read_file", "arguments": {"path": "/app/config.yaml"}}\n</tools>'},
        {"role": "user", "content": 'Tool result for read_file: "retries: 3\\ntimeout: 30". Now answer the original question.'}]
m = chat(msgs)
ans = m.get("content") or ""
print("followup answered:", "3" in ans and not parse_shim(ans), "| raw:", ans[:120].replace("\n", " "))

# restraint: no tool needed
m = chat([{"role": "system", "content": SYSTEM},
          {"role": "user", "content": "Say hello in Dutch."}])
print("restraint (no tool):", parse_shim(m.get("content")) is None, "| raw:", (m.get("content") or "")[:80])
