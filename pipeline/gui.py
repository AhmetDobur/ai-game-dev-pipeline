"""Single-page GUI: upload instruction.md + optional reference images, start a run,
watch the task table drain. Serves on localhost only."""
import json
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from . import config, db, eta
from .orchestrate import execute_run, resume_incomplete_runs, start_run

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app):
    """Crash-safe restart: any run interrupted by kill/shutdown/power cut
    continues automatically when the GUI comes back up."""
    def worker():
        c = db.connect(cfg["paths"]["db"])
        with _run_lock:
            resume_incomplete_runs(cfg, c)
    if db.incomplete_runs(conn()):
        threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="ai-game-dev-pipeline", lifespan=_lifespan)
cfg = config.load()
_conn = None
# one GPU, one run at a time — a second upload queues behind the active run
_run_lock = threading.Lock()


def conn():
    global _conn
    if _conn is None:
        _conn = db.connect(cfg["paths"]["db"])
    return _conn


PAGE = """<!doctype html><html><head><title>game pipeline</title><style>
body{font-family:system-ui;margin:2rem;max-width:70rem}
table{border-collapse:collapse;width:100%;margin-top:1rem}
td,th{border:1px solid #ccc;padding:.4rem .6rem;text-align:left;font-size:.9rem}
.done{background:#d4f7d4}.failed{background:#f7d4d4}.in_progress{background:#fdf3c9}
#drop{border:2px dashed #999;padding:1.5rem;border-radius:8px;margin:1rem 0}
button{padding:.5rem 1.2rem;font-size:1rem}
</style></head><body>
<h1>AI game pipeline</h1>
<div id=drop>
  <form id=f>
    <p>instruction.md: <input type=file name=instruction accept=".md,.txt" required></p>
    <p>reference images (optional): <input type=file name=refs accept="image/*" multiple></p>
    <button>Start run</button>
  </form>
</div>
<div id=runs></div>
<script>
const f=document.getElementById('f');
f.onsubmit=async e=>{e.preventDefault();
  const fd=new FormData();
  fd.append('instruction',f.instruction.files[0]);
  for(const r of f.refs.files)fd.append('refs',r);
  const res=await fetch('/api/runs',{method:'POST',body:fd});
  if(!res.ok)alert(await res.text());
  refresh();
};
async function refresh(){
  const runs=await (await fetch('/api/runs')).json();
  let html='';
  for(const r of runs){
    const tasks=await (await fetch('/api/runs/'+r.id+'/tasks')).json();
    html+=`<h2>run ${r.id} — ${r.status}${r.error?' — '+r.error:''}</h2>`;
    if(r.status==='in_progress'||r.status==='pending'){
      const e=await (await fetch('/api/runs/'+r.id+'/eta')).json();
      const pct=e.total_tasks?Math.round(100*e.done_tasks/e.total_tasks):0;
      const waves=e.breakdown.map(b=>`${b.wave}: ${fmt(b.seconds_p50)}`).join(' · ');
      html+=`<div class=eta><progress max=100 value=${pct}></progress> `+
        `${e.done_tasks}/${e.total_tasks} tasks — ETA ${fmt(e.seconds_p50)}`+
        `–${fmt(e.seconds_p90)} <small>(${waves}) [${e.confidence}]</small></div>`;
    }
    html+='<table><tr><th>id</th><th>type</th><th>status</th><th>attempts</th><th>output</th><th>error</th></tr>';
    for(const t of tasks)html+=`<tr class=${t.status}><td>${t.id}</td><td>${t.type}</td>`+
      `<td>${t.status}</td><td>${t.attempts}</td><td>${t.output_path}</td><td>${t.error}</td></tr>`;
    html+='</table>';
  }
  document.getElementById('runs').innerHTML=html;
}
function fmt(s){if(s<60)return s+'s';if(s<3600)return Math.floor(s/60)+'m';
  return Math.floor(s/3600)+'h '+Math.floor(s%3600/60)+'m';}
refresh();setInterval(refresh,5000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/api/runs")
async def create_run(instruction: UploadFile = File(...),
                     refs: list[UploadFile] = File(default=[])):
    uploads = Path(cfg["paths"]["workspace"]) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    # Path(...).name strips any client-supplied directory parts (path traversal)
    inst_path = uploads / (Path(instruction.filename or "instruction.md").name or "instruction.md")
    inst_path.write_bytes(await instruction.read())
    ref_paths = []
    for r in refs:
        if not r.filename:
            continue
        p = uploads / Path(r.filename).name
        p.write_bytes(await r.read())
        ref_paths.append(str(p))

    run_id = start_run(cfg, conn(), inst_path, ref_paths)
    threading.Thread(target=_run_in_background, args=(run_id,), daemon=True).start()
    return {"run_id": run_id}


def _run_in_background(run_id: str):
    c = db.connect(cfg["paths"]["db"])  # own connection for this thread
    with _run_lock:  # serialize runs: shared ports, shared GPU
        try:
            execute_run(cfg, c, run_id)
        except Exception:
            pass  # status/error already recorded on the run by execute_run


@app.get("/api/runs")
def list_runs():
    rows = conn().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/runs/{run_id}/tasks")
def run_tasks(run_id: str):
    tasks = db.list_tasks(conn(), run_id)
    for t in tasks:
        t["spec"] = json.dumps(t["spec"])[:200]
    return JSONResponse(tasks)


@app.get("/api/runs/{run_id}/eta")
def run_eta(run_id: str):
    return JSONResponse(eta.estimate(conn(), run_id, cfg["scheduler"]["wave_order"]))


