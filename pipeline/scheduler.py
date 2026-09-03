"""Stage 2: the wave scheduler.

Core policy (the whole point of this module): minimize model switches. One model is
loaded at a time; a wave drains EVERY ready task of that model's type, validating
in-wave so retries reuse the already-loaded model instead of costing a reload.
The motion wave (Blender + UniRig/Kimodo) is GPU-bound, so it runs serially like
every other wave — one thing on the 24GB card at a time. assemble tasks are
CPU-only and run between waves.
"""
import re
import time
import traceback
from pathlib import Path
from typing import Callable

from . import db
from .validate import validate

# task type -> wave name (which loaded resource serves it)
TASK_WAVE = {
    "code": "coder",
    "design_2d": "sdxl",
    "design_3d": "trellis",
    "rig_animate": "motion",      # local Blender rig+animate, own GPU wave
    "audio": "tts",
}
LOCAL_TYPES = {"assemble"}        # CPU subprocess, no model

# Executor: (task, out_dir) -> list of produced Paths.
Executor = Callable[[dict, Path], list[Path]]


class Scheduler:
    def __init__(self, conn, run_id: str, executors: dict[str, Executor],
                 workspace: Path, max_attempts: int = 3,
                 wave_order: list[str] | None = None,
                 wave_setup: dict[str, Callable[[], None]] | None = None,
                 wave_teardown: dict[str, Callable[[], None]] | None = None,
                 godot_binary: str = "godot"):
        self.conn, self.run_id = conn, run_id
        self.executors = executors
        self.workspace = workspace
        self.max_attempts = max_attempts
        self.wave_order = wave_order or ["coder", "sdxl", "trellis", "motion", "tts"]
        self.wave_setup = wave_setup or {}
        self.wave_teardown = wave_teardown or {}
        self.godot_binary = godot_binary
        self._heals = 0

    # -- public ---------------------------------------------------------------

    def run(self) -> None:
        # crash-safe resume: anything left in_progress by a kill/power cut goes
        # back to pending (attempts survive, so crash-loops still exhaust)
        reclaimed = db.reclaim_stale(self.conn, self.run_id)
        if reclaimed:
            print(f"[resume] run {self.run_id}: reclaimed {reclaimed} interrupted task(s)")
        db.set_run_status(self.conn, self.run_id, "in_progress")
        try:
            while not db.run_finished(self.conn, self.run_id):
                progressed = self._one_cycle()
                if not progressed:
                    # a zero-progress cycle is deterministic: anything still ready
                    # has no serving wave — fail it visibly instead of spinning
                    for t in db.ready_tasks(self.conn, self.run_id):
                        db.update_task(self.conn, t["id"], status="failed",
                                       error=f"no wave serves task type {t['type']!r}"
                                             f" (wave_order={self.wave_order})")
                    break
            tasks = db.list_tasks(self.conn, self.run_id)
            failed = [t for t in tasks if t["status"] == "failed"]
            blocked = [t for t in tasks if t["status"] == "pending"]
            if failed or blocked:
                db.set_run_status(self.conn, self.run_id, "failed",
                                  f"{len(failed)} failed, {len(blocked)} blocked")
            else:
                db.set_run_status(self.conn, self.run_id, "done")
        except Exception as e:
            db.set_run_status(self.conn, self.run_id, "failed",
                              f"{e}\n{traceback.format_exc()[-1500:]}")
            raise

    # -- internals ------------------------------------------------------------

    def _one_cycle(self) -> bool:
        """One pass over local tasks and every wave. Returns True if any task ran."""
        progressed = self._run_tasks_of_types(LOCAL_TYPES)

        for wave in self.wave_order:
            types = {t for t, w in TASK_WAVE.items() if w == wave}
            ready = [t for t in db.ready_tasks(self.conn, self.run_id)
                     if t["type"] in types]
            if not ready:
                continue
            setup = self.wave_setup.get(wave)
            teardown = self.wave_teardown.get(wave)
            if setup:
                t0 = time.monotonic()
                setup()  # load the model once for the whole wave
                db.record_duration(self.conn, f"load:{wave}", time.monotonic() - t0)
            try:
                while ready:
                    for task in ready:
                        self._execute_with_retries(task)
                        progressed = True
                    # dependents of just-finished tasks may now be ready in the SAME wave
                    ready = [t for t in db.ready_tasks(self.conn, self.run_id)
                             if t["type"] in types]
            finally:
                if teardown:
                    teardown()  # unload before the next wave
        return progressed

    def _run_tasks_of_types(self, types: set[str]) -> bool:
        ran = False
        for task in [t for t in db.ready_tasks(self.conn, self.run_id)
                     if t["type"] in types]:
            self._execute_with_retries(task)
            ran = True
        return ran

    def _execute_with_retries(self, task: dict) -> None:
        db.update_task(self.conn, task["id"], status="in_progress")
        executor = self.executors.get(task["type"])
        if executor is None:
            db.update_task(self.conn, task["id"], status="failed",
                           error=f"no executor for type {task['type']!r}")
            return
        out_dir = self.workspace / "artifacts" / task["id"]
        last_error = task.get("error", "")
        t0 = time.monotonic()
        # attempts persist across process death: resume continues the count
        # instead of granting a crash-looping task infinite retries
        # code retries are cheap (no model reload) and attempt 3 is often nearly
        # right — give the coder more room than the asset stages
        cap = self.max_attempts + 2 if task["type"] == "code" else self.max_attempts
        for attempt in range(task.get("attempts", 0) + 1, cap + 1):
            db.update_task(self.conn, task["id"], attempts=attempt)
            try:
                # pass the previous failure so the executor can prompt a fix,
                # and each dependency's outputs so data flows along edges
                task_view = dict(task)
                task_view["last_error"] = last_error
                dep_rows = {dep: db.get_task(self.conn, dep)
                            for dep in task["depends_on"]}
                task_view["dep_outputs"] = {
                    dep: (dt["output_path"].split(";") if dt and dt["output_path"] else [])
                    for dep, dt in dep_rows.items()
                }
                task_view["dep_types"] = {dep: (dt["type"] if dt else "")
                                          for dep, dt in dep_rows.items()}
                task_view["dep_specs"] = {dep: (dt["spec"] if dt else {})
                                          for dep, dt in dep_rows.items()}
                outputs = executor(task_view, out_dir)
                ok, detail = validate(task, outputs, godot_binary=self.godot_binary,
                                      project_dir=self.workspace / "game")
            except Exception as e:  # executor crash counts as a failed attempt
                ok, detail, outputs = False, f"{type(e).__name__}: {e}", []
            if ok:
                db.update_task(self.conn, task["id"], status="done",
                               output_path=";".join(str(p) for p in outputs), error="")
                db.record_duration(self.conn, task["type"], time.monotonic() - t0)
                self._print_eta()
                return
            last_error = detail
        db.update_task(self.conn, task["id"], status="failed",
                       error=last_error or f"exhausted {self.max_attempts} attempts")
        if task["type"] == "assemble":
            self._heal_from_boot_log(task["id"], last_error or "")
        self._print_eta()

    def _heal_from_boot_log(self, assemble_id: str, boot_log: str) -> None:
        """Self-heal: a boot failure that names a generated script re-opens THAT
        code task with the runtime error as its fix note, then re-opens assemble.
        The wave loop re-runs the coder wave on the next cycle. Bounded so a
        script the coder can never fix doesn't loop the run forever."""
        if self._heals >= 2:
            return
        scripts = set(re.findall(r"res://([\w/.-]+\.gd)", boot_log))
        if not scripts:
            return
        healed = []
        for t in db.list_tasks(self.conn, self.run_id):
            if t["type"] == "code" and t["status"] == "done" and any(
                    t["output_path"].replace("\\", "/").endswith(s) for s in scripts):
                db.update_task(self.conn, t["id"], status="pending", attempts=0,
                               error=f"the exported game failed at runtime:\n{boot_log[-1500:]}")
                healed.append(t["id"])
        if healed:
            self._heals += 1
            db.update_task(self.conn, assemble_id, status="pending", attempts=0, error="")
            print(f"[heal] boot failure -> re-coding {healed}, retrying assemble "
                  f"(heal {self._heals}/2)", flush=True)

    def _print_eta(self) -> None:
        from . import eta  # local import: eta imports this module's wave maps
        try:
            print(eta.line(self.conn, self.run_id, self.wave_order), flush=True)
        except Exception:
            pass  # an ETA hiccup must never take down the run
