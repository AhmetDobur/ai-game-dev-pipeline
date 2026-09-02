"""ETA engine.

Estimates come from the pipeline's own recorded history (per-task-type and
per-wave model-load durations, persisted in the durations table), aggregated
wave-aware:

- the GPU timeline is the sum of remaining waves: one model load + every
  remaining task of that wave (medians of history);
- rig_animate is network-bound and runs in parallel lanes, so it contributes
  max(per-task medians), overlapped with the GPU timeline;
- an in_progress task gets credit for time already spent;
- uncertainty: per-item spread ((p90 - p50) / 1.28 as a normal sigma) added in
  quadrature, reported as a p50..p90 band instead of one fake-precise number;
- with no history yet, calibrated defaults are used and the estimate says so.
"""
import math
import time

from . import db
from .scheduler import LANE_TYPES, LOCAL_TYPES, TASK_WAVE

# cold-start seconds per task type / model load, replaced by history as it accrues
DEFAULT_TASK_S = {"code": 240, "design_2d": 90, "design_3d": 300,
                  "audio": 30, "rig_animate": 600, "assemble": 120}
DEFAULT_LOAD_S = {"coder": 90, "sdxl": 60, "trellis": 90, "tts": 0}
DEFAULT_SPREAD = 0.5  # p90 assumed 50% above p50 when there is no history


def _stats(conn, kind: str, default_p50: float) -> tuple[float, float, bool]:
    """(p50, sigma, from_history)"""
    s = db.duration_stats(conn, kind)
    if s:
        return s["p50"], max(0.0, (s["p90"] - s["p50"]) / 1.28), True
    return default_p50, default_p50 * DEFAULT_SPREAD / 1.28, False


def estimate(conn, run_id: str, wave_order: list[str]) -> dict:
    tasks = db.list_tasks(conn, run_id)
    remaining = [t for t in tasks if t["status"] in ("pending", "in_progress")]
    done = [t for t in tasks if t["status"] == "done"]
    if not remaining:
        return {"seconds_p50": 0, "seconds_p90": 0, "confidence": "done",
                "breakdown": [], "remaining_tasks": 0,
                "done_tasks": len(done), "total_tasks": len(tasks)}

    now = time.time()
    from_history = 0
    total = 0
    gpu_p50 = 0.0
    var = 0.0
    breakdown = []

    def credit(t, p50):
        """An in_progress task gets credit for elapsed time (never below 10%)."""
        if t["status"] != "in_progress":
            return p50
        elapsed = max(0.0, now - t["updated_at"])
        return max(p50 * 0.1, p50 - elapsed)

    # GPU timeline: waves in order, one load each, plus CPU-local tasks
    for wave in wave_order:
        types = {ty for ty, w in TASK_WAVE.items() if w == wave}
        wave_tasks = [t for t in remaining if t["type"] in types]
        if not wave_tasks:
            continue
        load_p50, load_sig, load_hist = _stats(conn, f"load:{wave}",
                                               DEFAULT_LOAD_S.get(wave, 60))
        wave_s = load_p50
        var += load_sig ** 2
        from_history += load_hist
        total += 1
        for t in wave_tasks:
            p50, sig, hist = _stats(conn, t["type"], DEFAULT_TASK_S[t["type"]])
            wave_s += credit(t, p50)
            var += sig ** 2
            from_history += hist
            total += 1
        gpu_p50 += wave_s
        breakdown.append({"wave": wave, "tasks": len(wave_tasks),
                          "seconds_p50": round(wave_s)})

    for t in [t for t in remaining if t["type"] in LOCAL_TYPES]:
        p50, sig, hist = _stats(conn, t["type"], DEFAULT_TASK_S[t["type"]])
        gpu_p50 += credit(t, p50)
        var += sig ** 2
        from_history += hist
        total += 1
        breakdown.append({"wave": "local", "tasks": 1, "seconds_p50": round(p50)})

    # network lane runs alongside the GPU timeline; parallel threads -> max, not sum
    lane_p50 = 0.0
    lane_tasks = [t for t in remaining if t["type"] in LANE_TYPES]
    for t in lane_tasks:
        p50, sig, hist = _stats(conn, t["type"], DEFAULT_TASK_S[t["type"]])
        lane_p50 = max(lane_p50, credit(t, p50))
        var += sig ** 2
        from_history += hist
        total += 1
    if lane_tasks:
        breakdown.append({"wave": "meshy-lane", "tasks": len(lane_tasks),
                          "seconds_p50": round(lane_p50)})

    p50 = max(gpu_p50, lane_p50)
    p90 = p50 + 1.28 * math.sqrt(var)
    confidence = ("history" if from_history == total
                  else "defaults" if from_history == 0 else "mixed")
    return {"seconds_p50": round(p50), "seconds_p90": round(p90),
            "confidence": confidence, "breakdown": breakdown,
            "remaining_tasks": len(remaining), "done_tasks": len(done),
            "total_tasks": len(tasks)}


def fmt(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def line(conn, run_id: str, wave_order: list[str]) -> str:
    """One terminal-friendly ETA line."""
    e = estimate(conn, run_id, wave_order)
    if e["remaining_tasks"] == 0:
        return f"[eta] run {run_id}: all {e['total_tasks']} tasks done"
    waves = ", ".join(f"{b['wave']}:{fmt(b['seconds_p50'])}" for b in e["breakdown"])
    return (f"[eta] run {run_id}: {e['done_tasks']}/{e['total_tasks']} done, "
            f"{fmt(e['seconds_p50'])}–{fmt(e['seconds_p90'])} left "
            f"({waves}) [{e['confidence']}]")
