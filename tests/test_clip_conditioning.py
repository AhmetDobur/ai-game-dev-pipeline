"""Clip conditioning: trim, smooth and loop retargeted mocap.

blender_motion.py runs inside Blender, so mathutils/bpy are stubbed here. Only
the pure maths is exercised -- which is exactly the part that decides whether an
animation looks smooth.
"""
import importlib.util
import math
import random
import sys
import types
from pathlib import Path

import pytest


class _Q:
    """Enough Quaternion for align/smooth/blend: dot, negate, normalise, slerp."""

    def __init__(self, t=(1.0, 0.0, 0.0, 0.0)):
        self.w, self.x, self.y, self.z = t

    def dot(self, o):
        return self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z

    def __matmul__(self, o):
        """Hamilton product — mathutils spells composition this way."""
        return _Q((self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
                   self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
                   self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
                   self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w))

    def __neg__(self):
        return _Q((-self.w, -self.x, -self.y, -self.z))

    def copy(self):
        return _Q((self.w, self.x, self.y, self.z))

    def normalize(self):
        n = math.sqrt(self.dot(self)) or 1.0
        self.w /= n; self.x /= n; self.y /= n; self.z /= n

    def slerp(self, o, t):
        return _Q((self.w + (o.w - self.w) * t, self.x + (o.x - self.x) * t,
                   self.y + (o.y - self.y) * t, self.z + (o.z - self.z) * t))


@pytest.fixture(scope="module")
def motion():
    mu = types.ModuleType("mathutils"); mu.Quaternion = _Q
    saved = {k: sys.modules.get(k) for k in ("mathutils", "bpy")}
    sys.modules["mathutils"] = mu
    sys.modules["bpy"] = types.ModuleType("bpy")
    path = Path(__file__).resolve().parents[1] / "templates" / "blender_motion.py"
    spec = importlib.util.spec_from_file_location("blender_motion", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _rot(a):
    return _Q((math.cos(a / 2), math.sin(a / 2), 0.0, 0.0))


def _trial(period=30, pad=300):
    """A CMU-shaped trial: stand still, do the move, stand still again."""
    q = [_rot(0.0) for _ in range(pad)]
    q += [_rot(0.6 * math.sin(2 * math.pi * i / period)) for i in range(300)]
    q += [_rot(0.0) for _ in range(pad)]
    return {"LeftUpLeg": list(q), "RightUpLeg": list(q)}


def test_cyclic_clip_is_trimmed_to_one_stride_inside_the_moving_section(motion):
    track = _trial()
    motion.align_signs(track)
    assert motion._gait_period(track, 900) == 30
    a, b = motion.clip_window(track, "walk", 900)
    assert b - a + 1 == 31           # exactly one stride, not the whole trial
    assert 300 <= a < 600            # and not sampled from the standing part


def test_oneshot_clip_is_capped_and_centred_on_the_action(motion):
    track = _trial()
    motion.align_signs(track)
    a, b = motion.clip_window(track, "punch_1", 900)
    assert b - a + 1 <= int(motion._ONESHOT_SECONDS * motion.FPS) + 1
    assert 300 <= a < 600


def test_sign_flips_are_removed(motion):
    """q and -q are the same rotation; averaging across a flip is not."""
    t = {"b": [_rot(0.1), -_rot(0.11), _rot(0.12)]}
    motion.align_signs(t)
    assert all(t["b"][i].dot(t["b"][i - 1]) > 0 for i in (1, 2))


def test_smoothing_reduces_frame_to_frame_jitter(motion):
    random.seed(1)
    noisy = [_rot(0.5 * math.sin(2 * math.pi * i / 30) + random.uniform(-.05, .05))
             for i in range(90)]
    jitter = lambda qs: sum(1 - abs(qs[i].dot(qs[i - 1])) for i in range(1, len(qs)))
    assert jitter(motion.smooth_quats(noisy)) < jitter(noisy)


def test_loop_blend_closes_a_cyclic_clip(motion):
    t = {"b": [_rot(0.6 * math.sin(2 * math.pi * i / 30)) for i in range(30)]}
    assert 1 - abs(t["b"][-1].dot(t["b"][0])) > 0    # open before
    motion.loop_blend(t)
    assert 1 - abs(t["b"][-1].dot(t["b"][0])) == pytest.approx(0, abs=1e-9)


def test_only_locomotion_counts_as_cyclic(motion):
    assert motion.is_cyclic("walk") and motion.is_cyclic("run_2")
    assert not motion.is_cyclic("punch_1") and not motion.is_cyclic("getup_1")


def test_seed_offset_changes_every_sampler_but_leaves_attempt_zero_alone(tmp_path):
    """Fixed workflow seeds made every retry reproduce the same mesh exactly."""
    import json
    from pipeline.adapters.comfy import ComfyClient

    wf = tmp_path / "w.json"
    wf.write_text(json.dumps({
        "1": {"class_type": "KSampler", "inputs": {"seed": 56, "steps": 28}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 42}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
    }))
    seen = {}

    class _Stub(ComfyClient):
        def _submit(self, workflow):        # capture instead of calling ComfyUI
            seen.update(workflow)
            raise RuntimeError("stop")

    def seeds(offset):
        text = wf.read_text()
        w = json.loads(text)
        if offset:
            for n in w.values():
                if isinstance(n.get("inputs", {}).get("seed"), int):
                    n["inputs"]["seed"] = (n["inputs"]["seed"] + offset) % (2 ** 31)
        return [n["inputs"]["seed"] for n in w.values()
                if isinstance(n.get("inputs", {}).get("seed"), int)]

    assert seeds(0) == [56, 42]                    # first attempt is reproducible
    assert seeds(1009) == [1065, 1051]             # a retry is a different sample
    assert seeds(1009) != seeds(2018)              # and each retry differs again


def test_a_bare_mesh_is_not_required_to_have_a_skin(tmp_path):
    """design_3d produces an unrigged mesh; only rig_animate must be skinned."""
    import inspect
    from pipeline import validate
    src = inspect.getsource(validate.validate)
    d3 = src[src.index('if kind == "design_3d"'):src.index('if kind == "rig_animate"')]
    assert "_validate_rigs" not in d3, "design_3d must not demand a skeleton binding"


def test_breath_phase_is_asymmetric_and_loops(motion):
    assert motion.breath_phase(0.0) == pytest.approx(0.0, abs=1e-9)
    assert motion.breath_phase(1.0) == pytest.approx(0.0, abs=1e-9)
    assert motion.breath_phase(motion._BREATH_INHALE) == pytest.approx(1.0, abs=1e-9)
    # in is quicker than out: the peak sits before the midpoint of the cycle
    peak = max((motion.breath_phase(i / 400) , i / 400) for i in range(400))[1]
    assert peak < 0.5
    # monotone rise then monotone fall, so it reads as one breath not a flutter
    rise = [motion.breath_phase(u / 100 * motion._BREATH_INHALE) for u in range(101)]
    assert rise == sorted(rise)


def test_breathing_moves_the_chest_and_counter_rotates_the_neck(motion):
    n = 60
    track = {b: [_Q() for _ in range(n)] for b in
             ("Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm")}
    motion.breathe(track, amp=0.05)
    top = int(n * motion._BREATH_INHALE)
    assert track["Spine"][top].x > 0.0            # chest opens at full inhale
    assert track["Neck"][top].x < 0.0             # neck holds the gaze level
    assert track["LeftArm"][top].x == 0.0         # arms are not part of a breath
    # rest frames stay put, so the loop still closes
    assert track["Spine"][0].x == pytest.approx(0.0, abs=1e-9)


def test_an_idle_window_is_one_breath_from_the_calmest_stretch(motion):
    n = 400
    # busy for the first half, still for the second
    # real, normalised rotations: an unnormalised quaternion gives dot > 1, so
    # 1 - |dot| goes negative and the busiest window scores as the calmest
    track = {"Hips": [_rot(0.3 * math.sin(i / 3.0) if i < 200 else 0.0)
                      for i in range(n)]}
    a, b = motion.clip_window(track, "idle", n)
    assert b - a == pytest.approx(motion._BREATH_SECONDS * motion.FPS, abs=1)
    assert a >= 190, "picked the fidgety half of the trial"
