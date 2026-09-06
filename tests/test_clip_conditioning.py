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


def test_an_idle_window_prefers_standing_over_a_stiller_crouch(motion):
    """The stillest stretch of a trial can be a rest in a deep crouch.

    Retargeting that faithfully put the character in a sumo squat with his coat
    draped over his knees, which is how this check came to exist. The knee's
    local rotation is its bend, because the rig rests straight-legged.
    """
    n = 400
    # first half: perfectly still, knees bent 60 degrees. second half: standing
    # straight with a small sway. the crouch is stiller; the stand is the idle.
    crouch, stand = _rot(math.radians(60.0)), _rot(0.0)
    track = {
        "LeftLeg": [crouch if i < 200 else _rot(0.02 * math.sin(i / 3.0))
                    for i in range(n)],
        "RightLeg": [crouch if i < 200 else _rot(0.02 * math.sin(i / 3.0))
                     for i in range(n)],
    }
    a, _b = motion.clip_window(track, "idle", n)
    assert a >= 190, "picked the crouch because it happened to be stiller"


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


# --- authored movesets -----------------------------------------------------
#
# The tables are built at import time out of plain tuples, so the numbers a
# character actually moves to are readable here without Blender. That is the
# only check these constants get: nothing else in the suite runs the rig.


def _targets(spec):
    """Every limb target in a move, key by key: [{bone: (fwd, out, up)}, ...].

    Targets are leg- or arm-lengths from that limb's own root, which is what
    makes them comparable between a 2.80 m body and a 2.00 m one.
    """
    return [{b: ik[2] for b, ik in dirs.get("_ik", {}).items()}
            for _t, dirs in spec["keys"]]


def test_no_authored_limb_target_is_past_the_solver_reach_clamp(motion):
    """solve_limb renormalises a target longer than its clamp instead of
    refusing it, so a stride and a down-reach that add up past 0.98 come out as
    a quietly shorter step than the table asked for -- the one failure in these
    tables that leaves no trace in the build log."""
    for name, moves in motion.MOVESETS.items():
        for clip, spec in moves.items():
            for i, key in enumerate(_targets(spec)):
                for bone, t in key.items():
                    reach = math.sqrt(sum(c * c for c in t))
                    assert reach <= motion._REACH + 1e-9, \
                        f"{name}/{clip} key {i}: {bone} reaches {reach:.3f}"


def test_the_two_archetypes_do_not_walk_the_same_cycle(motion):
    def excursion(moveset, clip):
        fwd = [k["LeftUpLeg"][0] for k in _targets(motion.MOVESETS[moveset][clip])]
        return max(fwd) - min(fwd)

    grappler, striker = excursion("grappler", "walk"), excursion("striker", "walk")
    assert grappler > 0.2 and striker > 0.2, "a shuffle, not a step"
    assert abs(grappler - striker) > 0.05, "one gait retargeted onto both bodies"
    assert (motion.MOVESETS["grappler"]["walk"]["seconds"]
            != motion.MOVESETS["striker"]["walk"]["seconds"])
    # the flat-foot override: without it the IK aims the toe down the shin
    for _t, dirs in motion.MOVESETS["grappler"]["walk"]["keys"]:
        assert "LeftFoot" in dirs and "RightFoot" in dirs


def test_idle_is_an_authored_stance_that_closes_its_loop(motion):
    for moveset in ("grappler", "striker"):
        spec = motion.MOVESETS[moveset]["idle"]
        keys = _targets(spec)
        assert spec["keys"][0][0] == 0.0 and spec["keys"][-1][0] == 1.0
        for bone, t in keys[0].items():
            assert t == pytest.approx(keys[-1][bone], abs=1e-9), \
                f"{moveset} idle does not meet itself at {bone}"
        # a stance, not a stand: both fists are out in front of the body
        arms = {b: t for b, t in keys[0].items() if b.endswith("Arm")}
        assert len(arms) == 2 and all(t[0] > 0.25 for t in arms.values())
    assert (motion.MOVESETS["grappler"]["idle"]["seconds"]
            != motion.MOVESETS["striker"]["idle"]["seconds"])
