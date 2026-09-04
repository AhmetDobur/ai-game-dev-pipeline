"""Cutting named punches out of continuous boxing mocap.

The synthetic performer below stands at the origin facing +Y: hips at z=0,
shoulder joints at z=1 either side of x=0, so the frame origin is the shoulder
midpoint, torso == 1.0 and every threshold reads directly as a fraction. A hand
at body-local (lateral, reach, height) is therefore at world
(lateral, reach, 1 + height).
"""
import math

from templates.punch_mining import MOVESET, mine, select

FPS = 30


def _rig(n):
    return {"Hips": [(0.0, 0.0, 0.0)] * n,
            "LeftArm": [(-0.2, 0.0, 1.0)] * n,
            "RightArm": [(0.2, 0.0, 1.0)] * n}


def _world(local):
    lat, reach, h = local
    return (lat, reach, 1.0 + h)


def _throw(guard, impact, hold=6, lat_bow=0.0):
    """guard -> impact -> guard, with an optional sideways bow (a hook)."""
    path = []
    for t in [k / hold for k in range(hold + 1)] + \
             [1.0 - k / hold for k in range(1, hold + 1)]:
        p = tuple(g + (i - g) * t for g, i in zip(guard, impact))
        path.append(_world((p[0] + lat_bow * math.sin(math.pi * t), p[1], p[2])))
    return path


GUARD = (-0.15, 0.12, 0.05)


def _trial(*throws):
    """Stitch throws together with a beat of guard between them."""
    hand = []
    for t in throws:
        hand += [_world(GUARD)] * 8 + t
    hand += [_world(GUARD)] * 8
    return hand


def _tracks(left=None, right=None):
    """Both hands padded to the same length: a shorter track would truncate the
    other hand's punch out of the trial entirely."""
    left = list(left or [])
    right = list(right or [])
    n = max(len(left), len(right), 40)
    left += [_world(GUARD)] * (n - len(left))
    right += [_world(GUARD)] * (n - len(right))
    t = _rig(n)
    t["LeftHand"], t["RightHand"] = left, right
    return t


def test_a_straight_lead_hand_punch_is_a_jab():
    left = _trial(_throw(GUARD, (-0.05, 0.60, 0.05)))
    punches = mine(_tracks(left=left), FPS)
    assert punches, "no punch detected in an obvious jab"
    assert punches[0]["hand"] == "left"
    assert punches[0]["kind"] == "straight"
    assert punches[0]["target"] == "head"
    assert select(punches, "jab") is punches[0]


def test_a_rising_punch_is_an_uppercut_not_a_jab():
    left = _trial(_throw((-0.15, 0.12, -0.20), (-0.05, 0.50, 0.12)))
    punches = mine(_tracks(left=left), FPS)
    assert punches and punches[0]["kind"] == "uppercut"


def test_each_move_picks_its_own_exemplar_out_of_a_mixed_trial():
    """The real job: one trial holds every punch the performer threw, and each
    move has to come back with the right one."""
    left = _trial(
        _throw(GUARD, (-0.05, 0.62, 0.04)),                    # straight, level
        _throw((-0.15, 0.12, -0.22), (-0.05, 0.52, 0.14)),     # climbs
        _throw((-0.15, 0.12, 0.02), (-0.05, 0.56, -0.20)),     # to the ribs
        _throw(GUARD, (-0.05, 0.55, 0.04), lat_bow=0.30),      # bows out
    )
    punches = mine(_tracks(left=left), FPS)
    assert len(punches) == 4, [p["name"] for p in punches]
    order = {id(p): i for i, p in enumerate(punches)}
    assert order[id(select(punches, "jab"))] == 0
    assert order[id(select(punches, "left_uppercut"))] == 1
    assert order[id(select(punches, "left_bodyshot"))] == 2
    assert order[id(select(punches, "left_hook"))] == 3


def test_variant_walks_the_ranking_so_a_repeat_is_a_different_take():
    left = _trial(_throw(GUARD, (-0.05, 0.62, 0.04)),
                  _throw(GUARD, (-0.06, 0.58, 0.03)))
    punches = mine(_tracks(left=left), FPS)
    assert len(punches) == 2
    assert select(punches, "jab", 0) is not select(punches, "jab", 1)


def test_select_returns_none_when_that_hand_threw_nothing():
    punches = mine(_tracks(left=_trial(_throw(GUARD, (-0.05, 0.62, 0.04)))), FPS)
    assert select(punches, "jab") is not None
    assert select(punches, "cross") is None        # right hand never moved


def test_a_descending_punch_to_the_ribs_is_a_body_shot_not_an_overhand():
    """The trap: a body shot travels downward, so a naive "does it descend"
    test labels every one of them an overhand."""
    left = _trial(_throw((-0.15, 0.12, 0.02), (-0.05, 0.55, -0.18)))
    punches = mine(_tracks(left=left), FPS)
    assert punches and punches[0]["target"] == "body"
    assert punches[0]["kind"] != "overhand"


def test_a_punch_that_crests_overhead_and_falls_is_an_overhand():
    right = []
    for z, r in ((0.05, 0.12), (0.30, 0.25), (0.34, 0.45), (0.20, 0.58),
                 (0.06, 0.62), (0.05, 0.30), (0.05, 0.12)):
        right += [_world((0.10, r, z))] * 2
    punches = mine(_tracks(right=[_world(GUARD)] * 6 + right + [_world(GUARD)] * 6), FPS)
    assert punches and punches[0]["kind"] == "overhand"


def test_hand_separation_and_ordering():
    left = _trial(_throw(GUARD, (-0.05, 0.60, 0.05)))
    right = [_world(GUARD)] * 30 + _throw(GUARD, (0.05, 0.62, 0.05)) + \
            [_world(GUARD)] * 10
    punches = mine(_tracks(left=left, right=right), FPS)
    assert {p["hand"] for p in punches} == {"left", "right"}
    assert [p["impact"] for p in punches] == sorted(p["impact"] for p in punches)
    assert select(punches, "cross")["hand"] == "right"


def test_a_guard_fidget_is_not_a_punch():
    left = _trial(_throw(GUARD, (-0.12, 0.22, 0.05)))   # barely moves
    assert mine(_tracks(left=left), FPS) == []


def test_missing_joints_yield_nothing_rather_than_a_guess():
    assert mine({"LeftHand": [(0, 0, 0)] * 20}, FPS) == []   # no arms, no frame
    assert mine({}, FPS) == []


def test_every_requested_move_has_a_lookup_rule():
    for move in ("jab", "cross", "overhand", "left_uppercut",
                 "right_uppercut", "left_bodyshot"):
        hand, score = MOVESET[move]
        assert hand in ("left", "right")
        assert score({"reach": 1.0, "rise": 0.0, "drop": 0.0,
                      "arc": 0.0, "height": 0.0}) is not None
