"""Cut individual, named punches out of continuous boxing mocap.

CMU has real boxing (13_17, 13_18, 14_01..14_03, 15_13, 02_05, 76_01 and more),
but a trial is thirty seconds of unlabelled shadowboxing: jabs, crosses, hooks
and uppercuts run together with no marks saying which is which or when. The
existing one-shot windowing keeps "the busiest 2.5 seconds", which for a boxing
trial is an arbitrary handful of punches rather than the jab somebody asked for.

So this module segments a trial into single punches and labels each one by hand,
trajectory and target height, using only the joint positions. It is deliberately
free of bpy and of numpy: the same functions run inside Blender on imported BVH
tracks and inside pytest on hand-written ones.

Everything is measured in a body-local frame rebuilt per frame from the
shoulders and the spine, so a boxer who turns, steps or leans is described in
the same terms throughout -- world coordinates would call the same jab a
different shape every time the performer changed facing.
"""
import math

# thresholds as fractions of TORSO (hips->head), which is stable across
# performers in a way that absolute centimetres are not
_MIN_REACH = 0.42        # a punch commits the hand this far in front of the chest
_MIN_TRAVEL = 0.18       # ... and travels this far from its chambered position
_UPPERCUT_RISE = 0.16    # hand climbs this much between chamber and impact
_OVERHAND_DROP = 0.09    # ... or crests this much above the impact and comes down
_OVERHAND_CREST = 0.10   # ... from genuinely above the chest, not merely downward
_HOOK_ARC = 0.30         # lateral bow away from the chamber->impact chord
_BODY_LEVEL = -0.10      # impact below the chest by this much is a body shot
_HEAD_LEVEL = 0.02       # at or above the chest is a head shot
_MIN_GAP_S = 0.20        # two peaks closer than this are one punch, not two


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    m = math.sqrt(_dot(a, a))
    return (a[0] / m, a[1] / m, a[2] / m) if m > 1e-9 else (0.0, 0.0, 0.0)


def body_frame(left_arm, right_arm, hips):
    """Origin and an orthonormal (right, forward, up) triple for one frame.

    The origin is the midpoint of the two shoulder JOINTS, which is where
    punches actually start, and it makes "above the origin" mean "at head
    height" without any further calibration. `up` runs hips->shoulders rather
    than world Z, so a boxer slipping or ducking is still described relative to
    their own spine.

    The shoulder line comes from LeftArm/RightArm and not from
    LeftShoulder/RightShoulder: in a CMU skeleton those two are clavicles whose
    HEADS sit on the spine, 0.000 units apart, so using them produced a
    degenerate basis and silently zeroed every track. LeftArm/RightArm heads are
    the real shoulder joints, 6.96 units apart on the same performer.

    Returns None when the joints are degenerate on a malformed frame.
    """
    origin = tuple((a + b) / 2 for a, b in zip(left_arm, right_arm))
    up = _norm(_sub(origin, hips))
    right = _norm(_sub(right_arm, left_arm))
    if not any(up) or not any(right):
        return None
    # forward = up x right, NOT right x up. A performer facing +Y with up +Z has
    # right +X, and (0,0,1) x (1,0,0) = +Y; the other order points out of their
    # back, which made every measured reach on real mocap come out negative.
    forward = _norm(_cross(up, right))
    if not any(forward):
        return None
    right = _norm(_cross(forward, up))   # re-orthogonalise against a leaning spine
    return origin, right, forward, up


def local_track(hand, left_arm, right_arm, hips):
    """One hand's path in body-local coordinates: (lateral, reach, height).

    Frames whose basis could not be built inherit the previous one, so a single
    bad frame does not punch a hole in the middle of a trajectory. The returned
    scale is the mean hips->shoulders distance, which every threshold in this
    module is expressed as a fraction of.
    """
    out, torso, last = [], [], None
    for i in range(len(hand)):
        f = body_frame(left_arm[i], right_arm[i], hips[i]) or last
        if f is None:
            out.append((0.0, 0.0, 0.0))
            torso.append(1.0)
            continue
        last = f
        origin, right, forward, up = f
        d = _sub(hand[i], origin)
        out.append((_dot(d, right), _dot(d, forward), _dot(d, up)))
        t = math.sqrt(_dot(_sub(origin, hips[i]), _sub(origin, hips[i])))
        torso.append(t if t > 1e-6 else 1.0)
    return out, sum(torso) / max(len(torso), 1)


def _peaks(reach, min_gap):
    """Indices where the hand is furthest forward — one per punch.

    A strict local maximum is not enough: retargeted mocap wobbles, so a single
    thrust can register three neighbouring maxima. Peaks within min_gap frames
    collapse to the strongest, which is what separates a fast double-jab (two
    real peaks, ~0.3 s apart) from one noisy one.
    """
    cand = [i for i in range(1, len(reach) - 1)
            if reach[i] >= reach[i - 1] and reach[i] > reach[i + 1]]
    cand.sort(key=lambda i: -reach[i])
    kept = []
    for i in cand:
        if all(abs(i - j) >= min_gap for j in kept):
            kept.append(i)
    return sorted(kept)


def _chamber(reach, peak, eps=1e-3):
    """The frame the hand LEAVES guard on, which is where the punch starts.

    Walking back from the peak finds where the hand stopped coming forward, but
    a boxer holds guard for a beat first, and that hold is flat in reach. Taking
    the earliest flat frame dates the punch to the start of the hold, so the
    chamber height is sampled from a stale pose -- a hand dropped low for an
    uppercut then reads as having started at guard height, and the rise that
    defines an uppercut disappears. So: walk back to the plateau, then forward
    to its last frame, which is the moment the hand actually moves.
    """
    i = peak
    while i > 0 and reach[i - 1] <= reach[i]:
        i -= 1
    floor = reach[i]
    while i < peak and reach[i + 1] <= floor + eps:
        i += 1
    return i


def _recovery(reach, peak):
    """Where the hand finishes returning to guard after the peak."""
    i = peak
    n = len(reach)
    while i < n - 1 and reach[i + 1] <= reach[i]:
        i += 1
    return i


def _arc(track, a, b):
    """Peak sideways bow away from the straight chamber->impact chord, as a
    fraction of the chord. A hook bows; a jab does not."""
    p0, p1 = track[a], track[b]
    chord = _sub(p1, p0)
    length = math.sqrt(_dot(chord, chord))
    if length < 1e-6:
        return 0.0
    u = _norm(chord)
    worst = 0.0
    for i in range(a, b + 1):
        d = _sub(track[i], p0)
        along = _dot(d, u)
        perp = _sub(d, (u[0] * along, u[1] * along, u[2] * along))
        worst = max(worst, math.sqrt(_dot(perp, perp)))
    return worst / length


def classify(track, chamber, peak, recovery, torso):
    """Name the punch from its own trajectory.

    Order matters: an uppercut also reaches forward and a hook also rises a
    little, so the most distinctive geometry is tested first. Heights are read
    against the chest, which is the frame's origin, so "body shot" means below
    the performer's own sternum rather than below some absolute height.
    """
    rise = (track[peak][2] - track[chamber][2]) / torso
    crest = max(track[i][2] for i in range(chamber, peak + 1))
    drop = (crest - track[peak][2]) / torso
    arc = _arc(track, chamber, peak)
    height = track[peak][2] / torso

    if rise >= _UPPERCUT_RISE:
        kind = "uppercut"
    elif (drop >= _OVERHAND_DROP and crest / torso >= _OVERHAND_CREST
          and height > _BODY_LEVEL):
        # an overhand loops ABOVE the shoulder and falls onto the head. Testing
        # only for "descends" would label every straight body shot an overhand,
        # since a punch aimed at the ribs also travels downward.
        kind = "overhand"
    elif arc >= _HOOK_ARC:
        kind = "hook"
    else:
        kind = "straight"
    target = "body" if height <= _BODY_LEVEL else (
        "head" if height >= _HEAD_LEVEL else "mid")
    return {"kind": kind, "target": target, "rise": round(rise, 4),
            "drop": round(drop, 4), "arc": round(arc, 4), "height": round(height, 4),
            "reach": round(track[peak][1] / torso, 4)}


def mine_hand(track, torso, hand, fps):
    """Every punch thrown by one hand, as labelled frame windows."""
    reach = [p[1] for p in track]
    if not reach:
        return []
    out = []
    for peak in _peaks(reach, max(1, int(_MIN_GAP_S * fps))):
        chamber = _chamber(reach, peak)
        recovery = _recovery(reach, peak)
        if peak - chamber < 2 or recovery - peak < 1:
            continue
        if reach[peak] / torso < _MIN_REACH:
            continue                       # a guard adjustment, not a punch
        if (reach[peak] - reach[chamber]) / torso < _MIN_TRAVEL:
            continue                       # the hand was already out there
        p = classify(track, chamber, peak, recovery, torso)
        p.update(hand=hand, start=chamber, impact=peak, end=recovery,
                 name=f"{hand}_{p['kind']}_{p['target']}")
        out.append(p)
    return out


def mine(tracks, fps=30):
    """Every punch in a trial, both hands, ordered by time.

    `tracks` maps CMU/Mixamo bone names to per-frame world positions. Missing
    joints mean this is not a trial we can read, and an empty list sends the
    caller back to its existing behaviour rather than inventing a window.
    """
    need = ("LeftHand", "RightHand", "LeftArm", "RightArm", "Hips")
    if not all(tracks.get(b) for b in need):
        return []
    n = min(len(tracks[b]) for b in need)
    if n < 8:
        return []
    found = []
    for hand, bone in (("left", "LeftHand"), ("right", "RightHand")):
        track, torso = local_track(tracks[bone][:n], tracks["LeftArm"][:n],
                                   tracks["RightArm"][:n], tracks["Hips"][:n])
        found.extend(mine_hand(track, torso, hand, fps))
    found.sort(key=lambda p: p["impact"])
    return found


# The moveset the game asks for, expressed as what to look for in the mined set.
# A jab is the lead (left) hand going straight to the head; a cross is the same
# shape from the rear hand. Where an exact match is missing from a trial the
# fallbacks degrade along the axis that matters least: a head hook still reads
# as a cross far better than an uppercut does.
MOVESET = {
    "jab":             [("left", "straight", "head"), ("left", "straight", "mid")],
    "cross":           [("right", "straight", "head"), ("right", "straight", "mid")],
    "overhand":        [("right", "overhand", None), ("right", "hook", "head")],
    "left_uppercut":   [("left", "uppercut", None)],
    "right_uppercut":  [("right", "uppercut", None)],
    "left_bodyshot":   [("left", "straight", "body"), ("left", "hook", "body"),
                        ("left", None, "body")],
    "left_hook":       [("left", "hook", None)],
    "right_hook":      [("right", "hook", None)],
}


def select(punches, move, variant=0):
    """Pick one mined punch for a named move, or None.

    Preference order inside a tier is by reach: the most committed example of a
    move looks the most deliberate on a character whose proportions differ from
    the performer's. `variant` walks further down that list so a combo's second
    jab is a different take of a jab rather than the same clip replayed.
    """
    for hand, kind, target in MOVESET.get(move, []):
        tier = [p for p in punches
                if p["hand"] == hand
                and (kind is None or p["kind"] == kind)
                and (target is None or p["target"] == target)]
        if tier:
            # most committed example first: reach for everything except an
            # uppercut, which is defined by how far it climbs, not how far out
            tier.sort(key=lambda p: -(p["rise"] if kind == "uppercut" else p["reach"]))
            return tier[variant % len(tier)]
    return None
