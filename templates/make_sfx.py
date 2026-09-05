"""Combat sound effects, synthesised locally. No samples, no library, no fees.

A fight sound is three things stacked: a transient that says something
connected, a body that says how heavy it was, and a tail that says what it hit.
Synthesising them beats shipping a sample pack because every strike in
combat.gd's frame table can get a sound tuned to its own weight, and because a
generated wav costs nothing and can be regenerated for any character.

The vocal layer -- the grunt going out and the groan coming back -- is separate,
in orpheus_vocals.py, because it needs a model on disk. This module never does,
so a project always gets its combat audio even when that model is missing.

Run standalone:  python make_sfx.py <out_dir>
"""
import array
import math
import os
import random
import struct
import wave

RATE = 44100


def _noise(n, rng):
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def _lowpass(xs, cutoff_hz, rate=RATE):
    """One-pole lowpass. A biquad would be sharper, and a punch does not need
    it -- the ear reads the envelope, not the skirt of the filter."""
    a = math.exp(-2.0 * math.pi * cutoff_hz / rate)
    out, prev = [], 0.0
    for x in xs:
        prev = (1.0 - a) * x + a * prev
        out.append(prev)
    return out


def _highpass(xs, cutoff_hz, rate=RATE):
    low = _lowpass(xs, cutoff_hz, rate)
    return [x - l for x, l in zip(xs, low)]


def _env(n, attack, decay, curve=3.0):
    """Percussive envelope: near-instant attack, exponential fall."""
    at = max(1, int(attack * RATE))
    out = []
    for i in range(n):
        if i < at:
            out.append(i / at)
        else:
            t = (i - at) / max(1.0, decay * RATE)
            out.append(math.exp(-curve * t))
    return out


def _sweep(n, f0, f1, rate=RATE):
    """Sine sweeping f0 -> f1 exponentially: the 'body' of an impact."""
    out, phase = [], 0.0
    for i in range(n):
        t = i / max(1, n - 1)
        f = f0 * ((f1 / f0) ** t)
        phase += 2.0 * math.pi * f / rate
        out.append(math.sin(phase))
    return out


def _mix(*layers):
    n = max(len(x) for x in layers)
    out = [0.0] * n
    for layer in layers:
        for i, v in enumerate(layer):
            out[i] += v
    return out


def _normalise(xs, peak=0.89):
    """Leave headroom: several of these play at once during a combo, and a wav
    normalised to 1.0 clips the moment a second one lands on top of it."""
    hi = max((abs(x) for x in xs), default=0.0)
    if hi < 1e-9:
        return xs
    k = peak / hi
    return [x * k for x in xs]


def impact(weight=1.0, brightness=1.0, seed=0, tail=0.0):
    """One strike landing.

    weight     0.4 jab .. 1.6 overhand -- lower body frequency, longer decay
    brightness 0.6 body shot .. 1.4 head shot -- how much snap sits on top
    tail       how much cloth/robe rustle follows the hit
    """
    rng = random.Random(seed)
    dur = 0.14 + 0.10 * weight
    n = int(dur * RATE)

    # the crack: filtered noise, gone in a few tens of milliseconds
    snap = _highpass(_noise(n, rng), 1200.0 / max(0.3, brightness))
    snap = [s * e for s, e in zip(snap, _env(n, 0.001, 0.035 * brightness, 6.0))]

    # the body: a short pitched thump that falls as it hits
    lo = 190.0 / weight
    body = _sweep(n, lo, lo * 0.45)
    body = [b * e for b, e in zip(body, _env(n, 0.002, 0.09 * weight, 4.0))]

    # the meat: low broadband, what makes it sound like a person and not a drum
    flesh = _lowpass(_noise(n, rng), 520.0)
    flesh = [f * e for f, e in zip(flesh, _env(n, 0.001, 0.06 * weight, 5.0))]

    layers = [[s * 0.55 * brightness for s in snap],
              [b * 0.80 for b in body],
              [f * 0.70 for f in flesh]]
    if tail > 0.0:
        m = int(0.28 * RATE)
        rustle = _highpass(_lowpass(_noise(m, rng), 5200.0), 900.0)
        rustle = [r * e * tail * 0.28
                  for r, e in zip(rustle, _env(m, 0.010, 0.16, 2.2))]
        layers.append(rustle)
    return _normalise(_mix(*layers))


def whiff(weight=1.0, seed=0):
    """A strike moving through air and missing. No transient at all -- that is
    the whole difference between a miss and a hit, and it is what tells a player
    their punch did not land without them having to watch the health bar."""
    rng = random.Random(seed)
    n = int((0.20 + 0.08 * weight) * RATE)
    air = _lowpass(_highpass(_noise(n, rng), 500.0), 2600.0 * weight)
    # swells and falls away: loudest as the fist passes the camera
    swell = [math.sin(math.pi * (i / max(1, n - 1))) ** 1.6 for i in range(n)]
    return _normalise([a * s for a, s in zip(air, swell)], peak=0.55)


def block(seed=0):
    """Forearm on forearm: brighter and shorter than flesh, no low body."""
    rng = random.Random(seed)
    n = int(0.12 * RATE)
    click = _highpass(_noise(n, rng), 2200.0)
    click = [c * e for c, e in zip(click, _env(n, 0.0008, 0.020, 7.0))]
    thud = _sweep(n, 260.0, 150.0)
    thud = [t * e for t, e in zip(thud, _env(n, 0.002, 0.045, 5.0))]
    return _normalise(_mix([c * 0.9 for c in click], [t * 0.45 for t in thud]))


def write_wav(path, samples, rate=RATE):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = array.array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in samples))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


# Matched to combat.gd's FRAMES table: a jab is 4 damage and an overhand is 11,
# and the sound has to agree with the number or the hit reads as weightless.
STRIKES = {
    "jab":            dict(weight=0.5, brightness=1.35, tail=0.5),
    "cross":          dict(weight=0.8, brightness=1.20, tail=0.7),
    "left_bodyshot":  dict(weight=0.9, brightness=0.60, tail=0.9),
    "left_uppercut":  dict(weight=1.1, brightness=1.10, tail=0.8),
    "right_uppercut": dict(weight=1.2, brightness=1.10, tail=0.9),
    "overhand":       dict(weight=1.6, brightness=0.95, tail=1.0),
}


def generate(out_dir, seed=1):
    """Write every combat wav a scaffolded project expects. Returns the paths."""
    made = []
    for i, (name, kw) in enumerate(sorted(STRIKES.items())):
        made.append(write_wav(os.path.join(out_dir, f"hit_{name}.wav"),
                              impact(seed=seed + i, **kw)))
        made.append(write_wav(os.path.join(out_dir, f"whiff_{name}.wav"),
                              whiff(weight=kw["weight"], seed=seed + 100 + i)))
    made.append(write_wav(os.path.join(out_dir, "block.wav"), block(seed=seed)))
    return made


def _demo():
    """Self-check: the shapes that make these read as impacts, asserted."""
    hit = impact(weight=1.0, brightness=1.0, seed=7)
    assert 0.85 < max(abs(x) for x in hit) <= 0.9, "not normalised with headroom"
    head = int(0.01 * RATE)
    assert max(abs(x) for x in hit[:head]) > 0.5, "no attack transient"
    assert max(abs(x) for x in hit[-head:]) < 0.2, "does not decay away"

    light, heavy = impact(weight=0.5, seed=1), impact(weight=1.6, seed=1)
    assert len(heavy) > len(light), "a heavier strike must ring longer"

    air = whiff(seed=3)
    quarter = len(air) // 4
    assert max(abs(x) for x in air[:quarter]) < max(abs(x) for x in air[quarter:3 * quarter]), \
        "a whiff must swell rather than crack -- that is what marks it a miss"
    print("make_sfx self-check ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for p in generate(sys.argv[1]):
            print("wrote", p)
    else:
        _demo()
