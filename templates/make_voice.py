"""Character voice lines from Orpheus 3B, decoded through SNAC, written as wav.

The strike sounds next door in make_sfx.py are synthesised from stdlib DSP, but
a voice is not something a few hundred lines of oscillators can fake. Orpheus is
Apache-2.0 and SNAC is MIT, both run locally, and neither costs anything per
line -- so the voice lines are generated here on the same terms as everything
else in this pipeline.

Orpheus does not emit audio. It emits SNAC codec tokens as ordinary vocabulary
entries, which llama.cpp hands back as token ids, and SNAC turns those into
24 kHz samples. The layout below was read off a live server rather than taken
from a reference implementation:

    id 128256 + n  <->  "<custom_token_n>"
    the first two generated tokens are markers (custom_token_5, custom_token_1)
    every audio token then satisfies  code = id - 128266 - (index % 7) * 4096
    and lands inside 0..4095, in frames of seven

    python make_voice.py <out_dir> [--character pious_force]
                         [--url http://127.0.0.1:8090]
"""
import json
import sys
import urllib.request
import wave

# Orpheus emits seven codes per frame, split across SNAC's three codebooks at
# this stride: one coarse code, two mid, four fine.
FRAME = 7
CODE_BASE = 128266
CODE_SPAN = 4096
STOP = 128258
SAMPLE_RATE = 24000

# The pious grappler's lines. Written to be devout rather than to quote: a fight
# taunt built out of scripture is the thing to avoid, so these are a believer's
# own words about where he thinks his strength comes from.
#
# The striker next to him is not religious -- that register belongs to one
# character, not to the game -- so hers are about not being seen.
CHARACTERS = {
    "pious_force": {
        "voice": "leo",
        "lines": {
            "intro_praise":  "I only praise Him.",
            "intro_name":    "I raise no hand but in His name.",
            "intro_lent":    "Strength is lent. Never owned.",
            "victory_glory": "It was never mine to keep.",
            "victory_mercy": "Mercy first. Then force.",
        },
    },
    "veiled_shadow": {
        "voice": "tara",
        "lines": {
            "intro_unseen":  "You will not see the one that lands.",
            "intro_watch":   "Watch closely. It will not help.",
            "victory_never": "You were never fighting me.",
            "victory_kept":  "Nothing to see. As promised.",
        },
    },
}

# Striking and struck. Both fighters make the same noises -- what differs is the
# voice making them, which is the whole reason these are generated per character
# rather than shipped as one shared folder of grunts.
#
# These are Orpheus's own emotion tags rather than spelled-out interjections,
# because spelling them does not work. Measured: <groan> came back as 1.23s of
# real audio, while "Hah!", "Hyah!", "Agh!" and "Ngh!" all came back as takes of
# the right LENGTH and complete digital silence -- peak 0.00 -- which is why
# every take is now checked for signal and not just for duration.
#
# Variety comes from asking again rather than from a longer list: the model is
# sampled, so three requests for <gasp> are three different gasps.
GRUNTS = {"effort": "<gasp>", "hurt": "<groan>"}
GRUNT_TAKES = 3

# A take can be the right length and still be nothing at all. 0.05 is well below
# any real utterance here (the lines peak at 0.44-0.67) and well above the noise
# that a silent take rounds to.
MIN_PEAK = 0.05

LINES = CHARACTERS["pious_force"]["lines"]


def _post(url, path, obj, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def budget_tokens(text, headroom=1.2):
    """How many tokens this line could possibly need, from its own word count.

    The run-ons are not really a trimming problem, they are a budget problem:
    asked for 700 tokens the model will happily fill all 8.5 seconds of them
    with a three-word line, and then the trimmer is left guessing which of the
    silences inside that was the end. Sized from the text there is nothing to
    guess about -- a line gets its own length plus generous headroom, because
    clipping the last word is worse than a little trailing silence.
    """
    seconds = len(text.split()) * 0.45 + headroom
    return int(seconds * SAMPLE_RATE / 2048) * FRAME


def codes_for(text, voice, url, n_predict=None, temperature=0.6):
    """Ask the model for one line and return its SNAC codes, frame-aligned."""
    n_predict = n_predict or budget_tokens(text)
    ids = _post(url, "/tokenize", {"content": f"{voice}: {text}"})["tokens"]
    # 128259 opens the utterance; 128009 (end of turn) and 128260 close the
    # text and hand over to audio. Sent as ids rather than as a string so the
    # server cannot decide whether to parse the specials.
    out = _post(url, "/completion", {
        "prompt": [128259] + ids + [128009, 128260],
        "n_predict": n_predict, "temperature": temperature, "top_p": 0.9,
        "repeat_penalty": 1.1, "return_tokens": True, "cache_prompt": False,
    })
    codes = []
    for tok in out.get("tokens", []):
        if tok == STOP:
            break
        code = tok - CODE_BASE - (len(codes) % FRAME) * CODE_SPAN
        # The two leading markers and anything else out of range are not audio.
        # Testing the decoded value rather than the raw id is what keeps a
        # marker from being read as a plausible code in slot 0.
        if 0 <= code < CODE_SPAN:
            codes.append(code)
    return codes[: len(codes) - len(codes) % FRAME]


def decode(codes):
    """SNAC's three codebooks, then samples."""
    import torch
    from snac import SNAC

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    coarse, mid, fine = [], [], []
    for f in range(len(codes) // FRAME):
        c = codes[f * FRAME: f * FRAME + FRAME]
        coarse.append(c[0])
        mid += [c[1], c[4]]
        fine += [c[2], c[3], c[5], c[6]]
    layers = [torch.tensor(x, dtype=torch.int32).unsqueeze(0)
              for x in (coarse, mid, fine)]
    with torch.inference_mode():
        return model.decode(layers).squeeze().cpu().numpy()


# Speech runs at roughly this rate, and a take shorter than the words it was
# asked for has been cut in the middle of them.
SECONDS_PER_WORD = 0.28


def trim(samples, sr=SAMPLE_RATE, floor=0.06, gap=0.6, tail=0.08,
         min_seconds=0.0):
    """Cut the line back to the line.

    Orpheus does not reliably emit its end-of-speech token, and when it does not
    it keeps talking until n_predict runs out -- a four-word line came back as
    13.6 seconds, with the actual words in the first two. Waiting for the stop
    token is therefore not enough; the recording has to be cut on what is in it.
    Speech ends at a sustained silence, which is exactly what a run-on has and a
    clean take does not.

    A silence is not automatically the end, though. "Strength is lent. Never
    owned." has a full stop in the middle of it, and cutting at the first pause
    took that line down to 0.45 seconds -- one word and a breath. So the cut has
    to clear min_seconds, which the caller derives from how many words it asked
    for; below that the pause is punctuation, not the end of the take.
    """
    import numpy as np

    win = max(1, sr // 20)
    rms = np.sqrt(np.maximum(0.0, np.convolve(samples * samples,
                                              np.ones(win) / win, "same")))
    peak = rms.max()
    if peak <= 0.0:
        return samples
    loud = np.flatnonzero(rms > peak * floor)
    if loud.size == 0:
        return samples
    start = loud[0]
    end = loud[-1]
    for b in np.flatnonzero(np.diff(loud) > int(gap * sr)):
        if (loud[b] - start) / sr >= min_seconds:
            end = loud[b]
            break
    lo = max(0, start - int(0.03 * sr))
    hi = min(len(samples), end + int(tail * sr))
    return samples[lo:hi]


def write_wav(path, samples):
    import numpy as np

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def generate(out_dir, voice="leo", url="http://127.0.0.1:8090", lines=None,
             bounds=None):
    """Write one take per line. Returns the names that came back usable.

    `bounds` is (min_seconds, max_seconds) as a function of the text -- a take
    outside them is not written at all. Both ends matter and for opposite
    reasons: too long is the model carrying on past the line, too short is a
    take cut off inside it, and both are more common than a clean one.
    """
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = {}
    for name, text in (lines or LINES).items():
        codes = codes_for(text, voice, url)
        if len(codes) < FRAME:
            print(f"[voice] {name}: model returned no audio", flush=True)
            continue
        words = len(text.split())
        samples = trim(decode(codes), min_seconds=words * SECONDS_PER_WORD)
        seconds = len(samples) / SAMPLE_RATE
        lo, hi = bounds(text) if bounds else (0.0, float("inf"))
        if not lo <= seconds <= hi:
            print(f"[voice] {name}: {seconds:.2f}s, outside {lo:.2f}-{hi:.2f}s "
                  f"-- dropped", flush=True)
            continue
        peak = float(abs(samples).max()) if len(samples) else 0.0
        if peak < MIN_PEAK:
            print(f"[voice] {name}: silent take (peak {peak:.3f}) -- dropped",
                  flush=True)
            continue
        path = out / f"voice_{name}.wav"
        write_wav(path, samples)
        kept[name] = path
        print(f"[voice] {name}: {seconds:.2f}s -> {path}", flush=True)
    return kept


def _line_bounds(text):
    words = len(text.split())
    return words * 0.22, words * 0.75 + 1.0


def _grunt_bounds(text):
    # A groan legitimately runs past a second; a gasp on a jab is a fraction of
    # one. A single cap tight enough for the punch threw away every usable
    # groan at 1.3 and 1.5s.
    return (0.08, 1.6) if "groan" in text else (0.08, 0.9)


def generate_character(out_dir, name, url="http://127.0.0.1:8090", tries=3):
    """Every line plus the grunts for one fighter, into out_dir/<name>/.

    Retried, lines as well as grunts. The model misses in both directions often
    enough that asking once leaves a character with half a voice -- and a fight
    where one side grunts and the other does not is worse than neither.
    """
    from pathlib import Path

    spec = CHARACTERS.get(name)
    if spec is None:
        raise KeyError(f"no voice set for {name!r}; have {sorted(CHARACTERS)}")
    dest = Path(out_dir) / name
    # A dropped take leaves the previous run's file behind, and the game would
    # happily play an 8-second victory line that this run rejected.
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("voice_*.wav"):
        stale.unlink()
    grunts = {f"{kind}_{i}": text
              for kind, text in GRUNTS.items()
              for i in range(1, GRUNT_TAKES + 1)}
    written = {}
    for pending, bounds in ((dict(spec["lines"]), _line_bounds),
                            (grunts, _grunt_bounds)):
        for _ in range(tries):
            if not pending:
                break
            kept = generate(dest, spec["voice"], url, pending, bounds)
            written.update(kept)
            for got in kept:
                pending.pop(got, None)
        for missing in pending:
            print(f"[voice] {missing}: no usable take in {tries} tries",
                  flush=True)
    return list(written.values())


if __name__ == "__main__":
    args = sys.argv[1:]
    dest = args[0] if args else "."
    url = args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:8090"
    if "--character" in args:
        generate_character(dest, args[args.index("--character") + 1], url)
    else:
        for name in CHARACTERS:
            generate_character(dest, name, url)
