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

    python make_voice.py <out_dir> [--voice leo] [--url http://127.0.0.1:8090]
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
LINES = {
    "intro_praise":   "I only praise Him.",
    "intro_name":     "I raise no hand but in His name.",
    "intro_lent":     "Strength is lent. Never owned.",
    "victory_glory":  "I seek no glory. It was never mine to keep.",
    "victory_mercy":  "Mercy first. Then force.",
}


def _post(url, path, obj, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def codes_for(text, voice, url, n_predict=700, temperature=0.6):
    """Ask the model for one line and return its SNAC codes, frame-aligned."""
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


def trim(samples, sr=SAMPLE_RATE, floor=0.06, gap=0.35, tail=0.08):
    """Cut the line back to the line.

    Orpheus does not reliably emit its end-of-speech token, and when it does not
    it keeps talking until n_predict runs out -- a four-word line came back as
    13.6 seconds, with the actual words in the first two. Waiting for the stop
    token is therefore not enough; the recording has to be cut on what is in it.
    Speech ends at the first sustained silence after it starts, which is exactly
    what a run-on has and a clean take does not.
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
    breaks = np.flatnonzero(np.diff(loud) > int(gap * sr))
    if breaks.size:
        end = loud[breaks[0]]
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


def generate(out_dir, voice="leo", url="http://127.0.0.1:8090", lines=None):
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (lines or LINES).items():
        codes = codes_for(text, voice, url)
        if len(codes) < FRAME:
            print(f"[voice] {name}: model returned no audio", flush=True)
            continue
        path = out / f"voice_{name}.wav"
        write_wav(path, trim(decode(codes)))
        written.append(path)
        print(f"[voice] {name}: {len(codes) // FRAME} frames -> {path}", flush=True)
    return written


if __name__ == "__main__":
    args = sys.argv[1:]
    dest = args[0] if args else "."
    voice = args[args.index("--voice") + 1] if "--voice" in args else "leo"
    url = args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:8090"
    generate(dest, voice, url)
