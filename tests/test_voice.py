"""trim() decides where a generated line ends, and it is the only thing standing
between the game and a four-word taunt that runs for thirteen seconds."""
import numpy as np

from templates.make_voice import SAMPLE_RATE, SECONDS_PER_WORD, trim


def _tone(seconds, amp=1.0):
    n = int(seconds * SAMPLE_RATE)
    return np.sin(np.arange(n) / 40.0) * amp


def _silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE))


def test_a_sentence_pause_is_not_the_end_of_the_line():
    """"Strength is lent. Never owned." came back cut to 0.45s -- one word and a
    breath -- because the full stop in the middle reads as silence."""
    take = np.concatenate([_tone(0.7), _silence(0.4), _tone(0.9),
                           _silence(1.2), _tone(3.0)])          # the run-on
    words = 5
    out = trim(take, min_seconds=words * SECONDS_PER_WORD)
    seconds = len(out) / SAMPLE_RATE
    assert 1.9 < seconds < 2.4, seconds     # both clauses, not the run-on


def test_a_run_on_is_still_cut():
    take = np.concatenate([_tone(0.6), _silence(1.5), _tone(6.0)])
    out = trim(take, min_seconds=2 * SECONDS_PER_WORD)
    assert len(out) / SAMPLE_RATE < 1.0


def test_a_clean_take_is_left_alone():
    take = np.concatenate([_silence(0.1), _tone(1.4), _silence(0.1)])
    out = trim(take, min_seconds=4 * SECONDS_PER_WORD)
    assert 1.3 < len(out) / SAMPLE_RATE < 1.7


def test_the_request_is_sized_to_the_line():
    """A three-word line asked for 700 tokens gets 8.5 seconds of rope and hangs
    itself with it -- which is what produced an 8.31s take of "I only praise
    Him." Sized from the text, the run-on has nowhere to go."""
    from templates.make_voice import FRAME, budget_tokens

    short = budget_tokens("I only praise Him.")
    long = budget_tokens("I raise no hand but in His name.")
    assert short % FRAME == 0 and long % FRAME == 0     # whole SNAC frames
    assert short < long < 700
    # enough rope for the words themselves, or the last one is clipped off
    assert short / FRAME * 2048 / SAMPLE_RATE > 4 * SECONDS_PER_WORD


def test_a_silent_take_is_not_a_take(tmp_path, monkeypatch):
    """The failure that got through: right length, complete digital silence.

    "Hah!", "Hyah!", "Agh!" and "Ngh!" each came back at a plausible 0.15-0.34s
    and peaked at 0.00 -- four grunts that passed every duration check and made
    no sound. Only Orpheus's own <groan> tag produced audio."""
    from templates import make_voice

    takes = iter([np.zeros(int(0.3 * SAMPLE_RATE)), _tone(0.3, 0.5)])
    monkeypatch.setattr(make_voice, "codes_for", lambda *a, **k: [0] * 7)
    monkeypatch.setattr(make_voice, "decode", lambda _c: next(takes))

    assert make_voice.generate(tmp_path, lines={"effort_1": "<gasp>"},
                               bounds=make_voice._grunt_bounds) == {}
    kept = make_voice.generate(tmp_path, lines={"effort_1": "<gasp>"},
                               bounds=make_voice._grunt_bounds)
    assert list(kept) == ["effort_1"]
