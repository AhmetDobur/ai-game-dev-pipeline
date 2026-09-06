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
