"""word_onsets: word starts move forward to the speech after a pause, and only then."""
import shutil
import subprocess

import numpy as np
import pytest

import word_onsets as wo

QUIET, LOUD = -70.0, -20.0


def envelope(*spans, seconds=10.0):
    """A level trace, quiet except the given (start, end) loud spans."""
    env = np.full(int(seconds / wo.HOP_S), QUIET)
    for a, b in spans:
        env[int(a / wo.HOP_S):int(b / wo.HOP_S)] = LOUD
    return env


def test_start_after_a_pause_moves_to_the_speech():
    # The countdown shape: whisper says 1.15, the shout starts at 1.75.
    assert wo.word_onset(1.15, 1.99, envelope((1.75, 2.1))) == pytest.approx(1.75, abs=0.011)


def test_start_already_on_the_speech_is_left():
    assert wo.word_onset(1.0, 1.5, envelope((1.0, 1.5))) is None


def test_short_lead_is_left():
    assert wo.word_onset(1.0, 1.5, envelope((1.04, 1.5))) is None


def test_no_clear_rise_is_left():
    env = envelope((1.5, 2.0))
    env[env == LOUD] = QUIET + wo.MIN_RISE_DB - 1
    assert wo.word_onset(1.0, 2.0, env) is None


def test_start_never_passes_the_end_or_the_cap():
    assert wo.word_onset(1.0, 1.5, envelope((1.48, 1.6))) == pytest.approx(1.5 - wo.MIN_WORD_S, abs=0.011)
    assert wo.word_onset(1.0, 3.2, envelope((3.0, 3.2))) == pytest.approx(1.0 + wo.MAX_SHIFT_S, abs=0.011)


def test_envelope_offset_by_t0():
    env = envelope((1.35, 1.7))
    assert wo.word_onset(1121.15, 1121.99, env, t0=1120.4) == pytest.approx(1121.15 + 0.6, abs=0.011)


def test_snap_moves_only_late_words_and_keeps_ends():
    segments = [{"words": [
        {"word": " Five,", "start": 0.0, "end": 0.4},
        {"word": " four,", "start": 0.4, "end": 1.2},
        {"word": " go", "start": 1.2, "end": None},
    ]}]
    moved = wo.snap_word_onsets(segments, envelope((0.0, 0.4), (0.9, 1.2)))
    words = segments[0]["words"]
    assert moved == 1
    assert [w["start"] for w in words] == [0.0, pytest.approx(0.9, abs=0.011), 1.2]
    assert [w["end"] for w in words] == [0.4, 1.2, None]


def test_toggle(monkeypatch):
    monkeypatch.delenv("WORD_ONSET_SNAP", raising=False)
    assert wo.enabled()
    monkeypatch.setenv("WORD_ONSET_SNAP", "0")
    assert not wo.enabled()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_envelope_finds_a_tone_after_silence(tmp_path):
    wav = tmp_path / "t.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "aevalsrc='if(gte(t,1.5),0.5*sin(2*PI*800*t),0)':s=16000:d=3", str(wav)], check=True)
    env = wo.speech_envelope(str(wav))
    assert len(env) == pytest.approx(300, abs=3)
    assert wo.word_onset(1.0, 2.5, env) == pytest.approx(1.5, abs=0.03)
