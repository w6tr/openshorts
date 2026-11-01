"""Snap word starts forward to where the speech actually begins.

Whisper starts a word that follows a pause at the end of the previous word, so
after a gap the caption (and any beat timed off it) lands early. Measured on a
shouted countdown on 01-nov-2026: "three" started 0.60 s before it was said,
"one" 0.52 s, "four" 0.28 s; in continuous speech the starts were already
right. The fix is cheap: a speech-band level trace of the audio at 10 ms, and
each word's start moves forward to the first frame inside the word where the
level rises clearly above the quiet before it. Starts only ever move forward,
never past the word's own end, and word ends are left alone.
"""

import os
import subprocess
import sys

SAMPLE_RATE = 16000
HOP = 160            # 10 ms frames
WINDOW = 400         # 25 ms analysis window
NFFT = 512
BAND_HZ = (300.0, 5000.0)   # voice and most consonants, not rumble or hiss
HOP_S = HOP / SAMPLE_RATE

MIN_LEAD_S = 0.06    # a quiet lead shorter than this is left as it is
MIN_WORD_S = 0.08    # a moved word keeps at least this much before its end
MAX_SHIFT_S = 1.0
MIN_RISE_DB = 10.0   # word peak over the quiet before it; less = no clear onset
RISE_FRACTION = 0.35  # onset = first frame this far from the quiet up to the peak


def enabled():
    return os.environ.get("WORD_ONSET_SNAP", "1").strip().lower() not in ("0", "false", "no", "off")


def speech_envelope(media_path, start=0.0, duration=None):
    """Speech-band level in dB per 10 ms frame, streamed from ffmpeg in 30 s chunks.

    Frame i covers [start + i*0.01, start + i*0.01 + 0.025) seconds.
    """
    import numpy as np

    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", media_path]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]

    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    window = np.hanning(WINDOW).astype(np.float32)
    offsets = np.arange(WINDOW)[None, :]
    chunk_bytes = SAMPLE_RATE * 2 * 30

    out = []
    carry = np.zeros(0, dtype=np.float32)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(chunk_bytes)
            if not buf:
                break
            buf = buf[:len(buf) - len(buf) % 2]
            x = np.concatenate([carry, np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0])
            n = (len(x) - WINDOW) // HOP + 1
            if n <= 0:
                carry = x
                continue
            frames = x[offsets + HOP * np.arange(n)[:, None]] * window
            power = np.abs(np.fft.rfft(frames, NFFT)) ** 2
            out.append(10.0 * np.log10(power[:, band].sum(axis=1) + 1e-10))
            carry = x[n * HOP:]
        err = proc.stderr.read()
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err.decode(errors='replace')[-300:]}")
    return np.concatenate(out) if out else np.zeros(0)


def word_onset(start, end, envelope, t0=0.0):
    """Where the word's speech begins inside [start, end), or None to leave it."""
    import numpy as np

    i0 = int(round((start - t0) / HOP_S))
    i1 = int((end - t0) / HOP_S)
    i0, i1 = max(i0, 0), min(i1, len(envelope))
    if i1 - i0 < 3:
        return None
    level = np.asarray(envelope[i0:i1])
    peak = float(level.max())
    head = int(np.argmax(level))
    quiet = float(level[:head + 1].min())   # the quiet before the loudest part
    if peak - quiet < MIN_RISE_DB:
        return None
    threshold = quiet + RISE_FRACTION * (peak - quiet)
    first = int(np.argmax(level >= threshold))
    lead = first * HOP_S
    if lead < MIN_LEAD_S:
        return None
    new = min(start + lead, start + MAX_SHIFT_S, end - MIN_WORD_S)
    return round(new, 3) if new - start >= MIN_LEAD_S else None


def snap_word_onsets(segments, envelope, t0=0.0):
    """Move word starts forward to their speech onsets, in place; returns how many moved."""
    moved = 0
    for seg in segments:
        for w in seg.get("words") or []:
            if w.get("start") is None or w.get("end") is None:
                continue
            new = word_onset(float(w["start"]), float(w["end"]), envelope, t0)
            if new is not None:
                w["start"] = new
                moved += 1
    return moved


def snap_transcript(transcript, media_path):
    """Snap a transcript's word starts to the speech in media_path, in place."""
    envelope = speech_envelope(media_path)
    moved = snap_word_onsets(transcript.get("segments") or [], envelope)
    print(f"🎯 [ASR] word onsets: moved {moved} word start(s) forward to the speech")
    return transcript


if __name__ == "__main__":
    # python word_onsets.py <media> <transcript.json> <out.json>
    # Re-snaps a transcript made before this fix. Reads both inputs, writes only <out.json>.
    import json
    media, src, dst = sys.argv[1:4]
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    t = data.get("transcript") if "segments" not in data else data
    snap_transcript(t, media)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
