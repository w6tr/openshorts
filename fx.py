"""
Pop-culture FX layer: full-frame colour tints and one-shot sound effects laid
over a finished clip with FFmpeg. Used by /api/clip/fx.

Effect dicts accepted by apply_fx():
    {"type": "tint", "color": "red"|"green", "start": s, "end": s, "strength": 0..0.6}
    {"type": "sfx",  "sound": "<name in assets/sfx>", "at": s, "gain": 0..2}
    {"type": "punch_in"|"zoom_in"|"zoom_pulse"|"flash"|"vignette",
     "start": s, "end": s, "strength": ...}

The last five are camera moves, built and capped by edit_builder (zooms at
most 15%, no overlapping zooms, at most two flashes). A zoom that starts
before ``zoom_after`` is dropped: the caller sets it to the end of a hook or
captions still burned into the picture, so a zoom never crops them.

Hard caps: 12 tints, 12 sounds, tints 0.2-1.0 s long, tint alpha <= 0.6,
sound gain <= 2.0. A tint fades in over two frames and out over 0.3 s (a hard
on/off box reads as machine-made), and sounds default to gain 0.3 (about
-10 dB) so they sit under the voice. Anything outside the caps is clamped or dropped and the
caller is told which.
"""
import os
import shutil
import subprocess
import json

import edit_builder
from ffmpeg_utils import METADATA_SCRUB, video_encode_args

_HERE = os.path.dirname(os.path.abspath(__file__))
SFX_DIR = os.environ.get("SFX_DIR") or os.path.join(_HERE, "assets", "sfx")
SFX_EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

MAX_SFX = 12
MAX_TINTS = 12
MIN_TINT_SECONDS = 0.2
MAX_TINT_SECONDS = 1.0
DEFAULT_TINT_SECONDS = 0.5
DEFAULT_TINT_ALPHA = 0.2
MAX_TINT_ALPHA = 0.6
MAX_GAIN = 2.0
DEFAULT_GAIN = 0.3
TINT_FADE_IN = 0.07
TINT_FADE_OUT = 0.3
FFMPEG_TIMEOUT = 1800

TINT_COLORS = {"red": "0xFF2020", "green": "0x20E060"}
MOVE_TYPES = ("punch_in", "zoom_in", "zoom_pulse", "flash", "vignette")


def _env():
    env = os.environ.copy()
    env["LANG"] = env["LC_ALL"] = "C.UTF-8"
    return env


def list_sounds():
    """name -> absolute path for every sound file in SFX_DIR ({} if missing)."""
    out = {}
    if not os.path.isdir(SFX_DIR):
        return out
    for fn in sorted(os.listdir(SFX_DIR)):
        name, ext = os.path.splitext(fn)
        if ext.lower() in SFX_EXTS:
            out[name] = os.path.join(SFX_DIR, fn)
    return out


def probe(path):
    """(duration_seconds, has_audio) via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, text=True, env=_env(), timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:300]}")
    info = json.loads(r.stdout or "{}")
    duration = _num((info.get("format") or {}).get("duration"), 0.0) or 0.0
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return duration, has_audio


def probe_video(path):
    """(fps, width, height) of the first video stream via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format", "json",
         "-show_entries", "stream=width,height,r_frame_rate", path],
        capture_output=True, text=True, env=_env(), timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:300]}")
    stream = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    num, _, den = str(stream.get("r_frame_rate", "30/1")).partition("/")
    fps = (_num(num, 30.0) or 30.0) / (_num(den, 1.0) or 1.0)
    return fps, int(stream.get("width") or 1080), int(stream.get("height") or 1920)


def _num(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_effects(effects, duration, sounds=None):
    """Validate and clamp. Returns (tints, sfx, dropped)."""
    tints, sfx, dropped = [], [], []
    if not isinstance(effects, list) or duration <= 0:
        return tints, sfx, dropped
    sounds = list_sounds() if sounds is None else sounds

    for raw in effects:
        if not isinstance(raw, dict):
            dropped.append({"effect": raw, "reason": "not an object"})
            continue
        kind = str(raw.get("type", "")).strip().lower()

        if kind == "tint":
            color = str(raw.get("color", "")).strip().lower()
            if color not in TINT_COLORS:
                dropped.append({"effect": raw, "reason": f"unknown tint color {color!r}"})
                continue
            start = _num(raw.get("start"))
            if start is None:
                dropped.append({"effect": raw, "reason": "tint needs a start"})
                continue
            end = _num(raw.get("end"), start + DEFAULT_TINT_SECONDS)
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))
            if end - start < MIN_TINT_SECONDS:
                dropped.append({"effect": raw, "reason": "tint shorter than 0.2 s or outside the clip"})
                continue
            end = min(end, start + MAX_TINT_SECONDS)
            alpha = _num(raw.get("strength"), DEFAULT_TINT_ALPHA) or DEFAULT_TINT_ALPHA
            alpha = max(0.05, min(alpha, MAX_TINT_ALPHA))
            tints.append({"color": color, "start": start, "end": end, "alpha": alpha})

        elif kind == "sfx":
            sound = str(raw.get("sound", "")).strip()
            if sound not in sounds:
                dropped.append({"effect": raw, "reason": f"unknown sound {sound!r}"})
                continue
            at = _num(raw.get("at"))
            if at is None:
                dropped.append({"effect": raw, "reason": "sfx needs an 'at' time"})
                continue
            at = max(0.0, min(at, max(0.0, duration - 0.05)))
            gain = _num(raw.get("gain"), DEFAULT_GAIN)
            gain = max(0.0, min(gain, MAX_GAIN))
            sfx.append({"sound": sound, "path": sounds[sound], "at": at, "gain": gain})

        elif kind in MOVE_TYPES:
            continue   # normalize_moves owns these

        else:
            dropped.append({"effect": raw, "reason": f"unknown effect type {kind!r}"})

    tints.sort(key=lambda t: (t["start"], t["end"]))
    sfx.sort(key=lambda s: s["at"])
    for extra in tints[MAX_TINTS:]:
        dropped.append({"effect": {"type": "tint", **extra}, "reason": f"over the {MAX_TINTS}-tint cap"})
    for extra in sfx[MAX_SFX:]:
        dropped.append({"effect": {"type": "sfx", "sound": extra["sound"], "at": extra["at"]},
                        "reason": f"over the {MAX_SFX}-sound cap"})
    return tints[:MAX_TINTS], sfx[:MAX_SFX], dropped


def normalize_moves(effects, duration, zoom_after=0.0):
    """Camera moves from an effect list, capped by edit_builder.
    Returns (moves, dropped); moves are edit_builder edit dicts."""
    if not isinstance(effects, list) or duration <= 0:
        return [], []
    dropped, alone = [], []
    for raw in effects:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type", "")).strip().lower()
        if kind not in MOVE_TYPES:
            continue
        if edit_builder.EFFECT_LIMITS[kind]["zoom"] and (_num(raw.get("start"), 0.0) or 0.0) < zoom_after:
            dropped.append({"effect": raw, "reason": "zoom over a burned-in hook or captions"})
            continue
        edit = dict(raw, type=kind)
        if not edit_builder.normalize_edits([edit], duration):
            dropped.append({"effect": raw, "reason": "move shorter than 0.15 s or outside the clip"})
            continue
        alone.append((raw, edit))
    moves = edit_builder.normalize_edits([edit for _, edit in alone], duration)
    for raw, edit in alone:
        if edit_builder.normalize_edits([edit], duration)[0] not in moves:
            dropped.append({"effect": raw, "reason": "overlaps another zoom or over the flash/move cap"})
    return moves, dropped


def build_fx_command(input_path, output_path, tints, sfx, has_audio, duration,
                     moves=None, fps=30.0, width=1080, height=1920):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path]
    for s in sfx:
        cmd += ["-i", s["path"]]
    bed_index = None
    if sfx and not has_audio:
        bed_index = 1 + len(sfx)
        cmd += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]

    filters = []
    src = "[0:v]"
    if moves:
        move_filter, _ = edit_builder.build_filter_string(moves, duration, fps, width, height)
        if move_filter:
            src = "[m]" if tints else "[v]"
            filters.append(f"[0:v]{move_filter}{src}")
    if tints:
        # One solid-colour copy of the frame per tint, its alpha faded in and
        # out, laid over the picture only inside the tint's window.
        n = len(tints)
        filters.append(f"{src}split={n + 1}[t0]" + "".join(f"[c{k}]" for k in range(1, n + 1)))
        for k, t in enumerate(tints, 1):
            fin = min(TINT_FADE_IN, (t["end"] - t["start"]) / 2)
            fout = min(TINT_FADE_OUT, t["end"] - t["start"] - fin)
            filters.append(
                f"[c{k}]drawbox=x=0:y=0:w=iw:h=ih:color={TINT_COLORS[t['color']]}@1:t=fill,"
                f"format=rgba,colorchannelmixer=aa={t['alpha']:.2f},"
                f"fade=t=in:st={t['start']:.3f}:d={fin:.3f}:alpha=1,"
                f"fade=t=out:st={t['end'] - fout:.3f}:d={fout:.3f}:alpha=1[k{k}]")
            out = "[v]" if k == n else f"[t{k}]"
            filters.append(f"[t{k - 1}][k{k}]overlay=format=auto"
                           f":enable='between(t,{t['start']:.3f},{t['end']:.3f})'{out}")

    if sfx:
        base = f"[{bed_index}:a]" if bed_index is not None else "[0:a]"
        fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
        filters.append(f"{base}{fmt}[a0]")
        labels = ["[a0]"]
        for k, s in enumerate(sfx, 1):
            ms = int(round(s["at"] * 1000))
            filters.append(f"[{k}:a]{fmt},adelay={ms}:all=1,volume={s['gain']:.2f}[s{k}]")
            labels.append(f"[s{k}]")
        filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[aout]")

    if filters:
        cmd += ["-filter_complex", ";".join(filters)]

    if tints or src != "[0:v]":
        cmd += ["-map", "[v]"] + video_encode_args()
    else:
        cmd += ["-map", "0:v", "-c:v", "copy"]

    if sfx:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k"]
    elif has_audio:
        cmd += ["-map", "0:a", "-c:a", "copy"]

    cmd += METADATA_SCRUB + ["-movflags", "+faststart", output_path]
    return cmd


def apply_fx(input_path, output_path, effects, zoom_after=0.0):
    """Render tints, sounds and camera moves onto input_path -> output_path.

    Returns {"applied": [...], "dropped": [...], "changed": bool}. With no
    usable effect the input is copied unchanged and changed=False.
    """
    duration, has_audio = probe(input_path)
    tints, sfx, dropped = normalize_effects(effects, duration)
    moves, move_dropped = normalize_moves(effects, duration, zoom_after)
    dropped += move_dropped
    applied = [{"type": "tint", **t} for t in tints] + [
        {"type": "sfx", "sound": s["sound"], "at": s["at"], "gain": s["gain"]} for s in sfx
    ] + [dict(m) for m in moves]
    if not tints and not sfx and not moves:
        shutil.copyfile(input_path, output_path)
        return {"applied": applied, "dropped": dropped, "changed": False}

    fps, width, height = probe_video(input_path) if moves else (30.0, 1080, 1920)
    cmd = build_fx_command(input_path, output_path, tints, sfx, has_audio, duration,
                           moves, fps, width, height)
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(), timeout=FFMPEG_TIMEOUT)
    if r.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg fx failed: {r.stderr.strip()[-600:]}")
    return {"applied": applied, "dropped": dropped, "changed": True}
