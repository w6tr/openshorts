"""fx.py camera moves: zooms, flash and vignette in the tint/sound pass."""
from fx import build_fx_command, normalize_effects, normalize_moves


def _move(type_, start, end, strength=0.0):
    return {"type": type_, "start": start, "end": end, "strength": strength}


def test_moves_are_not_dropped_as_unknown():
    tints, sfx, dropped = normalize_effects([_move("punch_in", 1, 1.5)], 30, sounds={})
    assert (tints, sfx, dropped) == ([], [], [])


def test_zoom_capped_at_15_percent():
    moves, dropped = normalize_moves([_move("zoom_in", 1, 3, strength=0.9)], 30)
    assert moves[0]["strength"] == 0.15 and dropped == []


def test_zoom_inside_hook_window_dropped_flash_kept():
    moves, dropped = normalize_moves([_move("punch_in", 2, 2.5), _move("flash", 2, 2.2),
                                      _move("punch_in", 6, 6.5)], 30, zoom_after=5.0)
    assert [(m["type"], m["start"]) for m in moves] == [("flash", 2.0), ("punch_in", 6.0)]
    assert [d["reason"] for d in dropped] == ["zoom over a burned-in hook or captions"]


def test_no_zooms_over_unstripped_captions():
    moves, _ = normalize_moves([_move("zoom_pulse", 8, 9), _move("vignette", 8, 9)], 30,
                               zoom_after=float("inf"))
    assert [m["type"] for m in moves] == ["vignette"]


def test_overlap_and_flash_cap_reported():
    moves, dropped = normalize_moves([_move("punch_in", 1, 2), _move("zoom_in", 1.5, 3),
                                      _move("flash", 4, 4.2), _move("flash", 5, 5.2),
                                      _move("flash", 6, 6.2), _move("flash", 7, 7.01)], 30)
    assert [m["type"] for m in moves] == ["punch_in", "flash", "flash"]
    assert sorted(d["reason"] for d in dropped) == [
        "move shorter than 0.15 s or outside the clip",
        "overlaps another zoom or over the flash/move cap",
        "overlaps another zoom or over the flash/move cap",
    ]


def _filter(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


def test_moves_run_before_tints_and_reencode():
    moves, _ = normalize_moves([_move("punch_in", 6, 6.5)], 30)
    tint = [{"color": "red", "start": 6, "end": 6.5, "alpha": 0.2}]
    cmd = build_fx_command("in.mp4", "out.mp4", tint, [], True, 30, moves, 30, 1080, 1920)
    graph = _filter(cmd)
    assert graph.startswith("[0:v]zoompan=") and "[m];[m]split=2" in graph
    assert cmd[cmd.index("-map") + 1] == "[v]"


def test_moves_alone_map_their_own_output():
    moves, _ = normalize_moves([_move("vignette", 1, 2)], 30)
    cmd = build_fx_command("in.mp4", "out.mp4", [], [], True, 30, moves, 30, 1080, 1920)
    assert _filter(cmd).startswith("[0:v]vignette=") and _filter(cmd).endswith("[v]")
    assert "copy" not in cmd[cmd.index("-map") + 2:cmd.index("-map") + 4]


def test_no_moves_keeps_stream_copy():
    tint = [{"color": "green", "start": 1, "end": 1.5, "alpha": 0.2}]
    cmd = build_fx_command("in.mp4", "out.mp4", tint, [], True, 30)
    assert _filter(cmd).startswith("[0:v]split=2")
    no_fx = build_fx_command("in.mp4", "out.mp4", [], [], True, 30)
    assert no_fx[no_fx.index("-map") + 1:no_fx.index("-map") + 4] == ["0:v", "-c:v", "copy"]
