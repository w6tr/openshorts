"""Layer bookkeeping for burned hooks (app.py helpers).

The hook is a derived file (hooked_<ts>_<clean>) exactly like captions
(subtitled_<ts>_<clean>): replacing or removing a hook walks the prefix back
to the clean file instead of stacking a second overlay — the bug this design
replaced. These tests pin the walk-back and the canonical-file resolution.
"""
import json
import os
import time

import pytest

app_module = pytest.importorskip("app")

_strip_burned_captions = app_module._strip_burned_captions
_strip_burned_hook = app_module._strip_burned_hook
_canonical_clip_file = app_module._canonical_clip_file


def _touch(directory, name, mtime=None):
    path = os.path.join(str(directory), name)
    with open(path, "wb") as f:
        f.write(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


CLEAN = "base_clip_1.mp4"
HOOKED = f"hooked_123_{CLEAN}"
CAPTIONED_HOOKED = f"subtitled_456_{HOOKED}"


class TestStripBurnedHook:
    def test_walks_hooked_prefix_back_to_clean(self, tmp_path):
        _touch(tmp_path, CLEAN)
        assert _strip_burned_hook(str(tmp_path), HOOKED) == CLEAN

    def test_walks_legacy_hook_prefix(self, tmp_path):
        _touch(tmp_path, CLEAN)
        assert _strip_burned_hook(str(tmp_path), f"hook_{CLEAN}") == CLEAN

    def test_no_prefix_is_returned_unchanged(self, tmp_path):
        _touch(tmp_path, CLEAN)
        assert _strip_burned_hook(str(tmp_path), CLEAN) == CLEAN

    def test_missing_underlying_file_stops_the_walk(self, tmp_path):
        # Restored project that only kept the current version: never point at
        # a file that does not exist.
        assert _strip_burned_hook(str(tmp_path), HOOKED) == HOOKED


class TestLayerChain:
    def test_captioned_hook_strips_in_two_steps(self, tmp_path):
        """subtitled_<ts>_hooked_<ts>_<clean> -> hooked -> clean, the exact
        order /api/hook uses to REPLACE a hook without stacking."""
        _touch(tmp_path, CLEAN)
        _touch(tmp_path, HOOKED)
        _touch(tmp_path, CAPTIONED_HOOKED)
        no_captions = _strip_burned_captions(str(tmp_path), CAPTIONED_HOOKED)
        assert no_captions == HOOKED
        assert _strip_burned_hook(str(tmp_path), no_captions) == CLEAN


class TestHookedOnDemand:
    """The job keeps a recipe instead of encoding the hooked-only picture
    (hooks.materialize); stripping the captions encodes it when needed."""

    def _recipe(self, tmp_path, monkeypatch):
        import hooks
        _touch(tmp_path, CLEAN)
        _touch(tmp_path, CAPTIONED_HOOKED)
        json_path, png_path = hooks.recipe_paths(str(tmp_path), HOOKED)
        _touch(tmp_path, os.path.basename(png_path))
        with open(json_path, "w") as f:
            f.write('{"source": "%s", "x": 0, "y": 0, "duration": 5}' % CLEAN)
        made = []

        def fake(output_dir, name):
            made.append(name)
            _touch(output_dir, name)
            return True
        monkeypatch.setattr(hooks, "materialize", fake)
        return made

    def test_stripping_captions_encodes_the_missing_hooked_picture(self, tmp_path, monkeypatch):
        made = self._recipe(tmp_path, monkeypatch)
        assert _strip_burned_captions(str(tmp_path), CAPTIONED_HOOKED) == HOOKED
        assert made == [HOOKED] and os.path.exists(tmp_path / HOOKED)

    def test_replacing_the_hook_never_encodes_it(self, tmp_path, monkeypatch):
        made = self._recipe(tmp_path, monkeypatch)
        no_captions = _strip_burned_captions(str(tmp_path), CAPTIONED_HOOKED, materialize=False)
        assert no_captions == HOOKED and made == []
        assert _strip_burned_hook(str(tmp_path), no_captions) == CLEAN

    def test_a_failed_encode_keeps_the_current_file(self, tmp_path, monkeypatch):
        import hooks
        self._recipe(tmp_path, monkeypatch)
        monkeypatch.setattr(hooks, "materialize", lambda *a: False)
        assert _strip_burned_captions(str(tmp_path), CAPTIONED_HOOKED) == CAPTIONED_HOOKED

    def test_without_a_recipe_nothing_changes(self, tmp_path):
        _touch(tmp_path, CAPTIONED_HOOKED)
        assert _strip_burned_captions(str(tmp_path), CAPTIONED_HOOKED) == CAPTIONED_HOOKED

    def test_the_recipe_is_not_a_clip(self, tmp_path, monkeypatch):
        self._recipe(tmp_path, monkeypatch)
        assert _canonical_clip_file(str(tmp_path), "base", 0) == CAPTIONED_HOOKED


class TestCanonicalClipFile:
    def test_captioned_hook_chain_resolves(self, tmp_path):
        now = time.time()
        _touch(tmp_path, CLEAN, mtime=now - 60)
        _touch(tmp_path, HOOKED, mtime=now - 30)
        _touch(tmp_path, CAPTIONED_HOOKED, mtime=now)
        assert _canonical_clip_file(str(tmp_path), "base", 0) == CAPTIONED_HOOKED

    def test_uncaptioned_hook_resolves(self, tmp_path):
        now = time.time()
        _touch(tmp_path, CLEAN, mtime=now - 60)
        _touch(tmp_path, HOOKED, mtime=now)
        assert _canonical_clip_file(str(tmp_path), "base", 0) == HOOKED

    def test_legacy_manual_hook_resolves(self, tmp_path):
        now = time.time()
        _touch(tmp_path, CLEAN, mtime=now - 60)
        _touch(tmp_path, f"hook_{CLEAN}", mtime=now)
        assert _canonical_clip_file(str(tmp_path), "base", 0) == f"hook_{CLEAN}"

    def test_newest_derivation_wins(self, tmp_path):
        """After a hook replacement both chains exist on disk; the newer one
        must be served or the replace would appear to do nothing."""
        now = time.time()
        _touch(tmp_path, CLEAN, mtime=now - 90)
        _touch(tmp_path, HOOKED, mtime=now - 60)
        _touch(tmp_path, CAPTIONED_HOOKED, mtime=now - 40)
        new_chain = f"subtitled_900_hooked_800_{CLEAN}"
        _touch(tmp_path, f"hooked_800_{CLEAN}", mtime=now - 20)
        _touch(tmp_path, new_chain, mtime=now)
        assert _canonical_clip_file(str(tmp_path), "base", 0) == new_chain

    def test_clean_only_returns_clean(self, tmp_path):
        _touch(tmp_path, CLEAN)
        assert _canonical_clip_file(str(tmp_path), "base", 0) == CLEAN


_hook_free_twin = app_module._hook_free_twin
_hook_still_burned = app_module._hook_still_burned
_keep_hook_free_fx = app_module._keep_hook_free_fx
_fx_recipe_path = app_module._fx_recipe_path

FX_HOOKED = f"fx_{HOOKED}"          # sounds/tints/zooms laid over the first hook
FX_CLEAN = f"fx_{CLEAN}"            # the same pass without the hook


@pytest.fixture
def fake_fx(monkeypatch):
    """_apply_fx stand-in: writes the output, records each call."""
    calls = []

    def apply(src, dst, effects, zoom_after=0.0):
        calls.append((open(src, "rb").read(), effects, zoom_after))
        with open(dst, "wb") as f:
            f.write(b"fx")
        return {"applied": effects, "dropped": [], "changed": True}

    monkeypatch.setattr(app_module, "_apply_fx", apply)
    return calls


class TestHookUnderFx:
    """A hook burned in UNDER an fx pass (fx_hooked_<ts>_X) must come off on a
    redo too, or the new hook stacks on the old one (the K006 bug)."""

    def test_twin_names(self):
        assert _hook_free_twin(HOOKED) == CLEAN
        assert _hook_free_twin(FX_HOOKED) == FX_CLEAN
        assert _hook_free_twin(f"fx_fx_hook_{CLEAN}") == f"fx_fx_{CLEAN}"
        assert _hook_free_twin(f"hooked_900_{FX_CLEAN}") == FX_CLEAN
        assert _hook_free_twin(CLEAN) is None and _hook_free_twin(FX_CLEAN) is None

    def test_fx_pass_keeps_a_hook_free_recipe(self, tmp_path):
        _touch(tmp_path, CLEAN)
        effects = [{"type": "sfx", "sound": "pop", "at": 1.0}]
        assert _keep_hook_free_fx(str(tmp_path), HOOKED, FX_HOOKED, effects, 5.0) == FX_CLEAN
        with open(_fx_recipe_path(str(tmp_path), FX_CLEAN)) as f:
            assert json.load(f) == {"source": CLEAN, "effects": effects, "zoom_after": 5.0}
        assert not os.path.exists(tmp_path / FX_CLEAN)   # lazy: nothing encoded yet

    def test_a_new_fx_pass_replaces_a_stale_encode(self, tmp_path):
        _touch(tmp_path, CLEAN)
        _touch(tmp_path, FX_CLEAN)
        _keep_hook_free_fx(str(tmp_path), HOOKED, FX_HOOKED, [], 5.0)
        assert not os.path.exists(tmp_path / FX_CLEAN)

    def test_no_recipe_without_a_hook_or_its_clean_file(self, tmp_path):
        assert _keep_hook_free_fx(str(tmp_path), CLEAN, FX_CLEAN, [], 0.0) is None
        assert _keep_hook_free_fx(str(tmp_path), HOOKED, FX_HOOKED, [], 5.0) is None
        assert os.listdir(tmp_path) == []

    def test_strip_encodes_the_hook_free_pass(self, tmp_path, fake_fx):
        _touch(tmp_path, CLEAN)
        _touch(tmp_path, FX_HOOKED)
        effects = [{"type": "tint", "color": "red", "start": 6, "end": 6.5}]
        _keep_hook_free_fx(str(tmp_path), HOOKED, FX_HOOKED, effects, 5.0)
        assert _strip_burned_hook(str(tmp_path), FX_HOOKED) == FX_CLEAN
        assert fake_fx == [(b"x", effects, 5.0)]          # same pass, from the clean picture
        assert open(tmp_path / FX_CLEAN, "rb").read() == b"fx"
        assert not [n for n in os.listdir(tmp_path) if n.startswith(".part_")]

    def test_redo_of_a_redo_strips_the_second_hook(self, tmp_path, fake_fx):
        _touch(tmp_path, FX_CLEAN)
        assert _strip_burned_hook(str(tmp_path), f"hooked_900_{FX_CLEAN}") == FX_CLEAN
        assert fake_fx == []                               # already on disk: no encode

    def test_old_clip_without_a_twin_is_left_hooked(self, tmp_path, fake_fx):
        """K003/K004/K006: fx'd before the fix, so no recipe. The name stays
        hooked and /api/hook refuses (409) instead of stacking."""
        _touch(tmp_path, CLEAN)
        _touch(tmp_path, HOOKED)
        _touch(tmp_path, FX_HOOKED)
        assert _strip_burned_hook(str(tmp_path), FX_HOOKED) == FX_HOOKED
        assert _hook_still_burned(FX_HOOKED)
        assert not _hook_still_burned(FX_CLEAN) and not _hook_still_burned(CLEAN)

    def test_a_failed_encode_keeps_the_hooked_name(self, tmp_path, monkeypatch):
        _touch(tmp_path, CLEAN)
        _keep_hook_free_fx(str(tmp_path), HOOKED, FX_HOOKED, [], 5.0)

        def boom(*a, **k):
            raise RuntimeError("ffmpeg died")

        monkeypatch.setattr(app_module, "_apply_fx", boom)
        assert _strip_burned_hook(str(tmp_path), FX_HOOKED) == FX_HOOKED
        assert not os.path.exists(tmp_path / FX_CLEAN)
        assert not [n for n in os.listdir(tmp_path) if n.startswith(".part_")]
