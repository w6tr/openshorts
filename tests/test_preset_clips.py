"""HTTP tests for `clips` and `source_job_id` on /api/process.

Same conventions as test_analysis_only.py: real ASGI round-trip against the
imported app (BILLING_ENABLED=0 via conftest), jobs only ENQUEUED, so the
tests own the request contract — what lands in the job's cmd and on disk —
not the render. main.py's side (`load_preset_clips`) needs cv2/torch to
import and is not covered here.
"""

import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")

TRANSCRIPT = {
    "text": "hello world again",
    "language": "en",
    "segments": [
        {
            "start": 0.0, "end": 60.0, "text": "hello world again",
            "words": [
                {"word": "hello", "start": 12.0, "end": 12.5},
                {"word": "world", "start": 20.0, "end": 20.4},
                {"word": "again", "start": 50.0, "end": 50.5},
            ],
        },
    ],
}

CLIPS = [
    {"start": 10.0, "end": 40.0, "hook_text": "Watch this", "title": "First"},
    {"start": 45.0, "end": 58.5},
]


def _post_process(json_body=None, data=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.post(
                "/api/process", json=json_body, data=data,
                headers={"X-Gemini-Key": "test-key"},
            )
    return asyncio.run(_do())


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    out_root.mkdir()
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    return out_root, up_root


@pytest.fixture()
def session(dirs, monkeypatch):
    _, up_root = dirs
    video = up_root / "thumb_sess1_source.mp4"
    video.write_bytes(b"fake-video-bytes")
    sess = {"user_id": None, "video_path": str(video),
            "transcript_ready": True, "transcript": TRANSCRIPT}
    monkeypatch.setitem(app_module.thumbnail_sessions, "sess1", sess)
    return sess


@pytest.fixture()
def source_job(dirs, monkeypatch):
    """An earlier (analysis_only-style) job: source in uploads/{id}_*, a
    clipless metadata file carrying the transcript in its job dir."""
    out_root, up_root = dirs
    job_id = "src-job"
    (up_root / f"{job_id}_movie.mp4").write_bytes(b"source-bytes")
    job_dir = out_root / job_id
    job_dir.mkdir()
    with open(job_dir / "movie_metadata.json", "w") as f:
        json.dump({"shorts": [], "transcript": TRANSCRIPT,
                   "source_video": "movie.mp4", "duration": 60.0}, f)
    monkeypatch.setitem(app_module.jobs, job_id, {
        'status': 'completed', 'logs': app_module._TimedLog([]),
        'cmd': [], 'env': {}, 'output_dir': str(job_dir), 'user_id': None,
    })
    return job_id


def _submitted_job(resp):
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    return job_id, app_module.jobs[job_id]


def _flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_json_clips_written_and_passed(dirs, session):
    out_root, _ = dirs
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True,
                          "clips": CLIPS})
    job_id, job = _submitted_job(resp)
    path = _flag_value(job["cmd"], "--clips")
    assert path == str(out_root / job_id / "preset_clips.json")
    with open(path) as f:
        saved = json.load(f)
    assert saved == [
        {"start": 10.0, "end": 40.0, "hook_text": "Watch this", "title": "First"},
        {"start": 45.0, "end": 58.5, "hook_text": None, "title": None},
    ]
    assert "ANALYSIS_ONLY" not in job["env"]


def test_form_clips_as_json_string(dirs, session):
    resp = _post_process(data={"thumbnail_session_id": "sess1", "acknowledged": "true",
                               "clips": json.dumps(CLIPS)})
    _, job = _submitted_job(resp)
    assert "--clips" in job["cmd"]


def test_no_clips_by_default(dirs, session):
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True})
    _, job = _submitted_job(resp)
    assert "--clips" not in job["cmd"]


@pytest.mark.parametrize("bad", [
    [], "not json", [{"start": 5}], [{"start": 10, "end": 5}],
    [{"start": -1, "end": 5}], ["nope"], [{"start": "a", "end": "b"}],
])
def test_bad_clips_are_400(dirs, session, bad):
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True,
                          "clips": bad})
    assert resp.status_code == 400, resp.text


def test_source_job_reuses_source_and_transcript(dirs, source_job):
    out_root, up_root = dirs
    resp = _post_process({"source_job_id": source_job, "acknowledged": True,
                          "clips": CLIPS})
    job_id, job = _submitted_job(resp)
    cmd = job["cmd"]
    input_path = _flag_value(cmd, "-i")
    assert input_path == str(up_root / f"{job_id}_movie.mp4")
    assert os.path.exists(input_path)
    with open(input_path, "rb") as f:
        assert f.read() == b"source-bytes"
    transcript_path = _flag_value(cmd, "--transcript")
    assert transcript_path == str(out_root / job_id / "source_transcript.json")
    with open(transcript_path) as f:
        assert json.load(f) == TRANSCRIPT
    assert "--clips" in cmd
    assert job["attestation"]["source"] == "source_job"


def test_source_job_falls_back_to_checkpoint(dirs, source_job):
    out_root, _ = dirs
    os.remove(out_root / source_job / "movie_metadata.json")
    with open(out_root / source_job / ".transcript_checkpoint.json", "w") as f:
        json.dump({"source": {"name": "movie.mp4", "duration": 60.0},
                   "transcript": TRANSCRIPT}, f)
    resp = _post_process({"source_job_id": source_job, "acknowledged": True})
    _, job = _submitted_job(resp)
    assert "--transcript" in job["cmd"]


def test_source_job_without_transcript_skips_flag(dirs, source_job):
    out_root, _ = dirs
    os.remove(out_root / source_job / "movie_metadata.json")
    resp = _post_process({"source_job_id": source_job, "acknowledged": True})
    _, job = _submitted_job(resp)
    assert "-i" in job["cmd"]
    assert "--transcript" not in job["cmd"]


def test_unknown_source_job_is_404(dirs):
    resp = _post_process({"source_job_id": "nope", "acknowledged": True})
    assert resp.status_code == 404


def test_source_job_with_missing_file_is_404(dirs, source_job):
    _, up_root = dirs
    os.remove(up_root / f"{source_job}_movie.mp4")
    resp = _post_process({"source_job_id": source_job, "acknowledged": True})
    assert resp.status_code == 404


def test_nothing_given_is_400(dirs):
    resp = _post_process({"acknowledged": True})
    assert resp.status_code == 400
