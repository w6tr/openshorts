"""HTTP tests for `analysis_only` on /api/process.

Same conventions as test_process_handover.py: a real ASGI round-trip against
the imported app (BILLING_ENABLED=0 via conftest), submitting through a
Thumbnail Studio session so no network is touched. The job is only ENQUEUED —
the worker never runs under ASGITransport — so the first tests own the request
contract (the flag reaches the subprocess as ANALYSIS_ONLY=1, JSON and form).
The last one drives run_job against a fake, already-finished subprocess to
pin the completion contract: a clipless metadata file from an analysis-only
run is `completed` with the transcript in the result, not a failed job.
"""

import asyncio
import io
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


def _get_status(job_id):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.get(f"/api/status/{job_id}")
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


def _submitted_job(resp):
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    return job_id, app_module.jobs[job_id]


def test_json_flag_sets_env(dirs, session):
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True,
                          "analysis_only": True})
    _, job = _submitted_job(resp)
    assert job["env"].get("ANALYSIS_ONLY") == "1"


def test_form_flag_sets_env(dirs, session):
    resp = _post_process(data={"thumbnail_session_id": "sess1", "acknowledged": "true",
                               "analysis_only": "1"})
    _, job = _submitted_job(resp)
    assert job["env"].get("ANALYSIS_ONLY") == "1"


def test_flag_off_by_default(dirs, session):
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True})
    _, job = _submitted_job(resp)
    assert "ANALYSIS_ONLY" not in job["env"]


class _FinishedProcess:
    """What run_job sees from a main.py that already exited 0 silently."""
    returncode = 0
    pid = 0

    def __init__(self, *a, **k):
        self.stdout = io.BytesIO(b"")

    def poll(self):
        return 0

    def kill(self):
        pass


def test_clipless_metadata_completes_with_transcript(dirs, monkeypatch):
    out_root, _ = dirs
    job_id = "job-analysis"
    job_dir = out_root / job_id
    job_dir.mkdir()
    meta = {"shorts": [], "transcript": TRANSCRIPT, "source_video": "src.mp4",
            "output_format": "auto", "duration": 60.0}
    with open(job_dir / "src_metadata.json", "w") as f:
        json.dump(meta, f)

    monkeypatch.setattr(app_module.subprocess, "Popen", _FinishedProcess)
    monkeypatch.setattr(app_module, "upload_job_artifacts", lambda *a, **k: None)
    job = {
        'status': 'queued', 'logs': app_module._TimedLog([]),
        'cmd': ["python", "main.py"], 'env': {"ANALYSIS_ONLY": "1"},
        'output_dir': str(job_dir), 'user_id': None,
    }
    monkeypatch.setitem(app_module.jobs, job_id, job)
    asyncio.run(app_module.run_job(job_id, job))

    assert job['status'] == 'completed', job['logs']
    assert job['result']['clips'] == []
    assert job['result']['transcript'] == TRANSCRIPT
    assert job['result']['metadata']['source_video'] == "src.mp4"
    assert job['result']['metadata']['duration'] == 60.0
    assert 'transcript' not in job['result']['metadata']

    resp = _get_status(job_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["result"]["transcript"]["segments"][0]["words"][0]["word"] == "hello"


def test_clipless_metadata_without_flag_still_fails(dirs, monkeypatch):
    """The existing guard is untouched for normal jobs."""
    out_root, _ = dirs
    job_id = "job-normal"
    job_dir = out_root / job_id
    job_dir.mkdir()
    with open(job_dir / "src_metadata.json", "w") as f:
        json.dump({"shorts": [], "transcript": TRANSCRIPT}, f)
    monkeypatch.setattr(app_module.subprocess, "Popen", _FinishedProcess)
    monkeypatch.setattr(app_module, "upload_job_artifacts", lambda *a, **k: None)
    job = {'status': 'queued', 'logs': app_module._TimedLog([]),
           'cmd': ["python", "main.py"], 'env': {}, 'output_dir': str(job_dir),
           'user_id': None}
    monkeypatch.setitem(app_module.jobs, job_id, job)
    asyncio.run(app_module.run_job(job_id, job))
    assert job['status'] == 'failed'
