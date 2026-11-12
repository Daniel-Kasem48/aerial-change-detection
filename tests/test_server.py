"""Tests for the fraud-scanner web API (src/server)."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.forensics.trufor import Region, TruForReport
from src.server.app import Settings, create_app
from src.server.verdict import combine

SAMPLES = Path(__file__).parent.parent / "samples"


class FakeTruFor:
    """Stands in for the 25-second network in API tests."""

    def __init__(self, report: TruForReport) -> None:
        self.report = report
        self.calls = 0

    def analyze(self, path: str | Path) -> TruForReport:
        self.calls += 1
        h, w = cv2.imread(str(path)).shape[:2]
        return TruForReport(score=self.report.score, regions=self.report.regions,
                            manipulated_fraction=self.report.manipulated_fraction,
                            heatmap=np.zeros((h, w), np.float32))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    fake = FakeTruFor(TruForReport(score=0.9))
    app = create_app(Settings(data_dir=tmp_path, examples_dir=SAMPLES), trufor=fake)
    app.state.fake_trufor = fake
    return TestClient(app)


def _upload(client: TestClient, path: Path):
    return client.post("/api/scan", files={"file": (path.name, path.read_bytes(), "image/jpeg")})


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "trufor": True}


def test_index_serves_ui(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Fraud Scanner" in r.text


def test_scan_returns_fast_checks(client: TestClient) -> None:
    r = _upload(client, SAMPLES / "2.jpg")
    assert r.status_code == 200
    body = r.json()
    assert body["scan_id"]
    assert body["image"]["width"] == 1280
    assert body["metadata"]["flags"][0]["code"] == "no_camera_metadata"
    assert body["duplicates"] == []
    assert "score" in body["ela"] and body["ela"]["overlay"].startswith("data:image/jpeg;base64,")
    assert body["verdict"]["level"] in {"low", "medium", "high"}
    assert body["trufor"] is None  # deep scan is a separate call


def test_second_upload_of_same_photo_is_a_duplicate(client: TestClient) -> None:
    _upload(client, SAMPLES / "2.jpg")
    dupes = _upload(client, SAMPLES / "3.jpg").json()["duplicates"]
    assert dupes and dupes[0]["distance"] <= 10


def test_deep_scan_adds_trufor_and_updates_verdict(client: TestClient) -> None:
    scan_id = _upload(client, SAMPLES / "2.jpg").json()["scan_id"]
    r = client.post(f"/api/scan/{scan_id}/deep")
    assert r.status_code == 200
    body = r.json()
    assert body["trufor"]["score"] == 0.9
    assert body["trufor"]["suspicious"] is True
    assert body["trufor"]["overlay"].startswith("data:image/jpeg;base64,")
    assert body["verdict"]["level"] == "high"
    # cached: second call doesn't rerun the network
    client.post(f"/api/scan/{scan_id}/deep")
    assert client.app.state.fake_trufor.calls == 1


def test_deep_scan_unknown_id_is_404(client: TestClient) -> None:
    assert client.post("/api/scan/does-not-exist/deep").status_code == 404


def test_deep_scan_without_trufor_is_503(tmp_path: Path) -> None:
    c = TestClient(create_app(Settings(data_dir=tmp_path, examples_dir=SAMPLES), trufor=None))
    assert c.get("/api/health").json()["trufor"] is False
    scan_id = _upload(c, SAMPLES / "2.jpg").json()["scan_id"]
    assert c.post(f"/api/scan/{scan_id}/deep").status_code == 503


def test_rejects_non_image(client: TestClient) -> None:
    r = client.post("/api/scan", files={"file": ("x.jpg", b"hello", "image/jpeg")})
    assert r.status_code == 415


def test_rejects_too_large_upload(tmp_path: Path) -> None:
    c = TestClient(create_app(Settings(data_dir=tmp_path, examples_dir=SAMPLES,
                                       max_upload_bytes=1000), trufor=None))
    r = _upload(c, SAMPLES / "2.jpg")
    assert r.status_code == 413


def test_png_upload_works_and_ela_not_applicable(client: TestClient) -> None:
    _ok, buf = cv2.imencode(".png", cv2.imread(str(SAMPLES / "2.jpg")))
    r = client.post("/api/scan", files={"file": ("x.png", io.BytesIO(buf.tobytes()), "image/png")})
    assert r.status_code == 200
    assert r.json()["ela"]["applicable"] is False


def test_examples_list_and_fetch(client: TestClient) -> None:
    names = client.get("/api/examples").json()
    assert "2.jpg" in names
    r = client.get("/api/examples/2.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/examples/..%2Fsrc%2Fconfig.py").status_code == 404


# --- verdict ---------------------------------------------------------------

def test_verdict_low_for_clean_signals() -> None:
    v = combine(metadata_risk=0.2, duplicate=False, trufor=None)
    assert v.level == "low"
    assert any("camera" in r.lower() or "metadata" in r.lower() for r in v.reasons) or v.reasons == []


def test_verdict_high_when_trufor_suspicious() -> None:
    t = TruForReport(score=0.2, regions=[Region(0, 0, 9, 9, 0.9)], manipulated_fraction=0.05)
    v = combine(metadata_risk=0.0, duplicate=False, trufor=t)
    assert v.level == "high"
    assert v.score >= 0.7


def test_verdict_medium_for_editing_software_or_duplicate() -> None:
    assert combine(metadata_risk=0.6, duplicate=False, trufor=None).level == "medium"
    assert combine(metadata_risk=0.0, duplicate=True, trufor=None).level == "medium"


def test_verdict_high_for_ai_marker() -> None:
    assert combine(metadata_risk=0.95, duplicate=False, trufor=None).level == "high"
