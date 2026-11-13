"""Tests for the fraud-scanner web API (src/server)."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.forensics.dtd import DtdReport
from src.forensics.dtd import Region as DtdRegion
from src.forensics.trufor import Region, TruForReport
from src.server.config import Settings
from src.server.main import create_app
from src.server.services.verdict import combine

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


class FakeDtd:
    """Stands in for the document model."""

    def __init__(self, report: DtdReport) -> None:
        self.report = report
        self.calls = 0

    def analyze(self, path: str | Path) -> DtdReport:
        self.calls += 1
        h, w = cv2.imread(str(path)).shape[:2]
        r = self.report
        return DtdReport(r.applicable, r.score, r.reason, r.regions, r.tampered_fraction,
                         np.zeros((h, w), np.float32))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    fake = FakeTruFor(TruForReport(score=0.9))
    fake_dtd = FakeDtd(DtdReport(True, 1.0, "1 text area(s) look edited",
                                 [DtdRegion(10, 10, 40, 60, 0.99, 900)], 0.001))
    app = create_app(Settings(data_dir=tmp_path, examples_dir=SAMPLES), trufor=fake, dtd=fake_dtd)
    app.state.fake_trufor = fake
    app.state.fake_dtd = fake_dtd
    return TestClient(app)


def _upload(client: TestClient, path: Path):
    return client.post("/api/scan", files={"file": (path.name, path.read_bytes(), "image/jpeg")})


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "trufor": True, "dtd": True}


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
    c = TestClient(create_app(Settings(data_dir=tmp_path, examples_dir=SAMPLES), trufor=None,
                              dtd=None))
    assert c.get("/api/health").json()["trufor"] is False
    scan_id = _upload(c, SAMPLES / "2.jpg").json()["scan_id"]
    assert c.post(f"/api/scan/{scan_id}/deep").status_code == 503


def test_rejects_non_image(client: TestClient) -> None:
    r = client.post("/api/scan", files={"file": ("x.jpg", b"hello", "image/jpeg")})
    assert r.status_code == 415


def test_rejects_too_large_upload(tmp_path: Path) -> None:
    c = TestClient(create_app(Settings(data_dir=tmp_path, examples_dir=SAMPLES,
                                       max_upload_bytes=1000), trufor=None, dtd=None))
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


# --- check selection and document scan -----------------------------------

def test_scan_runs_only_selected_fast_checks(client: TestClient) -> None:
    path = SAMPLES / "2.jpg"
    r = client.post("/api/scan?checks=metadata",
                    files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    body = r.json()
    assert body["checks"] == ["metadata"]
    assert body["metadata"] is not None
    assert body["duplicates"] is None and body["ela"] is None


def test_scan_rejects_unknown_check(client: TestClient) -> None:
    path = SAMPLES / "2.jpg"
    r = client.post("/api/scan?checks=metadata,magic",
                    files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    assert r.status_code == 422


def test_skipped_duplicate_check_does_not_index_photo(client: TestClient) -> None:
    path = SAMPLES / "2.jpg"
    client.post("/api/scan?checks=metadata", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    assert _upload(client, path).json()["duplicates"] == []


def test_document_scan_adds_dtd_and_updates_verdict(client: TestClient) -> None:
    scan_id = _upload(client, SAMPLES / "2.jpg").json()["scan_id"]
    body = client.post(f"/api/scan/{scan_id}/document").json()
    assert body["dtd"]["suspicious"] is True
    assert body["dtd"]["overlay"].startswith("data:image/jpeg;base64,")
    assert body["verdict"]["level"] == "high"
    assert any("text" in r.lower() for r in body["verdict"]["reasons"])
    client.post(f"/api/scan/{scan_id}/document")
    assert client.app.state.fake_dtd.calls == 1


def test_deep_and_document_results_are_both_kept(client: TestClient) -> None:
    scan_id = _upload(client, SAMPLES / "2.jpg").json()["scan_id"]
    client.post(f"/api/scan/{scan_id}/deep")
    body = client.post(f"/api/scan/{scan_id}/document").json()
    assert body["trufor"] is not None and body["dtd"] is not None


def test_document_scan_without_dtd_is_503(tmp_path: Path) -> None:
    c = TestClient(create_app(Settings(data_dir=tmp_path, examples_dir=SAMPLES), trufor=None,
                              dtd=None))
    scan_id = _upload(c, SAMPLES / "2.jpg").json()["scan_id"]
    assert c.post(f"/api/scan/{scan_id}/document").status_code == 503


def test_verdict_high_when_dtd_suspicious() -> None:
    d = DtdReport(True, 1.0, "", [DtdRegion(0, 0, 9, 9, 0.99, 400)])
    v = combine(metadata_risk=0.0, duplicate=False, trufor=None, dtd=d)
    assert v.level == "high"


def test_verdict_is_unknown_when_nothing_could_judge_the_image(client: TestClient) -> None:
    """Only DTD ticked on a PNG: no check produced evidence, so don't say 'low risk'."""
    _ok, buf = cv2.imencode(".png", cv2.imread(str(SAMPLES / "2.jpg")))
    client.app.state.fake_dtd.report = DtdReport(False, 0.0, "Document scan needs a JPEG; this file is PNG")
    r = client.post("/api/scan?checks=", files={"file": ("x.png", io.BytesIO(buf.tobytes()), "image/png")})
    body = r.json()
    assert body["verdict"]["level"] == "unknown"
    body = client.post(f"/api/scan/{body['scan_id']}/document").json()
    assert body["verdict"]["level"] == "unknown"
    assert "not checked" in " ".join(body["verdict"]["reasons"]).lower()


def test_verdict_known_once_an_applicable_check_ran(client: TestClient) -> None:
    body = _upload(client, SAMPLES / "2.jpg").json()
    assert body["verdict"]["level"] in {"low", "medium", "high"}
