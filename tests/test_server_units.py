"""Unit tests for the server's config, store and service helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.server.config import Settings
from src.server.errors import InvalidChecksError, ScanNotFoundError
from src.server.main import create_app
from src.server.services.scanner import ScanService
from src.server.services.store import ScanRecord, ScanStore


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCANNER_DATA", str(tmp_path))
    monkeypatch.setenv("SCANNER_MAX_SCANS", "7")
    monkeypatch.setenv("TRUFOR_DIR", "/models/trufor")
    s = Settings()
    assert s.data_dir == tmp_path
    assert s.max_scans == 7
    assert s.trufor_dir == Path("/models/trufor")
    assert s.dtd_dir is None


def test_settings_reject_invalid_limits() -> None:
    with pytest.raises(ValueError):
        Settings(max_scans=0)


def test_store_evicts_oldest_and_deletes_its_upload(tmp_path: Path) -> None:
    store = ScanStore(max_scans=2)
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        p.write_bytes(b"x")
        paths.append(p)
        store.add(ScanRecord(str(i), p, f"{i}.jpg", (), fast={}))
    assert not paths[0].exists() and paths[1].exists() and paths[2].exists()
    with pytest.raises(ScanNotFoundError):
        store.get("0")
    assert store.get("2").filename == "2.jpg"


def test_parse_checks_strips_and_validates() -> None:
    assert ScanService.parse_checks(" metadata, ela ,") == ("metadata", "ela")
    assert ScanService.parse_checks("") == ()
    with pytest.raises(InvalidChecksError):
        ScanService.parse_checks("metadata,magic")


def test_openapi_documents_typed_scan_response(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path), trufor=None, dtd=None)
    schema = TestClient(app).get("/openapi.json").json()
    assert "ScanResponse" in schema["components"]["schemas"]
    ok = schema["paths"]["/api/scan"]["post"]["responses"]["200"]
    assert ok["content"]["application/json"]["schema"]["$ref"].endswith("/ScanResponse")
    assert {"/api/health", "/api/scan/{scan_id}/deep", "/api/scan/{scan_id}/document"} <= set(schema["paths"])
