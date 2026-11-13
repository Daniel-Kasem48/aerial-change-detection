"""Tests for the DocTamper DTD wrapper (src/forensics/dtd.py).

Model tests need the DocTamper code and weights (not in this repo,
non-commercial). Set DTD_DIR (the DocTamper/models folder) and DTD_WEIGHTS
(dtd_sroie.pth or dtd_doctamper.pth) to run them. Each takes 1-4 minutes on CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from src.forensics.dtd import DtdDetector, DtdReport, DtdUnavailableError, Region

DTD_DIR = os.environ.get("DTD_DIR", "")
DTD_WEIGHTS = os.environ.get("DTD_WEIGHTS", "")
RECEIPTS = Path(os.environ.get("DTD_RECEIPTS", Path.home() / "datasets/forensics/receipt_test"))
needs_model = pytest.mark.skipif(
    not (Path(DTD_DIR).is_dir() and Path(DTD_WEIGHTS).is_file()),
    reason="DTD_DIR / DTD_WEIGHTS not set",
)


def test_missing_install_gives_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "x.jpg"
    cv2.imwrite(str(path), np.full((64, 64, 3), 200, np.uint8))
    with pytest.raises(DtdUnavailableError):
        DtdDetector(dtd_dir=tmp_path / "nope", weights=tmp_path / "nope.pth").analyze(path)


def test_non_jpeg_is_not_applicable_without_loading_model(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    cv2.imwrite(str(path), np.full((64, 64, 3), 200, np.uint8))
    report = DtdDetector(dtd_dir=tmp_path / "nope", weights=tmp_path / "nope.pth").analyze(path)
    assert not report.applicable
    assert "JPEG" in report.reason
    assert not report.suspicious


def test_suspicious_needs_strong_and_not_tiny_region() -> None:
    strong = Region(0, 0, 20, 40, strength=0.95, pixels=500)
    speck = Region(0, 0, 3, 9, strength=0.77, pixels=27)
    assert DtdReport(True, 1.0, "", regions=[strong]).suspicious
    assert not DtdReport(True, 0.77, "", regions=[speck]).suspicious
    assert not DtdReport(True, 0.4, "", regions=[]).suspicious


def test_report_dict_is_json_without_heatmap() -> None:
    report = DtdReport(True, 1.0, "x", regions=[Region(1, 2, 3, 4, 0.9, 12)],
                       heatmap=np.zeros((4, 4), np.float32))
    data = json.loads(json.dumps(report.to_dict()))
    assert "heatmap" not in data
    assert data["suspicious"] is False  # 12 px is below the minimum size
    assert data["regions"][0]["pixels"] == 12


@pytest.fixture(scope="module")
def detector() -> DtdDetector:
    return DtdDetector(dtd_dir=DTD_DIR, weights=DTD_WEIGHTS)


@needs_model
@pytest.mark.skipif(not (RECEIPTS / "receipt_forged.jpg").is_file(), reason="receipt test file missing")
def test_forged_receipt_digits_are_found(detector: DtdDetector) -> None:
    report = detector.analyze(RECEIPTS / "receipt_forged.jpg")
    assert report.applicable and report.suspicious
    # both changed digits sit around x 935-962, y 994-1115
    assert any(900 <= r.x + r.w / 2 <= 1000 and 980 <= r.y + r.h / 2 <= 1130 for r in report.regions)
    assert report.heatmap.shape == cv2.imread(str(RECEIPTS / "receipt_forged.jpg")).shape[:2]
