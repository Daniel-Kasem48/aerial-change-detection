"""Tests for the TruFor wrapper (src/forensics/trufor.py).

Needs the TruFor code and weights, which are not in this repo (non-commercial
licence, 275 MB). Set TRUFOR_DIR (the TruFor_train_test folder) and
TRUFOR_WEIGHTS (numbers-only state dict, see README) to run them.
Each inference takes ~25 s on a laptop CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import pytest
from src.forensics.trufor import TruForDetector, TruForUnavailableError

TRUFOR_DIR = os.environ.get("TRUFOR_DIR", "")
TRUFOR_WEIGHTS = os.environ.get("TRUFOR_WEIGHTS", "")
SAMPLES = Path(__file__).parent.parent / "samples"
needs_model = pytest.mark.skipif(
    not (Path(TRUFOR_DIR).is_dir() and Path(TRUFOR_WEIGHTS).is_file()),
    reason="TRUFOR_DIR / TRUFOR_WEIGHTS not set",
)


def test_missing_install_gives_clear_error(tmp_path: Path) -> None:
    detector = TruForDetector(trufor_dir=tmp_path / "nope", weights=tmp_path / "nope.pt")
    with pytest.raises(TruForUnavailableError):
        detector.analyze(SAMPLES / "2.jpg")


@pytest.fixture(scope="module")
def detector() -> TruForDetector:
    return TruForDetector(trufor_dir=TRUFOR_DIR, weights=TRUFOR_WEIGHTS, max_side=768)


@pytest.fixture(scope="module")
def pasted(tmp_path_factory) -> Path:
    photo = cv2.imread(str(SAMPLES / "2.jpg"))
    donor = cv2.resize(cv2.imread(str(SAMPLES / "before.jpg")), (400, 300))
    photo[700:1000, 700:1100] = donor
    path = tmp_path_factory.mktemp("tf") / "pasted.jpg"
    cv2.imwrite(str(path), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


@needs_model
def test_untouched_photo_scores_low(detector: TruForDetector) -> None:
    assert detector.analyze(SAMPLES / "2.jpg").score < 0.5


@needs_model
def test_pasted_area_scores_high_and_is_located(detector: TruForDetector, pasted: Path) -> None:
    report = detector.analyze(pasted)
    assert report.score > 0.9
    assert report.regions
    r = report.regions[0]
    cx, cy = r.x + r.w / 2, r.y + r.h / 2
    assert 700 <= cx <= 1100 and 700 <= cy <= 1000
    assert report.heatmap.shape == (1280, 1280)  # back at full resolution
    data = json.loads(json.dumps(report.to_dict()))
    assert "heatmap" not in data and data["score"] == report.score


def test_suspicious_rule_uses_score_or_strong_large_region() -> None:
    from src.forensics.trufor import Region, TruForReport

    assert TruForReport(score=0.8).suspicious
    assert not TruForReport(score=0.2).suspicious
    big_strong = Region(0, 0, 10, 10, strength=0.85)
    assert TruForReport(score=0.2, regions=[big_strong], manipulated_fraction=0.03).suspicious
    # small or weak regions alone are normal on clean photos
    assert not TruForReport(score=0.2, regions=[big_strong], manipulated_fraction=0.01).suspicious
    weak = Region(0, 0, 10, 10, strength=0.7)
    assert not TruForReport(score=0.2, regions=[weak], manipulated_fraction=0.05).suspicious
    assert "suspicious" in TruForReport(score=0.8).to_dict()
