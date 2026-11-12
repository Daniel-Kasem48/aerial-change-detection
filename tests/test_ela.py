"""Tests for error level analysis (src/forensics/ela.py)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from src.forensics.ela import analyze_ela, render_heatmap

PATCH = (192, 128, 96, 96)  # x, y, w, h of the pasted area


def _texture(seed: int, h: int = 384, w: int = 384) -> np.ndarray:
    """Photo-like BGR image: soft shapes + fine grain."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 120, dtype=np.float32)
    for _ in range(15):
        color = rng.integers(40, 220, 3).tolist()
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(20, 120)), color, -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _write_jpeg(path: Path, img: np.ndarray, quality: int) -> Path:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


@pytest.fixture
def spliced(tmp_path: Path) -> Path:
    """Camera JPEG (q70) with an uncompressed patch pasted in, saved again at q92."""
    photo = _jpeg(_texture(1), 70)
    x, y, w, h = PATCH
    photo[y:y + h, x:x + w] = _texture(2)[y:y + h, x:x + w]
    return _write_jpeg(tmp_path / "spliced.jpg", photo, 92)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def test_pasted_patch_is_located(spliced: Path) -> None:
    report = analyze_ela(spliced)
    assert report.applicable
    assert report.regions, report.reason
    best = report.regions[0]
    assert _iou((best.x, best.y, best.w, best.h), PATCH) > 0.4
    assert report.score >= 0.5


def test_untouched_single_save_has_no_regions(tmp_path: Path) -> None:
    report = analyze_ela(_write_jpeg(tmp_path / "clean.jpg", _texture(3), 85))
    assert report.applicable
    assert report.regions == []
    assert report.score < 0.3


def test_untouched_double_save_has_no_regions(tmp_path: Path) -> None:
    """Whole photo recompressed (e.g. re-saved by an app) is not an edit."""
    img = _jpeg(_texture(4), 70)
    report = analyze_ela(_write_jpeg(tmp_path / "resaved.jpg", img, 92))
    assert report.regions == []
    assert report.score < 0.3


def test_estimates_jpeg_quality(tmp_path: Path) -> None:
    report = analyze_ela(_write_jpeg(tmp_path / "q85.jpg", _texture(5), 85))
    assert report.estimated_quality == pytest.approx(85, abs=3)


def test_png_is_not_applicable(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    cv2.imwrite(str(path), _texture(6))
    report = analyze_ela(path)
    assert not report.applicable
    assert report.score == 0.0
    assert "JPEG" in report.reason


def test_heatmap_matches_image_size(spliced: Path) -> None:
    report = analyze_ela(spliced)
    assert report.heatmap.shape == (384, 384)
    assert float(report.heatmap.min()) >= 0.0 and float(report.heatmap.max()) <= 1.0
    overlay = render_heatmap(cv2.imread(str(spliced)), report.heatmap)
    assert overlay.shape == (384, 384, 3)


def test_report_dict_is_json_without_heatmap(spliced: Path) -> None:
    data = json.loads(json.dumps(analyze_ela(spliced).to_dict()))
    assert "heatmap" not in data
    assert {"x", "y", "w", "h", "strength"} <= set(data["regions"][0])


def test_unreadable_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        analyze_ela(bad)


SAMPLES = Path(__file__).parent.parent / "samples"


def test_thin_painted_lines_on_real_photo_are_located(tmp_path: Path) -> None:
    """Painted scratches on a real messenger photo: weak per block, long connected trail."""
    photo = cv2.imread(str(SAMPLES / "2.jpg"))
    for i in range(6):
        cv2.line(photo, (300 + i * 25, 820), (620 + i * 20, 1050), (40, 40, 40), 4)
    report = analyze_ela(_write_jpeg(tmp_path / "scratched.jpg", photo, 90))
    assert any(_iou((r.x, r.y, r.w, r.h), (300, 820, 420, 230)) > 0.2 for r in report.regions), \
        report.regions


@pytest.mark.parametrize("name", sorted(p.name for p in SAMPLES.glob("*.jpg")))
def test_untouched_real_photos_have_no_regions(name: str) -> None:
    report = analyze_ela(SAMPLES / name)
    assert report.regions == [], report.regions
