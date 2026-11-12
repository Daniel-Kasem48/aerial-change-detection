"""Tests for duplicate-photo fingerprints (src/forensics/fingerprint.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
from src.forensics.fingerprint import FingerprintIndex, hamming_distance, perceptual_hash

if TYPE_CHECKING:
    from pathlib import Path


def _scene(seed: int) -> np.ndarray:
    """Natural-ish BGR image: smooth gradients plus shapes, different per seed."""
    rng = np.random.default_rng(seed)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(12):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        center = (int(rng.integers(0, 320)), int(rng.integers(0, 240)))
        cv2.circle(img, center, int(rng.integers(10, 80)), color, -1)
    return cv2.GaussianBlur(img, (7, 7), 0)


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_hash_is_64_bit_and_deterministic() -> None:
    img = _scene(1)
    h = perceptual_hash(img)
    assert 0 <= h < 2**64
    assert perceptual_hash(img) == h


@pytest.mark.parametrize("transform", [
    lambda im: cv2.resize(im, (160, 120)),                       # downscaled
    lambda im: _jpeg_roundtrip(im, 60),                          # recompressed
    lambda im: cv2.convertScaleAbs(im, alpha=1.0, beta=25),      # brighter
    lambda im: im[6:-6, 8:-8],                                   # small crop
])
def test_same_photo_after_common_changes_stays_close(transform) -> None:
    img = _scene(2)
    assert hamming_distance(perceptual_hash(img), perceptual_hash(transform(img))) <= 10


def test_different_photos_are_far_apart() -> None:
    distances = [hamming_distance(perceptual_hash(_scene(a)), perceptual_hash(_scene(b)))
                 for a, b in [(3, 4), (5, 6), (7, 8)]]
    assert min(distances) > 18


def test_index_finds_reused_photo() -> None:
    index = FingerprintIndex()
    index.add("claim-001.jpg", perceptual_hash(_scene(10)))
    index.add("claim-002.jpg", perceptual_hash(_scene(11)))

    reused = _jpeg_roundtrip(cv2.resize(_scene(10), (200, 150)), 70)
    matches = index.find_similar(perceptual_hash(reused))

    assert [m.image_id for m in matches] == ["claim-001.jpg"]
    assert matches[0].distance <= 10
    assert 0.8 <= matches[0].similarity <= 1.0


def test_index_returns_nothing_for_new_photo() -> None:
    index = FingerprintIndex()
    index.add("claim-001.jpg", perceptual_hash(_scene(20)))
    assert index.find_similar(perceptual_hash(_scene(21))) == []


def test_index_saves_and_loads(tmp_path: Path) -> None:
    index = FingerprintIndex()
    h = perceptual_hash(_scene(30))
    index.add("a.jpg", h)
    path = tmp_path / "index.json"
    index.save(path)

    loaded = FingerprintIndex.load(path)
    assert loaded.find_similar(h)[0].image_id == "a.jpg"
