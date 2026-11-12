"""
fingerprint.py
--------------
Perceptual hashing to catch reused photos (e.g. the same damage photo sent in
two insurance claims, or a picture taken from the internet).

``perceptual_hash`` is a 64-bit DCT hash (pHash): it survives resizing, JPEG
recompression, brightness changes and small crops, but changes completely for
a different scene. Compare hashes with ``hamming_distance`` (0 = identical,
<= 10 = very likely the same photo).

``FingerprintIndex`` is a simple in-memory index with JSON persistence. It does
a linear scan, which is fine for tens of thousands of images; swap in a
database (or a BK-tree) behind the same interface for more.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MAX_DISTANCE = 10
_HASH_BITS = 64


def perceptual_hash(image: np.ndarray) -> int:
    """64-bit pHash of a BGR or grayscale image."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])  # skip the DC term so brightness doesn't matter
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class Match:
    image_id: str
    distance: int

    @property
    def similarity(self) -> float:
        return round(1.0 - self.distance / _HASH_BITS, 3)


class FingerprintIndex:
    def __init__(self) -> None:
        self._hashes: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._hashes)

    def add(self, image_id: str, phash: int) -> None:
        self._hashes[image_id] = phash

    def find_similar(self, phash: int, max_distance: int = DEFAULT_MAX_DISTANCE) -> list[Match]:
        """All stored images within *max_distance* bits, closest first."""
        matches = [Match(image_id, hamming_distance(phash, stored))
                   for image_id, stored in self._hashes.items()]
        return sorted((m for m in matches if m.distance <= max_distance),
                      key=lambda m: m.distance)

    def save(self, path: str | Path) -> None:
        data = {image_id: f"{h:016x}" for image_id, h in self._hashes.items()}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FingerprintIndex":
        index = cls()
        for image_id, hex_hash in json.loads(Path(path).read_text()).items():
            index.add(image_id, int(hex_hash, 16))
        return index
