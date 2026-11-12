"""
trufor.py
---------
Wrapper around TruFor, a trained network for image-forgery detection and
localisation (Guillaro et al., CVPR 2023, https://grip-unina.github.io/TruFor).

TruFor combines the RGB image with a learned camera "noise fingerprint"
(Noiseprint++), so it works on real photos where hand-made checks such as ELA
fail. It returns an overall score (0 = authentic, 1 = manipulated) and a
per-pixel map of where the manipulation is.

The TruFor code and weights are NOT part of this repo: they are licensed for
non-commercial use only (GRIP-UNINA licence). Install them separately:

1. ``git clone https://github.com/grip-unina/TruFor`` and point ``trufor_dir``
   (or ``TRUFOR_DIR``) at ``TruFor/TruFor_train_test``.
2. Download the weights from the TruFor README and convert them once to a
   numbers-only state dict (see README "TruFor setup"), then point ``weights``
   (or ``TRUFOR_WEIGHTS``) at it. It loads with ``weights_only=True``, so the
   weights file cannot run code.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REGION_THRESHOLD = 0.5   # map value above which a pixel counts as manipulated
MIN_REGION_FRACTION = 0.002

# Yes/no rule, chosen on the real-world eval (scripts/eval_realworld.py) as the
# point with zero false alarms on 28 clean photos: catches 20/84 edits (24%).
# Clean photos often show small or weak regions, so those alone don't count.
SUSPICIOUS_SCORE = 0.7
SUSPICIOUS_FRACTION = 0.02
SUSPICIOUS_STRENGTH = 0.8


class TruForUnavailableError(RuntimeError):
    """TruFor code or weights are not installed."""


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int
    strength: float  # mean map value inside the region


@dataclass(frozen=True)
class TruForReport:
    score: float
    regions: list[Region] = field(default_factory=list)
    manipulated_fraction: float = 0.0
    heatmap: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def suspicious(self) -> bool:
        strongest = max((r.strength for r in self.regions), default=0.0)
        return self.score > SUSPICIOUS_SCORE or (
            self.manipulated_fraction >= SUSPICIOUS_FRACTION and strongest >= SUSPICIOUS_STRENGTH)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("heatmap")
        data["suspicious"] = self.suspicious
        return data


class TruForDetector:
    def __init__(self, trufor_dir: str | Path | None = None, weights: str | Path | None = None,
                 max_side: int = 1024, threads: int | None = None) -> None:
        self.trufor_dir = Path(trufor_dir or os.environ.get("TRUFOR_DIR", ""))
        self.weights = Path(weights or os.environ.get("TRUFOR_WEIGHTS", ""))
        self.max_side = max_side
        self.threads = threads
        self._model = None

    def analyze(self, path: str | Path) -> TruForReport:
        import torch
        import torch.nn.functional as F  # noqa: N812  (PyTorch convention)

        model = self._load()
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        full_w, full_h = rgb.size
        small = rgb.copy()
        small.thumbnail((self.max_side, self.max_side))
        x = torch.tensor(np.asarray(small).transpose(2, 0, 1), dtype=torch.float)[None] / 256.0

        with torch.no_grad():
            pred, _conf, det, _ = model(x)
        score = float(torch.sigmoid(det).item())
        heat = F.softmax(pred[0], dim=0)[1].numpy().astype(np.float32)
        heat = cv2.resize(heat, (full_w, full_h), interpolation=cv2.INTER_LINEAR)

        return TruForReport(score=round(score, 3), regions=_regions(heat),
                            manipulated_fraction=round(float((heat > REGION_THRESHOLD).mean()), 4),
                            heatmap=heat)

    def _load(self):
        if self._model is not None:
            return self._model
        if not (self.trufor_dir / "lib" / "config" / "trufor_ph3.yaml").is_file():
            raise TruForUnavailableError(f"TruFor code not found in '{self.trufor_dir}' (set TRUFOR_DIR)")
        if not self.weights.is_file():
            raise TruForUnavailableError(f"TruFor weights not found at '{self.weights}' (set TRUFOR_WEIGHTS)")

        import torch

        sys.path.insert(0, str(self.trufor_dir))
        try:
            from lib.config import config
            from lib.utils import get_model
        finally:
            sys.path.remove(str(self.trufor_dir))

        cfg = config.clone()
        cfg.defrost()
        cfg.merge_from_file(str(self.trufor_dir / "lib" / "config" / "trufor_ph3.yaml"))
        cfg.freeze()
        model = get_model(cfg)
        state = torch.load(self.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        if self.threads:
            torch.set_num_threads(self.threads)
        self._model = model
        return model


def _regions(heat: np.ndarray) -> list[Region]:
    mask = (heat > REGION_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = MIN_REGION_FRACTION * heat.size
    regions = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        regions.append(Region(int(x), int(y), int(w), int(h),
                              round(float(heat[labels == i].mean()), 3)))
    return sorted(regions, key=lambda r: r.w * r.h * r.strength, reverse=True)
