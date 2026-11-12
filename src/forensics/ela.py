"""
ela.py
------
Error level analysis (ELA) for JPEG photos.

Every JPEG save rounds the image onto a quality-dependent grid. An area that
was pasted in has a different save history than the rest of the photo, so
when the photo is re-saved it changes by a different amount.

Plain ELA (re-save once, show the difference) mostly highlights edges and
texture. This version is stricter:

1. Re-save at several qualities (55-95). The quality the original photo was
   first saved at makes untouched areas barely change, while a pasted area
   that never saw that grid changes a lot.
2. Work on luminance only (colour noise inflates error in flat areas).
   Per 16x16 block, compare the error with what the block's texture predicts
   (a log-log fit across all blocks). Only error that the texture does not
   explain counts.
3. Robust z-scores (median / MAD) per quality; a block's score is its maximum
   over qualities. Connected high-score blocks become regions.

Only meaningful for JPEG input. Heavy recompression (messengers) washes the
traces out, so a clean result is not proof of authenticity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

BLOCK = 16
QUALITIES = tuple(range(55, 100, 5))
Z_THRESHOLD = 5.0
MIN_REGION_BLOCKS = 4
MIN_MEDIAN_ERROR = 0.15  # mean abs pixel error per block
MIN_MAD = 0.15          # log units; stops tiny-noise images from exploding z

# Standard JPEG luminance table (quality 50), used to estimate quality.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int
    strength: float  # peak robust z-score inside the region


@dataclass(frozen=True)
class ElaReport:
    applicable: bool
    score: float
    reason: str
    estimated_quality: int | None = None
    regions: list[Region] = field(default_factory=list)
    heatmap: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("heatmap")
        return data


def analyze_ela(path: str | Path) -> ElaReport:
    """Run ELA on the image at *path*.

    Raises:
        ValueError: if the file is not a readable image.
    """
    path = Path(path)
    try:
        with Image.open(path) as img:
            fmt = img.format
            quant = getattr(img, "quantization", None) or {}
            rgb = np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Not a readable image: {path}") from exc

    if fmt != "JPEG":
        return ElaReport(False, 0.0, f"ELA needs a JPEG; this file is {fmt}")

    quality = _estimate_quality(quant)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if h < BLOCK * 4 or w < BLOCK * 4:
        return ElaReport(False, 0.0, "Image too small for ELA", quality)

    z = _block_scores(bgr)
    heatmap = _heatmap(z, h, w)
    regions = _regions(z)

    if regions:
        peak = regions[0].strength
        score = float(np.clip((peak - Z_THRESHOLD) / 6.0 + 0.5, 0.5, 1.0))
        reason = f"{len(regions)} region(s) compress differently from the rest of the photo"
    else:
        score = float(np.clip((z.max() - 2.0) / (Z_THRESHOLD - 2.0), 0.0, 1.0) * 0.25)
        reason = "No area compresses differently from the rest of the photo"
    if quality is not None and quality < 70:
        reason += f"; low JPEG quality ({quality}) may hide traces"

    return ElaReport(True, round(score, 3), reason, quality, regions, heatmap)


def render_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay *heatmap* (0..1, HxW) on a BGR *image*."""
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)


# ---------------------------------------------------------------------------

def _estimate_quality(quant: dict[int, Any]) -> int | None:
    """Invert the IJG quality scaling from the luminance table."""
    if 0 not in quant:
        return None
    table = np.asarray(quant[0], dtype=np.float64)
    scale = table.mean() / _STD_LUMA.mean() * 100.0
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return int(round(np.clip(quality, 1, 100)))


def _block_scores(bgr: np.ndarray) -> np.ndarray:
    """Robust z-score per block, max over re-save qualities."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Texture from a blurred copy: JPEG removes fine grain, so measuring texture
    # on the raw pixels would make recompressed areas look "smooth".
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.5)
    log_tex = np.log(_block_mean(np.abs(cv2.Laplacian(smooth, cv2.CV_32F))) + 1.0).ravel()

    best = np.full((gray.shape[0] // BLOCK, gray.shape[1] // BLOCK), -np.inf)
    for q in QUALITIES:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        resaved = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        err = _block_mean(np.abs(gray - cv2.cvtColor(resaved, cv2.COLOR_BGR2GRAY).astype(np.float32)))
        if np.median(err) < MIN_MEDIAN_ERROR:
            continue  # ~ the photo's own last-save quality: nothing to learn here
        log_err = np.log(err + 0.1)
        resid = log_err - _robust_fit(log_tex, log_err.ravel()).reshape(err.shape)
        med = np.median(resid)
        mad = max(np.median(np.abs(resid - med)) * 1.4826, MIN_MAD)
        best = np.maximum(best, (resid - med) / mad)
    return np.where(np.isfinite(best), best, 0.0)


def _robust_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Line fit of y on x, refitted without outliers so edited blocks don't bend it."""
    keep = np.ones_like(x, dtype=bool)
    for _ in range(3):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        resid = y - (slope * x + intercept)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) * 1.4826 + 1e-6
        keep = np.abs(resid - np.median(resid[keep])) < 2.5 * mad
    return slope * x + intercept


def _block_mean(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    bh, bw = h // BLOCK, w // BLOCK
    return arr[:bh * BLOCK, :bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK).mean(axis=(1, 3))


def _regions(z: np.ndarray) -> list[Region]:
    mask = (z > Z_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if area < MIN_REGION_BLOCKS:
            continue
        peak = float(z[labels == i].max())
        regions.append(Region(int(bx * BLOCK), int(by * BLOCK), int(bw * BLOCK),
                              int(bh * BLOCK), round(peak, 2)))
    return sorted(regions, key=lambda r: r.strength, reverse=True)


def _heatmap(z: np.ndarray, h: int, w: int) -> np.ndarray:
    heat = np.clip(z / (Z_THRESHOLD * 2), 0.0, 1.0).astype(np.float32)
    heat = cv2.resize(heat, (z.shape[1] * BLOCK, z.shape[0] * BLOCK),
                      interpolation=cv2.INTER_LINEAR)
    full = np.zeros((h, w), dtype=np.float32)
    full[:heat.shape[0], :heat.shape[1]] = heat
    return full
