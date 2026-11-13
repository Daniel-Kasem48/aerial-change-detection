"""Image helpers: upload sniffing and JPEG data URLs for previews and heatmaps."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import cv2
from PIL import Image, UnidentifiedImageError

from src.forensics.ela import render_heatmap

if TYPE_CHECKING:
    import numpy as np

EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
PREVIEW_MAX_SIDE = 1400


def sniff_format(data: bytes) -> str | None:
    """Real image format from the bytes (not the file name), or None."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
            return img.format
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None


def jpeg_data_url(bgr: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> str:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    _ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def overlay_data_url(bgr: np.ndarray, heatmap: np.ndarray) -> str:
    return jpeg_data_url(render_heatmap(bgr, heatmap, alpha=0.55))
