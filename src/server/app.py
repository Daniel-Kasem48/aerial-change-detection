"""
app.py
------
FastAPI app for single-image fraud screening.

    POST /api/scan              upload a photo -> fast checks (~1 s):
                                metadata, duplicate search, ELA heatmap, verdict
    POST /api/scan/{id}/deep    run TruFor on the same photo (~25 s on CPU),
                                cached per scan, updates the verdict
    GET  /api/health            status + whether TruFor is available
    GET  /api/examples[/{name}] bundled example photos for the demo UI
    GET  /                      single-page UI

Uploads are stored under ``data_dir`` and the last ``max_scans`` are kept.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

from src.forensics.ela import analyze_ela, render_heatmap
from src.forensics.fingerprint import FingerprintIndex, perceptual_hash
from src.forensics.metadata import analyze_metadata
from src.forensics.trufor import TruForDetector, TruForReport

from .verdict import combine

if TYPE_CHECKING:
    import numpy as np

STATIC = Path(__file__).parent / "static"
_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_OVERLAY_MAX_SIDE = 1400
_UNSET: Any = object()


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("SCANNER_DATA", "data")))
    examples_dir: Path = Path(__file__).resolve().parents[2] / "samples"
    max_upload_bytes: int = 15 * 1024 * 1024
    max_scans: int = 200


@dataclass
class _Scan:
    path: Path
    filename: str
    fast: dict[str, Any]
    metadata_risk: float
    duplicate: bool
    trufor: dict[str, Any] | None = None


def create_app(settings: Settings | None = None, trufor: Any = _UNSET) -> FastAPI:
    settings = settings or Settings()
    uploads = settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    index_path = settings.data_dir / "fingerprints.json"
    index = FingerprintIndex.load(index_path) if index_path.exists() else FingerprintIndex()
    if trufor is _UNSET:
        trufor = _trufor_from_env()

    scans: OrderedDict[str, _Scan] = OrderedDict()
    lock = threading.Lock()          # scans + index
    model_lock = threading.Lock()    # one TruFor inference at a time

    app = FastAPI(title="Fraud Scanner", version="0.1.0",
                  description="Single-image forensics: metadata, duplicates, ELA, TruFor")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "trufor": trufor is not None}

    @app.post("/api/scan")
    def scan(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"File larger than {settings.max_upload_bytes // 2**20} MB")
        fmt = _image_format(data)
        if fmt not in _EXT:
            raise HTTPException(415, "Upload a JPEG, PNG or WebP image")

        scan_id = uuid.uuid4().hex[:12]
        path = uploads / f"{scan_id}{_EXT[fmt]}"
        path.write_bytes(data)
        t0 = time.perf_counter()

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        meta = analyze_metadata(path)
        t_meta = time.perf_counter()

        phash = perceptual_hash(bgr)
        with lock:
            matches = index.find_similar(phash)
            index.add(scan_id, phash)
            index.save(index_path)
            names = {sid: s.filename for sid, s in scans.items()}
        duplicates = [{"scan_id": m.image_id, "filename": names.get(m.image_id),
                       "distance": m.distance, "similarity": m.similarity} for m in matches]
        t_dup = time.perf_counter()

        ela = analyze_ela(path)
        ela_out = ela.to_dict()
        ela_out["overlay"] = _overlay(bgr, ela.heatmap) if ela.heatmap is not None else None
        t_ela = time.perf_counter()

        verdict = combine(meta.risk, bool(duplicates), None)
        h, w = bgr.shape[:2]
        fast = {
            "scan_id": scan_id,
            "filename": file.filename,
            "image": {"width": w, "height": h, "format": fmt, "bytes": len(data),
                      "preview": _jpeg_data_url(bgr)},
            "metadata": meta.to_dict(),
            "duplicates": duplicates,
            "ela": ela_out,
            "verdict": verdict.to_dict(),
            "timings_ms": {"metadata": _ms(t0, t_meta), "duplicates": _ms(t_meta, t_dup),
                           "ela": _ms(t_dup, t_ela)},
        }
        with lock:
            scans[scan_id] = _Scan(path, file.filename or path.name, fast, meta.risk,
                                   bool(duplicates))
            while len(scans) > settings.max_scans:
                _, old = scans.popitem(last=False)
                old.path.unlink(missing_ok=True)
        return {**fast, "trufor": None}

    @app.post("/api/scan/{scan_id}/deep")
    def deep(scan_id: str) -> dict[str, Any]:
        with lock:
            s = scans.get(scan_id)
        if s is None:
            raise HTTPException(404, "Unknown scan id")
        if trufor is None:
            raise HTTPException(503, "TruFor is not installed on this server")
        if s.trufor is None:
            t0 = time.perf_counter()
            with model_lock:
                report: TruForReport = trufor.analyze(s.path)
            bgr = cv2.imread(str(s.path), cv2.IMREAD_COLOR)
            out = report.to_dict()
            out["overlay"] = _overlay(bgr, report.heatmap)
            out["ms"] = _ms(t0, time.perf_counter())
            s.trufor = out
            s.fast["verdict"] = combine(s.metadata_risk, s.duplicate, report).to_dict()
        return {**s.fast, "trufor": s.trufor}

    @app.get("/api/examples")
    def examples() -> list[str]:
        return _example_names(settings.examples_dir)

    @app.get("/api/examples/{name}")
    def example(name: str) -> FileResponse:
        if name not in _example_names(settings.examples_dir):
            raise HTTPException(404, "No such example")
        path = settings.examples_dir / name
        return FileResponse(path, media_type=_MEDIA[path.suffix.lower()])

    return app


# ---------------------------------------------------------------------------

def _trufor_from_env() -> TruForDetector | None:
    d, w = os.environ.get("TRUFOR_DIR"), os.environ.get("TRUFOR_WEIGHTS")
    if d and w and Path(d).is_dir() and Path(w).is_file():
        return TruForDetector(trufor_dir=d, weights=w)
    return None


def _image_format(data: bytes) -> str | None:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
            return img.format
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None


def _example_names(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in _MEDIA)


def _fit(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = _OVERLAY_MAX_SIDE / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr


def _jpeg_data_url(bgr: np.ndarray) -> str:
    _ok, buf = cv2.imencode(".jpg", _fit(bgr), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _overlay(bgr: np.ndarray, heatmap: np.ndarray) -> str:
    return _jpeg_data_url(render_heatmap(bgr, heatmap, alpha=0.55))


def _ms(a: float, b: float) -> int:
    return round((b - a) * 1000)
