"""
app.py
------
FastAPI app for single-image fraud screening.

    POST /api/scan              upload a photo -> fast checks (~1 s), selectable
                                with ?checks=metadata,duplicates,ela
    POST /api/scan/{id}/deep    run TruFor on the same photo (~30 s on CPU),
                                cached per scan, updates the verdict
    POST /api/scan/{id}/document  DocTamper DTD for edited text (JPEG only)
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
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

from src.forensics.dtd import DtdDetector, DtdReport
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


FAST_CHECKS = ("metadata", "duplicates", "ela")


@dataclass
class _Scan:
    path: Path
    filename: str
    fast: dict[str, Any]
    metadata_risk: float
    duplicate: bool
    checks: tuple[str, ...] = ()
    trufor_report: TruForReport | None = None
    trufor: dict[str, Any] | None = None
    dtd_report: DtdReport | None = None
    dtd: dict[str, Any] | None = None

    def judged(self) -> bool:
        """True once a check that feeds the verdict could actually analyse the image."""
        return ("metadata" in self.checks or "duplicates" in self.checks
                or self.trufor_report is not None
                or (self.dtd_report is not None and self.dtd_report.applicable))

    def response(self) -> dict[str, Any]:
        if self.judged():
            verdict = combine(self.metadata_risk, self.duplicate, self.trufor_report,
                              self.dtd_report).to_dict()
        else:
            verdict = {"level": "unknown", "score": 0.0, "reasons": [
                "Not checked: none of the selected scans could analyse this image. "
                "Tick Metadata or Earlier copies, or upload a JPEG for ELA and the document scan."]}
        return {**self.fast, "verdict": verdict, "trufor": self.trufor, "dtd": self.dtd}


def create_app(settings: Settings | None = None, trufor: Any = _UNSET,
               dtd: Any = _UNSET) -> FastAPI:
    settings = settings or Settings()
    uploads = settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    index_path = settings.data_dir / "fingerprints.json"
    index = FingerprintIndex.load(index_path) if index_path.exists() else FingerprintIndex()
    if trufor is _UNSET:
        trufor = _trufor_from_env()
    if dtd is _UNSET:
        dtd = _dtd_from_env()

    scans: OrderedDict[str, _Scan] = OrderedDict()
    lock = threading.Lock()          # scans + index
    model_lock = threading.Lock()    # one model inference at a time (CPU bound)

    app = FastAPI(title="Fraud Scanner", version="0.2.0",
                  description="Single-image forensics: metadata, duplicates, ELA, TruFor, DTD")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "trufor": trufor is not None, "dtd": dtd is not None}

    @app.post("/api/scan")
    def scan(file: UploadFile = File(...),  # noqa: B008
             checks: str = Query(",".join(FAST_CHECKS),
                                 description="Comma-separated: metadata, duplicates, ela")
             ) -> dict[str, Any]:
        selected = [c for c in (x.strip() for x in checks.split(",")) if c]
        unknown = sorted(set(selected) - set(FAST_CHECKS))
        if unknown:
            raise HTTPException(422, f"Unknown check(s): {', '.join(unknown)}")

        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"File larger than {settings.max_upload_bytes // 2**20} MB")
        fmt = _image_format(data)
        if fmt not in _EXT:
            raise HTTPException(415, "Upload a JPEG, PNG or WebP image")

        scan_id = uuid.uuid4().hex[:12]
        path = uploads / f"{scan_id}{_EXT[fmt]}"
        path.write_bytes(data)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        timings: dict[str, int] = {}

        meta_out, meta_risk = None, 0.0
        if "metadata" in selected:
            t0 = time.perf_counter()
            meta = analyze_metadata(path)
            meta_out, meta_risk = meta.to_dict(), meta.risk
            timings["metadata"] = _ms(t0, time.perf_counter())

        duplicates = None
        if "duplicates" in selected:
            t0 = time.perf_counter()
            phash = perceptual_hash(bgr)
            with lock:
                matches = index.find_similar(phash)
                index.add(scan_id, phash)
                index.save(index_path)
                names = {sid: sc.filename for sid, sc in scans.items()}
            duplicates = [{"scan_id": m.image_id, "filename": names.get(m.image_id),
                           "distance": m.distance, "similarity": m.similarity} for m in matches]
            timings["duplicates"] = _ms(t0, time.perf_counter())

        ela_out = None
        if "ela" in selected:
            t0 = time.perf_counter()
            ela = analyze_ela(path)
            ela_out = ela.to_dict()
            ela_out["overlay"] = _overlay(bgr, ela.heatmap) if ela.heatmap is not None else None
            timings["ela"] = _ms(t0, time.perf_counter())

        h, w = bgr.shape[:2]
        fast = {
            "scan_id": scan_id,
            "filename": file.filename,
            "checks": selected,
            "image": {"width": w, "height": h, "format": fmt, "bytes": len(data),
                      "preview": _jpeg_data_url(bgr)},
            "metadata": meta_out,
            "duplicates": duplicates,
            "ela": ela_out,
            "timings_ms": timings,
        }
        entry = _Scan(path, file.filename or path.name, fast, meta_risk, bool(duplicates),
                      tuple(selected))
        with lock:
            scans[scan_id] = entry
            while len(scans) > settings.max_scans:
                _, old = scans.popitem(last=False)
                old.path.unlink(missing_ok=True)
        return entry.response()

    def _get(scan_id: str) -> _Scan:
        with lock:
            s = scans.get(scan_id)
        if s is None:
            raise HTTPException(404, "Unknown scan id")
        return s

    @app.post("/api/scan/{scan_id}/deep")
    def deep(scan_id: str) -> dict[str, Any]:
        """TruFor: general photo forgery detector (~30 s on CPU)."""
        s = _get(scan_id)
        if trufor is None:
            raise HTTPException(503, "TruFor is not installed on this server")
        if s.trufor is None:
            t0 = time.perf_counter()
            with model_lock:
                report: TruForReport = trufor.analyze(s.path)
            out = report.to_dict()
            out["overlay"] = _overlay(cv2.imread(str(s.path), cv2.IMREAD_COLOR), report.heatmap)
            out["ms"] = _ms(t0, time.perf_counter())
            s.trufor_report, s.trufor = report, out
        return s.response()

    @app.post("/api/scan/{scan_id}/document")
    def document(scan_id: str) -> dict[str, Any]:
        """DocTamper DTD: edited text in documents and receipts (JPEG only, ~1 min on CPU)."""
        s = _get(scan_id)
        if dtd is None:
            raise HTTPException(503, "Document scan (DTD) is not installed on this server")
        if s.dtd is None:
            t0 = time.perf_counter()
            with model_lock:
                report: DtdReport = dtd.analyze(s.path)
            out = report.to_dict()
            out["overlay"] = (_overlay(cv2.imread(str(s.path), cv2.IMREAD_COLOR), report.heatmap)
                              if report.heatmap is not None else None)
            out["ms"] = _ms(t0, time.perf_counter())
            s.dtd_report, s.dtd = report, out
        return s.response()

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


def _dtd_from_env() -> DtdDetector | None:
    d, w = os.environ.get("DTD_DIR"), os.environ.get("DTD_WEIGHTS")
    if d and w and Path(d).is_dir() and Path(w).is_file():
        return DtdDetector(dtd_dir=d, weights=w)
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
