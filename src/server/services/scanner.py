"""
scanner.py
----------
ScanService: runs the checks for one upload and keeps the results.

Fast checks (metadata, duplicates, ELA) run on upload. The model scans
(TruFor, DTD) run on request, once per scan (cached), one inference at a
time because they are CPU bound.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import cv2

from src.forensics.ela import analyze_ela
from src.forensics.fingerprint import FingerprintIndex, perceptual_hash
from src.forensics.metadata import analyze_metadata
from src.server.errors import (
    ExampleNotFoundError,
    InvalidChecksError,
    ModelUnavailableError,
    UnsupportedMediaError,
    UploadTooLargeError,
)
from src.server.schemas import FAST_CHECKS

from .imaging import EXTENSIONS, MEDIA_TYPES, jpeg_data_url, overlay_data_url, sniff_format
from .store import ScanRecord, ScanStore
from .verdict import combine, not_checked

if TYPE_CHECKING:
    from pathlib import Path

    from src.server.config import Settings

    from .detectors import DtdLike, TruForLike


class ScanService:
    def __init__(self, settings: Settings, trufor: TruForLike | None,
                 dtd: DtdLike | None) -> None:
        self.settings = settings
        self.trufor = trufor
        self.dtd = dtd
        self.uploads = settings.data_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self._index_path = settings.data_dir / "fingerprints.json"
        self._index = (FingerprintIndex.load(self._index_path) if self._index_path.exists()
                       else FingerprintIndex())
        self._index_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self.store = ScanStore(settings.max_scans)

    # -- public API ------------------------------------------------------

    @staticmethod
    def parse_checks(raw: str) -> tuple[str, ...]:
        selected = tuple(c for c in (x.strip() for x in raw.split(",")) if c)
        unknown = sorted(set(selected) - set(FAST_CHECKS))
        if unknown:
            raise InvalidChecksError(f"Unknown check(s): {', '.join(unknown)}")
        return selected

    def scan(self, data: bytes, filename: str | None, checks: tuple[str, ...]) -> dict[str, Any]:
        if len(data) > self.settings.max_upload_bytes:
            raise UploadTooLargeError(
                f"File larger than {self.settings.max_upload_bytes // 2**20} MB")
        fmt = sniff_format(data)
        if fmt not in EXTENSIONS:
            raise UnsupportedMediaError("Upload a JPEG, PNG or WebP image")

        scan_id = uuid.uuid4().hex[:12]
        path = self.uploads / f"{scan_id}{EXTENSIONS[fmt]}"
        path.write_bytes(data)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        record = ScanRecord(scan_id, path, filename, checks, fast={})
        timings: dict[str, int] = {}

        metadata = None
        if "metadata" in checks:
            t0 = time.perf_counter()
            meta = analyze_metadata(path)
            metadata, record.metadata_risk = meta.to_dict(), meta.risk
            timings["metadata"] = _ms_since(t0)

        duplicates = None
        if "duplicates" in checks:
            t0 = time.perf_counter()
            duplicates = self._find_duplicates(scan_id, perceptual_hash(bgr))
            record.duplicate = bool(duplicates)
            timings["duplicates"] = _ms_since(t0)

        ela = None
        if "ela" in checks:
            t0 = time.perf_counter()
            report = analyze_ela(path)
            ela = report.to_dict()
            ela["overlay"] = (overlay_data_url(bgr, report.heatmap)
                              if report.heatmap is not None else None)
            timings["ela"] = _ms_since(t0)

        h, w = bgr.shape[:2]
        record.fast = {
            "scan_id": scan_id,
            "filename": filename,
            "checks": list(checks),
            "image": {"width": w, "height": h, "format": fmt, "bytes": len(data),
                      "preview": jpeg_data_url(bgr)},
            "metadata": metadata,
            "duplicates": duplicates,
            "ela": ela,
            "timings_ms": timings,
        }
        self.store.add(record)
        return self.result(record)

    def run_trufor(self, scan_id: str) -> dict[str, Any]:
        record = self.store.get(scan_id)
        if self.trufor is None:
            raise ModelUnavailableError("TruFor is not installed on this server")
        with record.lock:
            if record.trufor is None:
                t0 = time.perf_counter()
                with self._model_lock:
                    report = self.trufor.analyze(record.path)
                out = report.to_dict()
                out["overlay"] = overlay_data_url(_read(record.path), report.heatmap)
                out["ms"] = _ms_since(t0)
                record.trufor_report, record.trufor = report, out
        return self.result(record)

    def run_dtd(self, scan_id: str) -> dict[str, Any]:
        record = self.store.get(scan_id)
        if self.dtd is None:
            raise ModelUnavailableError("Document scan (DTD) is not installed on this server")
        with record.lock:
            if record.dtd is None:
                t0 = time.perf_counter()
                with self._model_lock:
                    report = self.dtd.analyze(record.path)
                out = report.to_dict()
                out["overlay"] = (overlay_data_url(_read(record.path), report.heatmap)
                                  if report.heatmap is not None else None)
                out["ms"] = _ms_since(t0)
                record.dtd_report, record.dtd = report, out
        return self.result(record)

    def result(self, record: ScanRecord) -> dict[str, Any]:
        verdict = (combine(record.metadata_risk, record.duplicate, record.trufor_report,
                           record.dtd_report) if record.judged else not_checked())
        return {**record.fast, "verdict": verdict.to_dict(),
                "trufor": record.trufor, "dtd": record.dtd}

    def example_names(self) -> list[str]:
        folder = self.settings.examples_dir
        if not folder.is_dir():
            return []
        return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in MEDIA_TYPES)

    def example_path(self, name: str) -> tuple[Path, str]:
        if name not in self.example_names():  # whitelist: blocks path traversal
            raise ExampleNotFoundError("No such example")
        path = self.settings.examples_dir / name
        return path, MEDIA_TYPES[path.suffix.lower()]

    # -- internals -------------------------------------------------------

    def _find_duplicates(self, scan_id: str, phash: int) -> list[dict[str, Any]]:
        with self._index_lock:
            matches = self._index.find_similar(phash)
            self._index.add(scan_id, phash)
            self._index.save(self._index_path)
        names = self.store.filenames()
        return [{"scan_id": m.image_id, "filename": names.get(m.image_id),
                 "distance": m.distance, "similarity": m.similarity} for m in matches]


def _read(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _ms_since(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)
