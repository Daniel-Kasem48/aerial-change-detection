from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Query, UploadFile

from src.server.dependencies import ScanServiceDep
from src.server.schemas import FAST_CHECKS, ScanResponse

router = APIRouter(prefix="/api/scan", tags=["scans"])

# Handlers are plain `def`: the work is CPU bound, so FastAPI runs them in its
# thread pool instead of blocking the event loop.


@router.post("", response_model=ScanResponse, summary="Upload a photo and run fast checks")
def create_scan(
    service: ScanServiceDep,
    file: Annotated[UploadFile, File(description="JPEG, PNG or WebP, max 15 MB")],
    checks: Annotated[str, Query(description="Comma-separated: metadata, duplicates, ela")]
        = ",".join(FAST_CHECKS),
) -> dict[str, Any]:
    selected = service.parse_checks(checks)
    data = file.file.read(service.settings.max_upload_bytes + 1)
    return service.scan(data, file.filename, selected)


@router.post("/{scan_id}/deep", response_model=ScanResponse,
             summary="TruFor photo-forgery scan (~30 s on CPU, cached)")
def deep_scan(scan_id: str, service: ScanServiceDep) -> dict[str, Any]:
    return service.run_trufor(scan_id)


@router.post("/{scan_id}/document", response_model=ScanResponse,
             summary="DocTamper DTD document scan (JPEG only, ~40 s on CPU, cached)")
def document_scan(scan_id: str, service: ScanServiceDep) -> dict[str, Any]:
    return service.run_dtd(scan_id)
