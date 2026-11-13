from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.server.dependencies import ScanServiceDep

router = APIRouter(prefix="/api/examples", tags=["examples"])


@router.get("", response_model=list[str])
def list_examples(service: ScanServiceDep) -> list[str]:
    return service.example_names()


@router.get("/{name}", response_class=FileResponse)
def get_example(name: str, service: ScanServiceDep) -> FileResponse:
    path, media_type = service.example_path(name)
    return FileResponse(path, media_type=media_type)
