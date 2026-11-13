from __future__ import annotations

from fastapi import APIRouter

from src.server.dependencies import ScanServiceDep
from src.server.schemas import Health

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=Health)
def health(service: ScanServiceDep) -> Health:
    return Health(trufor=service.trufor is not None, dtd=service.dtd is not None)
