"""FastAPI dependencies: hand routers the service built in the app factory."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.server.services.scanner import ScanService


def get_scan_service(request: Request) -> ScanService:
    return request.app.state.scan_service


ScanServiceDep = Annotated[ScanService, Depends(get_scan_service)]
