"""
main.py
-------
App factory for the Fraud Scanner.

    POST /api/scan?checks=metadata,duplicates,ela   upload + fast checks
    POST /api/scan/{id}/deep                       TruFor (optional install)
    POST /api/scan/{id}/document                   DocTamper DTD (optional install)
    GET  /api/health, /api/examples[/{name}], /    status, demo photos, UI
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from src.server.config import Settings
from src.server.errors import register_error_handlers
from src.server.routers import examples, health, scans, ui
from src.server.services.detectors import dtd_from_settings, trufor_from_settings
from src.server.services.scanner import ScanService

_FROM_SETTINGS: Any = object()


def create_app(settings: Settings | None = None, *, trufor: Any = _FROM_SETTINGS,
               dtd: Any = _FROM_SETTINGS) -> FastAPI:
    """Build the app. ``trufor``/``dtd`` override the detectors (tests pass fakes or None)."""
    settings = settings or Settings()
    service = ScanService(
        settings,
        trufor=trufor_from_settings(settings) if trufor is _FROM_SETTINGS else trufor,
        dtd=dtd_from_settings(settings) if dtd is _FROM_SETTINGS else dtd,
    )

    app = FastAPI(title="Fraud Scanner", version="0.3.0",
                  description="Single-image forensics: metadata, duplicates, ELA, TruFor, DTD")
    app.state.scan_service = service
    register_error_handlers(app)
    for router in (health.router, scans.router, examples.router, ui.router):
        app.include_router(router)
    return app
