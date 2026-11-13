from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

STATIC = Path(__file__).resolve().parents[1] / "static"

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")
