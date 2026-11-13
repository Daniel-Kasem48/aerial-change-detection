"""Optional model detectors (TruFor, DTD), built from settings when installed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.forensics.dtd import DtdDetector
from src.forensics.trufor import TruForDetector

if TYPE_CHECKING:
    from pathlib import Path

    from src.forensics.dtd import DtdReport
    from src.forensics.trufor import TruForReport
    from src.server.config import Settings


class TruForLike(Protocol):
    def analyze(self, path: str | Path) -> TruForReport: ...


class DtdLike(Protocol):
    def analyze(self, path: str | Path) -> DtdReport: ...


def _installed(code: Path | None, weights: Path | None) -> bool:
    return bool(code and weights and code.is_dir() and weights.is_file())


def trufor_from_settings(settings: Settings) -> TruForDetector | None:
    if _installed(settings.trufor_dir, settings.trufor_weights):
        return TruForDetector(trufor_dir=settings.trufor_dir, weights=settings.trufor_weights)
    return None


def dtd_from_settings(settings: Settings) -> DtdDetector | None:
    if _installed(settings.dtd_dir, settings.dtd_weights):
        return DtdDetector(dtd_dir=settings.dtd_dir, weights=settings.dtd_weights)
    return None
