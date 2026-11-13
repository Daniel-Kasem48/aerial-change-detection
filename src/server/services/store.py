"""In-memory scan store with bounded size; evicted scans delete their upload."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.server.errors import ScanNotFoundError

if TYPE_CHECKING:
    from pathlib import Path

    from src.forensics.dtd import DtdReport
    from src.forensics.trufor import TruForReport


@dataclass
class ScanRecord:
    """Everything known about one upload. Model results are filled in later."""

    scan_id: str
    path: Path
    filename: str | None
    checks: tuple[str, ...]
    fast: dict[str, Any]
    metadata_risk: float = 0.0
    duplicate: bool = False
    trufor_report: TruForReport | None = None
    trufor: dict[str, Any] | None = None
    dtd_report: DtdReport | None = None
    dtd: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def judged(self) -> bool:
        """True once a check that feeds the verdict could analyse the image."""
        return ("metadata" in self.checks or "duplicates" in self.checks
                or self.trufor_report is not None
                or (self.dtd_report is not None and self.dtd_report.applicable))


class ScanStore:
    def __init__(self, max_scans: int) -> None:
        self._max = max_scans
        self._scans: OrderedDict[str, ScanRecord] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, record: ScanRecord) -> None:
        with self._lock:
            self._scans[record.scan_id] = record
            while len(self._scans) > self._max:
                _, old = self._scans.popitem(last=False)
                old.path.unlink(missing_ok=True)

    def get(self, scan_id: str) -> ScanRecord:
        with self._lock:
            record = self._scans.get(scan_id)
        if record is None:
            raise ScanNotFoundError("Unknown scan id")
        return record

    def filenames(self) -> dict[str, str | None]:
        with self._lock:
            return {sid: r.filename for sid, r in self._scans.items()}
