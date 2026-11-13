"""Typed API response models (also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FastCheck = Literal["metadata", "duplicates", "ela"]
FAST_CHECKS: tuple[FastCheck, ...] = ("metadata", "duplicates", "ela")


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    trufor: bool = Field(description="TruFor deep scan installed")
    dtd: bool = Field(description="DocTamper document scan installed")


class ImageInfo(BaseModel):
    width: int
    height: int
    format: str
    bytes: int
    preview: str = Field(description="JPEG data URL, max 1400 px")


class Flag(BaseModel):
    code: str
    reason: str
    weight: float


class MetadataResult(BaseModel):
    camera: str | None
    software: str | None
    captured_at: str | None
    modified_at: str | None
    flags: list[Flag]
    risk: float


class Duplicate(BaseModel):
    scan_id: str
    filename: str | None
    distance: int = Field(description="Hamming distance of 64-bit fingerprints")
    similarity: float


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int
    strength: float


class DtdBox(Box):
    pixels: int


class ElaResult(BaseModel):
    applicable: bool
    score: float
    reason: str
    estimated_quality: int | None
    regions: list[Box]
    overlay: str | None


class TruForResult(BaseModel):
    score: float
    regions: list[Box]
    manipulated_fraction: float
    suspicious: bool
    overlay: str
    ms: int


class DtdResult(BaseModel):
    applicable: bool
    score: float
    reason: str
    regions: list[DtdBox]
    tampered_fraction: float
    suspicious: bool
    overlay: str | None
    ms: int


class Verdict(BaseModel):
    level: Literal["low", "medium", "high", "unknown"]
    score: float
    reasons: list[str]


class ScanResponse(BaseModel):
    scan_id: str
    filename: str | None
    checks: list[FastCheck]
    image: ImageInfo
    metadata: MetadataResult | None
    duplicates: list[Duplicate] | None
    ela: ElaResult | None
    trufor: TruForResult | None
    dtd: DtdResult | None
    verdict: Verdict
    timings_ms: dict[str, int]
