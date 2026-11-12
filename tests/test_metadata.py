"""Tests for single-image metadata forensics (src/forensics/metadata.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image, PngImagePlugin
from src.forensics.metadata import analyze_metadata

if TYPE_CHECKING:
    from pathlib import Path

# EXIF tag ids
MAKE, MODEL, SOFTWARE, DATETIME = 0x010F, 0x0110, 0x0131, 0x0132
EXIF_IFD, DATETIME_ORIGINAL = 0x8769, 0x9003


def _noise_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))


def _save_jpeg(path: Path, tags: dict[int, str] | None = None,
               original: str | None = None, xmp: bytes | None = None) -> Path:
    img = _noise_image()
    exif = Image.Exif()
    for tag, value in (tags or {}).items():
        exif[tag] = value
    if original:
        exif.get_ifd(EXIF_IFD)[DATETIME_ORIGINAL] = original
    kwargs = {"exif": exif.tobytes()} if (tags or original) else {}
    if xmp:
        kwargs["xmp"] = xmp
    img.save(path, "JPEG", quality=92, **kwargs)
    return path


def test_camera_photo_without_edits_has_no_flags(tmp_path: Path) -> None:
    path = _save_jpeg(tmp_path / "cam.jpg",
                      {MAKE: "Canon", MODEL: "EOS 80D", DATETIME: "2026:05:01 10:00:00"},
                      original="2026:05:01 10:00:00")
    report = analyze_metadata(path)
    assert report.camera == "Canon EOS 80D"
    assert report.flags == []
    assert report.risk == 0.0


def test_editing_software_is_flagged(tmp_path: Path) -> None:
    path = _save_jpeg(tmp_path / "ps.jpg",
                      {MAKE: "Canon", MODEL: "EOS 80D", SOFTWARE: "Adobe Photoshop 25.0 (Windows)"})
    report = analyze_metadata(path)
    assert report.software == "Adobe Photoshop 25.0 (Windows)"
    assert "editing_software" in report.flag_codes


def test_missing_camera_metadata_is_flagged(tmp_path: Path) -> None:
    report = analyze_metadata(_save_jpeg(tmp_path / "bare.jpg"))
    assert "no_camera_metadata" in report.flag_codes
    assert 0.0 < report.risk < 0.5  # weak signal: many apps strip EXIF


def test_modified_long_after_capture_is_flagged(tmp_path: Path) -> None:
    path = _save_jpeg(tmp_path / "late.jpg",
                      {MAKE: "Apple", MODEL: "iPhone 15", DATETIME: "2026:09:20 18:00:00"},
                      original="2026:05:01 10:00:00")
    assert "modified_after_capture" in analyze_metadata(path).flag_codes


def test_ai_marker_in_xmp_is_flagged(tmp_path: Path) -> None:
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
           b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/'
           b'trainedAlgorithmicMedia"/></rdf:RDF></x:xmpmeta>')
    report = analyze_metadata(_save_jpeg(tmp_path / "ai.jpg", xmp=xmp))
    assert "ai_generated_marker" in report.flag_codes
    assert report.risk >= 0.9


def test_stable_diffusion_png_parameters_are_flagged(tmp_path: Path) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a red car, Steps: 30, Sampler: Euler a, Seed: 42")
    path = tmp_path / "sd.png"
    _noise_image().save(path, "PNG", pnginfo=info)
    assert "ai_generated_marker" in analyze_metadata(path).flag_codes


def test_ai_tool_in_software_tag_is_flagged(tmp_path: Path) -> None:
    path = _save_jpeg(tmp_path / "mj.jpg", {SOFTWARE: "Midjourney v6"})
    assert "ai_generated_marker" in analyze_metadata(path).flag_codes


def test_report_is_json_serializable(tmp_path: Path) -> None:
    import json
    report = analyze_metadata(_save_jpeg(tmp_path / "x.jpg", {SOFTWARE: "GIMP 2.10"}))
    data = json.loads(json.dumps(report.to_dict()))
    assert data["flags"][0]["code"] == "editing_software"
    assert "reason" in data["flags"][0]


def test_unreadable_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        analyze_metadata(bad)


def test_camera_name_does_not_repeat_brand(tmp_path: Path) -> None:
    path = _save_jpeg(tmp_path / "c.jpg", {MAKE: "Canon", MODEL: "Canon EOS 70D"})
    assert analyze_metadata(path).camera == "Canon EOS 70D"
