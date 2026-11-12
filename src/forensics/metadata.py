"""
metadata.py
-----------
Metadata forensics for a single image.

Reads EXIF, XMP and PNG text chunks and raises flags that hint at editing
or AI generation. Metadata is easy to strip or fake, so these are *signals*,
not proof: a clean report does not mean the image is authentic.

Flags and their weights (combined as ``1 - prod(1 - w)``):

=======================  ======  ===========================================
code                     weight  meaning
=======================  ======  ===========================================
ai_generated_marker      0.95    AI tool name, IPTC "trainedAlgorithmicMedia"
                                 (also used by C2PA), or diffusion parameters
editing_software         0.60    Photoshop, GIMP, Snapseed, ... in Software tag
modified_after_capture   0.40    file changed more than a day after capture
no_camera_metadata       0.20    no camera make/model (often just stripped)
=======================  ======  ===========================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

# EXIF tag ids
_MAKE, _MODEL, _SOFTWARE, _DATETIME = 0x010F, 0x0110, 0x0131, 0x0132
_EXIF_IFD, _DATETIME_ORIGINAL = 0x8769, 0x9003

_WEIGHTS = {
    "ai_generated_marker": 0.95,
    "editing_software": 0.60,
    "modified_after_capture": 0.40,
    "no_camera_metadata": 0.20,
}

_AI_TOOLS = (
    "midjourney", "dall-e", "dall·e", "stable diffusion", "stablediffusion", "firefly",
    "imagen", "leonardo", "ideogram", "flux", "novelai", "comfyui", "automatic1111",
    "invokeai", "dreamstudio", "chatgpt", "gpt-4o", "gemini",
)
_EDITORS = (
    "photoshop", "gimp", "lightroom", "snapseed", "picsart", "pixelmator", "affinity",
    "canva", "paint.net", "fotor", "facetune", "photopea", "krita", "luminar", "capture one",
)
# IPTC digital source type for AI media; C2PA manifests use the same term.
_AI_SOURCE_MARKER = b"trainedalgorithmicmedia"
# PNG text keys written by diffusion UIs (AUTOMATIC1111, ComfyUI, InvokeAI, NovelAI)
_AI_PNG_KEYS = ("parameters", "prompt", "workflow", "invokeai_metadata", "sd-metadata")

_MAX_GAP = timedelta(days=1)


@dataclass(frozen=True)
class Flag:
    code: str
    reason: str
    weight: float


@dataclass(frozen=True)
class MetadataReport:
    camera: str | None = None
    software: str | None = None
    captured_at: str | None = None
    modified_at: str | None = None
    flags: list[Flag] = field(default_factory=list)

    @property
    def flag_codes(self) -> list[str]:
        return [f.code for f in self.flags]

    @property
    def risk(self) -> float:
        """0 = nothing suspicious, 1 = certain. Independent-signal combination."""
        clean = 1.0
        for f in self.flags:
            clean *= 1.0 - f.weight
        return round(1.0 - clean, 3)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["risk"] = self.risk
        return data


def analyze_metadata(path: str | Path) -> MetadataReport:
    """Inspect metadata of the image at *path*.

    Raises:
        ValueError: if the file is not a readable image.
    """
    path = Path(path)
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            info = dict(img.info)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Not a readable image: {path}") from exc

    make = _text(exif.get(_MAKE))
    model = _text(exif.get(_MODEL))
    software = _text(exif.get(_SOFTWARE))
    modified = _text(exif.get(_DATETIME))
    captured = _text(exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL))
    if make and model and model.lower().startswith(make.lower()):
        make = None  # e.g. Make "Canon", Model "Canon EOS 70D"
    camera = " ".join(p for p in (make, model) if p) or None

    flags: list[Flag] = []

    ai_reason = _ai_reason(path, software, info)
    if ai_reason:
        flags.append(_flag("ai_generated_marker", ai_reason))

    editor = _match(software, _EDITORS)
    if editor:
        flags.append(_flag("editing_software", f"Software tag says '{software}'"))

    gap = _gap(captured, modified)
    if gap is not None and gap > _MAX_GAP:
        flags.append(_flag("modified_after_capture",
                           f"Captured {captured}, last modified {modified} ({gap.days} days later)"))

    if not camera:
        flags.append(_flag("no_camera_metadata",
                           "No camera make or model (stripped by an app, screenshot, or generated)"))

    return MetadataReport(camera=camera, software=software, captured_at=captured,
                          modified_at=modified, flags=flags)


def _ai_reason(path: Path, software: str | None, info: dict[str, Any]) -> str | None:
    tool = _match(software, _AI_TOOLS)
    if tool:
        return f"Software tag names an AI tool: '{software}'"
    for key in _AI_PNG_KEYS:
        if key in info:
            return f"PNG text chunk '{key}' written by an image-generation tool"
    xmp = info.get("xmp") or b""
    if isinstance(xmp, str):
        xmp = xmp.encode("utf-8", "ignore")
    if _AI_SOURCE_MARKER in xmp.lower() or _AI_SOURCE_MARKER in path.read_bytes().lower():
        return "Embedded metadata declares AI-generated content (IPTC/C2PA trainedAlgorithmicMedia)"
    return None


def _flag(code: str, reason: str) -> Flag:
    return Flag(code=code, reason=reason, weight=_WEIGHTS[code])


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    value = str(value).strip("\x00 ").strip()
    return value or None


def _match(text: str | None, names: tuple[str, ...]) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    return next((n for n in names if n in lowered), None)


def _gap(captured: str | None, modified: str | None) -> timedelta | None:
    try:
        start = datetime.strptime(captured or "", "%Y:%m:%d %H:%M:%S")
        end = datetime.strptime(modified or "", "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    return end - start
