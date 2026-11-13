"""
dtd.py
------
Wrapper around DTD (Document Tampering Detector) from DocTamper
(Qu et al., "Towards Robust Tampered Text Detection in Document Image",
CVPR 2023, https://github.com/qcf-568/DocTamper).

DTD looks at the grey image *and* the JPEG's raw 8x8 DCT coefficients with
their quantization table, so it sees compression mismatches around edited
characters (for example a changed digit on a receipt). It returns a
per-pixel probability that text there was tampered with.

The DocTamper code and weights are NOT part of this repo (no licence file,
non-commercial dataset). Install them separately and point ``dtd_dir`` (or
``DTD_DIR``) at ``DocTamper/models`` and ``weights`` (or ``DTD_WEIGHTS``) at
``dtd_sroie.pth`` (receipts) or ``dtd_doctamper.pth`` (documents).

Safety: the released code loads two pickled *modules* (``vph_imagenet.pt``,
``swin_imagenet.pt``). They are not used here: those parts are rebuilt from
code with the shapes found in the checkpoint, and the checkpoint itself is
loaded with ``torch.load(weights_only=True)`` (tensors only).
"""

from __future__ import annotations

import os
import sys
import threading
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

TILE = 512
PROB_THRESHOLD = 0.5
BLANK_TILE_STD = 3.0          # skip tiles with no visible content (plain paper / background)
# Yes/no rule: a clear region (peak >= 0.9) of at least 150 px. The untouched
# test receipt only had a 27 px speck at 0.77; the two forged digits peaked at 1.0.
SUSPICIOUS_PEAK = 0.9
SUSPICIOUS_MIN_PIXELS = 150

_MEAN = np.array([0.485, 0.455, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_import_lock = threading.Lock()


class DtdUnavailableError(RuntimeError):
    """DocTamper code or weights are not installed."""


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int
    strength: float  # peak probability inside the region
    pixels: int      # pixels above PROB_THRESHOLD


@dataclass(frozen=True)
class DtdReport:
    applicable: bool
    score: float  # highest tamper probability in the image
    reason: str
    regions: list[Region] = field(default_factory=list)
    tampered_fraction: float = 0.0
    heatmap: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def suspicious(self) -> bool:
        return any(r.strength >= SUSPICIOUS_PEAK and r.pixels >= SUSPICIOUS_MIN_PIXELS
                   for r in self.regions)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("heatmap")
        data["suspicious"] = self.suspicious
        return data


class DtdDetector:
    def __init__(self, dtd_dir: str | Path | None = None, weights: str | Path | None = None,
                 threads: int | None = None) -> None:
        self.dtd_dir = Path(dtd_dir or os.environ.get("DTD_DIR", ""))
        self.weights = Path(weights or os.environ.get("DTD_WEIGHTS", ""))
        self.threads = threads
        self._model = None

    def analyze(self, path: str | Path) -> DtdReport:
        path = Path(path)
        with Image.open(path) as img:
            fmt = img.format
        if fmt != "JPEG":
            return DtdReport(False, 0.0, f"Document scan needs a JPEG; this file is {fmt}")

        import jpegio
        import torch
        import torch.nn.functional as F  # noqa: N812  (PyTorch convention)

        model = self._load()
        jpg = jpegio.read(str(path))
        dct = np.abs(jpg.coef_arrays[0]).clip(0, 20).astype(np.int64)
        qt = torch.from_numpy(np.clip(jpg.quant_tables[0], 0, 63).astype(np.int64)[None, None])
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        hp, wp = -(-h // TILE) * TILE, -(-w // TILE) * TILE
        g = np.full((hp, wp), 255, np.uint8)
        g[:h, :w] = gray
        d = np.zeros((hp, wp), np.int64)
        d[:min(h, dct.shape[0]), :min(w, dct.shape[1])] = dct[:h, :w]
        rgb = (np.repeat(g[..., None], 3, 2).astype(np.float32) / 255 - _MEAN) / _STD

        prob = np.zeros((hp, wp), np.float32)
        with torch.no_grad():
            for y in range(0, hp, TILE):
                for x in range(0, wp, TILE):
                    if g[y:y + TILE, x:x + TILE].std() < BLANK_TILE_STD:
                        continue
                    xi = torch.from_numpy(rgb[y:y + TILE, x:x + TILE].transpose(2, 0, 1)[None].copy())
                    di = torch.from_numpy(d[y:y + TILE, x:x + TILE][None].copy())
                    out = F.softmax(model(xi, di, qt), 1)[0, 1]
                    prob[y:y + TILE, x:x + TILE] = F.interpolate(
                        out[None, None], size=(TILE, TILE), mode="bilinear")[0, 0].numpy()
        prob = prob[:h, :w]

        regions = _regions(prob)
        reason = (f"{len(regions)} text area(s) look edited" if regions
                  else "No edited text found")
        return DtdReport(True, round(float(prob.max()), 3), reason, regions,
                         round(float((prob > PROB_THRESHOLD).mean()), 5), prob)

    # ------------------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model
        if not (self.dtd_dir / "dtd.py").is_file() or not (self.dtd_dir / "swins.py").is_file():
            raise DtdUnavailableError(f"DocTamper code not found in '{self.dtd_dir}' (set DTD_DIR)")
        if not self.weights.is_file():
            raise DtdUnavailableError(f"DTD weights not found at '{self.weights}' (set DTD_WEIGHTS)")

        import torch

        with _import_lock:
            model = _build_model(self.dtd_dir)
        state = torch.load(self.weights, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module.model."): v for k, v in state["state_dict"].items()}
        torch.nn.Module.load_state_dict(model, state, strict=True)
        torch.nn.Module.train(model, False)
        if self.threads:
            torch.set_num_threads(self.threads)
        self._model = model
        return model


def _build_model(dtd_dir: Path):
    """Rebuild DTD from DocTamper's code without its pickled modules."""
    import torch.nn as nn

    sys.path.insert(0, str(dtd_dir))
    stubs = ("lmdb", "albumentations", "albumentations.pytorch")
    added = [s for s in stubs if s not in sys.modules]
    for name in added:  # training-only imports at the top of dtd.py
        sys.modules[name] = types.ModuleType(name)
    if "albumentations.pytorch" in added:
        sys.modules["albumentations.pytorch"].ToTensorV2 = object
    try:
        import dtd as D  # noqa: N812
        from swins import BasicLayer, PatchMerging
    finally:
        sys.path.remove(str(dtd_dir))
        for name in added:
            sys.modules.pop(name, None)

    class Conv2dReLU(nn.Sequential):
        """segmentation_models_pytorch 0.2 block (newer versions renamed the args)."""

        def __init__(self, cin, cout, kernel_size, padding=0, stride=1, use_batchnorm=True):
            super().__init__(nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=padding,
                                       bias=not use_batchnorm),
                             nn.BatchNorm2d(cout) if use_batchnorm else nn.Identity(),
                             nn.ReLU(inplace=True))

    D.md.Conv2dReLU = Conv2dReLU

    def ln(c):
        return D.LayerNorm(c, eps=1e-6, data_format="channels_first")

    class VPH(nn.Module):
        """Rebuilt from checkpoint shapes; the released class is incomplete."""

        def __init__(self):
            super().__init__()
            self.downsample_layers = nn.ModuleList([
                nn.Sequential(nn.Conv2d(6, 96, 4, 4), ln(96)),
                nn.Sequential(ln(96), nn.Conv2d(96, 192, 2, 2)),
            ])
            self.stages = nn.ModuleList([
                nn.Sequential(*[D.ConvBlock(96) for _ in range(3)]),
                nn.Sequential(*[D.ConvBlock(192) for _ in range(3)]),
            ])
            for i, c in enumerate((96, 192, 384, 768)):
                self.add_module(f"norm{i}", ln(c))

        def forward(self, x):
            x = self.stages[0](self.downsample_layers[0](x))
            outs = [self.norm0(x)]
            x = self.stages[1](self.downsample_layers[1](x))
            outs.append(self.norm1(x))
            return outs

    class DTD(D.DTD):
        def __init__(self):
            nn.Module.__init__(self)
            self.vph = VPH()
            # resolutions match the attention masks stored in the checkpoint (256 px training crops)
            self.swin = nn.ModuleList([
                BasicLayer(192, (32, 32), 2, 6, 8, downsample=PatchMerging),
                BasicLayer(384, (16, 16), 18, 12, 8, downsample=PatchMerging),
                BasicLayer(768, (8, 8), 2, 24, 8),
            ])
            self.fph = D.FPH()
            self.decoder = D.MID(encoder_channels=(96, 192, 384, 768),
                                 decoder_channels=(384, 192, 96, 64))
            self.segmentation_head = D.SegmentationHead(in_channels=64, out_channels=2,
                                                        upsampling=2.0)
            self.addcoords = D.AddCoords()
            self.FU = nn.Sequential(D.SCSEModule(448), nn.Conv2d(448, 192, 3, 1, 1),
                                    nn.BatchNorm2d(192), nn.ReLU(True))
            self.classification_head = None

    return DTD()


def _regions(prob: np.ndarray) -> list[Region]:
    mask = (prob > PROB_THRESHOLD).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))  # join strokes of one character
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        inside = (labels == i) & (prob > PROB_THRESHOLD)
        px = int(inside.sum())
        if px == 0:
            continue
        regions.append(Region(int(x), int(y), int(w), int(h),
                              round(float(prob[labels == i].max()), 3), px))
    return sorted(regions, key=lambda r: (r.strength, r.pixels), reverse=True)
