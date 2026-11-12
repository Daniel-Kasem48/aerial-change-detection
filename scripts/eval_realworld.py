#!/usr/bin/env python3
"""
Real-world evaluation of single-image forensics.

Takes a folder of real camera JPEGs, creates realistic edits of each one and
reports how often ELA finds the edit (and how often it falsely flags clean
copies).

    python scripts/eval_realworld.py PHOTOS_DIR OUT_DIR [--trufor]

With --trufor (needs TRUFOR_DIR / TRUFOR_WEIGHTS) the TruFor network is
evaluated too (~25 s per image on a laptop CPU).

Per photo:
  clean_resave     opened and saved again at q90 (no edit)          -> must stay clean
  clean_messenger  resized to 1280 px, q80 (like WhatsApp)          -> must stay clean
  splice           area from ANOTHER photo pasted, soft edges, q90
  clone            area copied within the SAME photo, soft edges, q90
  inpaint          object removed (OpenCV inpainting), q90
  paint            painted scratches, q90
  text             text added (e.g. a fake amount), q90
  splice_messenger splice, then sent through a messenger (resize + q80)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.forensics.ela import analyze_ela, render_heatmap  # noqa: E402
from src.forensics.trufor import TruForDetector  # noqa: E402

EDITS = ["clean_resave", "clean_messenger", "splice", "clone", "inpaint", "paint", "text",
         "splice_messenger"]


def _box(rng: np.random.Generator, h: int, w: int, frac: float) -> tuple[int, int, int, int]:
    bw, bh = int(w * frac), int(h * frac)
    return int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh)), bw, bh


def _soft_paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    mask = np.zeros((h, w), np.float32)
    pad = max(4, min(h, w) // 12)
    mask[pad:-pad, pad:-pad] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), pad / 2)[..., None]
    roi = dst[y:y + h, x:x + w].astype(np.float32)
    dst[y:y + h, x:x + w] = (mask * src + (1 - mask) * roi).astype(np.uint8)


def _messenger(img: np.ndarray) -> np.ndarray:
    """Resize like a messenger app; the caller saves at q80."""
    h, w = img.shape[:2]
    s = 1280 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def make_edit(kind: str, img: np.ndarray, donor: np.ndarray, rng: np.random.Generator):
    """Return (edited image, edit box or None, JPEG quality to save at)."""
    h, w = img.shape[:2]
    out = img.copy()
    if kind == "clean_resave":
        return out, None, 90
    if kind == "clean_messenger":
        return _messenger(out), None, 80
    if kind in ("splice", "splice_messenger"):
        x, y, bw, bh = _box(rng, h, w, 0.22)
        d = cv2.resize(donor, (w, h))
        sx, sy, _, _ = _box(rng, h, w, 0.22)
        _soft_paste(out, d[sy:sy + bh, sx:sx + bw], x, y)
        if kind == "splice_messenger":
            s = min(1.0, 1280 / max(h, w))
            return _messenger(out), tuple(int(v * s) for v in (x, y, bw, bh)), 80
        return out, (x, y, bw, bh), 90
    if kind == "clone":
        x, y, bw, bh = _box(rng, h, w, 0.18)
        sx, sy, _, _ = _box(rng, h, w, 0.18)
        _soft_paste(out, img[sy:sy + bh, sx:sx + bw].copy(), x, y)
        return out, (x, y, bw, bh), 90
    if kind == "inpaint":
        x, y, bw, bh = _box(rng, h, w, 0.12)
        mask = np.zeros((h, w), np.uint8)
        mask[y:y + bh, x:x + bw] = 255
        small = cv2.resize(out, (w // 2, h // 2))
        filled = cv2.inpaint(small, cv2.resize(mask, (w // 2, h // 2)), 5, cv2.INPAINT_TELEA)
        out[y:y + bh, x:x + bw] = cv2.resize(filled, (w, h))[y:y + bh, x:x + bw]
        return out, (x, y, bw, bh), 90
    if kind == "paint":
        x, y, bw, bh = _box(rng, h, w, 0.25)
        t = max(2, w // 600)
        for i in range(6):
            cv2.line(out, (x + i * bw // 8, y), (x + bw // 3 + i * bw // 8, y + bh),
                     (45, 45, 45), t, cv2.LINE_AA)
        return out, (x, y, bw, bh), 90
    if kind == "text":
        scale = w / 600
        x, y = int(w * 0.1), int(h * rng.uniform(0.2, 0.8))
        (tw, th), _ = cv2.getTextSize("EUR 4.850,00", cv2.FONT_HERSHEY_SIMPLEX, scale, int(scale * 2))
        cv2.putText(out, "EUR 4.850,00", (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (20, 20, 160), int(scale * 2), cv2.LINE_AA)
        return out, (x, y - th, tw, th + int(scale * 10)), 90
    raise ValueError(kind)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def main(photos: Path, out: Path, use_trufor: bool = False) -> None:
    trufor = TruForDetector() if use_trufor else None
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(photos.glob("*.jpg"))
    rng = np.random.default_rng(7)
    rows = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        donor = cv2.imread(str(files[(i + 1) % len(files)]))
        if img is None or donor is None:
            continue
        for kind in EDITS:
            edited, box, q = make_edit(kind, img, donor, rng)
            path = out / f"{f.stem}_{kind}.jpg"
            cv2.imwrite(str(path), edited, [cv2.IMWRITE_JPEG_QUALITY, q])
            r = analyze_ela(path)
            hit = bool(box) and any(_iou((g.x, g.y, g.w, g.h), box) > 0.1 for g in r.regions)
            row = dict(photo=f.name, edit=kind, flagged=bool(r.regions), localized=hit,
                       score=r.score, regions=len(r.regions), quality=r.estimated_quality)
            if trufor:
                t = trufor.analyze(path)
                row.update(tf_score=t.score, tf_flagged=t.score > 0.5,
                           tf_localized=bool(box) and any(
                               _iou((g.x, g.y, g.w, g.h), box) > 0.1 for g in t.regions))
            rows.append(row)
            if i < 3:
                heat = render_heatmap(edited, r.heatmap)
                for g in r.regions:
                    cv2.rectangle(heat, (g.x, g.y), (g.x + g.w, g.y + g.h), (255, 255, 255), 6)
                if box:
                    bx, by, bw, bh = box
                    cv2.rectangle(edited, (bx, by), (bx + bw, by + bh), (0, 255, 0), 6)
                cv2.imwrite(str(out / f"viz_{f.stem}_{kind}.jpg"),
                            cv2.resize(np.hstack([edited, heat]), (1200, int(600 * edited.shape[0] / edited.shape[1]))))
        print(f"{f.name}: done", flush=True)
        (out / "results.json").write_text(json.dumps(rows, indent=1))

    (out / "results.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{len(files)} real photos\n")
    print_table(rows)


def print_table(rows: list[dict]) -> None:
    tf = "tf_score" in rows[0]
    head = f"{'edit':18} {'ELA flagged':>12} {'ELA spot':>9}"
    print(head + (f" {'TruFor flagged':>15} {'TruFor spot':>12}" if tf else ""))
    for kind in EDITS:
        sel = [r for r in rows if r["edit"] == kind]
        n = len(sel)
        clean = kind.startswith("clean")

        def frac(key: str) -> str:
            return f"{sum(r[key] for r in sel)}/{n}"
        line = f"{kind:18} {frac('flagged'):>12} {'-' if clean else frac('localized'):>9}"
        if tf:
            line += f" {frac('tf_flagged'):>15} {'-' if clean else frac('tf_localized'):>12}"
        print(line)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), "--trufor" in sys.argv[3:])
