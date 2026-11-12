# 🔍 image-change-detector

A Python project for detecting **meaningful changes** between two images, with a strong focus on **cross-view remote-sensing/drone imagery**.

It is designed to reduce false positives from:
- viewpoint/camera shifts
- illumination/exposure differences
- small registration errors

and produce usable outputs for both:
- region-level binary change detection
- object-level change reporting

---

## Features

| Mode | Method | Speed | Use Case |
|------|--------|-------|----------|
| `pixel` | Absolute pixel diff | ⚡ Fast | Simple brightness/color changes |
| `ssim` | Structural Similarity (SSIM) | ⚡ Fast | Perceptual / structural changes |
| `combined` | Pixel diff + SSIM | ⚡ Fast | General purpose (default) |
| `robust` | Align + Normalize + Multi-scale | ⚖️ Medium | Cross-view / different camera angle comparisons |
| `object` | YOLOv8 deep learning | 🐢 Slower | Object added / removed / moved |

---

## Project Structure

```
image-change-detector/
├── main.py                  # CLI entry point
├── config.yaml              # Configurable settings
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py
│   ├── detector.py          # Pixel/SSIM change detection
│   ├── object_detector.py   # YOLOv8 object-level detection
│   ├── benchmark.py         # Dataset-level evaluation (LEVIR-style)
│   ├── training.py          # Tiny UNet baseline training pipeline
│   └── utils.py             # Image I/O, preprocessing, visualisation
├── samples/                 # Put your test images here
├── outputs/                 # Results are saved here
└── tests/                   # Unit tests
```

---

## Installation

```bash
# 1. Clone / navigate to the project
cd image-change-detector

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU support:** Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) and set `device: cuda` in `config.yaml`.

---

## Usage

### Basic pixel/SSIM comparison

```bash
python main.py compare samples/before.jpg samples/after.jpg
```

### Choose a detection method

```bash
python main.py compare before.jpg after.jpg --mode pixel
python main.py compare before.jpg after.jpg --mode ssim
python main.py compare before.jpg after.jpg --mode combined   # default
```

### Object-level detection (YOLOv8)

```bash
python main.py compare before.jpg after.jpg --mode object
```

Detects which objects were **added**, **removed**, or **moved** between the two images.

### Robust cross-view detection (recommended production mode)

```bash
python main.py compare before.jpg after.jpg --mode robust
```

Uses feature alignment + photometric normalization + multi-scale differencing to reduce false positives from viewpoint/lighting changes.

### Save output image

```bash
python main.py compare before.jpg after.jpg --output outputs/result.png
```

### Apply preprocessing (CLAHE contrast enhancement)

```bash
python main.py compare before.jpg after.jpg --preprocess
```

### Suppress the plot window

```bash
python main.py compare before.jpg after.jpg --no-show
```

### Evaluate on a LEVIR-style dataset

```bash
python main.py evaluate /path/to/dataset --split test --mode robust
```

Process image pairs in batches (and optionally parallel workers):

```bash
python main.py evaluate /path/to/dataset --split test --mode robust --batch-size 8 --num-workers 4
```

Outputs dataset-level metrics: `precision`, `recall`, `f1`, `iou`, `accuracy`, plus confusion counts.

Optional JSON export:

```bash
python main.py evaluate /path/to/dataset --split test --mode robust --format json --json-out outputs/eval.json
```

Expected folder layout:

```text
dataset_root/
  test/
    A/      # before images
    B/      # after images
    label/  # binary masks (non-zero = change)
```

or without split:

```text
dataset_root/
  A/
  B/
  label/
```

### Compare two videos (different lengths supported)

```bash
python main.py compare-videos /path/to/video_a.mp4 /path/to/video_b.mp4 \
  --mode robust \
  --alignment feature-dtw \
  --min-alignment-score 0.08 \
  --min-detector-align-conf 0.25 \
  --stabilize \
  --background-window 2 \
  --static-std-thresh 3.0 \
  --min-static-coverage 0.15 \
  --illumination match \
  --sample-fps 1 \
  --min-change-percent 1.0 \
  --min-inner-change-percent 1.0 \
  --edge-ignore-px 24 \
  --min-regions 1 \
  --min-persistence 2 \
  --output-dir outputs/video_changes
```

This saves only changed frame-pair visualizations under:
`outputs/video_changes/changed_frames_YYYY-MM-DD_HH-MM-SS-ms/`
and writes a report:
`outputs/video_changes/summary.json`

Changed frame filename format:
`video1_HH-mm-ss-ms_____video2_HH-mm-ss-ms.png`

If you want simple time-ratio pairing instead of visual alignment:

```bash
python main.py compare-videos /path/to/video_a.mp4 /path/to/video_b.mp4 --alignment ratio
```

### Desktop app for video comparison

If you prefer a GUI instead of CLI, you have 2 options.

#### Option A (non-technical, recommended)

Run one command:

```bash
./install_desktop_app.sh
```

This will:
- install everything automatically
- create an app-menu launcher: `Video Change Detector`
- create a desktop shortcut (if `~/Desktop` exists)

Then open the app by clicking the launcher/desktop icon.

#### Option B (manual)

```bash
./venv/bin/pip install -r requirements-desktop.txt
./venv/bin/python video_compare_desktop.py
```

The desktop app lets you:
- pick Video A / Video B
- tune comparison options (`mode`, `alignment`, thresholds, stabilization)
- run comparison and view JSON summary in-app
- open output folder quickly

This GUI uses `PySide6` (installed from pip), so you do not need system Tk/Tkinter setup.
By default, it prefers `config.desktop.yaml` automatically (fallback: `config.yaml`).

#### Windows `.exe` (no terminal for end-users)

This repo includes GitHub Actions packaging for a native Windows executable:
- workflow file: `.github/workflows/build-windows-exe.yml`
- output artifact: `VideoChangeDetector.exe`

How to generate and download it:
1. Push your latest changes to GitHub.
2. Open the repo on GitHub.
3. Go to **Actions** -> **Build Windows EXE**.
4. Click **Run workflow**.
5. Open the finished run and download artifact **VideoChangeDetector-windows-exe**.
6. Share `VideoChangeDetector.exe` with non-technical users.

If the `.exe` fails to start on Windows, check:
`C:\Users\<username>\VideoChangeDetector.log`

Optional local Windows build (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_exe.ps1
```

### Train a deep baseline (Tiny UNet)

```bash
python main.py train-baseline /path/to/dataset --train-split train --val-split val --epochs 20 --batch-size 4
```

Outputs are saved under `outputs/baseline_runs/...`:
- `best.pt` / `final.pt`
- `history.json`
- `summary.json`

### Semantic change reasoning with GPT (or other VLMs)

```bash
export OPENAI_API_KEY="your-key"
python main.py reason dataset/test/A/001.jpg dataset/test/B/001.jpg --provider openai --format json --json-out outputs/reason_001.json
```

You can also use `--provider gemini`, `--provider huggingface`, or `--provider openrouter`.

Region-based mode (recommended): detect changed regions with the local CV pipeline,
then send only those crops to the VLM:

```bash
python main.py reason dataset/test/A/001.jpg dataset/test/B/001.jpg \
  --provider openai \
  --use-regions \
  --region-method robust \
  --max-regions 5 \
  --region-pad 24 \
  --format json
```

This mode saves a colorized/bordered region preview image to `outputs/` before
the VLM call (or provide `--regions-image-out /path/to/file.png`).

---

## Single-image forensics

Checks that work on **one** photo, with no "before" image. Useful for fraud screening
(for example, insurance claim photos).

```python
import cv2
from src.forensics import analyze_metadata, perceptual_hash, FingerprintIndex

# 1. Metadata: editing software, AI-generation markers, suspicious dates
report = analyze_metadata("claim.jpg")
print(report.risk, report.flag_codes)   # e.g. 0.6 ['editing_software']
print(report.to_dict())                 # JSON-ready, each flag has a reason

# 2. Duplicate / reused photos via perceptual hash
index = FingerprintIndex.load("index.json")
matches = index.find_similar(perceptual_hash(cv2.imread("claim.jpg")))
# -> [Match(image_id='old-claim.jpg', distance=3)]  (<= 10 = same photo)

# 3. Error level analysis: areas with a different JPEG save history
from src.forensics.ela import analyze_ela, render_heatmap
ela = analyze_ela("claim.jpg")
print(ela.score, ela.regions)           # e.g. 1.0 [Region(x=704, y=768, w=400, h=224, strength=8.07)]
cv2.imwrite("heat.jpg", render_heatmap(cv2.imread("claim.jpg"), ela.heatmap))
```

| Check | Flags | Notes |
|---|---|---|
| Metadata | `ai_generated_marker`, `editing_software`, `modified_after_capture`, `no_camera_metadata` | Reads EXIF, XMP (IPTC/C2PA `trainedAlgorithmicMedia`) and PNG text from diffusion tools. Metadata can be stripped or faked, so a clean report is not proof |
| ELA | suspicious regions + heatmap | Re-saves at 9 JPEG qualities, luminance only, corrects for texture with a robust fit, robust z-scores per 16x16 block. Works when the edit was saved at a clearly different quality than the original (synthetic: pasted patches z 13-17, clean <= 3.2), but **fails on real high-quality camera photos** (see results below). Use as a visual aid, not a verdict |
| Fingerprint | near-duplicate matches | 64-bit DCT pHash; survives resize, recompression, brightness changes, small crops |

### Real-world results

`scripts/eval_realworld.py` on 14 real camera photos from Wikimedia Commons
(car damage, windshields, water damage, interiors), each as 2 clean copies and
6 realistic edits (112 images). "Located" = a detected region overlaps the edit.

| Edit | ELA located | TruFor located |
|---|---|---|
| Area pasted from another photo | 0/14 | 10/14 |
| Area cloned within the photo | 0/14 | 11/14 |
| Object removed (inpainting) | 0/14 | 13/14 |
| Painted scratches | 0/14 | 2/14 |
| Added text | 0/14 | 7/14 |
| Pasted, then messenger-compressed | 0/14 | 10/14 |
| **Total** | **0/84** | **53/84 (63%)** |

False alarms on the 28 clean copies: ELA 6/28, TruFor score > 0.5 on 2/28 (both
from one photo that had passed through iPhoto). Thin edits are weak because
images are downscaled to 1024 px for CPU speed. Small sample; treat as indicative.

### TruFor (trained forgery detector, optional)

Hand-made checks like ELA break down on real high-quality photos (see
`scripts/eval_realworld.py`). For those, `src/forensics/trufor.py` wraps
**TruFor** (F. Guillaro, D. Cozzolino, A. Sud, N. Dufour, L. Verdoliva,
"TruFor: Leveraging all-round clues for trustworthy image forgery detection
and localization", CVPR 2023, https://grip-unina.github.io/TruFor). It gives
a 0-1 manipulation score plus a per-pixel map.

TruFor's code and weights are licensed by GRIP-UNINA for **non-commercial use
only** and are not included here. Setup:

```bash
git clone https://github.com/grip-unina/TruFor
# download TruFor_weights.zip from the TruFor README (MD5 7bee48f3476c75616c3c5721ab256ff8)
pip install "timm<1" yacs
```

Convert the checkpoint once to a numbers-only state dict, so it loads with
`torch.load(weights_only=True)` and cannot execute code:

```python
import numpy as np, numpy.core.multiarray as ma, torch
safe = [(ma.scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype("float64"))]
with torch.serialization.safe_globals(safe):
    ck = torch.load("trufor.pth.tar", map_location="cpu", weights_only=True)
torch.save(ck["state_dict"], "trufor_state_dict.pt")
```

```python
from src.forensics.trufor import TruForDetector
det = TruForDetector(trufor_dir="TruFor/TruFor_train_test", weights="trufor_state_dict.pt")
report = det.analyze("claim.jpg")   # ~25 s on a laptop CPU at 1024 px
print(report.score, report.regions)
```

## Python API

```python
from src.detector import ChangeDetector
from src.object_detector import ObjectDetector
from src.utils import load_image, print_report, plot_comparison

# Load images
img_a = load_image("samples/before.jpg")
img_b = load_image("samples/after.jpg")

# --- Pixel/SSIM detection ---
detector = ChangeDetector(config={"threshold": 25, "min_contour_area": 300})
result = detector.compare(img_a, img_b, method="combined")

print_report(result)
# result["score"]           → SSIM similarity score (1.0 = identical)
# result["change_percent"]  → % of image that changed
# result["bboxes"]          → list of changed region bounding boxes
# result["diff_image"]      → annotated comparison image (numpy array)

# --- Object detection ---
obj_detector = ObjectDetector(model_name="yolov8n", confidence=0.4)
obj_result = obj_detector.compare(img_a, img_b)

print_report(obj_result)
# obj_result["summary"]  → {"added": 2, "removed": 1, "moved": 0, ...}
# obj_result["changes"]  → list of ObjectChange objects
```

---

## Configuration

Edit `config.yaml` to tune detection behaviour:

```yaml
detector:
  threshold: 30          # pixel diff threshold (0-255)
  min_contour_area: 500  # ignore tiny changed regions
  blur_kernel: 5         # noise reduction kernel size

object_detector:
  model: yolov8n         # yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
  confidence: 0.4        # minimum detection confidence
  iou_threshold: 0.3     # IoU for matching objects across images
  device: cpu            # cpu / cuda / mps
```

---

## Dependencies

- **OpenCV** — image processing, contour detection
- **scikit-image** — SSIM computation
- **ultralytics** — YOLOv8 object detection
- **PyTorch** — deep learning inference backend
- **matplotlib** — result visualisation
- **click** — CLI interface

---

## License

MIT
