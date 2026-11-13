# Research notes: single-image fraud detection

What was tried, what worked, what did not, and what comes next.
Numbers come from small local experiments; treat them as indicative.

## 1. Checks in this repo

| Check | Module | Status | Works on messenger-compressed images? |
|---|---|---|---|
| Metadata (editing software, AI markers, dates) | `src/forensics/metadata.py` | in verdict | partly (apps strip metadata) |
| Duplicate / reused photo (perceptual hash) | `src/forensics/fingerprint.py` | in verdict | yes |
| Error level analysis | `src/forensics/ela.py` | shown, not in verdict | no |
| TruFor (trained detector, external) | `src/forensics/trufor.py` | in verdict when installed | weak |
| DocTamper DTD (document text edits, external) | `src/forensics/dtd.py` | in verdict when installed | no (in our test) |

## 2. Real-world photo test (`scripts/eval_realworld.py`)

14 real camera photos from Wikimedia Commons, each as 2 clean copies and 6
realistic edits (paste, clone, inpaint, painted lines, text, paste + messenger).
Details and tables: README, "Real-world results".

- ELA located 0/84 edits and raised false alarms on 6/28 clean copies.
  Camera photos are saved at quality 93-97 and edits are re-saved at about 90,
  so there is no compression gap for ELA to see.
- TruFor marks the edited spot in 53/84 cases, but also marks small regions on
  19/28 clean copies. As an automatic yes/no decision with zero false alarms it
  catches 20/84 (24%). `TruForReport.suspicious` uses that rule.
- Thin edits (scratches, text) are weak because images are downscaled to
  1024 px for CPU speed.

## 3. Receipt forgery test

A real German supermarket receipt photo (already compressed by Telegram,
JPEG quality 87). The total was changed from 13,07 to 18,07 in two places by
copying an "8" from another price on the same receipt (inpaint the old digit,
darken-blend the new one). Two versions were tested: saved once like a phone
editor (quality 92), and the same forgery sent through a messenger again
(quality 80).

| Check | Forged, saved once | Forged, then messenger |
|---|---|---|
| ELA | found both digits (one region) | missed |
| TruFor | missed (score 0.12) | missed (0.10) |
| DocTamper DTD (receipt weights) | **found both digits exactly**, original clean | missed (max 0.43) |
| Duplicate search | - | flagged as copy of the first upload |
| Arithmetic (items 13,07 vs total 18,07) | would catch it | would catch it |

Lessons:

1. ELA works when the image was compressed harder *before* the edit than
   after it (here 87 then 92). It cannot be trusted alone: its false alarms on
   clean photos reach the same score as this real hit.
2. General photo models (TruFor) miss tiny text edits; a document model (DTD)
   catches them.
3. A second messenger recompression removes the pixel traces for every method
   tried. Only content checks (arithmetic, duplicates, context) survive.

## 4. DocTamper DTD (integrated as optional "document scan")

DTD (Qu et al., "Towards Robust Tampered Text Detection in Document Image",
CVPR 2023, https://github.com/qcf-568/DocTamper) reads the grey image plus the
JPEG's raw 8x8 DCT coefficients and quantization table, so it sees compression
mismatches around edited characters. Weights exist for DocTamper and for
T-SROIE (tampered receipts).

How it was run safely:

- Only the trained checkpoints were loaded, with `torch.load(weights_only=True)`.
  Their pickle contents were inspected without executing them first: tensors
  and an `OrderedDict` only.
- The released code loads two pickled *modules* (`vph_imagenet.pt`,
  `swin_imagenet.pt`) that would need full unpickling. They were not used: the
  `VPH` part was rebuilt from the checkpoint's tensor shapes (the released class
  is incomplete), the Swin-V2 stages from `swins.py` with the training
  resolution taken from the stored attention masks. All 798 tensors load with
  0 missing / 0 unexpected keys.
- Two library incompatibilities were patched (segmentation-models-pytorch 0.2
  `Conv2dReLU` signature; its `load_state_dict`/`train` overrides).
- Inference uses 512x512 tiles like the authors' evaluation; tiles without
  visible content are skipped, which brought a receipt from about 4 minutes to
  about 40 seconds on a laptop CPU. JPEG input only.
- Yes/no rule (`DtdReport.suspicious`): a region with peak >= 0.9 and at least
  150 pixels. Tuned on one receipt (forged digits 1.0, untouched receipt only a
  27 px speck at 0.77), so it needs a larger test set.

Licence: the repo has no licence file and the dataset is non-commercial, so
the code and weights stay outside this repo (same approach as TruFor).

## 5. Noise check (prototype, not integrated)

Painted-over areas (for example white redaction strokes) have no sensor noise.
A block detector for "noise-free patch on noisy paper with a brightness jump"
found both white strokes on the receipt with 0 false alarms on 57 clean photos
(settings: noise-pixel fraction < 0.05, surrounding noise >= 0.4, brightness
jump >= 75). Tuned on a single edited image, so it needs a proper test set
before it goes into the verdict.

## 6. How the industry handles this (sources)

- Layered checks rather than one detector: metadata and file structure, OCR
  plus semantic consistency (arithmetic, merchant details), image forensics,
  and comparison against earlier submissions
  ([Resistant AI](https://resistant.ai/blog/how-to-spot-fake-receipts),
  [Inscribe](https://www.inscribe.ai/fraud-detection/inscribe-vs-resistant-ai)).
- Avoid recompression: in-app capture and C2PA content credentials
  ([ZCAM](https://zcam-sdk.succinct.xyz/usecases/insurance_claim_verification),
  [Guidewire](https://www.guidewire.com/resources/blog/industry-trends/combating-ai-generated-media-fraud-in-insurance-claims)).
- Train on platform-processed images: Wu et al., "Robust Image Forgery
  Detection Over Online Social Network Shared Images", CVPR 2022
  ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Robust_Image_Forgery_Detection_Over_Online_Social_Network_Shared_Images_CVPR_2022_paper.html)).
- Receipt benchmark: "Find it again", ICDAR 2023, 988 receipts with marked
  forgeries ([dataset](https://l3i-share.univ-lr.fr/2023Finditagain/index.html)).
- German receipts carry a TSE signature, counter and start/stop timestamps
  (KassenSichV); plausibility checks on these are possible without the
  register's keys.

## 7. Roadmap

1. **Compression warning**: detect messenger-typical files (quality 80-87,
   typical sizes, no metadata) and say "pixel checks limited, ask for the
   original" instead of reporting "clean".
2. **Receipt consistency check**: OCR, then items vs total, net + VAT = gross,
   date vs TSE timestamps. Works regardless of compression.
3. ~~DTD integration~~ done: `POST /api/scan/{id}/document`, UI tick box and
   heatmap tab.
4. **Noise check** into the verdict after testing on a larger set.
5. **Fair evaluation**: about 50 photos plus "Find it again", tune on half,
   report on the other half.
6. **Speed**: ONNX Runtime + INT8 for TruFor, or a GPU worker behind a queue.
