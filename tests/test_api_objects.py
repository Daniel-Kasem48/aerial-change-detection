"""Regression test: compare_objects with YOLO labels (class_agnostic_output: false)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from src import api


def test_compare_objects_with_labels_saves_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = np.full((120, 160, 3), 90, np.uint8)
    b = a.copy()
    cv2.rectangle(b, (60, 40), (100, 80), (255, 255, 255), -1)
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(pa), a)
    cv2.imwrite(str(pb), b)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("object_detector:\n  class_agnostic_output: false\n")

    class FakeYolo:
        def __init__(self, config=None) -> None: ...

        def compare(self, img_a, img_b, alignment_confidence=None):
            return {"detections_a": [], "detections_b": [], "changes": [],
                    "annotated_a": img_a, "annotated_b": img_b,
                    "summary": {"added": 0, "removed": 0, "moved": 0, "total_changes": 0}}

    monkeypatch.setattr(api, "ObjectDetector", FakeYolo)
    out = tmp_path / "side_by_side.jpg"
    api.compare_objects(pa, pb, config_path=cfg, output_path=out, align_first=False)
    assert out.is_file()
