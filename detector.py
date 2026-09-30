"""
detector.py
-----------
YOLO-based object detector for industrial parts.

This module wraps YOLOv8 (Ultralytics) and provides:
  - Real-time inference on frames from a USB camera or video file
  - Confidence-thresholded bounding-box extraction
  - Per-class centroid computation for downstream robot control

Author : Mukul Raj  (github.com/Mukul7Raj/robotic-object-detection)
Stack  : Python · OpenCV · PyTorch · YOLOv8
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Try to import Ultralytics YOLOv8; fall back to a lightweight stub so the
# rest of the code (demo, tests) can still run without GPU / model weights.
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO as _YOLO          # pip install ultralytics
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    _YOLO = None


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object in one frame."""
    class_id:   int
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2  (pixels)
    centroid:   Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self)  -> int: return self.bbox[2] - self.bbox[0]
    @property
    def height(self) -> int: return self.bbox[3] - self.bbox[1]
    @property
    def area(self)   -> int: return self.width * self.height


@dataclass
class FrameResult:
    """Detection results for a single frame, including timing."""
    detections:    List[Detection]
    inference_ms:  float          # raw model inference time
    total_ms:      float          # end-to-end frame time (capture → annotate)
    frame_id:      int

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


# ── Detector ─────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    YOLOv8 wrapper tuned for real-time inference on a Raspberry Pi / laptop.

    Parameters
    ----------
    model_path  : Path to YOLOv8 weights (.pt).  Use 'yolov8n.pt' for the
                  nano model (fastest on edge hardware).
    conf_thresh : Minimum confidence to accept a detection (0–1).
    iou_thresh  : IoU threshold for NMS.
    class_names : Optional dict {id: name} to override the model's built-in
                  labels (useful when the model was trained on custom data).
    device      : 'cpu' | 'cuda' | 'mps'  – auto-selected if None.
    """

    # Colour palette – one BGR colour per class index (cycles automatically)
    _PALETTE = [
        (0,   220, 255),   # cyan-yellow
        (0,   165, 255),   # orange
        (255, 100,   0),   # blue
        (100, 255,   0),   # lime
        (255,   0, 180),   # magenta
        (0,   255, 128),   # spring green
    ]

    def __init__(
        self,
        model_path:   str  = "yolov8n.pt",
        conf_thresh:  float = 0.45,
        iou_thresh:   float = 0.45,
        class_names:  Optional[dict] = None,
        device:       Optional[str]  = None,
    ):
        self.conf_thresh  = conf_thresh
        self.iou_thresh   = iou_thresh
        self._custom_names = class_names or {}
        self._frame_count  = 0

        # ── Load model ──────────────────────────────────────────────────────
        if _ULTRALYTICS_AVAILABLE:
            self._model = _YOLO(model_path)
            if device:
                self._model.to(device)
            # Warm-up pass so the first real frame isn't slow
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            self._stub = False
            print(f"[Detector] Loaded '{model_path}' via Ultralytics ✓")
        else:
            print("[Detector] ⚠ Ultralytics not found – running in STUB mode "
                  "(random detections for demo purposes).")
            self._model = None
            self._stub  = True

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> FrameResult:
        """
        Run detection on a BGR frame (as returned by cv2.VideoCapture.read).

        Returns a FrameResult with all accepted detections.
        """
        t_start = time.perf_counter()
        self._frame_count += 1

        if self._stub:
            detections, inf_ms = self._stub_detect(frame)
        else:
            detections, inf_ms = self._yolo_detect(frame)

        total_ms = (time.perf_counter() - t_start) * 1000
        return FrameResult(detections, inf_ms, total_ms, self._frame_count)

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """
        Draw bounding boxes, labels, centroids and overlay HUD on *frame*.
        Returns a new annotated copy (original is not modified).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        for det in result.detections:
            color = self._PALETTE[det.class_id % len(self._PALETTE)]
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.centroid

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Filled label background
            label = f"{det.class_name}  {det.confidence:.0%}"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - lh - bl - 4), (x1 + lw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Centroid cross-hair
            cv2.drawMarker(out, (cx, cy), color,
                           cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # ── HUD overlay ─────────────────────────────────────────────────────
        hud_lines = [
            f"FPS : {result.fps:5.1f}",
            f"Inf : {result.inference_ms:5.1f} ms",
            f"Det : {len(result.detections)}",
            f"Frm : {result.frame_id}",
        ]
        pad, line_h = 8, 22
        box_h = pad * 2 + line_h * len(hud_lines)
        overlay = out.copy()
        cv2.rectangle(overlay, (w - 155, 0), (w, box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
        for i, txt in enumerate(hud_lines):
            cv2.putText(out, txt, (w - 148, pad + (i + 1) * line_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

        return out

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _yolo_detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        t0 = time.perf_counter()
        results = self._model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )
        inf_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        for r in results:
            names = self._custom_names or r.names
            for box in r.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(cid, names.get(cid, str(cid)), conf,
                                            (x1, y1, x2, y2)))
        return detections, inf_ms

    def _stub_detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """Synthetic detections for demo / testing without real weights."""
        h, w = frame.shape[:2]
        rng   = np.random.default_rng(self._frame_count % 30)   # stable seed per frame
        names = self._custom_names or {0: "bolt", 1: "nut", 2: "gear", 3: "washer"}
        n_det = rng.integers(1, 4)
        detections: List[Detection] = []
        for _ in range(n_det):
            cid  = int(rng.integers(0, len(names)))
            conf = float(rng.uniform(0.55, 0.99))
            x1   = int(rng.integers(50, w - 150))
            y1   = int(rng.integers(50, h - 150))
            x2   = x1 + int(rng.integers(60, 140))
            y2   = y1 + int(rng.integers(60, 140))
            detections.append(Detection(cid, names[cid], conf, (x1, y1, x2, y2)))
        return detections, rng.uniform(8, 30)   # fake inference time


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    det = ObjectDetector()               # stub if ultralytics missing
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res   = det.detect(blank)
    ann   = det.annotate(blank, res)
    print(f"Detections: {len(res.detections)}  |  FPS: {res.fps:.1f}")
    cv2.imshow("Detector smoke-test", ann)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
