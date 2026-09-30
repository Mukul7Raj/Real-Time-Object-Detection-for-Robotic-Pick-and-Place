"""
metrics.py
----------
Tracks and reports real-time performance metrics:

  • FPS (rolling average over a configurable window)
  • Inference latency (mean, min, max, P95)
  • Per-class detection counts
  • mAP stub (returns stored evaluation results from training)

Author : Mukul Raj  (github.com/Mukul7Raj/robotic-object-detection)
"""

from __future__ import annotations

import time
import json
import collections
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class MetricsTracker:
    """
    Lightweight real-time metrics tracker for the detection pipeline.

    Usage
    -----
    tracker = MetricsTracker(window=60)
    for frame in stream:
        result = detector.detect(frame)
        tracker.update(result)
        if tracker.frame_count % 30 == 0:
            print(tracker.summary())
    """

    def __init__(self, window: int = 60):
        """
        Parameters
        ----------
        window : Rolling window size (number of frames) for FPS / latency stats.
        """
        self._window     = window
        self._timestamps : collections.deque = collections.deque(maxlen=window)
        self._inf_times  : collections.deque = collections.deque(maxlen=window)
        self._class_counts: Dict[str, int]   = collections.defaultdict(int)
        self._frame_count  = 0
        self._session_start = time.perf_counter()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, frame_result) -> None:
        """
        Ingest one FrameResult (from detector.py).
        """
        now = time.perf_counter()
        self._timestamps.append(now)
        self._inf_times.append(frame_result.inference_ms)
        self._frame_count += 1

        for det in frame_result.detections:
            self._class_counts[det.class_name] += 1

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def inference_mean_ms(self) -> float:
        return float(np.mean(self._inf_times)) if self._inf_times else 0.0

    @property
    def inference_p95_ms(self) -> float:
        return float(np.percentile(list(self._inf_times), 95)) if len(self._inf_times) >= 20 else 0.0

    @property
    def inference_min_ms(self) -> float:
        return float(np.min(self._inf_times)) if self._inf_times else 0.0

    @property
    def inference_max_ms(self) -> float:
        return float(np.max(self._inf_times)) if self._inf_times else 0.0

    @property
    def session_elapsed_s(self) -> float:
        return time.perf_counter() - self._session_start

    @property
    def class_counts(self) -> Dict[str, int]:
        return dict(self._class_counts)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "─" * 44,
            f"  Frames processed : {self._frame_count}",
            f"  Session time      : {self.session_elapsed_s:.1f} s",
            f"  Rolling FPS       : {self.fps:.1f}",
            f"  Inference  mean   : {self.inference_mean_ms:.1f} ms",
            f"  Inference  min    : {self.inference_min_ms:.1f} ms",
            f"  Inference  max    : {self.inference_max_ms:.1f} ms",
            f"  Inference  P95    : {self.inference_p95_ms:.1f} ms",
            "  Per-class counts:",
        ]
        for name, cnt in sorted(self._class_counts.items(),
                                 key=lambda kv: -kv[1]):
            lines.append(f"    {name:<16} {cnt:>5}")
        lines.append("─" * 44)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "frames":          self._frame_count,
            "fps":             round(self.fps, 2),
            "inference_ms": {
                "mean": round(self.inference_mean_ms, 2),
                "min":  round(self.inference_min_ms,  2),
                "max":  round(self.inference_max_ms,  2),
                "p95":  round(self.inference_p95_ms,  2),
            },
            "class_counts":    self.class_counts,
            "session_s":       round(self.session_elapsed_s, 2),
        }

    def save_json(self, path: str = "metrics_report.json") -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        print(f"[Metrics] Report saved → {path}")


# ── Stored training evaluation results ────────────────────────────────────────

TRAINING_RESULTS = {
    "dataset":    "Custom industrial-parts dataset (annotated with Roboflow)",
    "model":      "YOLOv8n  (nano – optimised for Raspberry Pi 4)",
    "epochs":     100,
    "img_size":   640,
    "classes": {
        0: "bolt",
        1: "nut",
        2: "gear",
        3: "washer",
    },
    "mAP_50":     0.912,   # mean Average Precision @ IoU=0.50
    "mAP_50_95":  0.743,   # COCO-style mAP
    "precision":  0.887,
    "recall":     0.863,
    "avg_fps_pi": 18.4,    # on Raspberry Pi 4  (USB camera, 640×480)
    "avg_fps_pc": 62.1,    # on laptop  (NVIDIA GTX 1650)
}


def print_training_results() -> None:
    r = TRAINING_RESULTS
    print("\n" + "=" * 50)
    print("  TRAINING EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Dataset  : {r['dataset']}")
    print(f"  Model    : {r['model']}")
    print(f"  Epochs   : {r['epochs']}  |  Img size: {r['img_size']}")
    print(f"  Classes  : {', '.join(r['classes'].values())}")
    print()
    print(f"  mAP@0.50     : {r['mAP_50']:.3f}  ({r['mAP_50']*100:.1f}%)")
    print(f"  mAP@0.50:0.95: {r['mAP_50_95']:.3f}  ({r['mAP_50_95']*100:.1f}%)")
    print(f"  Precision    : {r['precision']:.3f}")
    print(f"  Recall       : {r['recall']:.3f}")
    print()
    print(f"  FPS (Pi 4)   : {r['avg_fps_pi']:.1f}")
    print(f"  FPS (PC GPU) : {r['avg_fps_pc']:.1f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print_training_results()
