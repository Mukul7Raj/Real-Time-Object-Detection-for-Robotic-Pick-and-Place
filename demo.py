"""
demo.py
-------
Interactive real-time demo for interview / presentation.

Modes
─────
  --source 0          Live webcam (default)
  --source video.mp4  Pre-recorded video
  --source demo       Synthetic animated frames (no camera needed)

Controls (while window is open)
──────────────────────────────
  SPACE   Pause / resume
  P       Trigger a single pick-and-place cycle (on highest-confidence detection)
  M       Print metrics summary to console
  R       Reset metrics
  Q / ESC Quit

Author : Mukul Raj  (github.com/Mukul7Raj/robotic-object-detection)
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Project modules
from detector        import ObjectDetector
from robot_controller import RobotArm
from metrics         import MetricsTracker, print_training_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Synthetic frame generator (no camera needed) ──────────────────────────────

def _make_demo_frame(frame_idx: int, w: int = 640, h: int = 480) -> np.ndarray:
    """
    Generates an animated conveyor-belt scene with moving coloured blobs that
    represent industrial parts (bolt, nut, gear, washer).
    Used when --source demo is passed so the demo runs without any camera.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Dark background with gradient
    for row in range(h):
        v = int(20 + 40 * row / h)
        frame[row] = (v, v + 5, v + 10)

    # Conveyor belt lines
    belt_y1, belt_y2 = int(h * 0.35), int(h * 0.75)
    cv2.rectangle(frame, (0, belt_y1), (w, belt_y2), (50, 45, 40), -1)
    for x in range(0, w, 60):
        offset = (frame_idx * 3) % 60
        cv2.line(frame, (x - offset, belt_y1), (x - offset, belt_y2), (70, 65, 60), 2)

    # Parts – each moves at a different speed
    parts = [
        {"name": "bolt",   "color": (30,  200, 255), "radius": 22,
         "speed": 1.8, "y_frac": 0.48},
        {"name": "nut",    "color": (50,  255, 120), "radius": 18,
         "speed": 1.2, "y_frac": 0.60},
        {"name": "gear",   "color": (255, 120,  30), "radius": 28,
         "speed": 0.9, "y_frac": 0.52},
        {"name": "washer", "color": (200,  60, 255), "radius": 15,
         "speed": 2.1, "y_frac": 0.57},
    ]
    for i, p in enumerate(parts):
        x = int((frame_idx * p["speed"] + i * 160) % (w + 60)) - 30
        y = int(h * p["y_frac"])
        cv2.circle(frame, (x, y), p["radius"], p["color"], -1)
        cv2.circle(frame, (x, y), p["radius"], (255, 255, 255), 1)
        # inner detail
        cv2.circle(frame, (x, y), p["radius"] // 3, (30, 30, 30), -1)

    # Top banner
    banner = "Real-Time Object Detection — Robotic Pick-and-Place Demo"
    cv2.putText(frame, banner, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


# ── Camera / source abstraction ───────────────────────────────────────────────

class FrameSource:
    def __init__(self, source: str | int):
        self._demo   = (source == "demo")
        self._idx    = 0
        self._cap    = None
        if not self._demo:
            self._cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
            if not self._cap.isOpened():
                logger.warning("Could not open source '%s'. Switching to demo mode.", source)
                self._demo = True

    def read(self):
        if self._demo:
            self._idx += 1
            frame = _make_demo_frame(self._idx)
            time.sleep(1 / 30)           # ~30 FPS synthetic stream
            return True, frame
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video files
            ok, frame = self._cap.read()
        return ok, frame

    def release(self):
        if self._cap:
            self._cap.release()


# ── Main demo loop ────────────────────────────────────────────────────────────

def run_demo(source: str = "demo",
             model:  str = "yolov8n.pt",
             conf:   float = 0.45,
             show_metrics_every: int = 150) -> None:

    print_training_results()

    detector = ObjectDetector(model_path=model, conf_thresh=conf)
    arm      = RobotArm()
    tracker  = MetricsTracker(window=60)
    src      = FrameSource(source)

    paused        = False
    pick_thread   = None

    logger.info("Demo started. Controls: SPACE=pause  P=pick  M=metrics  Q=quit")

    while True:
        if not paused:
            ok, frame = src.read()
            if not ok:
                break

            result = detector.detect(frame)
            tracker.update(result)
            annotated = detector.annotate(frame, result)

            # Arm status overlay
            status = arm.status_dict()
            status_txt = (f"Arm: {status['state']}   "
                          f"Picks: {status['picks']}   "
                          f"Gripper: {status['gripper_mm']:.0f} mm")
            cv2.putText(annotated, status_txt,
                        (8, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 255, 180), 1, cv2.LINE_AA)

            if tracker.frame_count % show_metrics_every == 0:
                print(tracker.summary())

            cv2.imshow("Pick-and-Place Object Detection", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):        # Q or ESC
            break

        elif key == ord(' '):            # pause / resume
            paused = not paused
            logger.info("%-8s", "PAUSED" if paused else "RESUMED")

        elif key == ord('p'):            # trigger pick
            if result.detections and (pick_thread is None or not pick_thread.is_alive()):
                best = max(result.detections, key=lambda d: d.confidence)
                cx, cy = best.centroid
                logger.info("Pick triggered → '%s' at pixel (%d, %d)", best.class_name, cx, cy)
                pick_thread = threading.Thread(
                    target=arm.pick_and_place,
                    args=(cx, cy, best.class_name),
                    daemon=True,
                )
                pick_thread.start()
            else:
                logger.info("No detections to pick (or arm busy).")

        elif key == ord('m'):            # metrics summary
            print(tracker.summary())

        elif key == ord('r'):            # reset metrics
            tracker = MetricsTracker()
            logger.info("Metrics reset.")

    src.release()
    cv2.destroyAllWindows()
    print("\nFinal metrics:")
    print(tracker.summary())
    tracker.save_json("metrics_report.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-Time Object Detection for Robotic Pick-and-Place")
    p.add_argument("--source", default="demo",
                   help="Video source: 0 (webcam), path/to/video.mp4, or 'demo'")
    p.add_argument("--model",  default="yolov8n.pt",
                   help="YOLOv8 weights file (default: yolov8n.pt)")
    p.add_argument("--conf",   type=float, default=0.45,
                   help="Confidence threshold (default: 0.45)")
    p.add_argument("--metrics-only", action="store_true",
                   help="Only print training evaluation results and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.metrics_only:
        print_training_results()
        sys.exit(0)

    run_demo(source=args.source, model=args.model, conf=args.conf)
