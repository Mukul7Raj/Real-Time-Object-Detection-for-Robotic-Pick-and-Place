"""
app.py
------
Flask web server — serves the detection stream and REST API.

  http://localhost:5000          →  Live dashboard
  http://localhost:5000/video    →  Raw MJPEG stream
  http://localhost:5000/api/metrics   →  JSON metrics
  http://localhost:5000/api/pick      →  POST: trigger pick-and-place
  http://localhost:5000/api/status    →  JSON arm + system status

Author : Mukul Raj  (github.com/Mukul7Raj/robotic-object-detection)
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from detector         import ObjectDetector
from robot_controller import RobotArm
from metrics          import MetricsTracker, TRAINING_RESULTS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Global state ─────────────────────────────────────────────────────────────
detector  = ObjectDetector(model_path="yolov8n.pt", conf_thresh=0.45)
arm       = RobotArm()
tracker   = MetricsTracker(window=90)
_pick_lock     = threading.Lock()
_pick_thread   = None
_latest_result = None          # shared between generator and /api/pick


# ── Synthetic demo frame generator ────────────────────────────────────────────

def _make_demo_frame(frame_idx: int, w: int = 640, h: int = 480) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        v = int(18 + 38 * row / h)
        frame[row] = (v, v + 4, v + 10)

    belt_y1, belt_y2 = int(h * 0.33), int(h * 0.73)
    cv2.rectangle(frame, (0, belt_y1), (w, belt_y2), (48, 42, 36), -1)
    for x in range(0, w + 60, 60):
        off = int(frame_idx * 2.5) % 60
        cv2.line(frame, (x - off, belt_y1), (x - off, belt_y2), (65, 58, 50), 2)

    parts = [
        {"name": "bolt",   "color": (30,  210, 255), "r": 22, "speed": 1.8, "yf": 0.46},
        {"name": "nut",    "color": (50,  255, 130), "r": 17, "speed": 1.1, "yf": 0.60},
        {"name": "gear",   "color": (255, 125,  30), "r": 28, "speed": 0.8, "yf": 0.52},
        {"name": "washer", "color": (200,  55, 255), "r": 14, "speed": 2.0, "yf": 0.56},
    ]
    for i, p in enumerate(parts):
        x = int((frame_idx * p["speed"] + i * 170) % (w + 70)) - 35
        y = int(h * p["yf"])
        cv2.circle(frame, (x, y), p["r"],          p["color"], -1)
        cv2.circle(frame, (x, y), p["r"],          (255, 255, 255), 1)
        cv2.circle(frame, (x, y), p["r"] // 3,     (25, 25, 25), -1)

    return frame


# ── MJPEG frame generator ─────────────────────────────────────────────────────

def _frame_generator(source: str = "demo"):
    global _latest_result

    cap       = None
    frame_idx = 0

    if source != "demo":
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            logger.warning("Cannot open source '%s', falling back to demo", source)
            source = "demo"

    try:
        while True:
            frame_idx += 1

            if source == "demo":
                frame = _make_demo_frame(frame_idx)
                time.sleep(1 / 30)
            else:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()

            result = detector.detect(frame)
            tracker.update(result)
            _latest_result = result

            annotated = detector.annotate(frame, result)

            # Arm status bar
            st = arm.status_dict()
            bar = (f"  Arm: {st['state']}   "
                   f"Picks: {st['picks']}   "
                   f"Gripper: {st['gripper_mm']:.0f} mm  ")
            h = annotated.shape[0]
            cv2.rectangle(annotated, (0, h - 26), (annotated.shape[1], h), (15, 15, 15), -1)
            cv2.putText(annotated, bar, (6, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 160), 1, cv2.LINE_AA)

            # JPEG encode
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() +
                   b"\r\n")
    finally:
        if cap:
            cap.release()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", training=TRAINING_RESULTS)


@app.route("/video")
def video():
    source = request.args.get("source", "demo")
    return Response(
        _frame_generator(source),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/metrics")
def api_metrics():
    d = tracker.to_dict()
    d["arm"] = arm.status_dict()
    return jsonify(d)


@app.route("/api/status")
def api_status():
    return jsonify({
        "arm":      arm.status_dict(),
        "frames":   tracker.frame_count,
        "fps":      round(tracker.fps, 1),
        "training": TRAINING_RESULTS,
    })


@app.route("/api/pick", methods=["POST"])
def api_pick():
    global _pick_thread
    with _pick_lock:
        if _pick_thread and _pick_thread.is_alive():
            return jsonify({"ok": False, "message": "Arm is busy"}), 409

        result = _latest_result
        if not result or not result.detections:
            return jsonify({"ok": False, "message": "No objects detected"}), 400

        best = max(result.detections, key=lambda d: d.confidence)
        cx, cy = best.centroid

        def _do_pick():
            arm.pick_and_place(cx, cy, best.class_name)

        _pick_thread = threading.Thread(target=_do_pick, daemon=True)
        _pick_thread.start()

        return jsonify({
            "ok":      True,
            "object":  best.class_name,
            "conf":    round(best.confidence, 3),
            "pixel":   [cx, cy],
            "message": f"Picking '{best.class_name}' at pixel ({cx}, {cy})",
        })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global tracker
    tracker = MetricsTracker(window=90)
    return jsonify({"ok": True, "message": "Metrics reset"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Pick-and-Place Detection Server")
    logger.info("  http://localhost:5000")
    logger.info("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
