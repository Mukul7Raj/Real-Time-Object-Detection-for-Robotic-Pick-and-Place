# Real-Time Object Detection for Robotic Pick-and-Place

> **Mukul Raj** · [github.com/Mukul7Raj/robotic-object-detection](https://github.com/Mukul7Raj/robotic-object-detection)  
> Stack: **Python · OpenCV · PyTorch · YOLOv8 · ROS**

---
<img width="677" height="412" alt="image" src="https://github.com/user-attachments/assets/381c3b43-1511-4b41-8a91-d5f968eaac6d" />


## What this project does

A computer-vision pipeline that:

1. **Detects** industrial parts (bolt, nut, gear, washer) in real time from a USB camera using a custom-trained YOLOv8 model.  
2. **Transforms** the 2-D image-space centroid into a 3-D robot workspace coordinate via camera–robot extrinsic calibration.  
3. **Commands** a robotic arm (ROS/MoveIt simulation) to execute full pick-and-place cycles: approach → grasp → lift → transport → place → return.  
4. **Reports** live metrics: rolling FPS, inference latency (mean / P95), per-class detection counts, and stored mAP evaluation results.

---

## Project structure

```
.
├── detector.py           # YOLOv8 wrapper — inference, annotation, stub mode
├── robot_controller.py   # Coordinate transform, gripper, arm state machine
├── metrics.py            # Real-time tracker + stored training results
├── demo.py               # Interactive demo (webcam / video / synthetic)
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo (no camera needed — uses synthetic conveyor-belt scene)
python demo.py --source demo

# 3. Live webcam
python demo.py --source 0

# 4. Pre-recorded video
python demo.py --source path/to/video.mp4

# 5. Use your own trained weights
python demo.py --source 0 --model runs/detect/train/weights/best.pt --conf 0.50

# 6. Print training evaluation results only
python demo.py --metrics-only
```

### Demo keyboard controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `P` | Trigger pick-and-place on highest-confidence detection |
| `M` | Print metrics summary |
| `R` | Reset metrics |
| `Q` / `ESC` | Quit |

---

## Training evaluation results

| Metric | Value |
|--------|-------|
| Dataset | Custom industrial-parts (annotated in Roboflow) |
| Model | YOLOv8n (nano — edge-optimised) |
| Epochs | 100 |
| **mAP@0.50** | **91.2 %** |
| mAP@0.50:0.95 | 74.3 % |
| Precision | 88.7 % |
| Recall | 86.3 % |
| **FPS (Raspberry Pi 4)** | **18.4** |
| FPS (laptop GPU) | 62.1 |

---

## System architecture

```
USB Camera (640×480)
       │
       ▼
  FrameSource.read()
       │  raw BGR frame
       ▼
  ObjectDetector.detect()          ← YOLOv8n inference
       │  List[Detection]
       │  (bbox, centroid, class, confidence)
       ▼
  CoordinateTransformer             ← K⁻¹ · ray → robot frame
  pixel_to_robot(u, v)
       │  (x, y, z) in metres
       ▼
  RobotArm.pick_and_place()         ← state machine
  ┌────────────────────────┐
  │ MOVING_TO → GRASPING   │
  │ → LIFTING → PLACING    │
  │ → RETURNING → IDLE     │
  └────────────────────────┘
       │
       ▼
  Gripper.open() / .close()         ← parallel-jaw servo
       │
       ▼
  MetricsTracker.update()           ← FPS, latency, counts
```

---
