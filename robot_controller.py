"""
robot_controller.py
-------------------
Simulated robotic arm controller for pick-and-place tasks.

In a real deployment this module publishes ROS topics to a physical arm
(e.g. MoveIt! trajectory planning).  Here it simulates the full pipeline:

  Image-space centroid  →  Camera-to-robot coordinate transform
                        →  Gripper open/close commands
                        →  Pick-and-place state machine

Author : Mukul Raj  (github.com/Mukul7Raj/robotic-object-detection)
Stack  : Python · ROS (simulated) · NumPy
"""

from __future__ import annotations

import time
import math
import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Camera intrinsics & extrinsics ───────────────────────────────────────────

# Raspberry Pi camera (640×480) – replace with your calibrated values
CAMERA_MATRIX = np.array([
    [554.26,   0.0, 320.0],
    [  0.0, 554.26, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

# Rotation + translation: camera frame → robot base frame
# (derived from hand-eye calibration on the physical setup)
R_CAM_TO_ROBOT = np.array([
    [ 0.0, -1.0,  0.0],
    [ 1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
], dtype=np.float64)

T_CAM_TO_ROBOT = np.array([0.15, -0.05, 0.40], dtype=np.float64)   # metres

# Fixed working-plane height above robot base (objects lie on a flat table)
WORK_PLANE_Z_M = 0.02   # 2 cm


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class RobotPose:
    x: float; y: float; z: float    # metres in robot base frame
    roll: float = 0.0               # radians
    pitch: float = math.pi          # point down
    yaw: float = 0.0

    def __str__(self):
        return (f"Pose(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f} m  |  "
                f"roll={math.degrees(self.roll):.1f}°  "
                f"pitch={math.degrees(self.pitch):.1f}°  "
                f"yaw={math.degrees(self.yaw):.1f}°)")


class ArmState(Enum):
    IDLE        = auto()
    MOVING_TO   = auto()
    GRASPING    = auto()
    LIFTING     = auto()
    PLACING     = auto()
    RETURNING   = auto()
    ERROR       = auto()


# ── Coordinate transform ─────────────────────────────────────────────────────

class CoordinateTransformer:
    """
    Projects a 2-D image-space centroid onto the 3-D robot workspace.

    Steps
    -----
    1. Back-project pixel (u, v) to a normalised camera ray using K^{-1}.
    2. Scale the ray so that its Z-component hits the known work-plane Z.
    3. Apply the rigid-body transform  R * p_cam + T  to get robot coords.
    """

    def __init__(self,
                 K:  np.ndarray = CAMERA_MATRIX,
                 R:  np.ndarray = R_CAM_TO_ROBOT,
                 T:  np.ndarray = T_CAM_TO_ROBOT,
                 z_plane: float = WORK_PLANE_Z_M):
        self._K_inv  = np.linalg.inv(K)
        self._R      = R
        self._T      = T
        self._z_plane = z_plane

    def pixel_to_robot(self, u: int, v: int) -> Tuple[float, float, float]:
        """
        Convert image pixel (u, v) → robot-frame (x, y, z) in metres.
        """
        # 1. Homogeneous pixel → normalised camera ray
        pixel_h = np.array([u, v, 1.0], dtype=np.float64)
        ray_cam = self._K_inv @ pixel_h          # direction in camera frame

        # 2. Scale to intersect the work plane
        #    We want  (R @ (lambda * ray_cam) + T)[2] == z_plane
        #    => lambda = (z_plane - T[2]) / (R @ ray_cam)[2]
        ray_robot = self._R @ ray_cam
        lam = (self._z_plane - self._T[2]) / ray_robot[2]
        p_robot = lam * ray_robot + self._T

        return float(p_robot[0]), float(p_robot[1]), float(p_robot[2])


# ── Gripper ──────────────────────────────────────────────────────────────────

class Gripper:
    """
    Simulates a parallel-jaw gripper.
    In ROS this would publish to /gripper/command (std_msgs/Float64).
    """

    OPEN_APERTURE  = 0.08   # metres (8 cm)
    CLOSE_APERTURE = 0.005  # metres (5 mm)
    ACTUATE_DELAY  = 0.6    # seconds (simulate servo travel time)

    def __init__(self):
        self._aperture = self.OPEN_APERTURE
        self._holding  = False

    def open(self):
        logger.info("[Gripper] Opening  → %.0f mm", self.OPEN_APERTURE * 1000)
        time.sleep(self.ACTUATE_DELAY)
        self._aperture = self.OPEN_APERTURE
        self._holding  = False

    def close(self):
        logger.info("[Gripper] Closing  → %.0f mm", self.CLOSE_APERTURE * 1000)
        time.sleep(self.ACTUATE_DELAY)
        self._aperture = self.CLOSE_APERTURE
        self._holding  = True

    @property
    def is_holding(self) -> bool:
        return self._holding

    @property
    def aperture_mm(self) -> float:
        return self._aperture * 1000


# ── Robot arm (state machine) ─────────────────────────────────────────────────

class RobotArm:
    """
    Simulated 6-DOF robotic arm with a full pick-and-place state machine.

    In a live ROS deployment, `_move_to` would call:
        MoveGroupCommander.set_pose_target(pose)
        MoveGroupCommander.go(wait=True)

    The simulated motion uses a simple linear interpolation with a
    configurable speed limit (metres per second).
    """

    PLACE_POSITION = RobotPose(x=0.40, y=-0.25, z=0.15)   # drop-off bin
    HOME_POSITION  = RobotPose(x=0.30, y=0.0,   z=0.35)
    LIFT_HEIGHT_M  = 0.20    # metres above object before lateral motion
    SPEED_M_S      = 0.25    # simulated TCP speed

    def __init__(self):
        self._transformer = CoordinateTransformer()
        self._gripper     = Gripper()
        self._state       = ArmState.IDLE
        self._current_pos = self.HOME_POSITION
        self._pick_count  = 0

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def state(self) -> ArmState:
        return self._state

    @property
    def pick_count(self) -> int:
        return self._pick_count

    def pick_and_place(self, pixel_u: int, pixel_v: int,
                       object_name: str = "object") -> bool:
        """
        Execute a full pick-and-place cycle for the object whose image-space
        centroid is at (pixel_u, pixel_v).

        Returns True on success, False on failure.
        """
        logger.info("=" * 60)
        logger.info("[Arm] Starting pick-and-place for '%s' at pixel (%d, %d)",
                    object_name, pixel_u, pixel_v)

        try:
            # 1. Transform pixel → robot-frame target
            x, y, z = self._transformer.pixel_to_robot(pixel_u, pixel_v)
            pick_approach = RobotPose(x=x, y=y, z=self.LIFT_HEIGHT_M)
            pick_target   = RobotPose(x=x, y=y, z=z)

            logger.info("[Arm] Robot target: (%.3f, %.3f, %.3f) m", x, y, z)

            # 2. Pre-grasp approach
            self._state = ArmState.MOVING_TO
            self._gripper.open()
            self._move_to(pick_approach, label="Approach")

            # 3. Lower to object
            self._move_to(pick_target, label="Descend")

            # 4. Grasp
            self._state = ArmState.GRASPING
            self._gripper.close()
            if not self._gripper.is_holding:
                raise RuntimeError("Grasp failed – gripper did not close.")

            # 5. Lift
            self._state = ArmState.LIFTING
            self._move_to(pick_approach, label="Lift")

            # 6. Move to place position
            self._state = ArmState.PLACING
            place_approach = RobotPose(x=self.PLACE_POSITION.x,
                                       y=self.PLACE_POSITION.y,
                                       z=self.LIFT_HEIGHT_M)
            self._move_to(place_approach, label="Transport")
            self._move_to(self.PLACE_POSITION, label="Place-down")

            # 7. Release
            self._gripper.open()

            # 8. Return home
            self._state = ArmState.RETURNING
            self._move_to(self.HOME_POSITION, label="Return home")

            self._pick_count += 1
            self._state = ArmState.IDLE
            logger.info("[Arm] ✓ Pick-and-place #%d complete.", self._pick_count)
            return True

        except Exception as exc:
            logger.error("[Arm] ✗ Error: %s", exc)
            self._state = ArmState.ERROR
            self._gripper.open()        # safety: always open on error
            self._state = ArmState.IDLE
            return False

    # ── Simulation helpers ───────────────────────────────────────────────────

    def _move_to(self, target: RobotPose, label: str = ""):
        """Simulate linear TCP motion with speed-limited timing."""
        dx = target.x - self._current_pos.x
        dy = target.y - self._current_pos.y
        dz = target.z - self._current_pos.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        travel_time = dist / self.SPEED_M_S
        logger.info("[Arm]   → %-14s  dist=%.3f m  time=%.2f s  %s",
                    label, dist, travel_time, target)
        time.sleep(min(travel_time, 0.3))   # cap sim delay at 0.3 s
        self._current_pos = target

    def status_dict(self) -> dict:
        return {
            "state":      self._state.name,
            "picks":      self._pick_count,
            "gripper_mm": self._gripper.aperture_mm,
            "pos_x":      self._current_pos.x,
            "pos_y":      self._current_pos.y,
            "pos_z":      self._current_pos.z,
        }


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    arm = RobotArm()
    print("\n--- Coordinate Transform Test ---")
    tf = CoordinateTransformer()
    for (u, v) in [(320, 240), (100, 100), (500, 380)]:
        x, y, z = tf.pixel_to_robot(u, v)
        print(f"  pixel ({u:3d}, {v:3d})  →  robot ({x:.3f}, {y:.3f}, {z:.3f}) m")

    print("\n--- Pick-and-Place Simulation ---")
    ok = arm.pick_and_place(320, 240, object_name="bolt")
    print(f"\nResult: {'SUCCESS ✓' if ok else 'FAILURE ✗'}")
    print(f"Status: {arm.status_dict()}")
