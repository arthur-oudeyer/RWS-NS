import numpy as np
import copy
from sim_config import N

# ---------------------------------------------------------------------------
# Robot physical description
# Number of rotational hip joints on the robot.
# Used by read_robot_sensors() to slice qpos/qvel without hardcoded indices.
#
# qpos layout (freejoint always occupies the first 7 slots):
#   [0:3]          freejoint position  (x, y, z)
#   [3:7]          freejoint quaternion (w, x, y, z)
#   [7:7+N_HIP]    hip joint angles
#
# qvel layout (freejoint always occupies the first 6 slots):
#   [0:3]          linear  velocity (vx, vy, vz)
#   [3:6]          angular velocity (wx, wy, wz)
#   [6:6+N_HIP]    hip joint velocities
# ---------------------------------------------------------------------------
N_HIP = 3

# ---------------------------------------------------------------------------
# Robot control configs — one dict per robot
#
# Keys:
#   amplitude   — peak swing angle in radians
#   frequency   — oscillations per second (Hz)
#   static      — constant angle offset per leg (radians), shifts resting pose
#   phase_legs  — per-leg phase offset in the sine wave (radians)
#
# If N > len(ROBOT_CONFIGS), the last entry is repeated automatically.
# ---------------------------------------------------------------------------
ROBOT_CONFIGS = [
    {
        "amplitude":  30 / 180 * np.pi,
        "frequency":  1.0,
        "static":     np.array([-10, -10, -10]) / 180 * np.pi,
        "phase_legs": np.array([0.0, 0.0, 4 * np.pi / 3]),
    },
    {
        "amplitude":  40 / 180 * np.pi,
        "frequency":  1.5,
        "static":     np.array([-5, -5, -5]) / 180 * np.pi,
        "phase_legs": np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]),
    },
    {
        "amplitude":  20 / 180 * np.pi,
        "frequency":  2.0,
        "static":     np.array([0, 0, 0]) / 180 * np.pi,
        "phase_legs": np.array([0.0, np.pi, 0.0]),
    },
    {
        "amplitude":  50 / 180 * np.pi,
        "frequency":  0.8,
        "static":     np.array([-15, -15, -15]) / 180 * np.pi,
        "phase_legs": np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]),
    },
]

# Pad list to length N by repeating the last entry
if len(ROBOT_CONFIGS) < N:
    print("Config ERROR : N > len(ROBOT_CONFIGS) !")
while len(ROBOT_CONFIGS) < N:
    ROBOT_CONFIGS.append(copy.deepcopy(ROBOT_CONFIGS[-1]))

