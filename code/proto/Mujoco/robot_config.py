import numpy as np
import copy
from sim_config import N, ROBOT_CONTROL, MORPHOLOGIES
from morphology import pad_morphologies

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
# N_HIP is now derived from the morphology — kept for backward compat with ROBOT_CONFIGS
N_HIP = pad_morphologies(1, MORPHOLOGIES)[0].n_joints

# ---------------------------------------------------------------------------
# Robot control configs — one dict per robot
#
# Keys:
#   amplitude   — peak swing angle in radians
#   frequency   — oscillations per second (Hz)
#   static      — constant angle offset per leg (radians), shifts resting pose
#   phase_legs  — per-leg phase offset in the sine wave (radians)
#                 leg order: front(0°), right(90°), back(180°), left(270°)
#
# Gait notes:
#   Trot  : diagonal pairs in sync → [0, π, 0, π]  (legs 1&3 vs 2&4)
#   Walk  : sequential wave        → [0, π/2, π, 3π/2]
#
# If N > len(ROBOT_CONFIGS), the last entry is repeated automatically.
# ---------------------------------------------------------------------------
ROBOT_CONFIGS = [
    {
        # Trot gait — diagonal pairs (front/back vs left/right) in antiphase
        "amplitude":  30 / 180 * np.pi,
        "frequency":  1.0,
        "static":     np.array([-10, -10, -10, -10]) / 180 * np.pi,
        "phase_legs": np.array([0.0, np.pi, 0.0, np.pi]),
    },
]

# Pad list to length N by repeating the last entry
if ROBOT_CONTROL == "pre-configured":
    if len(ROBOT_CONFIGS) < N :
        print("Config ERROR : N > len(ROBOT_CONFIGS) !")
    while len(ROBOT_CONFIGS) < N:
        ROBOT_CONFIGS.append(copy.deepcopy(ROBOT_CONFIGS[-1]))

