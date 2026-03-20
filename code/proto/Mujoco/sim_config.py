"""
sim_config.py
=============
All simulation parameters in one place.
This is the only file you need to edit for most experiments.
"""

import copy
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Feature Activation
# ---------------------------------------------------------------------------
VIEWER_ON = False
VIDEO_RENDERER_ON = True
SHOW_LIVE_POS_ON = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR     = Path(__file__).parent
PHYSICS_XML = ROOT_DIR / "tripod_robot.xml"
RENDER_DIR  = ROOT_DIR / "render"

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
N                   = 10     # number of robots to simulate in parallel
ROBOT_SPACING       = 0.7    # distance between robots in the viewer (metres)
SIMULATION_DURATION = 5.0    # seconds

# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------
VIDEO_FPS     = 10    # target frames per second (20 is smooth enough for gait analysis)
RENDER_WIDTH  = 240   # pixel width  of each robot's video
RENDER_HEIGHT = 192   # pixel height of each robot's video

# ---------------------------------------------------------------------------
# Robot control configs — one dict per robot (= genome in MAP-Elites).
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
while len(ROBOT_CONFIGS) < N:
    print("Config ERROR : N > len(ROBOT_CONFIGS) !")
    ROBOT_CONFIGS.append(copy.deepcopy(ROBOT_CONFIGS[-1]))
