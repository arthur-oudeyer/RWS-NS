"""
sim_config.py
=============
All simulation parameters in one place.
This is the only file you need to edit for most experiments.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature Activation
# ---------------------------------------------------------------------------
VIEWER_ON = True
VIDEO_RENDERER_ON = False
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
ROBOT_CONTROL = "pre-configured"    # pre-configured / external

# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------
VIDEO_FPS     = 10    # target frames per second (20 is smooth enough for gait analysis)
RENDER_WIDTH  = 240   # pixel width  of each robot's video
RENDER_HEIGHT = 192   # pixel height of each robot's video

