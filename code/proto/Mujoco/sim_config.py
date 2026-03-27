"""
sim_config.py
=============
All simulation parameters in one place.
This is the only file you need to edit for most experiments.
"""
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent / 'Brain'))
from morphology import QUADRIPOD, TRIPOD, HEXAPOD, RobotMorphology

# ---------------------------------------------------------------------------
# Feature Activation
# ---------------------------------------------------------------------------
VIEWER_ON = True
VIDEO_RENDERER_ON = False

DATA_MODE = "Full" # StartStop / Full
SAVE_BEST = False # Create a "last_best.pkl"
UNIQUE_SAVE_BEST = False # Create a "best_392304702147.pkl"

SHOW_LIVE_POS_ON = False

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
N                   = 11    # number of robots to simulate in parallel
ROBOT_SPACING       = 1.    # distance between robots in the viewer (metres)
SIMULATION_DURATION = 5.0    # seconds
ROBOT_CONTROL = "external"    # pre-configured / external

# ---------------------------------------------------------------------------
# Controller initialisation
# ---------------------------------------------------------------------------
# None                                    → all robots start with fresh random weights
# "last_best"                             → load all from the latest best robots save
# "last_sim"                              → load all from the previous simulation
# {"source": "last_sim",                  → load only listed indices from source,
#  "indices": [0, 4, 5, 6, 9]}              fresh random weights for the rest
# {"source": "best_20250323_143012",      → same, from a specific named save
#  "indices": "all"}
# {"source": "last_best",                 → load all from the latest best robot saved and mutate it according to parameter
#  "indices": "mutation",
#  "amplitude": 0.2, "variation": 0.1}
CONTROLLER_INIT = "last_best" #{"source": "last_best", "indices": "mutation", "amplitude": 0.3, "variation": 0.3, "morph_amp": 0.2, "morph_var": 0.3, "morph_mod": 0.25}
CLEAR_ARCHIVE = False

# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------
# One RobotMorphology, or a list (padded to length N with the last entry).
# Pre-defined: QUADRIPOD (4 legs), TRIPOD (3 legs), HEXAPOD (6 legs), None (random)
# Example mixed population:
#   MORPHOLOGIES = [QUADRIPOD] * 15 + [TRIPOD] * 10
MORPHOLOGIES = None #[TRIPOD] * 33 + [QUADRIPOD] * 33 + [HEXAPOD] * 33
MAX_LEGS = 6

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR     = Path(__file__).parent
PHYSICS_XML = ROOT_DIR / "quadripod_robot.xml" # tripod_robot.xml / quadripod_robot.xml
RENDER_DIR  = ROOT_DIR / "render"

# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------
VIDEO_FPS     = 2    # target frames per second (20 is smooth enough for gait analysis)
RENDER_WIDTH  = 240   # pixel width of each robot's video
RENDER_HEIGHT = 192   # pixel height of each robot's video

