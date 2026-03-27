# RWS-NS-Proto

Prototype for evolving locomotion controllers on simulated legged robots using MuJoCo physics and a custom neural network. Robots have programmable morphologies (leg count, joint sizes) that can be mutated alongside their neural network weights.

---

## Project structure

```
code/proto/
├── Mujoco/                   # Simulation layer
│   ├── main_sim.py           # Entry point — run this
│   ├── sim_config.py         # All configuration (edit this for experiments)
│   ├── control.py            # Sensor reading + controller dispatch
│   ├── robot_config.py       # Derived robot properties (N_HIP from morphology)
│   ├── display.py            # Multi-robot viewer (square grid layout)
│   ├── data.py               # DataManager — records sensors, computes metrics, saves best
│   ├── video_render.py       # Optional per-robot video export
│   └── demo/                 # Standalone demo scripts
│
└── Brain/                    # Controller layer
    ├── controller.py         # Selects and initialises the active controller
    ├── morphology.py         # Morphology descriptors, XML generation, MutateMorphology()
    ├── simple_brain.py       # Neural network controller + Mutate()
    ├── saver.py              # Save / load NeuralNetwork + morphologies to Brain/saves/
    ├── saves/                # Persisted controllers (.pkl files)
    └── simplebrain_loc/
        ├── brain.py          # NeuralNetwork, Layer, Neuron classes
        ├── butils.py         # Activation functions (tanh, sigmoid)
        ├── bmath.py          # Math helpers, random, normal distribution
        ├── bgradient.py      # Gradient utilities
        └── bmutation.py      # Mutation helpers
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mujoco` | Physics simulation and viewer |
| `numpy` | Array operations throughout |

Install with:
```bash
pip install mujoco numpy
```

The viewer requires `mjpython` (bundled with the MuJoCo Python package on macOS):
```bash
pip install mujoco
```

---

## How to run

```bash
cd code/proto/Mujoco
mjpython main_sim.py
```

> Use `mjpython` instead of `python` to get the MuJoCo viewer on macOS.
> On Linux `python main_sim.py` works directly.

---

## How to configure

**`sim_config.py`** is the single file to edit between experiments.

### Simulation

| Parameter | Description |
|---|---|
| `N` | Number of robots simulated in parallel |
| `SIMULATION_DURATION` | Duration in seconds |
| `ROBOT_SPACING` | Distance between robots in the viewer (metres) |
| `ROBOT_CONTROL` | `"external"` (neural net) or `"pre-configured"` (fixed sine gait) |

### Display & recording

| Parameter | Description |
|---|---|
| `VIEWER_ON` | Open the MuJoCo viewer window |
| `VIDEO_RENDERER_ON` | Export one `.mp4` per robot to `Mujoco/render/` |
| `SHOW_LIVE_POS_ON` | Print joint positions to the terminal each second |

### Data & saves

| Parameter | Description |
|---|---|
| `DATA_MODE` | `"Full"` (every step) or `"StartStop"` (first + last only) |
| `SAVE_BEST` | Save the best robot to `Brain/saves/last_best.pkl` after each sim |
| `UNIQUE_SAVE_BEST` | Also save a timestamped archive `best_YYYYMMDD_HHMMSS.pkl` |

### Morphology

Each robot's body is defined by a `RobotMorphology` descriptor — no static XML files. The simulation generates MuJoCo XML at runtime.

```python
MORPHOLOGIES = QUADRIPOD   # 4 legs at 0/90/180/270°
MORPHOLOGIES = TRIPOD      # 3 legs at 0/120/240°
MORPHOLOGIES = HEXAPOD     # 6 legs at 0/60/120/180/240/300°
MORPHOLOGIES = [QUADRIPOD, TRIPOD, TRIPOD]  # heterogeneous population
```

Pre-built presets are imported from `Brain/morphology.py`. A custom morphology can be built from `RobotMorphology`, `LegDescriptor`, and `JointDescriptor` dataclasses.

### Controller initialisation

Controls what weights (and morphologies) each robot starts with:

```python
CONTROLLER_INIT = None
# → all N robots start with fresh random weights + MORPHOLOGIES

CONTROLLER_INIT = "last_best"
# → all N robots are copies of the best robot from the previous simulation
# → morphology is restored from the save file (MORPHOLOGIES is ignored)

CONTROLLER_INIT = "last_sim"
# → each robot i reloads its own weights from the previous simulation
# → morphologies are restored from the save file (MORPHOLOGIES is ignored)

CONTROLLER_INIT = {"source": "last_sim", "indices": [0, 4, 5, 6, 9]}
# → robots 0,4,5,6,9 reload from last_sim, the rest start fresh

CONTROLLER_INIT = {"source": "best_20260323_171407", "indices": "all"}
# → load from a specific timestamped archive

CONTROLLER_INIT = {
    "source": "last_best",
    "indices": "mutation",
    "amplitude": 0.2,    # weight perturbation scale
    "variation": 0.2,    # per-weight variation scale
    "morphology": 0.1,   # morphological mutation amplitude (see below)
}
# → robot 0 is a copy of last_best; robots 1..N-1 are mutated clones
# → if "morphology" is set, each mutated robot also gets a perturbed body
```

---

## Morphological mutation

`MutateMorphology(morph, amplitude)` in `Brain/morphology.py` creates a modified copy of a morphology:

- **Continuous**: each joint's `length` and `radius` are scaled by `1 + N(0, amplitude)`, clamped to safe bounds.
- **Discrete** (probability ∝ amplitude): randomly add a leg (placed at the largest angular gap between existing legs) or remove a random leg (minimum 2 legs kept).

When `CONTROLLER_INIT` uses `"indices": "mutation"` and a `"morphology"` key is present:
- Robot 0 keeps the saved morphology unchanged.
- Robots 1..N-1 each get an independently mutated body.
- If the mutation changes the joint count, the robot gets a fresh randomly-initialised brain (instead of a mutated copy of robot 0's brain).

---

## Neural network controller

The default controller (`simple_brain.py`) is a fully-connected network with `tanh` activations.

- **Inputs**: 3 clock signals `sin(ω·t)` at different frequencies + 2 × n_joints sensor values (angles + velocities) — automatically sized to the robot's morphology
- **Outputs**: n_joints Δangle increments added to current hip angles each step — automatically sized
- **Architecture**: configured via `NB_NEURONS_BY_LAYER` in `simple_brain.py`

Weights are initialised with Xavier (`N(0, 1/√n_inputs)`). The `Mutate()` function creates a perturbed copy of a network for evolutionary search.

### Save files (`Brain/saves/`)

| File | Content |
|---|---|
| `last_sim.pkl` | All N networks + morphologies from the most recent simulation |
| `last_best.pkl` | Best network + morphology from the most recent simulation (overwritten each sim) |
| `best_YYYYMMDD_HHMMSS.pkl` | Timestamped archive of a best network + morphology |

Save files include both the neural network weights and the robot morphology, so a brain can always be loaded back into its correct body.

Load a save manually:
```python
from saver import load_controller
payload = load_controller("last_best")
networks     = payload["networks"]      # list[NeuralNetwork]
morphologies = payload["morphologies"]  # list[RobotMorphology]
context      = payload["context"]       # score, robot_index, …
```

---

## Performance metrics

`DataManager` computes per-robot after each simulation:

| Metric | Description |
|---|---|
| `displacement_xy` | XY distance travelled from start (metres) |
| `avg_speed_xy` | Mean XY speed (m/s) — precise in `Full` mode |
| `is_standing_end` | Whether the robot is still upright at the end |
| `fell_at_time` | Time of first fall in seconds (`Full` mode only) |

Results are printed automatically at the end of each simulation.