# RWS-NS-Proto

Prototype for evolving locomotion controllers on a simulated tripod robot using MuJoCo physics and a custom neural network.

---

## Project structure

```
code/proto/
├── Mujoco/                   # Simulation layer
│   ├── main_sim.py           # Entry point — run this
│   ├── sim_config.py         # All configuration (edit this for experiments)
│   ├── control.py            # Sensor reading + controller dispatch
│   ├── robot_config.py       # Robot physical description (N_HIP, pre-configured gaits)
│   ├── display.py            # Multi-robot viewer (grid layout)
│   ├── data.py               # DataManager — records sensors, computes metrics, saves best
│   ├── video_render.py       # Optional per-robot video export
│   ├── tripod_robot.xml      # MuJoCo model — 3-legged robot with one hip joint per leg
│   └── demo/                 # Standalone demo scripts
│
└── Brain/                    # Controller layer
    ├── controller.py         # Selects and initialises the active controller
    ├── simple_brain.py       # Neural network controller + Mutate()
    ├── saver.py              # Save / load NeuralNetwork to Brain/saves/
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

### Controller initialisation

Controls what weights each robot starts with:

```python
CONTROLLER_INIT = None
# → all N robots start with fresh random weights

CONTROLLER_INIT = "last_best"
# → all N robots are copies of the best robot from the previous simulation

CONTROLLER_INIT = "last_sim"
# → each robot i reloads its own weights from the previous simulation

CONTROLLER_INIT = {"source": "last_sim", "indices": [0, 4, 5, 6, 9]}
# → robots 0,4,5,6,9 reload from last_sim, the rest start fresh

CONTROLLER_INIT = {"source": "best_20260323_171407", "indices": "all"}
# → load from a specific archived save
```

---

## Neural network controller

The default controller (`simple_brain.py`) is a fully-connected network with `tanh` activations.

- **Inputs**: 3 clock signals `sin(ω·t)` at different frequencies + 6 hip sensor values (angles + velocities)
- **Outputs**: 3 Δangle increments added to current hip angles each step
- **Architecture**: configured via `NB_NEURONS_BY_LAYER` in `simple_brain.py`

Weights are initialised with Xavier (`N(0, 1/√n_inputs)`). The `Mutate()` function in `simple_brain.py` creates a perturbed copy of a network for evolutionary search.

### Save files (`Brain/saves/`)

| File | Content |
|---|---|
| `last_sim.pkl` | All N networks from the most recent simulation |
| `last_best.pkl` | Best network from the most recent simulation |
| `best_YYYYMMDD_HHMMSS.pkl` | Timestamped archive of a best network |

Load a save manually:
```python
from saver import load_controller
payload = load_controller("last_best")
networks = payload["networks"]   # list[NeuralNetwork]
context  = payload["context"]    # score, robot_index, …
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