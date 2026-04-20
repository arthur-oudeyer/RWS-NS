# RWS-NS-Proto — Proof of Concept

Locomotion evolution on simulated legged robots using MuJoCo physics and a custom neural-network brain, with standalone VLM scoring scripts (Gemini, CLIP) for video and image evaluation.

---

## Project structure

```
code/proto/
├── Mujoco/                   # Simulation layer
│   ├── main_sim.py           # Entry point — run this
│   ├── sim_config.py         # All configuration (edit this for experiments)
│   ├── control.py            # Sensor reading + controller dispatch
│   ├── robot_config.py       # Derived robot properties (joint count from morphology)
│   ├── display.py            # Multi-robot viewer (square grid layout)
│   ├── data.py               # DataManager — records sensors, computes metrics, saves best
│   ├── video_render.py       # Optional per-robot video export (.mp4)
│   └── demo/                 # Standalone demo scripts
│
├── Robot/                    # Robot morphology + brain
│   ├── morphology.py         # RobotMorphology, LegDescriptor, JointDescriptor, presets
│   ├── controller.py         # Selects and initialises the active controller
│   ├── simple_brain.py       # Neural network controller + Mutate()
│   ├── saver.py              # Save / load NeuralNetwork + morphologies to Robot/saves/
│   └── simplebrain_loc/
│       ├── brain.py          # NeuralNetwork, Layer, Neuron classes
│       ├── butils.py         # Activation functions (tanh, sigmoid)
│       ├── bmath.py          # Math helpers, random, normal distribution
│       ├── bgradient.py      # Gradient utilities
│       └── bmutation.py      # Mutation helpers
│
├── VLM/                      # Standalone VLM scoring scripts
│   ├── gemini_flash.py       # Score a robot video with Gemini (locomotion quality)
│   ├── CLIP.py               # Score a robot image with CLIP (morphology similarity)
│   ├── Gemma.py              # Gemma local model scoring (experimental)
│   ├── count_tokens.py       # Token counter utility
│   ├── qwen_7b.py            # Qwen 7B local model (experimental)
│   └── img/ video/           # Sample images and videos for testing
│
├── Selection/                # Archive and selection helpers
│   ├── archive_explorer.py   # Browse saved runs and individuals
│   └── selector.py           # Individual selector utilities
│
└── api_keys.py               # API keys — APIKEY_GEMINI, etc. (not committed)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mujoco` | Physics simulation and viewer |
| `numpy` | Array operations |
| `google-genai` | Gemini VLM scoring |
| `open_clip_torch` | CLIP image scoring |

```bash
pip install mujoco numpy google-genai open_clip_torch
```

The MuJoCo viewer requires `mjpython` on macOS:
```bash
mjpython Mujoco/main_sim.py
```
On Linux, `python main_sim.py` works directly.

---

## How to run

```bash
cd code/proto/Mujoco
mjpython main_sim.py
```

---

## How to configure

**`Mujoco/sim_config.py`** is the single file to edit between experiments.

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
| `SAVE_BEST` | Save the best robot to `Robot/saves/last_best.pkl` after each sim |
| `UNIQUE_SAVE_BEST` | Also save a timestamped archive `best_YYYYMMDD_HHMMSS.pkl` |

### Morphology

Robot bodies are generated at runtime from `RobotMorphology` descriptors — no static XML files.

```python
MORPHOLOGIES = QUADRIPOD   # 4 legs at 0/90/180/270°
MORPHOLOGIES = TRIPOD      # 3 legs at 0/120/240°
MORPHOLOGIES = HEXAPOD     # 6 legs at 0/60/120/180/240/300°
MORPHOLOGIES = [QUADRIPOD, TRIPOD, TRIPOD]  # heterogeneous population
```

Presets are imported from `Robot/morphology.py`. Custom morphologies are built from `RobotMorphology`, `LegDescriptor`, and `JointDescriptor` dataclasses.

### Controller initialisation

```python
CONTROLLER_INIT = None
# → all N robots start with fresh random weights + MORPHOLOGIES

CONTROLLER_INIT = "last_best"
# → all N robots are copies of the best robot from the previous simulation
# → morphology is restored from the save file (MORPHOLOGIES ignored)

CONTROLLER_INIT = "last_sim"
# → each robot i reloads its own weights from the previous simulation
# → morphologies are restored from the save file (MORPHOLOGIES ignored)

CONTROLLER_INIT = {"source": "last_sim", "indices": [0, 4, 5, 6, 9]}
# → robots 0,4,5,6,9 reload from last_sim, the rest start fresh

CONTROLLER_INIT = {"source": "best_20260323_171407", "indices": "all"}
# → load from a specific timestamped archive

CONTROLLER_INIT = {
    "source": "last_best",
    "indices": "mutation",
    "amplitude": 0.2,
    "variation": 0.2,
    "morphology": 0.1,   # morphological mutation amplitude
}
# → robot 0 is a copy of last_best; robots 1..N-1 are mutated clones
# → "morphology" key triggers MutateMorphology on each mutated clone
```

---

## Morphological mutation

`MutateMorphology(morph, amplitude)` in `Robot/morphology.py` creates a modified copy:

- **Continuous**: each joint's `length` and `radius` are scaled by `1 + N(0, amplitude)`, clamped to safe bounds.
- **Discrete** (probability ∝ amplitude): randomly add a leg (at the largest angular gap) or remove a random leg (minimum 2 kept).

When `CONTROLLER_INIT` uses `"indices": "mutation"` and a `"morphology"` key is set:
- Robot 0 keeps the saved morphology unchanged.
- Robots 1..N-1 each get an independently mutated body.
- If mutation changes the joint count, the robot gets a fresh randomly-initialised brain.

---

## Neural network controller

Default controller (`Robot/simple_brain.py`): fully-connected network with `tanh` activations.

- **Inputs**: 3 clock signals `sin(ω·t)` + 2 × n_joints sensor values (angles + velocities)
- **Outputs**: n_joints Δangle increments per step
- **Architecture**: configured via `NB_NEURONS_BY_LAYER` in `simple_brain.py`

Weights are Xavier-initialised. `Mutate()` creates a perturbed copy for evolutionary search.

### Save files (`Robot/saves/`)

| File | Content |
|---|---|
| `last_sim.pkl` | All N networks + morphologies from the most recent simulation |
| `last_best.pkl` | Best network + morphology (overwritten each sim) |
| `best_YYYYMMDD_HHMMSS.pkl` | Timestamped archive of a specific best |

Load manually:
```python
from saver import load_controller
payload = load_controller("last_best")
networks     = payload["networks"]      # list[NeuralNetwork]
morphologies = payload["morphologies"]  # list[RobotMorphology]
context      = payload["context"]       # score, robot_index, …
```

---

## VLM scoring scripts

### `VLM/gemini_flash.py` — video scoring

Uploads a `.mp4` robot locomotion video to the Gemini API and returns structured scores:
locomotion quality, fall detection, gait regularity, etc.

```bash
python VLM/gemini_flash.py path/to/robot.mp4
```

### `VLM/CLIP.py` — image scoring

Loads a CLIP model and scores a robot image against a set of text prompts.

```bash
python VLM/CLIP.py path/to/image.png
```

API keys are read from `api_keys.py` (not committed):
```python
APIKEY_GEMINI = "your-key-here"
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
