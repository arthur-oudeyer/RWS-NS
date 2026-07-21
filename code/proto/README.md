# proto — Proof of Concept

Part 1 of [RWS-NS-Proto](../../README.md). The original prototype: evolve
locomotion on simulated legged robots using **MuJoCo** physics and a custom
neural-network "brain", plus a set of **standalone VLM scoring scripts** (Gemini,
CLIP) that score a robot's video or image. Runs on a laptop (CPU).

This part is exploratory and script-driven (edit a config, run a script). The
cleaner, experiment-managed studies live in [`Morphology/`](../Morphology/README.md),
[`Controller/`](../Controller/README.md) and
[`Controller_MJX/`](../Controller_MJX/README.md).

---

## Structure

```
code/proto/
├── requirements.txt          # dependencies for this part
├── Mujoco/                   # Simulation layer
│   ├── main_sim.py           # Entry point — run this
│   ├── sim_config.py         # All simulation configuration (edit this)
│   ├── robot_config.py       # Derived robot properties + pre-set gaits
│   ├── control.py            # Sensor reading + controller dispatch
│   ├── display.py            # Multi-robot viewer (grid layout)
│   ├── data.py               # DataManager — records sensors, metrics, saves best
│   ├── video_render.py       # Optional per-robot video export (.mp4)
│   ├── quadripod_robot.xml   # MuJoCo scene (also tripod_robot.xml)
│   └── demo/                 # Standalone demo scripts
│
├── Robot/                    # Robot morphology + brain
│   ├── morphology.py         # RobotMorphology, presets (QUADRIPOD/TRIPOD/HEXAPOD)
│   ├── controller.py         # Selects/initialises the active controller
│   ├── simple_brain.py       # Neural-network controller + Mutate()
│   ├── saver.py              # Save / load networks + morphologies to Robot/saves/
│   └── simplebrain_loc/      # Low-level NN + math/mutation helpers
│
├── VLM/                      # Standalone VLM scoring scripts
│   ├── gemini_flash.py       # Score a robot VIDEO with Gemini (locomotion quality)
│   ├── gemini_batch.py       # Batch-score many morphology IMAGES in one Gemini call
│   ├── CLIP.py               # Score a robot IMAGE with CLIP (morphology similarity)
│   ├── Gemma.py              # Gemma via OpenRouter (experimental)
│   ├── count_tokens.py       # Token-counter utility
│   ├── qwen_7b.py            # Qwen 7B via local Ollama (experimental)
│   └── img/ , video/         # Sample inputs for the scripts
│
└── Selection/                # Archive + selection helpers
    ├── archive_explorer.py   # Browse saved runs / plot the MAP-Elites archive
    └── selector.py           # Individual selector utilities
```

> **API keys** are read from `code/api_keys.py` (one level above `proto/`, shared
> by the whole repo — see the [main README](../../README.md)). The VLM scripts add
> `code/` to the import path themselves, so run them from inside `proto/VLM/`.

---

## Setup

```bash
cd code/proto
pip install -r requirements.txt
```

The MuJoCo viewer needs **`mjpython`** on macOS (ships with the `mujoco` pip
package); on Linux plain `python` works. The VLM scripts additionally need
`code/api_keys.py` with your `APIKEY_GEMINI` (and `APIKEY_OPENROUTER` only for
`Gemma.py`).

---

## How to run the simulation

```bash
cd code/proto/Mujoco
mjpython main_sim.py       # macOS (viewer)
# python main_sim.py       # Linux
```

All behaviour is controlled by **`Mujoco/sim_config.py`** — the single file you
edit between experiments. `main_sim.py` itself should rarely change.

---

## Configuration — edit `Mujoco/sim_config.py`

### Feature toggles
| Parameter | Description |
|---|---|
| `VIEWER_ON` | Open the interactive MuJoCo viewer (off = run at full speed). |
| `VIDEO_RENDERER_ON` | Export one `.mp4` per robot to `Mujoco/render/`. |
| `SHOW_LIVE_POS_ON` | Print joint positions to the terminal each second. |
| `DATA_MODE` | `"Full"` (record every step) or `"StartStop"` (first + last only). |
| `SAVE_BEST` | Save the best robot to `Robot/saves/last_best.pkl` after each sim. |
| `UNIQUE_SAVE_BEST` | Also save a timestamped `best_YYYYMMDD_HHMMSS.pkl`. |

### Simulation
| Parameter | Description |
|---|---|
| `N` | Number of robots simulated in parallel. |
| `SIMULATION_DURATION` | Duration in seconds. |
| `ROBOT_SPACING` | Distance between robots in the viewer (metres). |
| `ROBOT_CONTROL` | `"external"` (neural net) or `"pre-configured"` (fixed sine gait). |

### Morphology
```python
MORPHOLOGIES = QUADRIPOD                       # 4 legs
MORPHOLOGIES = TRIPOD                          # 3 legs
MORPHOLOGIES = HEXAPOD                         # 6 legs
MORPHOLOGIES = [TRIPOD] * 40 + [QUADRIPOD] * 40  # mixed population (padded to N)
MORPHOLOGIES = None                            # random morphologies
```
Presets come from `Robot/morphology.py`. `MAX_LEGS` caps procedural bodies.

### Controller initialisation
`CONTROLLER_INIT` decides where each robot's brain comes from:
```python
CONTROLLER_INIT = None          # all robots: fresh random weights
CONTROLLER_INIT = "last_best"   # all robots: the best robot from the previous sim
CONTROLLER_INIT = "last_sim"    # each robot reloads its own weights from last sim
CONTROLLER_INIT = {"source": "last_sim", "indices": [0, 4, 5]}   # only these reload, rest fresh
CONTROLLER_INIT = {"source": "best_20260323_171407", "indices": "all"}  # from a named save
CONTROLLER_INIT = {"source": "last_best", "indices": "mutation",         # mutate the best
                   "amplitude": 0.3, "variation": 0.3, "morph_mod": 0.2}
```
With `"indices": "mutation"`, robot 0 is a copy of the source and robots 1..N-1
are mutated clones. Adding morphology keys (`morph_*`) also mutates their bodies;
if the joint count changes, that robot gets a fresh brain.

### Paths & visuals
`PHYSICS_XML` (which robot XML to load), `RENDER_DIR`, `FLOOR_TEXREPEAT`,
`VIDEO_FPS`, `RENDER_WIDTH/HEIGHT`.

---

## Saved robots (`Robot/saves/`)

| File | Content |
|---|---|
| `last_sim.pkl` | All N networks + morphologies from the most recent simulation. |
| `last_best.pkl` | Best network + morphology (overwritten each sim). |
| `best_YYYYMMDD_HHMMSS.pkl` | Timestamped archive of a specific best. |

Performance metrics computed per robot (`data.py`): `displacement_xy`,
`avg_speed_xy`, `is_standing_end`, `fell_at_time`.

---

## VLM scoring scripts (`VLM/`)

Run from inside `proto/VLM/`. Each reads the API key from `code/api_keys.py`.

| Command | What it does |
|---|---|
| `python gemini_flash.py path/to/robot.mp4` | Score a locomotion **video** with Gemini. |
| `python gemini_batch.py img/batch/ 5` | Batch-score many morphology **images** in one Gemini call (batch size 5). |
| `python CLIP.py path/to/image.png` | Score a robot **image** against text prompts with CLIP. |
| `python Gemma.py` | Score via an OpenRouter-hosted Gemma model (needs `APIKEY_OPENROUTER`). |
| `python qwen_7b.py` | Score via a local Qwen model through Ollama. |
| `python count_tokens.py` | Count tokens for a Gemini request. |

`Gemma.py` and `qwen_7b.py` are experimental and need extra services (OpenRouter /
a running Ollama server) — see the optional lines in `requirements.txt`.

---

## Selection helpers (`Selection/`)

```bash
cd code/proto/Selection
python archive_explorer.py            # plot the last_best archive
python archive_explorer.py my_save    # a named save
python archive_explorer.py --list     # list available saves
```
