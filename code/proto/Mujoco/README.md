# Tripod Robot Simulation — Full Guide

## How to run

```bash
mjpython main_sim.py
```

---

## File structure

```
Mujoco/
├── main_sim.py          ← entry point — simulation loop (rarely needs editing)
├── sim_config.py        ← feature flags and simulation parameters
├── robot_config.py      ← robot physical description + control configs (genomes)
├── control.py           ← robot brain: sensors, compute_control()
├── display.py           ← builds the multi-robot viewer model
├── video_render.py      ← offscreen rendering + MP4 writing
├── tripod_robot.xml     ← MuJoCo robot definition (physics)
└── render/              ← r0.mp4, r1.mp4, ... saved here
```

**Dependency flow** (no circular imports):
```
sim_config.py / robot_config.py   ← no local imports
        ↑
   control.py
   display.py
   video_render.py
        ↑
   main_sim.py
```

---

## sim_config.py — feature flags & parameters

The first thing to edit for any experiment.

### Feature flags

| Flag | Effect when `True` |
|---|---|
| `VIEWER_ON` | Opens the live 3D viewer. Simulation runs at real-time speed (`time.sleep` is active). |
| `VIDEO_RENDERER_ON` | Records one MP4 per robot to `render/`. |
| `SHOW_LIVE_POS_ON` | Prints servo targets to the terminal every second. |

When `VIEWER_ON = False` and `SHOW_LIVE_POS_ON = False`, a progress bar is shown instead:
```
  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  30.0%  t_sim=3.0/10.0s
```

### Simulation parameters

| Parameter | Description |
|---|---|
| `N` | Number of robots to simulate in parallel |
| `ROBOT_SPACING` | Distance between robots in the viewer (metres) |
| `SIMULATION_DURATION` | Length of each run (seconds) |
| `ROBOT_CONTROL` | `"pre-configured"` or `"external"` — selects the control mode |

### Video parameters

| Parameter | Description |
|---|---|
| `VIDEO_FPS` | Frames per second of saved videos (20 is smooth for gait) |
| `RENDER_WIDTH` | Pixel width of each robot's video |
| `RENDER_HEIGHT` | Pixel height of each robot's video |

---

## robot_config.py — robot description & genomes

### `N_HIP`

```python
N_HIP = 3   # number of rotational hip joints on the robot
```

This is the only place where the robot's joint count is declared.
`read_robot_sensors()` and `compute_control()` use it automatically —
change it here and everything adapts without touching any index.

### `ROBOT_CONFIGS`

One dict per robot. In MAP-Elites terminology, each dict **is a genome**.

```python
{
    "amplitude":  30 / 180 * np.pi,   # peak swing angle (radians)
    "frequency":  1.0,                 # oscillations per second (Hz)
    "static":     np.array([-10, -10, -10]) / 180 * np.pi,  # resting offset per leg
    "phase_legs": np.array([0.0, 2*np.pi/3, 4*np.pi/3]),    # per-leg phase (radians)
}
```

If `N > len(ROBOT_CONFIGS)`, the last entry is repeated automatically with a warning.

---

## control.py — the robot's brain

### `RobotSensorData`

`read_robot_sensors(state)` is called every step before `compute_control()`.
It converts the raw `MjData` into a named, human-readable object:

```python
@dataclass
class RobotSensorData:
    torso_pos         # (3,)      world position [x, y, z]  (metres)
    torso_height      # float     z only — is the robot falling?
    torso_orientation # (4,)      quaternion [w, x, y, z]   — is it tilting?
    torso_velocity    # (3,)      linear velocity [vx, vy, vz]  (m/s)
    hip_angles        # (N_HIP,)  current joint angles  (radians)
    hip_velocities    # (N_HIP,)  current joint speeds  (rad/s)
```

`print(sensors)` gives a formatted output:
```
RobotSensorData
  torso_pos    : [+0.012  +0.003  +0.418] m
  torso_height : +0.418 m
  torso_orient : [w=+1.00  x=+0.00  y=+0.00  z=+0.00]
  torso_vel    : [+0.01  -0.00  +0.02] m/s
  hip_angles   : [-0.120  +0.053  -0.081] rad  ([-6.9°  +3.0°  -4.6°] °)
  hip_velocities: [+0.231  -0.140  +0.312] rad/s
```

`torso_pos[0:2]` (x, y displacement after the episode) is the natural
**MAP-Elites behavioral descriptor**.

### qpos / qvel layout (for reference)

```
qpos (nq = 7 + N_HIP):
  [0:3]           freejoint position  (x, y, z)
  [3:7]           freejoint quaternion (w, x, y, z)
  [7 : 7+N_HIP]   hip joint angles

qvel (nv = 6 + N_HIP):
  [0:3]           linear velocity  (vx, vy, vz)
  [3:6]           angular velocity (wx, wy, wz)
  [6 : 6+N_HIP]   hip joint velocities
```

### Control modes (`ROBOT_CONTROL` in sim_config.py)

#### `"pre-configured"` (default)

Reads genomes from `ROBOT_CONFIGS` in `robot_config.py` and runs a sinusoidal gait:

```python
target_angles = static + amplitude * sin(2π * frequency * t + phase_legs)
```

Each robot gets its own config entry (its genome).

#### `"external"`

Your custom controller. Replace `external_control` in `control.py`:

```python
def my_controller(robot_index, current_time, n_hip, sensors):
    # sensors.hip_angles   — where the joints are now
    # sensors.torso_pos    — where the robot is in the world
    # sensors.torso_height — use to detect a fall
    return np.array([...])   # one target angle per hip

external_control = my_controller
```

Examples of what you can implement here:
- **Feedback controller**: `torque = KP * (target - sensors.hip_angles)`
- **Neural network**: forward pass with `sensors` as input
- **CPG (Central Pattern Generator)**: oscillator network driven by time

---

## display.py — multi-robot viewer

`DisplayManager` builds a combined MuJoCo model at runtime by duplicating
the physics XML `N` times with `r0_`, `r1_`, ... name prefixes.
It is never stepped — only used to copy state into for visualisation.

```
robot_states[0]  ──→  sync_from_physics()  ──→  display.state  ──→  viewer
robot_states[1]  ──┘                             (all robots shown side-by-side)
```

---

## video_render.py — MP4 recording

`VideoRecorder` renders each robot independently to `render/r{i}.mp4`.

**Optimisations active:**
- Low FPS (`VIDEO_FPS`) — fewer `renderer.render()` calls
- Low resolution (`RENDER_WIDTH × RENDER_HEIGHT`) — fewer pixels per frame
- `libx264 ultrafast` preset — fast encoding, slightly larger files
- **Background encoder thread** — `writer.append_data()` runs in a separate
  thread; the sim loop is never blocked by encoding

---

## Feature combinations

| `VIEWER_ON` | `VIDEO_RENDERER_ON` | Use case |
|---|---|---|
| `True` | `False` | Watching / debugging a gait |
| `True` | `True` | Watch + record |
| `False` | `False` | Fastest possible evaluation (MAP-Elites) |
| `False` | `True` | Headless batch run with output videos |

---

## MAP-Elites integration path

```
genome dict  (amplitude, frequency, static, phase_legs)
     ↓
ROBOT_CONFIGS[i]
     ↓
compute_control()  called every timestep
     ↓
mj_step()  ×  TOTAL_STEPS
     ↓
sensors.torso_pos[0:2]  →  behavioral descriptor  (x, y displacement)
     ↓
MAP-Elites grid update
```

To evaluate a new genome: set `ROBOT_CONFIGS[i]` to the genome dict,
call `mujoco.mj_resetData(physics_model, robot_states[i])` to reset the
robot, run the episode, read `robot_states[i].qpos[0:2]` as the descriptor.
