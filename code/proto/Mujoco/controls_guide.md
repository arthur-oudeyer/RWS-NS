# Tripod Robot — Controls Guide

## How to run

```bash
mjpython tripod_sim.py
```

---

## What is `data.ctrl`?

`robot_state.ctrl` is a NumPy array with **one value per servo**.

```
robot_state.ctrl = [  leg1_target_angle,  leg2_target_angle,  leg3_target_angle  ]
                      index 0              index 1              index 2
```

Because the actuators are **position servos** (not raw torque motors), each value is a
**target joint angle in radians**. The servo automatically applies:

```
torque = kp × (target_angle − current_angle)
       + kv × (0 − current_velocity)
```

| Value | Meaning |
|-------|---------|
| `+0.6` | Move leg to +0.6 rad (≈ +34°) from its rest position |
| `0.0`  | Hold leg at 0° (straight down) |
| `-0.6` | Move leg to −0.6 rad (≈ −34°) |

The allowed range is `[-0.8, +0.8]` radians (set by `ctrlrange` in `tripod_robot.xml`).

> **Why servos and not raw torque motors?**
> A sinusoidal *torque* integrates to a non-zero average velocity — the leg just
> keeps spinning in one direction instead of oscillating. A sinusoidal *target angle*
> creates true back-and-forth motion because the leg always chases the moving target.

---

## How to change the control over time

Everything happens inside `compute_control(current_time)` in `tripod_sim.py`.
You receive the **current simulation time in seconds**, and you return a NumPy array of 3 torque values.

### Pattern 1 — Constant angle (hold all legs at the same position)

```python
def compute_control(current_time):
    return np.array([0.4, 0.4, 0.4])   # hold at +0.4 rad (≈ 23°) forward
```

All 3 legs move to 23° forward and hold there.

---

### Pattern 2 — Sine wave (legs oscillate in sync)

```python
AMPLITUDE = 0.6   # peak angle in radians (≈ 34°)
FREQUENCY = 1.0   # 1 oscillation per second

def compute_control(current_time):
    angle = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * current_time)
    return np.array([angle, angle, angle])
```

All 3 legs swing back and forth together, ±34°.

---

### Pattern 3 — Phase-offset gait (each leg offset in time)

This is what the default `tripod_sim.py` uses.
Each leg starts its cycle 1/3 of a period later than the previous one.

```python
AMPLITUDE = 0.6
FREQUENCY = 1.0

def compute_control(current_time):
    t = 2 * np.pi * FREQUENCY * current_time
    return np.array([
        AMPLITUDE * np.sin(t + 0.0),            # leg 1: starts immediately
        AMPLITUDE * np.sin(t + 2*np.pi/3),      # leg 2: starts 1/3 cycle later
        AMPLITUDE * np.sin(t + 4*np.pi/3),      # leg 3: starts 2/3 cycle later
    ])
```

This gives the classic tripod gait pattern.

---

### Pattern 4 — Pre-computed sequence (frame by frame)

If you want full manual control over every timestep (useful in MAP-Elites):

```python
# Build a list of control vectors before the simulation starts
TOTAL_STEPS = 1000
control_sequence = []
for step in range(TOTAL_STEPS):
    t = step * robot_model.opt.timestep
    control_sequence.append(np.array([
        0.5 * np.sin(t),
        0.5 * np.sin(t + 1.0),
        -0.3
    ]))

# Then in the sim loop:
def compute_control(current_time):
    step = int(current_time / robot_model.opt.timestep)
    return control_sequence[step]
```

---

### Pattern 5 — Ramp up, hold, ramp down

```python
def compute_control(current_time):
    TOTAL_DURATION = 5.0

    if current_time < 1.0:
        strength = current_time / 1.0        # ramp up over first second
    elif current_time < 4.0:
        strength = 1.0                        # hold full torque
    else:
        strength = (TOTAL_DURATION - current_time) / 1.0  # ramp down

    torque = 0.7 * strength * np.sin(2 * np.pi * 1.5 * current_time)
    return np.array([torque, torque, torque])
```

---

## Key parameters to tweak

| Parameter | Location | Effect |
|-----------|----------|--------|
| `AMPLITUDE` | `tripod_sim.py` | Peak swing angle in radians (0 → 0.8) |
| `FREQUENCY` | `tripod_sim.py` | How fast the legs swing (Hz) |
| `PHASE_LEG1/2/3` | `tripod_sim.py` | Timing offset between legs (radians) |
| `kp` | `tripod_robot.xml` | Servo stiffness: higher = snappier response |
| `kv` | `tripod_robot.xml` | Servo damping: higher = less oscillation overshoot |
| `damping` | `tripod_robot.xml` | Joint resistance (slows oscillations) |
| `range` | `tripod_robot.xml` | Hard joint angle limits (degrees) |
| `SIMULATION_DURATION` | `tripod_sim.py` | Total length of the run |

---

## MAP-Elites connection

In a MAP-Elites context, the **genome** is the `control_sequence` (or the parameters that generate it — amplitude, frequency, phases). The simulation loop is exactly the same; only the values inside `compute_control()` change between evaluations.

```
genome (e.g. [amplitude, frequency, phase1, phase2, phase3])
    ↓
compute_control(t)   ←  parameterised by genome
    ↓
run full simulation (5 s)
    ↓
read foot positions / distance traveled  →  behavioral descriptor
    ↓
MAP-Elites grid update
```
