"""
tripod_sim.py
=============
Main entry point. Runs the simulation loop.

All configuration lives in sim_config.py.
To experiment, edit that file — this file should rarely change.

Run with: mjpython tripod_sim.py
"""

import time
import contextlib

import mujoco
import mujoco.viewer
import numpy as np

from control import compute_control
from display import DisplayManager
from sim_config import (
    N, ROBOT_SPACING, SIMULATION_DURATION, PHYSICS_XML, ROBOT_CONFIGS,
    VIEWER_ON, VIDEO_RENDERER_ON, SHOW_LIVE_POS_ON
)
from video_render import VideoRecorder

# ---------------------------------------------------------------------------
# Load the physics model (one robot, shared across all simulations)
# ---------------------------------------------------------------------------
physics_model = mujoco.MjModel.from_xml_path(str(PHYSICS_XML))
physics_model.opt.iterations = 100  # default=100 — solver iterations (accuracy vs speed)
physics_model.opt.ls_iterations = 50  # default=50  — line-search iterations
physics_model.opt.timestep = 0.005  # default=0.005 —

print(f"Physics model : nq={physics_model.nq}, nu={physics_model.nu}, "
      f"timestep={physics_model.opt.timestep} s")
print(f"Robots        : {N}  (spacing={ROBOT_SPACING} m)")
print(f"Viewer        : {'ON' if VIEWER_ON else 'OFF (full speed)'}")
print(f"Video render  : {'ON' if VIDEO_RENDERER_ON else 'OFF'}\n")

for i, cfg in enumerate(ROBOT_CONFIGS[:N]):
    print(f"  R{i}: freq={cfg['frequency']} Hz  "
          f"amp={np.degrees(cfg['amplitude']):.0f}°  "
          f"static={np.degrees(cfg['static'][0]):.0f}°")
print()

# ---------------------------------------------------------------------------
# One independent physics state per robot
# ---------------------------------------------------------------------------
robot_states = [mujoco.MjData(physics_model) for _ in range(N)]

# ---------------------------------------------------------------------------
# Optional: display and recorder (only created when their flag is ON)
# ---------------------------------------------------------------------------
display  = DisplayManager(n=N, physics_model=physics_model, robot_states=robot_states) if VIEWER_ON else None
recorder = VideoRecorder(n=N, physics_model=physics_model) if VIDEO_RENDERER_ON else None

# ---------------------------------------------------------------------------
# Simulation loop — runs with or without viewer using a null context manager
# when VIEWER_ON is False, so the loop body stays identical in both cases.
# ---------------------------------------------------------------------------
TOTAL_STEPS = int(SIMULATION_DURATION / physics_model.opt.timestep)
center_x    = (N - 1) * ROBOT_SPACING / 2

viewer_context = (
    mujoco.viewer.launch_passive(display.model, display.state)
    if VIEWER_ON
    else contextlib.nullcontext()
)

Time_start = time.time()
print("Simulation started... ")

with viewer_context as viewer:

    if VIEWER_ON:
        viewer.cam.azimuth   = 20
        viewer.cam.elevation = -20
        viewer.cam.distance  = N * ROBOT_SPACING * 1.4
        viewer.cam.lookat[:] = [center_x, 0.0, 0.4]
        print("Viewer open. Close the window to stop.\n")

    for step in range(TOTAL_STEPS):
        current_time = step * physics_model.opt.timestep

        # Step each robot independently
        for robot_index, state in enumerate(robot_states):
            state.ctrl[:] = compute_control(robot_index, current_time, state.qpos)
            mujoco.mj_step(physics_model, state)

        # Viewer update (skipped when OFF)
        if VIEWER_ON:
            display.sync_from_physics()
            viewer.sync()
            if not viewer.is_running():
                break
            time.sleep(physics_model.opt.timestep)   # real-time pacing

        # Video capture (skipped when OFF)
        if VIDEO_RENDERER_ON:
            recorder.capture(step, robot_states)

        if SHOW_LIVE_POS_ON:
            if step % int(1.0 / physics_model.opt.timestep) == 0:
                print(f"  t={current_time:.1f}s", end="")
                for i, state in enumerate(robot_states):
                    c = state.ctrl
                    print(f"  | R{i}:[{c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}]", end="")
                print()
        else:
            BAR_WIDTH = 40
            filled    = int(BAR_WIDTH * (step + 1) / TOTAL_STEPS)
            bar       = "█" * filled + "░" * (BAR_WIDTH - filled)
            pct       = 100.0 * step / TOTAL_STEPS
            print(f"\r  [{bar}] {pct:5.1f}%  t={current_time:.1f}/{SIMULATION_DURATION:.1f}s", end="", flush=True)


if not SHOW_LIVE_POS_ON:
    print()   # newline after the progress bar

if VIDEO_RENDERER_ON:
    recorder.close()

print(f"\nSimulation finished. (t= {time.time() - Time_start}s)")
