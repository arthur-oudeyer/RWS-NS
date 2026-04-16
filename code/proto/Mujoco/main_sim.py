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

import mujoco.viewer
import numpy as np
from colorama import Fore, Style

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Robot'))
from simple_brain import controllers as nn_controllers
from morphology import MorphologyManager, resolve_morphologies

from control import compute_control, read_robot_sensors, init_robots_controllers
from data import DataManager
from display import DisplayManager
from sim_config import *
from robot_config import ROBOT_CONFIGS
from video_render import VideoRecorder

print(Fore.LIGHTBLUE_EX + "\nMujoco Simulation Started\n" + Style.RESET_ALL)

print(Fore.LIGHTGREEN_EX + f"Robots        : {N}  (spacing={ROBOT_SPACING} m)")
print(f"Viewer        : {'ON' if VIEWER_ON else 'OFF (full speed)'}")
print(f"Video render  : {'ON' if VIDEO_RENDERER_ON else 'OFF'}\n" + Style.RESET_ALL)

# ---------------------------------------------------------------------------
# Load the physics model (one robot, shared across all simulations)
# ---------------------------------------------------------------------------
print(Fore.YELLOW + f"Building robots models ({N}, {CONTROLLER_INIT})...")
morph_manager      = MorphologyManager(env_xml_path=str(PHYSICS_XML), floor_texrepeat=FLOOR_TEXREPEAT)
robot_morphologies = resolve_morphologies(N, CONTROLLER_INIT, MORPHOLOGIES)
for m in robot_morphologies:
    m.torso_rgba = (0.9, 0.9, 0.9, 1.0)
robot_models       = [morph_manager.get_model(m) for m in robot_morphologies]
for model in robot_models:
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    model.opt.timestep = 0.005
    model.vis.global_.offwidth  = RENDER_WIDTH
    model.vis.global_.offheight = RENDER_HEIGHT

print(f"\nSetup Controllers ({N}, {CONTROLLER_INIT})..")
init_robots_controllers(robot_morphologies)

print(f"\nPhysics model (first) : nq={robot_models[0].nq}, nu={robot_models[0].nu}, "
      f"timestep={robot_models[0].opt.timestep} s\n")

if ROBOT_CONTROL == "pre-configured":
    for i, cfg in enumerate(ROBOT_CONFIGS[:N]):
        print(f"  R{i}: freq={cfg['frequency']} Hz  "
              f"amp={np.degrees(cfg['amplitude']):.0f}°  "
              f"static={np.degrees(cfg['static'][0]):.0f}°")
    print()
else:
    print("Externally controlled robot.")

# ---------------------------------------------------------------------------
# One independent physics state per robot
# ---------------------------------------------------------------------------
robot_states = [mujoco.MjData(robot_models[i]) for i in range(N)]
data_manager = DataManager(N, mode=DATA_MODE, controllers=nn_controllers, save_best=(SAVE_BEST, UNIQUE_SAVE_BEST, CLEAR_ARCHIVE), morphologies=robot_morphologies)

# ---------------------------------------------------------------------------
# Optional: display and recorder (only created when their flag is ON)
# ---------------------------------------------------------------------------
display = DisplayManager(
    n                 = N,
    robot_morphologies= robot_morphologies,
    robot_states      = robot_states,
    morph_manager     = morph_manager,
) if VIEWER_ON else None
recorder = VideoRecorder(n=N, physics_models=robot_models) if VIDEO_RENDERER_ON else None

# ---------------------------------------------------------------------------
# Simulation loop — runs with or without viewer using a null context manager
# when VIEWER_ON is False, so the loop body stays identical in both cases.
# ---------------------------------------------------------------------------
TOTAL_STEPS = int(SIMULATION_DURATION / robot_models[0].opt.timestep)

viewer_context = (
    mujoco.viewer.launch_passive(display.model, display.state)
    if VIEWER_ON
    else contextlib.nullcontext()
)

Time_start = time.time()
print(Fore.LIGHTRED_EX + "\nSimulation started... " + Style.RESET_ALL)

with viewer_context as viewer:

    if VIEWER_ON:
        center_x = (display.cols - 1) * ROBOT_SPACING / 2
        center_y = (display.rows - 1) * ROBOT_SPACING / 2
        viewer.cam.azimuth   = 20
        viewer.cam.elevation = -20
        viewer.cam.distance  = max(display.cols, display.rows) * ROBOT_SPACING * 1.8
        viewer.cam.lookat[:] = [center_x, center_y, 0.4]
        print("Viewer open. Close the window to stop.\n")

    for step in range(TOTAL_STEPS):
        current_time = step * robot_models[0].opt.timestep

        # Step each robot independently
        for robot_index, state in enumerate(robot_states):
            n_joints = robot_morphologies[robot_index].n_joints
            sensors  = read_robot_sensors(state, n_joints)
            data_manager.record(current_time, robot_index, sensors)
            state.ctrl[:] = compute_control(robot_index, current_time, sensors, n_joints)
            mujoco.mj_step(robot_models[robot_index], state)

        # Viewer update (skipped when OFF)
        if VIEWER_ON:
            display.sync_from_physics()
            viewer.sync()
            if not viewer.is_running():
                break
            time.sleep(robot_models[0].opt.timestep)   # real-time pacing

        # Video capture (skipped when OFF)
        if VIDEO_RENDERER_ON:
            recorder.capture(step, robot_states)

        if SHOW_LIVE_POS_ON:
            if step % int(1.0 / robot_models[0].opt.timestep) == 0:
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
            print(f"\r  [{bar}] {pct:5.1f}%  t_sim={current_time:.1f}/{SIMULATION_DURATION:.1f}s", end="", flush=True)


if not SHOW_LIVE_POS_ON:
    print()   # newline after the progress bar

if VIDEO_RENDERER_ON:
    recorder.close()

print(Fore.LIGHTRED_EX + f"\nSimulation finished. (t= {time.time() - Time_start}s)" + Style.RESET_ALL)
print(Fore.LIGHTWHITE_EX, end="")
data_manager.print_summary()
print(Style.RESET_ALL, end="")
