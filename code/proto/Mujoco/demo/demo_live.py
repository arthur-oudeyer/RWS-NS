"""
MuJoCo — Live Viewer Demo
==========================
Runs the 2-link arm simulation in real time with an interactive 3D window.

Controls in the viewer window:
  Left-drag   : rotate camera
  Right-drag  : pan
  Scroll      : zoom
  Ctrl+A      : toggle wireframe
  Space       : (not used here, but you can hook it)
  Esc / close : quit
"""

import time
import mujoco
import mujoco.viewer
import numpy as np

# ---------------------------------------------------------------------------
# Same MJCF model as demo.py
# ---------------------------------------------------------------------------
XML = """
<mujoco model="2link_arm">
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="150" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="groundplane"/>

    <!-- Shoulder joint attached to a fixed pillar -->
    <body name="pillar" pos="0 0 0">
      <geom type="cylinder" fromto="0 0 0  0 0 1" size="0.05" rgba="0.4 0.4 0.4 1"/>

      <body name="upper_arm" pos="0 0 1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
        <geom name="link1" type="capsule" fromto="0 0 0  0.4 0 0" size="0.04" rgba="0.3 0.6 0.9 1"/>

        <body name="forearm" pos="0.4 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-150 150"/>
          <geom name="link2" type="capsule" fromto="0 0 0  0.3 0 0" size="0.03" rgba="0.9 0.5 0.2 1"/>
          <geom name="hand"  type="sphere"  pos="0.3 0 0" size="0.04" rgba="1.0 0.8 0.2 1"/>

          <site name="tip" pos="0.3 0 0" size="0.02" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="act1" joint="joint1" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="act2" joint="joint2" gear="30" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)

# ---------------------------------------------------------------------------
# Control sequences to play back (cycles through them automatically)
# ---------------------------------------------------------------------------
EPISODES = [
    {"ctrl": np.array([ 0.6,  0.4]), "label": "forward sweep",    "steps": 400},
    {"ctrl": np.array([-0.6,  0.4]), "label": "reverse shoulder",  "steps": 400},
    {"ctrl": np.array([ 0.3, -0.9]), "label": "elbow fold",        "steps": 400},
    {"ctrl": np.array([ 0.0,  0.0]), "label": "free fall",         "steps": 300},
]

# ---------------------------------------------------------------------------
# Launch the passive viewer and run the simulation loop
# ---------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:

    # Optional: adjust initial camera
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -25
    viewer.cam.distance  = 2.5
    viewer.cam.lookat[:] = [0.2, 0, 1.0]

    episode_idx = 0
    step_count  = 0

    print("Viewer open. Close the window to stop.\n")

    while viewer.is_running():
        ep = EPISODES[episode_idx % len(EPISODES)]

        # Reset at the start of each episode
        if step_count == 0:
            mujoco.mj_resetData(model, data)
            print(f"Episode {episode_idx + 1}: {ep['label']}  ctrl={ep['ctrl']}")

        # Apply control & step
        data.ctrl[:] = ep["ctrl"]
        mujoco.mj_step(model, data)
        step_count += 1

        # Sync viewer with current sim state
        viewer.sync()

        # Real-time pacing: sleep to match wall-clock time
        time.sleep(model.opt.timestep)

        # Advance to next episode when done
        if step_count >= ep["steps"]:
            step_count   = 0
            episode_idx += 1

print("Viewer closed.")