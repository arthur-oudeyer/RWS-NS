"""
MuJoCo Python — Simple Robot Simulation Demo
==============================================
Simulates a 2-link planar robot arm (2 rotational joints).
Demonstrates the core loop used in MAP-Elites:
  1. Load model
  2. Set control inputs (genome / policy)
  3. Run simulation for N steps
  4. Read final state as behavioral descriptor
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Define the robot in MJCF (MuJoCo XML)
# ---------------------------------------------------------------------------
# A simple 2-joint planar arm attached to the world.
XML = """
<mujoco model="2link_arm">
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <worldbody>
    <!-- Base (fixed) -->
    <body name="base" pos="0 0 1">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
      <geom name="link1" type="capsule" fromto="0 0 0  0.4 0 0" size="0.04" rgba="0.3 0.6 0.9 1"/>

      <!-- Forearm -->
      <body name="forearm" pos="0.4 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-150 150"/>
        <geom name="link2" type="capsule" fromto="0 0 0  0.3 0 0" size="0.03" rgba="0.9 0.5 0.2 1"/>

        <!-- End-effector site (used to read tip position) -->
        <site name="tip" pos="0.3 0 0" size="0.02" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>

  <!-- Actuators on each joint -->
  <actuator>
    <motor name="act1" joint="joint1" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="act2" joint="joint2" gear="30" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# ---------------------------------------------------------------------------
# 2. Load model & create data
# ---------------------------------------------------------------------------
model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)

print(f"Model loaded: {model.nq} DOF, {model.nu} actuators, timestep={model.opt.timestep}s")

# ---------------------------------------------------------------------------
# 3. Helper: run one episode with a fixed control vector
# ---------------------------------------------------------------------------
def run_episode(ctrl: np.ndarray, n_steps: int = 300) -> dict:
    """
    Reset the simulation, apply constant torques, step N times.
    Returns a dict with trajectory and final behavioral descriptor.

    In MAP-Elites, ctrl would be your genome; the returned descriptor
    maps into the behavior space grid.
    """
    mujoco.mj_resetData(model, data)   # <-- key for MAP-Elites rollouts

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")

    positions = []   # (x, y) of end-effector over time

    for _ in range(n_steps):
        data.ctrl[:] = ctrl            # apply control
        mujoco.mj_step(model, data)    # advance physics

        # Read tip position (world frame)
        tip_pos = data.site_xpos[tip_id].copy()
        positions.append(tip_pos[:2])  # x, y only (planar arm)

    positions = np.array(positions)

    return {
        "positions":         positions,
        "final_pos":         positions[-1],
        "joint_angles_final": data.qpos.copy(),
        # Behavioral descriptor for MAP-Elites: (final_x, final_y)
        "descriptor":        positions[-1],
    }

# ---------------------------------------------------------------------------
# 4. Run a few episodes with different controls
# ---------------------------------------------------------------------------
controls = [
    np.array([ 0.3,  0.3]),   # push both joints "forward"
    np.array([-0.5,  0.3]),   # pull joint1, push joint2
    np.array([ 0.2, -0.8]),   # mostly bend elbow back
    np.array([ 0.0,  0.0]),   # no torque — arm falls under gravity
]

results = []
for ctrl in controls:
    res = run_episode(ctrl, n_steps=300)
    results.append(res)
    print(f"ctrl={ctrl}  →  tip final pos: ({res['final_pos'][0]:.3f}, {res['final_pos'][1]:.3f})")

# ---------------------------------------------------------------------------
# 5. Visualise trajectories (end-effector paths)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -- Left: trajectories
ax = axes[0]
colors = plt.cm.tab10(np.linspace(0, 0.4, len(results)))
for i, (res, ctrl) in enumerate(zip(results, controls)):
    traj = res["positions"]
    ax.plot(traj[:, 0], traj[:, 1], color=colors[i], lw=1.5,
            label=f"ctrl=[{ctrl[0]:.1f},{ctrl[1]:.1f}]")
    ax.scatter(*traj[0],  color=colors[i], marker="o", s=40)   # start
    ax.scatter(*traj[-1], color=colors[i], marker="*", s=120)  # end

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("End-effector trajectories")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# -- Right: behavioral descriptor space (MAP-Elites view)
ax2 = axes[1]
for i, (res, ctrl) in enumerate(zip(results, controls)):
    ax2.scatter(*res["descriptor"], color=colors[i], s=200, marker="*",
                label=f"ctrl=[{ctrl[0]:.1f},{ctrl[1]:.1f}]", zorder=3)

ax2.set_xlabel("Final X (m)")
ax2.set_ylabel("Final Y (m)")
ax2.set_title("Behavioral descriptors\n(MAP-Elites grid would overlay this)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("demo_output.png", dpi=150)
plt.show()
print("\nPlot saved to demo_output.png")

# ---------------------------------------------------------------------------
# 6. Quick introspection helpers (useful when building MAP-Elites)
# ---------------------------------------------------------------------------
print("\n--- Model introspection ---")
print(f"Joint names : {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}")
print(f"Actuator names: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]}")
print(f"qpos shape  : {data.qpos.shape}  (joint positions)")
print(f"qvel shape  : {data.qvel.shape}  (joint velocities)")
print(f"ctrl shape  : {data.ctrl.shape}  (actuator commands)")