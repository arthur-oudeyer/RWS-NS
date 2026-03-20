"""
control.py
==========
The robot's "brain" — computes servo target angles at each timestep.

compute_control() is called once per robot per simulation step.
It receives the robot index, the current time, and a RobotSensorData object,
and returns a numpy array of target angles (one per servo).

To implement a different controller (e.g. feedback, neural network, CPG),
replace or extend compute_control().
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from sim_config import ROBOT_CONTROL
from robot_config import N_HIP


# ---------------------------------------------------------------------------
# RobotSensorData — structured view of MjData for one robot.
#
# Passed to compute_control() instead of raw qpos so the controller
# can read named fields rather than magic array indices.
# ---------------------------------------------------------------------------
@dataclass
class RobotSensorData:
    torso_pos:         np.ndarray   # (3,)      world position  [x, y, z]  (metres)
    torso_height:      float        # z only  — convenience shortcut for torso_pos[2]
    torso_orientation: np.ndarray   # (4,)      quaternion [w, x, y, z]
    torso_velocity:    np.ndarray   # (3,)      linear velocity  [vx, vy, vz]  (m/s)
    hip_angles:        np.ndarray   # (N_HIP,)  hip joint angles  (rad)
    hip_velocities:    np.ndarray   # (N_HIP,)  hip angular velocities  (rad/s)

    def __str__(self) -> str:
        def fmt_vec(v):
            return "[" + "  ".join(f"{x:+.3f}" for x in v) + "]"

        hip_deg = np.degrees(self.hip_angles)
        return (
            f"RobotSensorData\n"
            f"  torso_pos    : {fmt_vec(self.torso_pos)} m\n"
            f"  torso_height : {self.torso_height:+.3f} m\n"
            f"  torso_orient : [w={self.torso_orientation[0]:+.2f}  "
            f"x={self.torso_orientation[1]:+.2f}  "
            f"y={self.torso_orientation[2]:+.2f}  "
            f"z={self.torso_orientation[3]:+.2f}]\n"
            f"  torso_vel    : {fmt_vec(self.torso_velocity)} m/s\n"
            f"  hip_angles   : {fmt_vec(self.hip_angles)} rad"
            f"  ({fmt_vec(hip_deg)} °)\n"
            f"  hip_velocities: {fmt_vec(self.hip_velocities)} rad/s"
        )

def read_robot_sensors(state: mujoco.MjData) -> RobotSensorData:
    """
    Extract useful values from a raw MjData object and return them
    as a named RobotSensorData, ready to pass into compute_control().

    Indices are derived from N_HIP (set in robot_config.py) so nothing
    is hardcoded here — change N_HIP and this function adapts automatically.

    qpos layout:
        [0 : 3]              freejoint position  (x, y, z)
        [3 : 7]              freejoint quaternion (w, x, y, z)
        [7 : 7 + N_HIP]      hip joint angles

    qvel layout:
        [0 : 3]              linear velocity  (vx, vy, vz)
        [3 : 6]              angular velocity (wx, wy, wz)
        [6 : 6 + N_HIP]      hip joint velocities
    """
    _HIP_QPOS = 7           # freejoint always occupies qpos[0:7]
    _HIP_QVEL = 6           # freejoint always occupies qvel[0:6]

    return RobotSensorData(
        torso_pos         = state.qpos[0:3].copy(),
        torso_height      = float(state.qpos[2]),
        torso_orientation = state.qpos[3:7].copy(),
        torso_velocity    = state.qvel[0:3].copy(),
        hip_angles        = state.qpos[_HIP_QPOS : _HIP_QPOS + N_HIP].copy(),
        hip_velocities    = state.qvel[_HIP_QVEL : _HIP_QVEL + N_HIP].copy(),
    )

if ROBOT_CONTROL == "pre-configured":
    from robot_config import ROBOT_CONFIGS
else:
    ROBOT_CONFIGS = None

def compute_control(robot_index: int, current_time: float, sensors: RobotSensorData) -> np.ndarray:
    """
    Returns the target joint angle (radians) for each servo.

    Parameters
    ----------
    robot_index  : which robot / genome
    current_time : simulation time in seconds
    sensors      : structured sensor readings from read_robot_sensors()
                   (use sensors.hip_angles, sensors.torso_pos, etc.)
    """
    if ROBOT_CONTROL == "pre-configured":
        target_angles = pre_configured_control(robot_index, current_time)
    elif ROBOT_CONTROL == "external":
        target_angles = external_control(robot_index, current_time, N_HIP, sensors)
    else:
        raise Exception("Error : ROBOT_CONTROL not recognized.")

    if target_angles.size != N_HIP:
        raise Exception(f"Error : ROBOT_CONTROL did not gave the right number of angle targets. ({target_angles.size} != {N_HIP} needed)")

    return target_angles


def pre_configured_control(robot_index: int, current_time: float) -> np.ndarray:
    cfg = ROBOT_CONFIGS[robot_index]

    target_angles = cfg["static"] + cfg["amplitude"] * np.sin(
        2 * np.pi * cfg["frequency"] * current_time + cfg["phase_legs"]
    )
    return target_angles

def default_external_control(robot_index: int, current_time: float, n_hip: int, sensors: RobotSensorData) -> np.ndarray:
    return np.array([0.] * N_HIP)

external_control = default_external_control
