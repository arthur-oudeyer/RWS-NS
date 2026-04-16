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

from sim_config import ROBOT_CONTROL, N, CONTROLLER_INIT, MORPHOLOGIES

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Robot'))
from controller import getController
from morphology import resolve_morphologies

# ---------------------------------------------------------------------------
# RobotSensorData — structured view of MjData for one robot.
#
# Passed to compute_control() instead of raw qpos so the controller
# can read named fields rather than magic array indices.
# ---------------------------------------------------------------------------
@dataclass
class RobotSensorData:
    torso_pos:              np.ndarray   # (3,)  world position  [x, y, z]  (metres)
    torso_height:           float        # z only — convenience shortcut for torso_pos[2]
    torso_orientation:      np.ndarray   # (4,)  quaternion [w, x, y, z]
    torso_velocity:         np.ndarray   # (3,)  linear velocity  [vx, vy, vz]  (m/s)
    torso_angular_velocity: np.ndarray   # (3,)  angular velocity [wx, wy, wz]  (rad/s)
    com_pos:                np.ndarray   # (3,)  whole-robot center of mass  [x, y, z]  (metres)
    hip_angles:             np.ndarray   # (N_HIP,)  hip joint angles  (rad)
    hip_velocities:         np.ndarray   # (N_HIP,)  hip angular velocities  (rad/s)

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
            f"  torso_ω      : {fmt_vec(self.torso_angular_velocity)} rad/s\n"
            f"  com_pos      : {fmt_vec(self.com_pos)} m\n"
            f"  hip_angles   : {fmt_vec(self.hip_angles)} rad"
            f"  ({fmt_vec(hip_deg)} °)\n"
            f"  hip_velocities: {fmt_vec(self.hip_velocities)} rad/s"
        )

    def getTorsoData(self):
        return np.array((*self.torso_pos, *self.torso_orientation, *self.torso_velocity))

    def getHipsData(self):
        return np.array((*self.hip_angles, *self.hip_velocities))

    def getData(self):
        return np.array((*self.getTorsoData(), *self.getHipsData()))

def read_robot_sensors(state: mujoco.MjData, n_joints: int) -> RobotSensorData:
    """
    Extract useful values from a raw MjData object and return them
    as a named RobotSensorData, ready to pass into compute_control().

    Indices are derived from n_joints so nothing is hardcoded here —
    change the morphology and this function adapts automatically.

    qpos layout:
        [0 : 3]                freejoint position  (x, y, z)
        [3 : 7]                freejoint quaternion (w, x, y, z)
        [7 : 7 + n_joints]     hip joint angles

    qvel layout:
        [0 : 3]                linear velocity  (vx, vy, vz)
        [3 : 6]                angular velocity (wx, wy, wz)
        [6 : 6 + n_joints]     hip joint velocities
    """
    _HIP_QPOS = 7           # freejoint always occupies qpos[0:7]
    _HIP_QVEL = 6           # freejoint always occupies qvel[0:6]

    return RobotSensorData(
        torso_pos              = state.qpos[0:3].copy(),
        torso_height           = float(state.qpos[2]),
        torso_orientation      = state.qpos[3:7].copy(),
        torso_velocity         = state.qvel[0:3].copy(),
        torso_angular_velocity = state.qvel[3:6].copy(),
        com_pos                = state.subtree_com[1].copy(),
        hip_angles             = state.qpos[_HIP_QPOS : _HIP_QPOS + n_joints].copy(),
        hip_velocities         = state.qvel[_HIP_QVEL : _HIP_QVEL + n_joints].copy(),
    )

if ROBOT_CONTROL == "pre-configured":
    from robot_config import ROBOT_CONFIGS
else:
    ROBOT_CONFIGS = None

def compute_control(robot_index: int, current_time: float, sensors: RobotSensorData, n_joints: int) -> np.ndarray:
    """
    Returns the target joint angle (radians) for each servo.

    Parameters
    ----------
    robot_index  : which robot / genome
    current_time : simulation time in seconds
    sensors      : structured sensor readings from read_robot_sensors()
                   (use sensors.hip_angles, sensors.torso_pos, etc.)
    n_joints     : number of joints for this robot's morphology
    """
    if ROBOT_CONTROL == "pre-configured":
        target_angles = pre_configured_control(robot_index, current_time)
    elif ROBOT_CONTROL == "external":
        target_angles = external_control(robot_index, current_time, n_joints, sensors)
    else:
        raise Exception("Error : ROBOT_CONTROL not recognized.")

    if target_angles.size != n_joints:
        raise Exception(f"Error : Controller did not gave the right number of angle targets. ({target_angles.size} != {n_joints} needed)")

    return target_angles


def pre_configured_control(robot_index: int, current_time: float) -> np.ndarray:
    cfg = ROBOT_CONFIGS[robot_index]

    target_angles = cfg["static"] + cfg["amplitude"] * np.sin(
        2 * np.pi * cfg["frequency"] * current_time + cfg["phase_legs"]
    )
    return target_angles

def default_external_control(robot_index: int, current_time: float, n_joints: int, sensors: RobotSensorData) -> np.ndarray:
    return np.zeros(n_joints)

external_control = default_external_control

def init_robots_controllers(robot_morphologies):
    global external_control
    external_control = getController(N, CONTROLLER_INIT, robot_morphologies)
    if external_control is None:
        external_control = default_external_control
