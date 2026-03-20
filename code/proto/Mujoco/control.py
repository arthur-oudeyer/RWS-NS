"""
control.py
==========
The robot's "brain" — computes servo target angles at each timestep.

compute_control() is called once per robot per simulation step.
It receives the robot index, the current time, and the full joint state (qpos),
and returns a numpy array of target angles (one per servo).

To implement a different controller (e.g. feedback, neural network, CPG),
replace or extend compute_control().
"""

import numpy as np

from sim_config import ROBOT_CONFIGS


def compute_control(robot_index: int, current_time: float, qpos: np.ndarray) -> np.ndarray:
    """
    Returns the target joint angle (radians) for each servo.

    Parameters
    ----------
    robot_index : which robot / which genome from ROBOT_CONFIGS
    current_time: simulation time in seconds
    qpos        : current joint positions (available for feedback controllers)

    Returns
    -------
    np.ndarray of shape (n_servos,) — target angle per servo in radians
    """
    cfg = ROBOT_CONFIGS[robot_index]

    target_angles = cfg["static"] + cfg["amplitude"] * np.sin(
        2 * np.pi * cfg["frequency"] * current_time + cfg["phase_legs"]
    )
    return target_angles
