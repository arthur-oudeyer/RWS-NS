"""
display.py
==========
Manages the display model shown in the MuJoCo viewer.

Because MuJoCo only allows one viewer window, we build a single "display"
model that contains N robots placed side-by-side. The physics simulation
runs independently in separate MjData objects; this module copies their
state into the display model every frame so the viewer shows all robots.

Usage
-----
    dm = DisplayManager(n=4, physics_model=..., robot_states=...)

    with mujoco.viewer.launch_passive(dm.model, dm.state) as viewer:
        # each step:
        dm.sync_from_physics()
        viewer.sync()
"""

import copy
import math
import xml.etree.ElementTree as ET
from typing import List

import mujoco
import numpy as np

from sim_config import PHYSICS_XML, ROBOT_SPACING


def _grid_dims(n: int) -> tuple[int, int]:
    """Return (cols, rows) for the tightest square grid fitting n robots."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def _build_display_xml(n: int, spacing: float) -> str:
    """
    Parses the physics XML and returns an MJCF string containing N robots
    arranged in a square grid (cols × rows).

    All named elements are prefixed with r{i}_ to avoid conflicts.
    No <actuator> block — the display model is never stepped.
    """
    cols, _ = _grid_dims(n)

    tree   = ET.parse(PHYSICS_XML)
    source = tree.getroot()

    display = ET.Element("mujoco", model="tripod_display")

    for tag in ("option", "visual", "asset"):
        elem = source.find(tag)
        if elem is not None:
            display.append(copy.deepcopy(elem))

    src_worldbody  = source.find("worldbody")
    disp_worldbody = ET.SubElement(display, "worldbody")

    for child in src_worldbody:
        if child.tag != "body":
            disp_worldbody.append(copy.deepcopy(child))

    robot_body = src_worldbody.find("body")

    for i in range(n):
        robot_copy = copy.deepcopy(robot_body)

        for elem in robot_copy.iter():
            if "name" in elem.attrib:
                elem.attrib["name"] = f"r{i}_{elem.attrib['name']}"

        col = i % cols
        row = i // cols
        robot_copy.attrib["pos"] = f"{col * spacing} {row * spacing} 0.5"
        disp_worldbody.append(robot_copy)

    return ET.tostring(display, encoding="unicode")


class DisplayManager:
    """
    Holds the display MjModel/MjData and handles state synchronisation.

    Parameters
    ----------
    n            : number of robots
    physics_model: the single-robot MjModel used for physics
    robot_states : list of MjData, one per robot
    """

    def __init__(self, n: int, physics_model: mujoco.MjModel, robot_states: List[mujoco.MjData]):
        self.n            = n
        self.nq_per_robot = physics_model.nq   # qpos size for one robot
        self.robot_states = robot_states
        self.spacing      = ROBOT_SPACING
        self.cols, self.rows = _grid_dims(n)

        xml          = _build_display_xml(n, ROBOT_SPACING)
        self.model   = mujoco.MjModel.from_xml_string(xml)
        self.state   = mujoco.MjData(self.model)

    def sync_from_physics(self):
        """
        Copy each robot's qpos into the display model, then call mj_forward
        so the viewer reflects the current physics state.

        The freejoint x-position (qpos[0] per robot) is shifted by
        i * spacing so robots appear side-by-side in the viewer.
        Each robot's own physics world is always centred at the origin.
        """
        for i, state in enumerate(self.robot_states):
            start = i * self.nq_per_robot
            self.state.qpos[start : start + self.nq_per_robot] = state.qpos
            self.state.qpos[start]     += (i % self.cols) * self.spacing   # X offset
            self.state.qpos[start + 1] += (i // self.cols) * self.spacing  # Y offset

        mujoco.mj_forward(self.model, self.state)
