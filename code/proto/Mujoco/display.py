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
import os
import sys
import xml.etree.ElementTree as ET
from typing import List

import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Brain'))
from morphology import MorphologyManager, RobotMorphology

from sim_config import PHYSICS_XML, ROBOT_SPACING


def _grid_dims(n: int) -> tuple[int, int]:
    """Return (cols, rows) for the tightest square grid fitting n robots."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def _build_display_xml(
    morphologies: list[RobotMorphology],
    spacing:      float,
    morph_manager: MorphologyManager,
    env_xml_path: str,
) -> str:
    n    = len(morphologies)
    cols, _ = _grid_dims(n)

    # Environment (floor, lights, visual, asset) from base XML
    env_tree = ET.parse(env_xml_path)
    env_root = env_tree.getroot()

    display = ET.Element("mujoco", model="display")
    for tag in ("option", "visual", "asset"):
        elem = env_root.find(tag)
        if elem is not None:
            copied = copy.deepcopy(elem)
            if tag == "asset":
                morph_manager._apply_texrepeat(copied)
            display.append(copied)

    worldbody = ET.SubElement(display, "worldbody")
    src_worldbody = env_root.find("worldbody")
    for child in src_worldbody:
        if child.tag != "body":
            worldbody.append(copy.deepcopy(child))

    for i, morph in enumerate(morphologies):
        col = i % cols
        row = i // cols
        body_elem = morph_manager.generate_body_element(
            morph   = morph,
            prefix  = f"r{i}_",
            col     = col,
            row     = row,
            spacing = spacing,
        )
        worldbody.append(body_elem)

    return ET.tostring(display, encoding="unicode")


class DisplayManager:
    """
    Holds the display MjModel/MjData and handles state synchronisation.

    Parameters
    ----------
    n                  : number of robots
    robot_morphologies : list of RobotMorphology, one per robot
    robot_states       : list of MjData, one per robot
    morph_manager      : MorphologyManager used to build the display XML
    """

    def __init__(
        self,
        n:                  int,
        robot_morphologies: list[RobotMorphology],
        robot_states:       List[mujoco.MjData],
        morph_manager:      MorphologyManager,
    ):
        self.n                  = n
        self.robot_morphologies = robot_morphologies
        self.robot_states       = robot_states
        self.spacing            = ROBOT_SPACING
        self.cols, self.rows    = _grid_dims(n)
        self.nq_per_robot       = [m.n_qpos for m in robot_morphologies]

        xml        = _build_display_xml(robot_morphologies, ROBOT_SPACING, morph_manager, str(PHYSICS_XML))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.state = mujoco.MjData(self.model)

        print(f"Display Manager initialized (spacing={self.spacing})")

    def sync_from_physics(self):
        offset = 0
        for i, (state, nq) in enumerate(zip(self.robot_states, self.nq_per_robot)):
            self.state.qpos[offset : offset + nq] = state.qpos
            self.state.qpos[offset]     += (i % self.cols) * self.spacing   # X grid offset
            self.state.qpos[offset + 1] += (i // self.cols) * self.spacing  # Y grid offset
            offset += nq

        mujoco.mj_forward(self.model, self.state)
