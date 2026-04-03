"""
morphology.py
=============
Programmatic robot morphology descriptors and XML generator.

Instead of static XML files, robots are described by RobotMorphology
dataclasses. MorphologyManager generates MJCF XML on the fly, allowing
different robot bodies to coexist in the same simulation.

Pre-defined morphologies: QUADRIPOD, TRIPOD
"""

from __future__ import annotations
import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

from simplebrain_loc.bmath import normal, uniform, rndInt, proba

MAX_INIT_LEGS = 6

# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

@dataclass
class JointDescriptor:
    """One rotational joint on a leg segment."""
    damping:    float              = 4.0
    kp:         float              = 25.0          # position servo stiffness (N·m/rad)
    ctrl_range: tuple[float,float] = (-1.5, 0.8)   # actuator limits (radians)
    length:     float              = 0.25           # segment length below this joint (m)
    radius:     float              = 0.025          # capsule radius (m)
    rgba:       tuple              = (0.5, 0.5, 0.5, 1.0)


@dataclass
class LegDescriptor:
    """One leg attached to the torso."""
    placement_angle_deg: float
    joints: list[JointDescriptor] = field(default_factory=lambda: [JointDescriptor()])


@dataclass
class RobotMorphology:
    """
    Full description of a robot body.
    MorphologyManager uses this to generate MJCF XML on the fly.
    """
    name:          str
    legs:          list[LegDescriptor]
    torso_radius:  float = 0.12                    # cylinder radius (m)
    torso_height:  float = 0.04                    # cylinder half-height (m)
    torso_rgba:    tuple = (0.9, 0.9, 0.9, 1.0)
    spawn_height:  float = 0.5                     # initial Z position (m)
    foot_radius:   float = 0.038
    foot_rgba:     tuple = (0.9, 0.7, 0.2, 1.0)

    # ---- derived properties -----------------------------------------------

    @property
    def n_joints(self) -> int:
        return sum(len(leg.joints) for leg in self.legs)

    @property
    def n_outputs(self) -> int:
        return self.n_joints

    @property
    def n_sensor_inputs(self) -> int:
        """Joint angles + joint velocities."""
        return self.n_joints * 2

    @property
    def n_qpos(self) -> int:
        """freejoint(7) + one per joint."""
        return 7 + self.n_joints

    @property
    def n_qvel(self) -> int:
        """freejoint(6) + one per joint."""
        return 6 + self.n_joints


# ---------------------------------------------------------------------------
# Pre-defined morphologies
# ---------------------------------------------------------------------------

QUADRIPOD = RobotMorphology(
    name  = "quadripod",
    legs  = [
        LegDescriptor(  0.0, [JointDescriptor(rgba=(0.3, 0.6, 0.9, 1.0))]),  # front  (blue)
        LegDescriptor( 90.0, [JointDescriptor(rgba=(0.3, 0.7, 0.4, 1.0))]),  # right  (green)
        LegDescriptor(180.0, [JointDescriptor(rgba=(0.8, 0.3, 0.3, 1.0))]),  # back   (red)
        LegDescriptor(270.0, [JointDescriptor(rgba=(0.8, 0.6, 0.3, 1.0))]),  # left   (orange)
    ],
)

TRIPOD = RobotMorphology(
    name  = "tripod",
    legs  = [
        LegDescriptor(  0.0, [JointDescriptor(rgba=(0.3, 0.6, 0.9, 1.0))]),
        LegDescriptor(120.0, [JointDescriptor(rgba=(0.3, 0.7, 0.4, 1.0))]),
        LegDescriptor(240.0, [JointDescriptor(rgba=(0.8, 0.3, 0.3, 1.0))]),
    ],
)

HEXAPOD = RobotMorphology(
    name  = "hexapod",
    legs  = [
        LegDescriptor( 30.0, [JointDescriptor(rgba=(0.3, 0.6, 0.9, 1.0))]),
        LegDescriptor( 90.0, [JointDescriptor(rgba=(0.5, 0.5, 0.9, 1.0))]),
        LegDescriptor(150.0, [JointDescriptor(rgba=(0.3, 0.7, 0.4, 1.0))]),
        LegDescriptor(210.0, [JointDescriptor(rgba=(0.5, 0.8, 0.4, 1.0))]),
        LegDescriptor(270.0, [JointDescriptor(rgba=(0.8, 0.3, 0.3, 1.0))]),
        LegDescriptor(330.0, [JointDescriptor(rgba=(0.8, 0.6, 0.3, 1.0))]),
    ],
    torso_radius=0.14,
)

def NewMorph(name="new_morph", legs: list[LegDescriptor] = None, torso_radius: float = 0.14, n_legs: int = 0) -> RobotMorphology :

    if legs is None:
        l = []
        for i in range(n_legs if n_legs > 0 else rndInt(1, MAX_INIT_LEGS)):
            l.append(LegDescriptor(rndInt(0, 360), [JointDescriptor(rgba=(uniform(0.1, 0.9), uniform(0.1, 0.9), uniform(0.1, 0.9), 1.0), length=uniform(0.1, 0.4))]))
    else:
        l = legs

    return RobotMorphology(
        name=name,
        legs=l,
        torso_radius=torso_radius,
    )

# ---------------------------------------------------------------------------
# Serialisation helpers (used by saver.py)
# ---------------------------------------------------------------------------

def morphology_to_dict(morph: RobotMorphology) -> dict:
    """Convert a RobotMorphology to a plain dict safe for pickle."""
    import dataclasses
    return dataclasses.asdict(morph)   # tuples → lists, fine for storage


def dict_to_morphology(d: dict) -> RobotMorphology:
    """Reconstruct a RobotMorphology from a plain dict (inverse of morphology_to_dict)."""
    legs = [
        LegDescriptor(
            placement_angle_deg = leg["placement_angle_deg"],
            joints = [
                JointDescriptor(
                    damping    = j["damping"],
                    kp         = j["kp"],
                    ctrl_range = tuple(j["ctrl_range"]),
                    length     = j["length"],
                    radius     = j["radius"],
                    rgba       = tuple(j["rgba"]),
                )
                for j in leg["joints"]
            ],
        )
        for leg in d["legs"]
    ]
    return RobotMorphology(
        name         = d["name"],
        legs         = legs,
        torso_radius = d["torso_radius"],
        torso_height = d["torso_height"],
        torso_rgba   = tuple(d["torso_rgba"]),
        spawn_height = d["spawn_height"],
        foot_radius  = d["foot_radius"],
        foot_rgba    = tuple(d["foot_rgba"]),
    )


# ---------------------------------------------------------------------------
# Resolver — picks morphologies from save file or falls back to config
# ---------------------------------------------------------------------------

def resolve_morphologies(
    n:                   int,
    controller_init,
    default_morphologies,
) -> list[RobotMorphology]:
    """
    Return the list of N morphologies to use for this simulation.

    If controller_init points to a save that contains morphologies, those
    are used (so the physics + brain always match what was saved).
    Otherwise default_morphologies is used.
    """

    if controller_init is None:
        print(f"[init morphology] starting default morphologies.")
        return pad_morphologies(n, default_morphologies)

    if isinstance(controller_init, str):
        source       = controller_init
        load_indices = "all"
    else:
        source       = controller_init["source"]
        load_indices = controller_init.get("indices", "all")

    # --- load the save ---
    saved_morph = None
    try:
        from saver import load_controller  # lazy — avoids circular import
        payload = load_controller(source)
        saved_morph = payload.get("morphologies", [])
    except FileNotFoundError:
        print(f"[init morphology] Save '{source}' not found — starting default morphologies.")
        return pad_morphologies(n, default_morphologies)

    if not saved_morph:
        print(f"[init morphology] Save '{source}' found but empty or corrupted. starting with default morphologies.")
        return pad_morphologies(n, default_morphologies)

    # --- build controller list ---
    if load_indices == "mutation":
        # Robot 0: copy of a random elite (unchanged seed)
        # Robots 1..N-1: each mutated from a randomly chosen elite
        import random
        morphs = []
        for _ in range(n):
            source_morph = random.choice(saved_morph)
            morphs.append(
                MutateMorphology(source_morph, amplitude=controller_init.get("morph_amp"), variation=controller_init.get("morph_var"), morph_mod=controller_init.get("morph_mod")))
        print(f"[init morphology] Loaded {len(saved_morph)} elite(s) from '{source}', mutated randomly {n} time(s).")
        return morphs

    if load_indices == "all":
        load_indices = list(range(min(n, len(saved_morph))))

    load_set = set(load_indices)
    morphs = []
    for i in range(n):
        if i in load_set and i < len(saved_morph):
            morphs.append(saved_morph[i])

    loaded = len(load_set & set(range(len(morphs))))
    print(f"[init morphology] Loaded {loaded} morphology(s) from '{source}', {n - loaded} padded.")
    morphs = pad_morphologies(n, morphs)

    return morphs

def MutateMorphology(base: RobotMorphology, amplitude, variation, morph_mod):
    new = copy.deepcopy(base)
    if proba(morph_mod * 100):
        if len(new.legs) < MAX_INIT_LEGS + 2 and proba(50): # New leg
            new.legs.append(LegDescriptor(uniform(0, 360), [JointDescriptor(rgba=(uniform(0.1, 0.9), uniform(0.1, 0.9), uniform(0.1, 0.9), 1.0), length=uniform(0.15, 0.35))]))
        else: # Remove leg
            if len(new.legs) > 1:
                new.legs.pop(rndInt(0, len(base.legs) - 1))
    for leg in new.legs:
        for j in leg.joints:
            j.length = min(0.4, max(0.1, j.length + amplitude * normal(0, variation)))
    return new

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def pad_morphologies(n: int, morphologies) -> list[RobotMorphology]:
    """
    Return a list of exactly n morphologies.
    - If morphologies is a single RobotMorphology → replicate n times.
    - If it's a list shorter than n → pad with last element.
    """
    if morphologies is None:
        return [NewMorph() for _ in range(n)]

    if isinstance(morphologies, RobotMorphology):
        return [morphologies] * n
    result = list(morphologies)
    while len(result) < n:
        result.append(result[-1])
    return result[:n]


# ---------------------------------------------------------------------------
# MorphologyManager
# ---------------------------------------------------------------------------

class MorphologyManager:
    """
    Generates MJCF XML from RobotMorphology descriptors.

    Parameters
    ----------
    env_xml_path : path to a base XML file from which to extract
                   <option>, <visual>, <asset>, floor and lights.
                   If None, built-in defaults are used.
    """

    def __init__(self, env_xml_path: Optional[str] = None, floor_texrepeat: int = 64):
        self._env_xml_path    = env_xml_path
        self.floor_texrepeat  = floor_texrepeat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_texrepeat(self, asset_elem: ET.Element) -> None:
        """Patch the floor material's texrepeat in-place after an asset block is copied."""
        mat = asset_elem.find(".//material[@name='floor_material']")
        if mat is not None:
            mat.set("texrepeat", f"{self.floor_texrepeat} {self.floor_texrepeat}")

    @staticmethod
    def _rgba(rgba: tuple) -> str:
        return " ".join(f"{v:.3f}" for v in rgba)

    @staticmethod
    def _hip_axis(angle_deg: float) -> tuple[float, float, float]:
        """
        Hip swing axis perpendicular to the radial leg direction.
        For leg at angle θ: axis = (-sin θ, cos θ, 0).
        """
        r = math.radians(angle_deg)
        return (-math.sin(r), math.cos(r), 0.0)

    @staticmethod
    def _attachment_pos(angle_deg: float, torso_radius: float) -> tuple[float, float, float]:
        r = math.radians(angle_deg)
        return (
            round(math.cos(r) * torso_radius, 5),
            round(math.sin(r) * torso_radius, 5),
            0.0,
        )

    # ------------------------------------------------------------------
    # Per-leg XML builder
    # ------------------------------------------------------------------

    def _build_leg(
        self,
        leg: LegDescriptor,
        leg_idx: int,
        morph: RobotMorphology,
        prefix: str,
    ) -> tuple[ET.Element, list[str]]:
        """
        Returns (leg body element, list of joint names).
        Handles single and multi-joint legs via nested bodies.
        """
        ax    = self._hip_axis(leg.placement_angle_deg)
        at    = self._attachment_pos(leg.placement_angle_deg, morph.torso_radius)

        leg_body = ET.Element("body",
            name = f"{prefix}leg{leg_idx + 1}",
            pos  = f"{at[0]} {at[1]} {at[2]}",
        )

        joint_names: list[str] = []
        current    = leg_body

        for j_idx, jd in enumerate(leg.joints):
            jname = f"{prefix}hip{leg_idx + 1}" if len(leg.joints) == 1 \
                    else f"{prefix}leg{leg_idx + 1}_j{j_idx + 1}"
            joint_names.append(jname)

            ET.SubElement(current, "joint",
                name    = jname,
                type    = "hinge",
                axis    = f"{ax[0]:.5f} {ax[1]:.5f} {ax[2]:.5f}",
                damping = str(jd.damping),
            )

            ET.SubElement(current, "geom",
                name    = f"{prefix}leg{leg_idx + 1}_geom{j_idx}",
                type    = "capsule",
                fromto  = f"0 0 0  0 0 -{jd.length}",
                size    = str(jd.radius),
                rgba    = self._rgba(jd.rgba),
            )

            if j_idx == len(leg.joints) - 1:
                # Foot sphere at the tip
                foot = ET.SubElement(current, "body",
                    name = f"{prefix}foot{leg_idx + 1}",
                    pos  = f"0 0 -{jd.length}",
                )
                ET.SubElement(foot, "geom",
                    name    = f"{prefix}foot{leg_idx + 1}_geom",
                    type    = "sphere",
                    size    = str(morph.foot_radius),
                    rgba    = self._rgba(morph.foot_rgba),
                    condim  = "3",
                    friction= "0.7 0.005 0.0001",
                )
            else:
                # Nested body for the next joint
                next_body = ET.SubElement(current, "body",
                    name = f"{prefix}leg{leg_idx + 1}_seg{j_idx + 2}",
                    pos  = f"0 0 -{jd.length}",
                )
                current = next_body

        return leg_body, joint_names

    # ------------------------------------------------------------------
    # Torso + all legs builder (shared by generate_xml and generate_body_element)
    # ------------------------------------------------------------------

    def _build_torso(
        self,
        morph: RobotMorphology,
        prefix: str = "",
        x: float = 0.0,
        y: float = 0.0,
    ) -> tuple[ET.Element, list[str]]:
        """
        Build the full torso body element (with freejoint and all legs).
        Returns (torso element, all joint names).
        x, y : grid offset for display positioning.
        """
        torso = ET.Element("body",
            name = f"{prefix}torso",
            pos  = f"{x} {y} {morph.spawn_height}",
        )

        ET.SubElement(torso, "freejoint", name=f"{prefix}root")
        ET.SubElement(torso, "geom",
            name  = f"{prefix}torso_geom",
            type  = "cylinder",
            size  = f"{morph.torso_radius} {morph.torso_height}",
            rgba  = self._rgba(morph.torso_rgba),
        )

        all_joint_names: list[str] = []
        for leg_idx, leg in enumerate(morph.legs):
            leg_elem, jnames = self._build_leg(leg, leg_idx, morph, prefix)
            torso.append(leg_elem)
            all_joint_names.extend(jnames)

        return torso, all_joint_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_xml(self, morph: RobotMorphology) -> str:
        """
        Generate a complete MJCF XML string for one robot.
        Visual/physics environment is taken from env_xml_path if provided.
        """
        root = ET.Element("mujoco", model=morph.name)

        if self._env_xml_path:
            env = ET.parse(self._env_xml_path).getroot()
            for tag in ("option", "visual", "asset"):
                elem = env.find(tag)
                if elem is not None:
                    copied = copy.deepcopy(elem)
                    if tag == "asset":
                        self._apply_texrepeat(copied)
                    root.append(copied)
        else:
            option = ET.SubElement(root, "option", timestep="0.005", gravity="0 0 -9.81")
            visual = ET.SubElement(root, "visual")
            ET.SubElement(visual, "headlight",
                diffuse="0.7 0.7 0.7", ambient="0.3 0.3 0.3", specular="0 0 0")
            asset = ET.SubElement(root, "asset")
            ET.SubElement(asset, "texture", type="2d", name="floor_texture",
                builtin="checker", rgb1="0.3 0.3 0.3", rgb2="0.2 0.2 0.2",
                width="512", height="512")
            ET.SubElement(asset, "material", name="floor_material",
                texture="floor_texture",
                texrepeat=f"{self.floor_texrepeat} {self.floor_texrepeat}")

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", pos="0 0 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
        ET.SubElement(worldbody, "geom",
            name="floor", type="plane", size="100 100 0.1", material="floor_material")

        torso, joint_names = self._build_torso(morph)
        worldbody.append(torso)

        # Actuators
        actuator = ET.SubElement(root, "actuator")
        for leg_idx, leg in enumerate(morph.legs):
            for j_idx, jd in enumerate(leg.joints):
                jname = joint_names[sum(len(morph.legs[k].joints) for k in range(leg_idx)) + j_idx]
                ET.SubElement(actuator, "position",
                    name        = f"servo_{jname}",
                    joint       = jname,
                    kp          = str(jd.kp),
                    ctrllimited = "true",
                    ctrlrange   = f"{jd.ctrl_range[0]} {jd.ctrl_range[1]}",
                )

        return ET.tostring(root, encoding="unicode")

    def generate_body_element(
        self,
        morph:  RobotMorphology,
        prefix: str,
        col:    int,
        row:    int,
        spacing: float,
    ) -> ET.Element:
        """
        For display: returns the torso body element placed at grid position (col, row).
        No actuator block — display is never stepped.
        """
        torso, _ = self._build_torso(
            morph,
            prefix = prefix,
            x      = col * spacing,
            y      = row * spacing,
        )
        return torso

    def get_model(self, morph: RobotMorphology):
        """Build and return a mujoco.MjModel for the given morphology."""
        import mujoco  # lazy import — allows importing this module without mujoco
        xml = self.generate_xml(morph)
        return mujoco.MjModel.from_xml_string(xml)