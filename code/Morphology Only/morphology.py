"""
morphology.py
=============
Robot morphology descriptors and MuJoCo XML generator for the
morphology-only experiment.

A robot body is described by a RobotMorphology dataclass:
  - a torso (cylinder)
  - a list of LegDescriptors, each with a list of JointDescriptors

Legs can be attached to the torso (root legs) or to the end of a
joint segment of another leg (branched legs), enabling tree-shaped
limb structures.

Constraint for branching: a branched leg's parent_leg_idx must be
strictly less than its own index in RobotMorphology.legs, so the XML
can be built in a single forward pass.

MorphologyManager generates valid MJCF XML from any RobotMorphology.
The XML is suitable for static offscreen rendering (no physics stepping
required).

Mutation:
    MutateMorphology(base, ...) applies random perturbations to segment
    lengths, placement angles, and optionally adds/removes legs.

Serialisation:
    morphology_to_dict / dict_to_morphology — plain dicts, safe for
    JSON and pickle. New fields (parent_leg_idx, parent_joint_idx) are
    handled with defaults so old saves can still be loaded.

Debug:
    Run this file directly to test morphology creation, mutation,
    branching, encoding, and XML generation.
"""

from __future__ import annotations
import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Maximum total legs allowed during mutation
MAX_LEGS = 8
MIN_LENGTH, MAX_LENGTH = 0.15, 0.5


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

@dataclass
class JointDescriptor:
    """One rotational hinge joint on a leg segment."""
    damping:    float               = 4.0
    kp:         float               = 25.0           # position servo stiffness (N·m/rad)
    ctrl_range: tuple[float, float] = (-1.5, 0.8)    # actuator limits (radians)
    length:     float               = 0.25            # segment length below this joint (m)
    radius:     float               = 0.025           # capsule radius (m)
    rgba:       tuple               = (0.5, 0.5, 0.5, 1.0)
    rest_angle: float               = 0.0             # joint angle used for static rendering (radians)

@dataclass
class LegDescriptor:
    """
    One leg attached either to the torso or to a joint on another leg.

    placement_angle_deg:
        For root legs (parent_leg_idx is None):
            Absolute angle in degrees around the Z axis from the robot's
            front (0 = front, 90 = right, 180 = back, 270 = left).
            Defines both the torso rim attachment point and the hip swing
            axis: axis = (-sin θ, cos θ, 0).

        For branched legs:
            Relative rotation angle φ (degrees) around the parent segment's
            own downward axis (-Z in parent local frame).  The branched
            leg's swing axis is the parent's swing axis rotated by φ:

                axis_branched = x_parent · cos φ + y_parent · sin φ
                              = _hip_axis(θ_parent_effective − φ)

            where x_parent = parent's swing axis,
                  y_parent = parent's radial-outward direction (x_parent rotated
                             90° CCW around world-Z).

            φ = 0   → same swing direction as the parent leg.
            φ = 90  → swing perpendicular to parent (rotated 90° CCW from above).
            φ = 180 → swing opposite to the parent.

            The effective angle propagates recursively, so a branch-of-branch
            uses the same formula with its own parent's effective angle.

    parent_leg_idx:
        Index in RobotMorphology.legs of the parent leg.
        None → root leg, attaches to torso rim.
        Must be < this leg's own index (enforced by MorphologyManager).

    parent_joint_idx:
        Index of the joint in the parent leg at whose end-body this leg
        attaches.  Negative indices are supported (-1 = last joint).
        Ignored when parent_leg_idx is None.
    """
    placement_angle_deg: float
    joints:              list[JointDescriptor] = field(default_factory=lambda: [JointDescriptor()])
    parent_leg_idx:      Optional[int]         = None
    parent_joint_idx:    Optional[int]         = None


@dataclass
class RobotMorphology:
    """
    Full description of a robot body.
    MorphologyManager uses this to generate MJCF XML on the fly.
    """
    name:          str
    legs:          list[LegDescriptor]
    torso_radius:  float = 0.12
    torso_height:  float = 0.04
    torso_rgba:    tuple = (0.9, 0.9, 0.9, 1.0)
    spawn_height:  float = 0.5
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

    def encoding(self) -> dict:
        """
        Named scalar descriptors for this morphology.

        These are computed purely from the structure (no simulation).
        Used by MapElite as feature dimensions and for analysis/plotting.

        Returns a plain dict so it can be logged directly to JSON.
        """
        root_legs   = [l for l in self.legs if l.parent_leg_idx is None]
        branch_legs = [l for l in self.legs if l.parent_leg_idx is not None]
        all_lengths = [j.length for l in self.legs for j in l.joints]

        # Symmetry: how evenly spaced root-leg angles are around 360°
        if len(root_legs) > 1:
            angles   = sorted(l.placement_angle_deg % 360 for l in root_legs)
            n        = len(angles)
            gaps     = [(angles[(i + 1) % n] - angles[i]) % 360 for i in range(n)]
            expected = 360.0 / n
            symmetry = float(max(0.0, 1.0 - np.std(gaps) / expected))
        elif len(root_legs) == 1:
            symmetry = 1.0
        else:
            symmetry = 0.0

        return {
            "n_legs":               len(self.legs),
            "n_root_legs":          len(root_legs),
            "n_branch_legs":        len(branch_legs),
            "n_total_joints":       self.n_joints,
            "symmetry_score":       round(symmetry, 4),
            "total_segment_length": round(sum(all_lengths), 4) if all_lengths else 0.0,
            "mean_segment_length":  round(float(np.mean(all_lengths)), 4) if all_lengths else 0.0,
            "max_segment_length":   round(float(max(all_lengths)), 4) if all_lengths else 0.0,
            "torso_radius":         self.torso_radius,
        }

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

# ---------------------------------------------------------------------------
# Factory — random morphology
# ---------------------------------------------------------------------------

def NewMorph(
    name:         str   = "morph",
    n_legs:       int   = 0,
    min_init_legs: int   = 2,
    max_init_legs: int   = 6,
    torso_radius: float = 0.12,
) -> RobotMorphology:
    """
    Create a random root-only morphology (no branching).
    If n_legs == 0, a random count in [min_legs, max_legs] is chosen.
    """
    count = n_legs if n_legs > 0 else int(np.random.randint(min_init_legs, max_init_legs + 1))
    legs  = []
    for _ in range(count):
        angle = float(np.random.uniform(0, 360))
        color = (float(np.random.uniform(0.2, 0.9)),
                 float(np.random.uniform(0.2, 0.9)),
                 float(np.random.uniform(0.2, 0.9)),
                 1.0)
        length      = float(np.random.uniform(MIN_LENGTH, MAX_LENGTH))
        rest_angle  = float(np.random.uniform(-0.6, 0.4))   # within typical ctrl_range
        legs.append(LegDescriptor(angle, [JointDescriptor(rgba=color, length=length, rest_angle=rest_angle)]))

    return RobotMorphology(name=name, legs=legs, torso_radius=torso_radius)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def MutateMorphology(
    base:             RobotMorphology,
    length_std:       float = 0.04,
    angle_std:        float = 12.0,
    rest_angle_std:   float = 0.15,
    add_remove_prob:  float = 0.15,
    allow_branching:  bool  = False,
    branching_prob:   float = 0.3,
    rng:              Optional[np.random.Generator] = None,
) -> RobotMorphology:
    """
    Return a mutated copy of base.  The original is never modified.

    Parameters
    ----------
    length_std      : std dev (metres) for Gaussian length perturbations.
    angle_std       : std dev (degrees) for placement angle jitter.
    add_remove_prob : probability [0, 1] of adding or removing one leg.
    allow_branching : when adding a leg, whether it can attach to another
                      leg's segment rather than the torso.
    rng             : numpy Generator for reproducibility (uses global
                      default if None).
    """
    if rng is None:
        rng = np.random.default_rng()

    new = copy.deepcopy(base)

    # --- Add or remove a leg ---
    if rng.random() < add_remove_prob:
        if rng.random() < 0.5 and len(new.legs) < MAX_LEGS:
            # Add leg
            angle  = float(rng.uniform(0, 360))
            color  = (float(rng.uniform(0.2, 0.9)),
                      float(rng.uniform(0.2, 0.9)),
                      float(rng.uniform(0.2, 0.9)),
                      1.0)
            length = float(rng.uniform(0.12, 0.32))

            rest_angle = float(rng.uniform(-1.5, 1.5))

            if allow_branching and len(new.legs) > 0 and rng.random() < branching_prob:
                parent_idx   = int(rng.integers(0, len(new.legs)))
                parent_joint = int(rng.integers(0, len(new.legs[parent_idx].joints)))
                new.legs.append(LegDescriptor(
                    placement_angle_deg = angle,
                    joints              = [JointDescriptor(rgba=color, length=length, rest_angle=rest_angle)],
                    parent_leg_idx      = parent_idx,
                    parent_joint_idx    = parent_joint,
                ))
            else:
                new.legs.append(LegDescriptor(
                    placement_angle_deg = angle,
                    joints              = [JointDescriptor(rgba=color, length=length, rest_angle=rest_angle)],
                ))
        else:
            # Remove a random leg if more than one remains
            if len(new.legs) > 1:
                idx = int(rng.integers(0, len(new.legs)))
                new.legs.pop(idx)
                # Fix broken parent references caused by the removal
                for leg in new.legs:
                    if leg.parent_leg_idx is not None:
                        if leg.parent_leg_idx == idx:
                            # Parent was removed — promote to root
                            leg.parent_leg_idx   = None
                            leg.parent_joint_idx = None
                        elif leg.parent_leg_idx > idx:
                            leg.parent_leg_idx -= 1

    # --- Perturb segment lengths and placement angles ---
    for leg in new.legs:
        if leg.parent_leg_idx is None:
            # Jitter placement angle for root legs only
            leg.placement_angle_deg = float(
                leg.placement_angle_deg + rng.normal(0, angle_std)
            ) % 360

        for j in leg.joints:
            j.length = float(np.clip(
                j.length + rng.normal(0, length_std),
                MIN_LENGTH, MAX_LENGTH,
            ))
            j.rest_angle = float(np.clip(
                j.rest_angle + rng.normal(0, rest_angle_std),
                j.ctrl_range[0], j.ctrl_range[1],
            ))

    return new


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def morphology_to_dict(morph: RobotMorphology) -> dict:
    """Convert a RobotMorphology to a plain dict safe for JSON/pickle."""
    return {
        "name":          morph.name,
        "torso_radius":  morph.torso_radius,
        "torso_height":  morph.torso_height,
        "torso_rgba":    list(morph.torso_rgba),
        "spawn_height":  morph.spawn_height,
        "foot_radius":   morph.foot_radius,
        "foot_rgba":     list(morph.foot_rgba),
        "legs": [
            {
                "placement_angle_deg": leg.placement_angle_deg,
                "parent_leg_idx":      leg.parent_leg_idx,
                "parent_joint_idx":    leg.parent_joint_idx,
                "joints": [
                    {
                        "damping":    j.damping,
                        "kp":         j.kp,
                        "ctrl_range": list(j.ctrl_range),
                        "length":     j.length,
                        "radius":     j.radius,
                        "rgba":       list(j.rgba),
                        "rest_angle": j.rest_angle,
                    }
                    for j in leg.joints
                ],
            }
            for leg in morph.legs
        ],
    }


def dict_to_morphology(d: dict) -> RobotMorphology:
    """
    Reconstruct a RobotMorphology from a plain dict.
    The new parent_leg_idx / parent_joint_idx fields default to None
    so that saves from the prototype (which lack those keys) still load.
    """
    legs = [
        LegDescriptor(
            placement_angle_deg = leg["placement_angle_deg"],
            parent_leg_idx      = leg.get("parent_leg_idx"),
            parent_joint_idx    = leg.get("parent_joint_idx"),
            joints = [
                JointDescriptor(
                    damping    = j["damping"],
                    kp         = j["kp"],
                    ctrl_range = tuple(j["ctrl_range"]),
                    length     = j["length"],
                    radius     = j["radius"],
                    rgba       = tuple(j["rgba"]),
                    rest_angle = j.get("rest_angle", 0.0),  # default for old saves
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
# MorphologyManager — MJCF XML builder
# ---------------------------------------------------------------------------

class MorphologyManager:
    """
    Generates MJCF XML from RobotMorphology descriptors.

    Supports both root legs (attached to the torso) and branched legs
    (attached to the end-body of a joint on another leg).

    Parameters
    ----------
    floor_texrepeat : checker tile density on the floor texture.
    """

    def __init__(self, floor_texrepeat: int = 64):
        self.floor_texrepeat = floor_texrepeat

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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
    def _rim_pos(angle_deg: float, torso_radius: float) -> tuple[float, float, float]:
        """XYZ position on the torso rim at the given angle."""
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
        leg:              LegDescriptor,
        leg_idx:          int,
        morph:            RobotMorphology,
        prefix:           str,
        attachment_pos:   tuple[float, float, float],
        effective_angle:  float,
    ) -> tuple[ET.Element, list[str], list[ET.Element]]:
        """
        Build the XML subtree for one leg.

        effective_angle : the resolved world-equivalent angle (degrees) used
                          to compute the hip axis.  For root legs this equals
                          placement_angle_deg.  For branched legs it is
                          parent_effective_angle − placement_angle_deg.

        Returns
        -------
        leg_body        : the root ET.Element for this leg.
        joint_names     : flat list of joint names (in order).
        segment_ends    : segment_ends[j] is the body element at the
                          end of joint j. A branched child leg may be
                          appended here by _build_torso.
        """
        ax = self._hip_axis(effective_angle)
        px, py, pz = attachment_pos

        leg_body = ET.Element("body",
            name = f"{prefix}leg{leg_idx + 1}",
            pos  = f"{px} {py} {pz}",
        )

        joint_names:  list[str]       = []
        segment_ends: list[ET.Element] = []
        current = leg_body

        for j_idx, jd in enumerate(leg.joints):
            # Joint name
            if len(leg.joints) == 1:
                jname = f"{prefix}hip{leg_idx + 1}"
            else:
                jname = f"{prefix}leg{leg_idx + 1}_j{j_idx + 1}"
            joint_names.append(jname)

            # Hinge joint
            ET.SubElement(current, "joint",
                name    = jname,
                type    = "hinge",
                axis    = f"{ax[0]:.5f} {ax[1]:.5f} {ax[2]:.5f}",
                damping = str(jd.damping),
            )

            # Segment capsule
            ET.SubElement(current, "geom",
                name   = f"{prefix}leg{leg_idx + 1}_seg{j_idx + 1}",
                type   = "capsule",
                fromto = f"0 0 0  0 0 -{jd.length}",
                size   = str(jd.radius),
                rgba   = self._rgba(jd.rgba),
            )

            # End body at the tip of this segment
            is_last = (j_idx == len(leg.joints) - 1)
            end_name = (f"{prefix}foot{leg_idx + 1}"
                        if is_last
                        else f"{prefix}leg{leg_idx + 1}_end{j_idx + 1}")
            end_body = ET.SubElement(current, "body",
                name = end_name,
                pos  = f"0 0 -{jd.length}",
            )

            if is_last:
                # Foot sphere — touch point with the ground
                ET.SubElement(end_body, "geom",
                    name     = f"{prefix}foot{leg_idx + 1}_geom",
                    type     = "sphere",
                    size     = str(morph.foot_radius),
                    rgba     = self._rgba(morph.foot_rgba),
                    condim   = "3",
                    friction = "0.7 0.005 0.0001",
                )

            segment_ends.append(end_body)
            current = end_body   # next joint in the chain hangs from here

        return leg_body, joint_names, segment_ends

    # ------------------------------------------------------------------
    # Torso + all legs
    # ------------------------------------------------------------------

    def _build_torso(
        self,
        morph:  RobotMorphology,
        prefix: str  = "",
        x:      float = 0.0,
        y:      float = 0.0,
    ) -> tuple[ET.Element, list[str]]:
        """
        Build the full torso body with all legs (root and branched).
        Returns (torso_element, all_joint_names_flat).
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

        all_joint_names:     list[str]              = []
        segment_ends_by_leg: dict[int, list[ET.Element]] = {}

        # Compute the effective swing-axis angle for every leg.
        # Root legs:    effective = placement_angle_deg  (absolute, unchanged)
        # Branched legs: effective = parent_effective − placement_angle_deg
        #   → axis_branched = _hip_axis(parent_eff − φ)
        #     which is the parent's swing axis rotated by φ around the parent
        #     segment's downward axis.  Propagates recursively for nested branches.
        effective_angle: dict[int, float] = {}
        for leg_idx, leg in enumerate(morph.legs):
            if leg.parent_leg_idx is None:
                effective_angle[leg_idx] = leg.placement_angle_deg
            else:
                parent_eff = effective_angle[leg.parent_leg_idx]
                effective_angle[leg_idx] = parent_eff - leg.placement_angle_deg

        for leg_idx, leg in enumerate(morph.legs):
            if leg.parent_leg_idx is None:
                # Root leg: attach to the torso rim
                rx, ry, rz = self._rim_pos(leg.placement_angle_deg, morph.torso_radius)
                attachment_body = torso
                attach_pos      = (rx, ry, rz)
            else:
                # Branched leg: attach to the end-body of a parent joint
                parent_idx      = leg.parent_leg_idx
                joint_idx       = leg.parent_joint_idx if leg.parent_joint_idx is not None else -1
                attachment_body = segment_ends_by_leg[parent_idx][joint_idx]
                attach_pos      = (0.0, 0.0, 0.0)

            leg_elem, jnames, seg_ends = self._build_leg(
                leg, leg_idx, morph, prefix, attach_pos,
                effective_angle=effective_angle[leg_idx],
            )
            attachment_body.append(leg_elem)
            segment_ends_by_leg[leg_idx] = seg_ends
            all_joint_names.extend(jnames)

        return torso, all_joint_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_xml(self, morph: RobotMorphology) -> str:
        """Generate a complete MJCF XML string for one robot."""
        root = ET.Element("mujoco", model=morph.name)

        # Physics + visuals
        ET.SubElement(root, "option", timestep="0.005", gravity="0 0 -9.81")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight",
            diffuse="0.7 0.7 0.7", ambient="0.3 0.3 0.3", specular="0 0 0")

        # Floor assets
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture",
            type="2d", name="floor_tex", builtin="checker",
            rgb1="0.3 0.3 0.3", rgb2="0.22 0.22 0.22",
            width="512", height="512")
        ET.SubElement(asset, "material",
            name="floor_mat", texture="floor_tex",
            texrepeat=f"{self.floor_texrepeat} {self.floor_texrepeat}")

        # World
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light",
            pos="0 0 4", dir="0 0 -1", diffuse="0.8 0.8 0.8", specular="0.2 0.2 0.2")
        ET.SubElement(worldbody, "geom",
            name="floor", type="plane", size="100 100 0.1", material="floor_mat")

        torso, joint_names = self._build_torso(morph)
        worldbody.append(torso)

        # Actuators (one position servo per joint)
        actuator = ET.SubElement(root, "actuator")
        flat_idx = 0
        for leg in morph.legs:
            for jd in leg.joints:
                jname = joint_names[flat_idx]
                ET.SubElement(actuator, "position",
                    name        = f"servo_{jname}",
                    joint       = jname,
                    kp          = str(jd.kp),
                    ctrllimited = "true",
                    ctrlrange   = f"{jd.ctrl_range[0]} {jd.ctrl_range[1]}",
                )
                flat_idx += 1

        return ET.tostring(root, encoding="unicode")

    def get_model(self, morph: RobotMorphology):
        """Build and return a mujoco.MjModel for the given morphology."""
        import mujoco
        return mujoco.MjModel.from_xml_string(self.generate_xml(morph))

    def generate_body_element(
        self,
        morph:   RobotMorphology,
        prefix:  str,
        col:     int,
        row:     int,
        spacing: float,
    ) -> ET.Element:
        """
        For multi-robot display: returns the torso body element placed
        at grid position (col, row).  No actuator block.
        """
        torso, _ = self._build_torso(
            morph,
            prefix = prefix,
            x      = col * spacing,
            y      = row * spacing,
        )
        return torso


# ---------------------------------------------------------------------------
# Debug — run directly to test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  morphology.py — debug mode")
    print("=" * 60)

    manager = MorphologyManager()

    # --- 1. Built-in morphologies and their encodings ---
    print("\n[1] Pre-defined morphologies\n")
    for morph in (TRIPOD, QUADRIPOD, HEXAPOD):
        enc = morph.encoding()
        print(f"  {morph.name:<12}  legs={enc['n_legs']}  "
              f"sym={enc['symmetry_score']:.3f}  "
              f"total_len={enc['total_segment_length']:.3f}m")

    # --- 2. Random morphology ---
    print("\n[2] Random morphology\n")
    rand  = NewMorph(name="random")
    rand2 = NewMorph(name="random2", n_legs=5)
    for m in (rand, rand2):
        enc = m.encoding()
        print(f"  {m.name:<12}  legs={enc['n_legs']}  "
              f"angles={[round(l.placement_angle_deg, 1) for l in m.legs]}")

    # --- 3. Mutation ---
    print("\n[3] Mutation (3 steps from QUADRIPOD)\n")
    current = QUADRIPOD
    for i in range(3):
        current = MutateMorphology(current, length_std=0.06, angle_std=20.0, add_remove_prob=0.4)
        enc = current.encoding()
        print(f"  gen {i + 1}: legs={enc['n_legs']}  "
              f"sym={enc['symmetry_score']:.3f}  "
              f"total_len={enc['total_segment_length']:.3f}m")

    # --- 4. Branched morphology ---
    print("\n[4] Branched morphology\n")
    branched = RobotMorphology(
        name="branched",
        legs=[
            LegDescriptor( 0.0, [JointDescriptor(length=0.3), JointDescriptor(length=0.2)]),   # leg 0 — 2 joints
            LegDescriptor(90.0, [JointDescriptor(length=0.25)]),                                 # leg 1 — root
            LegDescriptor(180.0,[JointDescriptor(length=0.25)]),                                 # leg 2 — root
            LegDescriptor(270.0,[JointDescriptor(length=0.25)]),                                 # leg 3 — root
            LegDescriptor(45.0, [JointDescriptor(length=0.15, rgba=(0.9, 0.2, 0.2, 1.0))],     # leg 4 — branches from
                          parent_leg_idx=0, parent_joint_idx=0),                                 #   leg 0 at joint 0
        ],
    )
    enc = branched.encoding()
    print(f"  root_legs={enc['n_root_legs']}  branch_legs={enc['n_branch_legs']}")
    xml = manager.generate_xml(branched)
    print(f"  XML length: {len(xml)} chars  (first 120: {xml[:120]}...)")

    # --- 5. Serialisation round-trip ---
    print("\n[5] Serialisation round-trip\n")
    d    = morphology_to_dict(branched)
    back = dict_to_morphology(d)
    assert back.name == branched.name
    assert len(back.legs) == len(branched.legs)
    assert back.legs[4].parent_leg_idx   == 0
    assert back.legs[4].parent_joint_idx == 0
    print("  morphology_to_dict → dict_to_morphology : OK")
    print(f"  JSON keys: {list(d.keys())}")

    # --- 6. XML validity check ---
    print("\n[6] XML validity (requires mujoco)\n")
    try:
        import mujoco
        for morph in (TRIPOD, QUADRIPOD, branched):
            model = manager.get_model(morph)
            print(f"  {morph.name:<12}  nq={model.nq}  nu={model.nu}  OK")
    except ImportError:
        print("  mujoco not available — skipping XML validation")

    print("\nAll debug checks passed.")
