"""
morphology.py
=============
Robot morphology descriptors and MuJoCo XML generator for the
morphology-only experiment.

A robot body is described by a RobotMorphology dataclass:
  - a torso (ellipsoid)
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
from sympy.strategies.core import switch

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
    One leg attached to the torso, to a secondary body part, or to a joint
    on another leg (branched).

    placement_angle_deg:
        For root legs (parent_leg_idx is None):
            Absolute angle in degrees around the Z axis from the robot's
            front (0 = front, 90 = right, 180 = back, 270 = left).
            Defines both the rim attachment point and the hip swing axis:
            axis = (-sin θ, cos θ, 0).  When body_part_idx is set, the angle
            is relative to the body part's own frame.

        For branched legs:
            Relative rotation angle φ (degrees) around the parent segment's
            own downward axis (-Z in parent local frame).  The effective angle
            propagates recursively; see MorphologyManager for details.

    parent_leg_idx:
        Index in RobotMorphology.legs of the parent leg.
        None → root leg (attaches to torso or body part rim).
        Must be < this leg's own index.

    parent_joint_idx:
        Index of the joint in the parent leg at whose end-body this branched
        leg attaches.  Negative indices supported (-1 = last joint).
        Ignored when parent_leg_idx is None.

    body_part_idx:
        Index in RobotMorphology.body_parts of the body part this leg's rim
        attaches to.  None → attach to the main torso rim (default).
        Ignored when parent_leg_idx is not None (branched legs always attach
        to their parent leg's joint end-body, not a body part rim).
        The referenced body part's parent_leg_idx must be < this leg's index.
    """
    placement_angle_deg: float
    joints:              list[JointDescriptor] = field(default_factory=lambda: [JointDescriptor()])
    parent_leg_idx:      Optional[int]         = None
    parent_joint_idx:    Optional[int]         = None
    body_part_idx:       Optional[int]         = None


@dataclass
class BodyPartDescriptor:
    """
    A secondary ellipsoidal body rigidly attached at the tip of a parent leg.

    Body parts act as structural nodes from which additional root legs can
    spawn, enabling multi-segment body plans (e.g. insect thorax+abdomen).
    The body part is welded to the last segment end-body of its parent leg
    (no joint), so it moves rigidly with the leg in simulation.

    Attributes
    ----------
    parent_leg_idx : index in RobotMorphology.legs of the leg whose last
                     segment tip this body part attaches to.
                     Must be < any leg that references this body part via
                     body_part_idx, so the MJCF can be built in one pass.
    a              : ellipsoid X semi-axis (m).
    b              : ellipsoid Y semi-axis (m).
    c              : ellipsoid Z semi-axis (m).
    euler_deg      : (roll, pitch, yaw) orientation relative to the parent
                     leg's tip frame, in degrees.
    rgba           : RGBA colour.
    """
    parent_leg_idx: int
    a:              float = 0.08
    b:              float = 0.08
    c:              float = 0.03
    euler_deg:      tuple = field(default_factory=lambda: (0., 0., 0.))
    rgba:           tuple = field(default_factory=lambda: (0.75, 0.75, 0.75, 1.0))


@dataclass
class RobotMorphology:
    """
    Full description of a robot body.
    MorphologyManager uses this to generate MJCF XML on the fly.
    """
    name:          str
    legs:          list[LegDescriptor]
    body_parts:    list[BodyPartDescriptor] = field(default_factory=list)
    torso_a:       float = 0.12   # ellipsoid X semi-axis (m)
    torso_b:       float = 0.12   # ellipsoid Y semi-axis (m)
    torso_c:       float = 0.04   # ellipsoid Z semi-axis (m)
    torso_rgba:    tuple = (0.9, 0.9, 0.9, 1.0)
    torso_euler:   tuple = (0.0, 0.0, 0.0)   # roll, pitch, yaw (degrees) of the torso body
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
            "n_body_parts":         len(self.body_parts),
            "n_total_joints":       self.n_joints,
            "symmetry_score":       round(symmetry, 4),
            "total_segment_length": round(sum(all_lengths), 4) if all_lengths else 0.0,
            "mean_segment_length":  round(float(np.mean(all_lengths)), 4) if all_lengths else 0.0,
            "max_segment_length":   round(float(max(all_lengths)), 4) if all_lengths else 0.0,
            "torso_a":              self.torso_a,
            "torso_b":              self.torso_b,
            "torso_c":              self.torso_c,
            "torso_euler":          list(self.torso_euler),
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
    torso_a=0.14,
    torso_b=0.14,
)

def get_preconfigured_morph(name: str):
    for mo in (QUADRIPOD, TRIPOD, HEXAPOD):
        if mo.name == name:
            return mo
    print(f"ERROR : morphology '{name}' not found. -> Default {QUADRIPOD.name}")
    return QUADRIPOD

# ---------------------------------------------------------------------------
# Spawn-height computation — keeps robot above the floor
# ---------------------------------------------------------------------------

def compute_spawn_height(morph: RobotMorphology, floor_clearance: float = 0.05) -> float:
    """
    Compute the torso spawn height (z position) so that:
      1. No part of the robot penetrates the floor (z=0).
      2. At least one foot touches the floor exactly (foot sphere bottom = z=0),
         provided terminal legs exist.

    Feet land at z=0 without floor_clearance offset — clearance is only applied
    to the torso bottom and body parts to avoid rendering artefacts.
    If no terminal leg exists, clearance is applied uniformly as before.

    The calculation uses the rest-pose geometry:
      - Each leg segment with rest_angle α and length L drops by L·cos(α).
      - Drops along a chain accumulate additively.
      - Body parts at leg tips extend the drop by the body part's Z semi-axis (c).
      - Leg tips end in a foot sphere of radius foot_radius.

    Parameters
    ----------
    morph           : RobotMorphology to analyse.
    floor_clearance : clearance margin applied to the torso and body parts (m).

    Returns
    -------
    Required torso spawn_height in metres.
    """
    leg_tip_z: dict[int, float] = {}
    body_part_z: dict[int, float] = {}
    body_part_at_leg: dict[int, int] = {
        bp.parent_leg_idx: bp_idx for bp_idx, bp in enumerate(morph.body_parts)
    }

    # Torso bottom and body parts need clearance; feet do not.
    non_foot_drop: float = morph.torso_c
    foot_drop:     float = 0.0
    has_terminal_leg:    bool  = False

    for leg_idx, leg in enumerate(morph.legs):
        z_drop = sum(j.length * math.cos(j.rest_angle) for j in leg.joints)

        if leg.parent_leg_idx is None and leg.body_part_idx is None:
            attach_z = 0.0
        elif leg.body_part_idx is not None:
            attach_z = body_part_z.get(leg.body_part_idx, 0.0)
        else:
            attach_z = leg_tip_z.get(leg.parent_leg_idx, 0.0)

        tip_z = attach_z - z_drop
        leg_tip_z[leg_idx] = tip_z

        if leg_idx in body_part_at_leg:
            bp_idx = body_part_at_leg[leg_idx]
            bp = morph.body_parts[bp_idx]
            body_part_z[bp_idx] = tip_z
            non_foot_drop = max(non_foot_drop, -tip_z + bp.c)
        else:
            has_terminal_leg = True
            foot_drop = max(foot_drop, -tip_z + morph.foot_radius)

    if has_terminal_leg:
        # Lowest foot touches z=0; torso/body parts still get their clearance.
        return max(foot_drop, non_foot_drop + floor_clearance)
    else:
        return non_foot_drop + floor_clearance

# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def morphology_to_dict(morph: RobotMorphology) -> dict:
    """Convert a RobotMorphology to a plain dict safe for JSON/pickle."""
    return {
        "name":        morph.name,
        "torso_a":     morph.torso_a,
        "torso_b":     morph.torso_b,
        "torso_c":     morph.torso_c,
        "torso_rgba":  list(morph.torso_rgba),
        "torso_euler": list(morph.torso_euler),
        "spawn_height": morph.spawn_height,
        "foot_radius":  morph.foot_radius,
        "foot_rgba":    list(morph.foot_rgba),
        "body_parts": [
            {
                "parent_leg_idx": bp.parent_leg_idx,
                "a":              bp.a,
                "b":              bp.b,
                "c":              bp.c,
                "euler_deg":      list(bp.euler_deg),
                "rgba":           list(bp.rgba),
            }
            for bp in morph.body_parts
        ],
        "legs": [
            {
                "placement_angle_deg": leg.placement_angle_deg,
                "parent_leg_idx":      leg.parent_leg_idx,
                "parent_joint_idx":    leg.parent_joint_idx,
                "body_part_idx":       leg.body_part_idx,
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
    All new fields default gracefully so old saves still load.
    """
    body_parts = [
        BodyPartDescriptor(
            parent_leg_idx = bp["parent_leg_idx"],
            a              = bp.get("a", bp.get("radius", 0.08)),
            b              = bp.get("b", bp.get("a", bp.get("radius", 0.08))),
            c              = bp.get("c", bp.get("height", 0.03)),
            euler_deg      = tuple(bp.get("euler_deg", [0., 0., 0.])),
            rgba           = tuple(bp.get("rgba", [0.75, 0.75, 0.75, 1.0])),
        )
        for bp in d.get("body_parts", [])
    ]
    legs = [
        LegDescriptor(
            placement_angle_deg = leg["placement_angle_deg"],
            parent_leg_idx      = leg.get("parent_leg_idx"),
            parent_joint_idx    = leg.get("parent_joint_idx"),
            body_part_idx       = leg.get("body_part_idx"),
            joints = [
                JointDescriptor(
                    damping    = j["damping"],
                    kp         = j["kp"],
                    ctrl_range = tuple(j["ctrl_range"]),
                    length     = j["length"],
                    radius     = j["radius"],
                    rgba       = tuple(j["rgba"]),
                    rest_angle = j.get("rest_angle", 0.0),
                )
                for j in leg["joints"]
            ],
        )
        for leg in d["legs"]
    ]
    _old_r = d.get("torso_radius", 0.12)
    return RobotMorphology(
        name         = d["name"],
        legs         = legs,
        body_parts   = body_parts,
        torso_a      = d.get("torso_a", _old_r),
        torso_b      = d.get("torso_b", d.get("torso_a", _old_r)),
        torso_c      = d.get("torso_c", d.get("torso_height", 0.04)),
        torso_rgba   = tuple(d["torso_rgba"]),
        torso_euler  = tuple(d.get("torso_euler", [0.0, 0.0, 0.0])),
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

    def __init__(self, floor_texrepeat: int = 128, photorealistic: bool = False):
        self.floor_texrepeat  = floor_texrepeat
        self.photorealistic   = photorealistic

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
    def _rim_pos(angle_deg: float, a: float, b: float) -> tuple[float, float, float]:
        """XYZ position on the ellipsoidal equatorial rim at the given angle."""
        r = math.radians(angle_deg)
        return (
            round(math.cos(r) * a, 5),
            round(math.sin(r) * b, 5),
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
        skip_foot:        bool = False,
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

            if is_last and not skip_foot:
                # Foot sphere — touch point with the ground.
                # Omitted when a body part attaches at this tip (body part
                # geom replaces the foot as the terminal contact element).
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
    # Secondary body part builder
    # ------------------------------------------------------------------

    def _build_body_part_element(
        self,
        bp:     BodyPartDescriptor,
        bp_idx: int,
        prefix: str,
    ) -> ET.Element:
        """
        Build the XML body element for one secondary body part.

        The body has no joint — it is rigidly welded to the parent leg's
        last end-body.  The cylinder geom doubles as the contact surface,
        replacing the foot sphere of the parent leg.
        """
        rx, ry, rz = bp.euler_deg
        bp_body = ET.Element("body",
            name = f"{prefix}bpart{bp_idx + 1}",
            pos  = "0 0 0",
        )
        if any(abs(v) > 1e-6 for v in (rx, ry, rz)):
            bp_body.set("euler", f"{rx:.4f} {ry:.4f} {rz:.4f}")

        ET.SubElement(bp_body, "geom",
            name     = f"{prefix}bpart{bp_idx + 1}_geom",
            type     = "ellipsoid",
            size     = f"{bp.a} {bp.b} {bp.c}",
            rgba     = self._rgba(bp.rgba),
            condim   = "3",
            friction = "0.7 0.005 0.0001",
        )
        return bp_body

    # ------------------------------------------------------------------
    # Torso + all legs + body parts
    # ------------------------------------------------------------------

    def _build_torso(
        self,
        morph:  RobotMorphology,
        prefix: str  = "",
        x:      float = 0.0,
        y:      float = 0.0,
    ) -> tuple[ET.Element, list[str]]:
        """
        Build the full torso body with all legs (root, branched, body-part-mounted)
        and all secondary body parts.
        Returns (torso_element, all_joint_names_flat).

        Build order
        -----------
        Legs are processed in index order.  When a leg's tip carries a body
        part, the body part element is inserted inline immediately after that
        leg is built — ensuring body_part_elements[N] is always populated
        before any leg with body_part_idx == N is processed.
        Invariant: body_parts[N].parent_leg_idx < any leg with body_part_idx == N.
        """
        torso = ET.Element("body",
            name = f"{prefix}torso",
            pos  = f"{x} {y} {morph.spawn_height}",
        )
        te_rx, te_ry, te_rz = morph.torso_euler
        if any(abs(v) > 1e-6 for v in (te_rx, te_ry, te_rz)):
            torso.set("euler", f"{te_rx:.4f} {te_ry:.4f} {te_rz:.4f}")
        ET.SubElement(torso, "freejoint", name=f"{prefix}root")
        ET.SubElement(torso, "geom",
            name  = f"{prefix}torso_geom",
            type  = "ellipsoid",
            size  = f"{morph.torso_a} {morph.torso_b} {morph.torso_c}",
            rgba  = self._rgba(morph.torso_rgba),
        )

        all_joint_names:     list[str]                   = []
        segment_ends_by_leg: dict[int, list[ET.Element]] = {}
        body_part_elements:  dict[int, ET.Element]       = {}

        # Map leg_idx → (bp_idx, BodyPartDescriptor) for legs that carry a body part
        body_part_at_leg_tip: dict[int, tuple[int, BodyPartDescriptor]] = {
            bp.parent_leg_idx: (bp_idx, bp)
            for bp_idx, bp in enumerate(morph.body_parts)
        }

        # Effective swing-axis angle per leg.
        # Root / body-part legs: effective = placement_angle_deg  (in attachment body's frame)
        # Branched legs:         effective = parent_effective − placement_angle_deg
        effective_angle: dict[int, float] = {}
        for leg_idx, leg in enumerate(morph.legs):
            if leg.parent_leg_idx is None:
                effective_angle[leg_idx] = leg.placement_angle_deg
            else:
                parent_eff = effective_angle[leg.parent_leg_idx]
                effective_angle[leg_idx] = parent_eff - leg.placement_angle_deg

        for leg_idx, leg in enumerate(morph.legs):
            skip_foot = (leg_idx in body_part_at_leg_tip)

            if leg.parent_leg_idx is None and leg.body_part_idx is None:
                # Root leg on the main torso
                rx, ry, rz = self._rim_pos(leg.placement_angle_deg, morph.torso_a, morph.torso_b)
                attachment_body = torso
                attach_pos      = (rx, ry, rz)

            elif leg.body_part_idx is not None:
                # Leg mounted on a secondary body part
                bp      = morph.body_parts[leg.body_part_idx]
                rx, ry, rz = self._rim_pos(leg.placement_angle_deg, bp.a, bp.b)
                attachment_body = body_part_elements[leg.body_part_idx]
                attach_pos      = (rx, ry, rz)

            else:
                # Branched leg: attach to the end-body of a parent joint
                parent_idx      = leg.parent_leg_idx
                joint_idx       = leg.parent_joint_idx if leg.parent_joint_idx is not None else -1
                attachment_body = segment_ends_by_leg[parent_idx][joint_idx]
                attach_pos      = (0.0, 0.0, 0.0)

            leg_elem, jnames, seg_ends = self._build_leg(
                leg, leg_idx, morph, prefix, attach_pos,
                effective_angle = effective_angle[leg_idx],
                skip_foot       = skip_foot,
            )
            attachment_body.append(leg_elem)
            segment_ends_by_leg[leg_idx] = seg_ends
            all_joint_names.extend(jnames)

            # If this leg tip carries a body part, build it inline now so that
            # any subsequent leg with body_part_idx pointing here can find it.
            if skip_foot:
                bp_idx, bp = body_part_at_leg_tip[leg_idx]
                bp_body    = self._build_body_part_element(bp, bp_idx, prefix)
                seg_ends[-1].append(bp_body)
                body_part_elements[bp_idx] = bp_body

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

        asset = ET.SubElement(root, "asset")

        if self.photorealistic:
            # Blue-sky skybox: gradient from deep blue (top) to pale blue (horizon)
            # with random white marks to suggest clouds.
            ET.SubElement(asset, "texture",
                type="skybox", builtin="gradient",
                rgb1="0.18 0.48 0.88", rgb2="0.82 0.91 0.98",
                width="512", height="512")
            # Grass floor: two-tone dark green checker
            ET.SubElement(asset, "texture",
                type="2d", name="floor_tex", builtin="checker",
                rgb1="0.22 0.38 0.09", rgb2="0.18 0.32 0.07",
                width="512", height="512")
            ET.SubElement(asset, "material",
                name="floor_mat", texture="floor_tex",
                texrepeat=f"{self.floor_texrepeat} {self.floor_texrepeat}",
                shininess="0.0", specular="0.0", reflectance="0.0")
        else:
            ET.SubElement(asset, "texture",
                type="2d", name="floor_tex", builtin="checker",
                rgb1="0.3 0.3 0.3", rgb2="0.22 0.22 0.22",
                width="512", height="512")
            ET.SubElement(asset, "material",
                name="floor_mat", texture="floor_tex",
                texrepeat=f"{self.floor_texrepeat} {self.floor_texrepeat}")

        # World
        worldbody = ET.SubElement(root, "worldbody")

        if self.photorealistic:
            # Sun-like directional light from upper-side angle, warm tint
            ET.SubElement(worldbody, "light",
                pos="5 3 8", dir="-0.5 -0.3 -1",
                diffuse="0.95 0.90 0.75", specular="0.3 0.3 0.2",
                directional="true", castshadow="true")
            # Soft fill light from opposite side (sky bounce)
            ET.SubElement(worldbody, "light",
                pos="-3 -2 5", dir="0.3 0.2 -1",
                diffuse="0.35 0.45 0.55", specular="0 0 0",
                directional="true", castshadow="false")
        else:
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
