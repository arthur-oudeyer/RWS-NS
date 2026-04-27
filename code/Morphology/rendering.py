"""
rendering.py
============
Static offscreen rendering of a RobotMorphology to a PIL Image.

No physics stepping — mj_forward is called once to place the robot in
its rest pose (all joints at 0). This is sufficient to evaluate body
shape visually via CLIP.

Multiple camera angles are supported. When more than one angle is
requested, the images are arranged in a horizontal strip so CLIP
receives a single composite image that shows the robot from several
viewpoints simultaneously.

Usage
-----
    from rendering import MorphologyRenderer, RenderConfig

    config   = RenderConfig(width=512, height=512, debug=True)
    renderer = MorphologyRenderer(config)
    image    = renderer.render(morph)                     # PIL.Image
    image    = renderer.render(morph, save_path="out.png")

Debug mode
----------
    Set RenderConfig.debug = True to:
      - print model info (nq, nu, body count) after building
      - save every render to  debug_renders/  with a timestamped name
      - display the image with the system viewer (PIL Image.show)
    Or pass debug=True to an individual render() call to override.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from morphology import MorphologyManager, RobotMorphology, NewMorph, MutateMorphology, compute_spawn_height

# Lazy-import mujoco so the module can be imported even if mujoco is absent
# (e.g. for reading configs or prompt sets without rendering).
try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Camera descriptor
# ---------------------------------------------------------------------------

@dataclass
class CameraView:
    """One camera angle for offscreen rendering."""
    azimuth:   float = 45.0    # degrees: 0 = front, 90 = right, 180 = back
    elevation: float = -20.0   # degrees: negative = looking down
    distance:  float = 1.8     # metres from lookat point
    lookat:    tuple = (0.0, 0.0, 0.3)  # world-space focus point


# ---------------------------------------------------------------------------
# RenderConfig
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    """
    All parameters that control how morphologies are rendered.

    width, height   : pixel dimensions of each camera view.
    camera_views    : list of CameraView.  Multiple views are tiled
                      horizontally into a single composite image.
    debug           : global debug flag.  Can be overridden per render() call.
    debug_dir       : directory where debug renders are saved.
    """
    width:           int               = 512
    height:          int               = 512
    camera_views:    list[CameraView]  = field(default_factory=lambda: [
        CameraView(azimuth=0,  elevation=5, distance=1.8),
        CameraView(azimuth=45, elevation=-50, distance=1.8),
    ])
    debug:           bool              = False
    debug_dir:       str               = "debug_renders"
    floor_clearance: float             = 0.05   # metres of clearance above z=0
    photorealistic:  bool              = False  # grass floor + blue-sky skybox


# Sensible default used in __main__ and quick tests
DEFAULT_CONFIG = RenderConfig()


# ---------------------------------------------------------------------------
# MorphologyRenderer
# ---------------------------------------------------------------------------

class MorphologyRenderer:
    """
    Renders a RobotMorphology to a PIL Image using MuJoCo offscreen rendering.

    A single MorphologyManager and a single set of MuJoCo renderers are
    reused across calls to avoid repeated model rebuilding when iterating
    over a population.  Call rebuild(morph) explicitly when the morphology
    changes, or just call render() which rebuilds automatically.
    """

    def __init__(self, config: RenderConfig = DEFAULT_CONFIG):
        if not _MUJOCO_AVAILABLE:
            raise ImportError("mujoco is required for MorphologyRenderer.")
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow (PIL) is required for MorphologyRenderer.")

        self.config   = config
        self._manager = MorphologyManager(photorealistic=config.photorealistic)
        self._renderers: list[mujoco.Renderer] = []
        self._model:  Optional[mujoco.MjModel] = None
        self._data:   Optional[mujoco.MjData]  = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self, morph: RobotMorphology, debug: bool) -> None:
        """Compile the MJCF model and initialise one renderer per camera view."""
        # Close any previous renderers
        for r in self._renderers:
            r.close()
        self._renderers.clear()

        self._model = self._manager.get_model(morph)
        self._data  = mujoco.MjData(self._model)

        # Apply rest pose: set each joint to its rest_angle.
        #
        # We cannot assume that the flat iteration order [leg0_j0, leg1_j0, ...]
        # matches MuJoCo's qpos layout.  MuJoCo assigns qpos indices via a
        # depth-first traversal of the XML body tree: when a leg is branched
        # from another leg, its joint appears *inside* the parent leg's
        # subtree and therefore gets a lower qpos index than the remaining
        # root legs — regardless of its position in morph.legs.
        #
        # Fix: build a {joint_name → rest_angle} lookup from the morphology,
        # then use mj_id2name + jnt_qposadr to find each joint's actual
        # qpos address in the compiled model.
        joint_rest = {}
        for leg_idx, leg in enumerate(morph.legs):
            for j_idx, jd in enumerate(leg.joints):
                if len(leg.joints) == 1:
                    jname = f"hip{leg_idx + 1}"
                else:
                    jname = f"leg{leg_idx + 1}_j{j_idx + 1}"
                joint_rest[jname] = jd.rest_angle

        applied = {}
        for j_id in range(self._model.njnt):
            jname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if jname in joint_rest:
                qadr = self._model.jnt_qposadr[j_id]
                self._data.qpos[qadr] = joint_rest[jname]
                applied[jname] = (qadr, joint_rest[jname])

        mujoco.mj_forward(self._model, self._data)

        # Empirical floor snap: measure the actual lowest geom surface after
        # physics is resolved, then translate the torso so it sits exactly at z=0.
        # This corrects for cases where the geometric formula in compute_spawn_height
        # is inaccurate (tilted torso, near-horizontal legs, branching effects).
        min_bottom = self._find_min_geom_bottom()
        if abs(min_bottom) > 1e-4:
            root_id      = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "root")
            torso_z_adr  = self._model.jnt_qposadr[root_id] + 2
            self._data.qpos[torso_z_adr] -= min_bottom
            mujoco.mj_forward(self._model, self._data)

        for _ in self.config.camera_views:
            self._renderers.append(
                mujoco.Renderer(self._model,
                                height=self.config.height,
                                width=self.config.width)
            )

        if debug:
            print(f"  [render] model built: "
                  f"nq={self._model.nq}  nu={self._model.nu}  "
                  f"nbody={self._model.nbody}  "
                  f"views={len(self.config.camera_views)}")
            print(f"  [render] rest angles by joint (qpos_addr: value):")
            for jname, (qadr, val) in sorted(applied.items(), key=lambda x: x[1][0]):
                print(f"    qpos[{qadr}]  {jname:<20}  {val:+.3f} rad")
            print(f"  [render] floor snap: min_bottom={min_bottom:+.4f} m")

    def _find_min_geom_bottom(self) -> float:
        """
        Return the world-Z of the lowest geom surface (excluding the floor plane).

        For each non-plane geom:
          - sphere:           bottom = center_z - radius
          - capsule/cylinder: find the lower endpoint, then subtract radius
          - other:            conservative center_z - max(size)

        The result is used to shift qpos[2] so the lowest surface sits at z=0.
        A positive return value means the robot is floating above the ground;
        a negative value means it has already clipped through.
        """
        min_z = float('inf')
        for geom_id in range(self._model.ngeom):
            gtype = self._model.geom_type[geom_id]
            if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            center = self._data.geom_xpos[geom_id]
            size   = self._model.geom_size[geom_id]
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                bottom = center[2] - size[0]
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # Hemispherical ends: lowest point = lower endpoint - radius.
                rot    = self._data.geom_xmat[geom_id].reshape(3, 3)
                axis_z = rot[2, 2]
                p1_z   = center[2] + axis_z * size[1]
                p2_z   = center[2] - axis_z * size[1]
                bottom = min(p1_z, p2_z) - size[0]
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # Flat ends: lowest point = lower endpoint face center.
                rot    = self._data.geom_xmat[geom_id].reshape(3, 3)
                axis_z = rot[2, 2]
                p1_z   = center[2] + axis_z * size[1]
                p2_z   = center[2] - axis_z * size[1]
                bottom = min(p1_z, p2_z)
            else:
                bottom = center[2] - float(np.max(size[:3]))
            if bottom < min_z:
                min_z = bottom
        return min_z if min_z != float('inf') else 0.0

    def _make_camera(self, view: CameraView) -> "mujoco.MjvCamera":
        cam           = mujoco.MjvCamera()
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth   = view.azimuth
        cam.elevation = view.elevation
        cam.distance  = view.distance
        cam.lookat[:] = list(view.lookat)
        return cam

    def _frames_to_image(self, frames: list[np.ndarray]) -> "PILImage.Image":
        """Tile frames horizontally into one PIL Image."""
        pil_frames = [PILImage.fromarray(f, mode="RGB") for f in frames]
        if len(pil_frames) == 1:
            return pil_frames[0]
        total_w = sum(im.width for im in pil_frames)
        max_h   = max(im.height for im in pil_frames)
        strip   = PILImage.new("RGB", (total_w, max_h))
        x_off   = 0
        for im in pil_frames:
            strip.paste(im, (x_off, 0))
            x_off += im.width
        return strip

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        morph:     RobotMorphology,
        save_path: Optional[str] = None,
        debug:     Optional[bool] = None,
    ) -> "PILImage.Image":
        """
        Render morph and return a PIL Image.

        Parameters
        ----------
        morph      : morphology to render.
        save_path  : if given, the image is saved to this path.
        debug      : overrides RenderConfig.debug for this call if not None.

        Returns
        -------
        PIL.Image in RGB mode.  Width = config.width * len(camera_views).
        """
        dbg = self.config.debug if debug is None else debug

        if dbg:
            enc = morph.encoding()
            print(f"\n  [render] morphology: {morph.name}  "
                  f"legs={enc['n_legs']}  "
                  f"sym={enc['symmetry_score']:.3f}")

        # Auto-adjust spawn height so no part clips the floor.
        import copy as _copy
        morph = _copy.copy(morph)
        morph.spawn_height = compute_spawn_height(morph, self.config.floor_clearance)
        if dbg:
            print(f"  [render] spawn_height={morph.spawn_height:.4f} m  "
                  f"(floor_clearance={self.config.floor_clearance} m)")

        self._build(morph, dbg)

        frames = []
        for renderer, view in zip(self._renderers, self.config.camera_views):
            cam = self._make_camera(view)
            renderer.update_scene(self._data, camera=cam)
            frames.append(renderer.render().copy())

        image = self._frames_to_image(frames)

        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)
            if dbg:
                print(f"  [render] saved → {save_path}")

        # Debug save + display
        if dbg:
            debug_dir = Path(self.config.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dbg_path  = debug_dir / f"{morph.name}_{ts}.png"
            image.save(dbg_path)
            print(f"  [render] debug save → {dbg_path}")
            print(f"  [render] image size: {image.size}")
            image.show()

        return image

    def close(self):
        """Release MuJoCo renderer resources."""
        for r in self._renderers:
            r.close()
        self._renderers.clear()


# ---------------------------------------------------------------------------
# Debug — run directly to test rendering
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from morphology import QUADRIPOD, TRIPOD, HEXAPOD, RobotMorphology, LegDescriptor, JointDescriptor

    print("=" * 60)
    print("  rendering.py — debug mode")
    print("=" * 60)

    if not _MUJOCO_AVAILABLE:
        print("ERROR: mujoco is not installed.")
        sys.exit(1)

    config   = RenderConfig(width=256, height=256, debug=True)
    renderer = MorphologyRenderer(config)

    # --- 0. Render standard morphologies ---
    print("\n[0] Rendering NewMorph()\n")
    for morph in (NewMorph(name="new_morph") for _ in range(5)):
        morph = MutateMorphology(morph, add_remove_prob=0.5, allow_branching=True, branching_prob=0.5)
        img = renderer.render(morph)
        print(f"  → {morph.name}: image {img.size}")

    # --- 1. Render standard morphologies ---
    print("\n[1] Rendering TRIPOD, QUADRIPOD, HEXAPOD\n")
    for morph in (TRIPOD, QUADRIPOD, HEXAPOD):
        img = renderer.render(morph)
        print(f"  → {morph.name}: image {img.size}")

    # --- 2. Render a branched morphology ---
    print("\n[2] Rendering branched morphology\n")
    branched = RobotMorphology(
        name="branched_test",
        legs=[
            LegDescriptor(  0.0, [JointDescriptor(length=0.30, rgba=(0.9, 0.1, 0.1, 1.0), rest_angle=-0.3)]),
            LegDescriptor( 90.0, [JointDescriptor(length=0.25, rgba=(0.1, 0.9, 0.1, 1.0), rest_angle=-0.3)]),
            LegDescriptor(180.0, [JointDescriptor(length=0.25, rgba=(0.1, 0.1, 0.9, 1.0), rest_angle=-0.3)]),
            LegDescriptor(270.0, [JointDescriptor(length=0.25, rgba=(0.9, 0.9, 0.1, 1.0), rest_angle=-0.3)]),
            LegDescriptor(0.0, [JointDescriptor(length=0.15, rest_angle=0.3)], parent_leg_idx=0, parent_joint_idx=0),
            LegDescriptor(0.0, [JointDescriptor(length=0.15, rest_angle=0.3)], parent_leg_idx=1, parent_joint_idx=0),
            LegDescriptor(0.0, [JointDescriptor(length=0.15, rest_angle=0.3)], parent_leg_idx=2, parent_joint_idx=0),
            LegDescriptor(0.0, [JointDescriptor(length=0.15, rest_angle=0.3)], parent_leg_idx=3, parent_joint_idx=0),
        ],
    )
    img = renderer.render(branched)
    print(f"  → {branched.name}: image {img.size}")

    # --- 3. Custom camera angles ---
    print("\n[3] Custom single-view render (top-down)\n")
    top_down_cfg = RenderConfig(
        width=512, height=512, debug=True,
        camera_views=[CameraView(azimuth=0, elevation=-90, distance=2.0)],
    )
    top_renderer = MorphologyRenderer(top_down_cfg)
    img = top_renderer.render(HEXAPOD)
    print(f"  → hexapod top-down: {img.size}")

    renderer.close()
    top_renderer.close()
    print("\nAll render tests done.")
