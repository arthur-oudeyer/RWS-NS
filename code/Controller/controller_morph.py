"""
controller_morph.py
===================
Static morphology used by every individual in the controller study.

Stage 3 freezes the body to QUADRIPOD (4 legs at 0/90/180/270°, 1 hinge each).
The evolutionary search varies the reward weight vector (and, indirectly, the
PPO-trained policy); the body is identical for every individual so any change
in behaviour can be attributed to the controller, not the morphology.

This module is a thin wrapper around `Morphology/morphology.py` so we do not
duplicate MJCF generation logic.

Usage
-----
    from controller_morph import build_model, STATIC_MORPH

    model = build_model()    # ready-to-step mujoco.MjModel
    print(model.nq, model.nu)

Debug
-----
Run this file directly to print the model's nq / nu and write a single
offscreen PNG render of the static body to results/_smoke/static_morph.png.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

import config as cfg
import morphology

QUADRIPOD            = morphology.QUADRIPOD
MorphologyManager    = morphology.MorphologyManager
RobotMorphology      = morphology.RobotMorphology
compute_spawn_height = morphology.compute_spawn_height


# ---------------------------------------------------------------------------
# The frozen morphology used by every individual
# ---------------------------------------------------------------------------

STATIC_MORPH: RobotMorphology = QUADRIPOD if cfg.ExperimentConfig.morphology is None else morphology.get_preconfigured_morph(cfg.ExperimentConfig.morphology)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(
    morph:             RobotMorphology = STATIC_MORPH,
    floor_clearance:   float = 0.0,
    photorealistic:    bool  = cfg.ExperimentConfig.photorealistic,
    origin_tile_rgba:  tuple = cfg.ExperimentConfig.origin_tile_rgba,
    origin_tile_size:  float = cfg.ExperimentConfig.origin_tile_size,
):
    """
    Build a ready-to-step `mujoco.MjModel` for the static morphology.

    The morphology's `spawn_height` is overwritten with the value computed
    by `compute_spawn_height` so the robot is dropped just above the floor
    and at least one foot rests on it. We mutate a deep copy so the module-
    level `STATIC_MORPH` is never modified.

    Parameters
    ----------
    morph             : the static morphology (defaults to STATIC_MORPH).
    floor_clearance   : extra clearance above z=0 for the torso/body parts.
    photorealistic    : grass + sky textures (mirrors Morphology's flag).
    origin_tile_rgba  : RGBA colour of the spawn-position marker tile.
    origin_tile_size  : half-extent (m) of the marker tile.
    """
    import copy
    m = copy.deepcopy(morph)
    m.spawn_height = compute_spawn_height(m, floor_clearance=floor_clearance)
    manager = MorphologyManager(
        floor_texrepeat  = 256,
        photorealistic   = photorealistic,
        origin_tile_rgba = origin_tile_rgba,
        origin_tile_size = origin_tile_size,
    )
    return manager.get_model(m), m


def get_static_morph() -> RobotMorphology:
    """Return the static morphology with `spawn_height` set for stepping."""
    import copy
    m = copy.deepcopy(STATIC_MORPH)
    m.spawn_height = compute_spawn_height(m, floor_clearance=0.0)
    return m


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("  controller_morph.py — debug mode")
    print("=" * 60)

    model, morph = build_model()
    print(f"\n  STATIC_MORPH      : {morph.name}")
    print(f"  legs              : {len(morph.legs)}")
    print(f"  joints (n_joints) : {morph.n_joints}")
    print(f"  spawn_height      : {morph.spawn_height:.4f} m")
    print(f"  model.nq          : {model.nq}")
    print(f"  model.nu          : {model.nu}")
    print(f"  model.nv          : {model.nv}")

    # Sanity checks for the QUADRIPOD baseline
    assert model.nu == 4,    f"expected 4 actuators, got {model.nu}"
    assert model.nq == 7 + 4, f"expected nq=11, got {model.nq}"
    assert model.nv == 6 + 4, f"expected nv=10, got {model.nv}"
    print("\n  Dimensions match QUADRIPOD (4 actuators).")

    # Single offscreen render so we can eyeball the body
    try:
        import mujoco
        import imageio
        out_dir  = Path(__file__).resolve().parent / "results" / "_smoke"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "static_morph.png"

        renderer = mujoco.Renderer(model, height=192, width=192)
        data     = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 45, -25, 1.6
        cam.lookat[:] = [0.0, 0.0, 0.2]

        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        imageio.imwrite(str(out_path), frame)
        renderer.close()

        print(f"\n  Saved render → {out_path}")
    except Exception as e:
        print(f"\n  (skipped offscreen render: {e})")

    print("\nAll controller_morph checks passed.")
