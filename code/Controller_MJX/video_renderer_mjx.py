"""
video_renderer_mjx.py
=====================
Render a short MP4 from a trained Flax policy + MJX environment.

Flow
----
1. Run one episode using the single-env JAX functions from mujoco_env_mjx.py
   (CPU-side JAX; reset_fn / step_fn are JIT-compiled but run on CPU).
2. At each step, call `mjx.get_data(mj_model, mjx_state.data)` to transfer
   the MJX physics state back to a regular `mujoco.MjData` on the CPU.
3. Render two camera views side-by-side with `mujoco.Renderer` — the same
   two-camera layout used by the original `mujoco_env.py`.
4. Write frames to an MP4 with imageio + libx264 (async encoder thread so
   physics and encoding overlap).

The rendering is intentionally CPU-only and happens AFTER training.  MJX
physics is only invoked during training for speed; here correctness and visual
quality matter more than throughput.

Public API
----------
    rollout_to_video_mjx(
        params, cfg, mj_model,
        save_path,
        fps=20, render_width=192, render_height=192,
        cam1_azimuth=90, cam1_elevation=-5, cam1_distance=4,
        cam1_lookat_z=0.1,
        cam2_azimuth=60, cam2_elevation=-30, cam2_distance=4,
        camera_track_torso=False,
        seed=0,
        policy_arch=(256, 256),
    ) → (path, info_dict)

    build_policy_fn(params, net) → Callable[[obs], action]

Debug
-----
Run this file directly for a smoke test (trains a tiny policy then renders).
"""

from __future__ import annotations

import os
# Headless rendering on GPU servers (no X11 display).
# EGL uses the NVIDIA driver directly; osmesa is the CPU fallback.
# Set MUJOCO_GL before mujoco is imported — setdefault so the caller can override.
os.environ.setdefault("MUJOCO_GL", "egl")

import queue
import threading
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from mujoco_env_mjx import (
    MJXEnvConfig,
    EnvState,
    build_env_config,
    make_env_fns,
    _pick_mjx_device,
)
from ppo_trainer_mjx import ActorCritic, init_actor_critic
from reward import RewardWeights


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

def build_policy_fn(
    params:      Any,
    net:         ActorCritic,
    deterministic: bool = True,
) -> Callable[[Any], Any]:
    """
    Return a callable  obs → action  using the Flax actor-critic.

    With `deterministic=True` (default) the mean of the Gaussian policy is
    returned — no sampling noise.  This produces the cleanest video for the
    VLM grader, mirroring the convention in `video_renderer.py`.

    Parameters
    ----------
    params      : Flax param dict from train_from_scratch_mjx / train_warm_start_mjx.
    net         : ActorCritic instance (must match the architecture used during training).
    deterministic: if True returns mean action; if False samples from the policy.
    """
    if deterministic:
        @jax.jit
        def _policy(obs: Any) -> Any:
            mean, _log_std, _value = net.apply(params, obs)
            return jnp.clip(mean, -1.0, 1.0)
    else:
        @jax.jit
        def _policy(obs: Any) -> Any:
            key = jax.random.PRNGKey(0)   # fixed key — only for visualization
            mean, log_std, _value = net.apply(params, obs)
            std = jnp.exp(log_std)
            action = mean + std * jax.random.normal(key, mean.shape)
            return jnp.clip(action, -1.0, 1.0)

    return _policy


# ---------------------------------------------------------------------------
# MJX → CPU data transfer
# ---------------------------------------------------------------------------

def mjx_state_to_mj_data(
    mj_model: mujoco.MjModel,
    state:    EnvState,
) -> mujoco.MjData:
    """
    Transfer an MJX EnvState's physics data to a regular `mujoco.MjData`.

    `mjx.get_data` copies all kinematic arrays (qpos, qvel, geom_xpos, …)
    from JAX device arrays to CPU numpy.  The returned MjData is fully valid
    for `mujoco.Renderer.update_scene()`.
    """
    return mjx.get_data(mj_model, state.data)


# ---------------------------------------------------------------------------
# Two-camera renderer (mirrors mujoco_env.py)
# ---------------------------------------------------------------------------

def _make_cameras(
    cam1_azimuth:   float,
    cam1_elevation: float,
    cam1_distance:  float,
    cam1_lookat_z:  float,
    cam2_azimuth:   float,
    cam2_elevation: float,
    cam2_distance:  float,
) -> Tuple[mujoco.MjvCamera, mujoco.MjvCamera]:
    cam1 = mujoco.MjvCamera()
    cam1.azimuth   = cam1_azimuth
    cam1.elevation = cam1_elevation
    cam1.distance  = cam1_distance
    cam1.lookat[:] = [0.0, 0.0, cam1_lookat_z]

    cam2 = mujoco.MjvCamera()
    cam2.azimuth   = cam2_azimuth
    cam2.elevation = cam2_elevation
    cam2.distance  = cam2_distance
    cam2.lookat[:] = [0.0, 0.0, 0.2]

    return cam1, cam2


def _render_frame(
    renderer1:           mujoco.Renderer,
    renderer2:           mujoco.Renderer,
    mj_data:             mujoco.MjData,
    cam1:                mujoco.MjvCamera,
    cam2:                mujoco.MjvCamera,
    camera_track_torso:  bool,
) -> np.ndarray:
    """Render two camera views and return them side-by-side as (H, 2W, 3)."""
    torso_x = float(mj_data.qpos[0])

    if camera_track_torso:
        cam1.lookat[0] = torso_x
        cam2.lookat[0] = torso_x

    renderer1.update_scene(mj_data, camera=cam1)
    frame1 = renderer1.render()

    renderer2.update_scene(mj_data, camera=cam2)
    frame2 = renderer2.render()

    return np.concatenate([frame1, frame2], axis=1)


# ---------------------------------------------------------------------------
# Main rollout + render function
# ---------------------------------------------------------------------------

def rollout_to_video_mjx(
    params:      Any,
    cfg:         MJXEnvConfig,
    mj_model:    mujoco.MjModel,
    save_path:   str,
    # Video settings
    fps:               int   = 20,
    render_width:      int   = 192,
    render_height:     int   = 192,
    # Camera 1 — ground-level side view
    cam1_azimuth:      float = 90.0,
    cam1_elevation:    float = -5.0,
    cam1_distance:     float = 4.0,
    cam1_lookat_z:     float = 0.1,
    # Camera 2 — diagonal front view
    cam2_azimuth:      float = 60.0,
    cam2_elevation:    float = -30.0,
    cam2_distance:     float = 4.0,
    # Behaviour settings
    camera_track_torso: bool = False,
    deterministic:      bool = True,
    seed:               int  = 0,
    policy_arch:        tuple = (256, 256),
    max_steps:          Optional[int] = None,
) -> Tuple[str, dict]:
    """
    Roll out the Flax policy for one episode and write an MP4.

    Parameters
    ----------
    params      : Flax param dict (from train_from_scratch_mjx / warm_start).
    cfg         : MJXEnvConfig built for this individual (contains morphology,
                  reward weights, episode length, …).
    mj_model    : CPU-side mujoco.MjModel for the same morphology — used both
                  for mjx.get_data() transfer and for mujoco.Renderer.
    save_path   : output MP4 path (parent directories are created).
    fps         : output video fps.
    render_width/height : per-camera resolution; total width = 2 × render_width.
    cam1_*      : parameters for the ground-level side-view camera.
    cam2_*      : parameters for the diagonal front-view camera.
    camera_track_torso : if True, both cameras follow the torso along X.
    deterministic : use mean action (True) or sample from the policy (False).
    seed        : PRNG seed for the reset.
    policy_arch : must match the architecture used during training.
    max_steps   : cap episode at this many steps (default: cfg.max_steps).

    Returns
    -------
    (save_path, info)
        info = {n_frames, terminated, truncated, total_reward, n_steps}
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    cap = max_steps if max_steps is not None else cfg.max_steps

    # ---- Build policy fn ---------------------------------------------------
    net = ActorCritic(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.n_joints,
        hidden=tuple(policy_arch),
    )
    policy_fn = build_policy_fn(params, net, deterministic=deterministic)

    # ---- Build single-env JAX fns ------------------------------------------
    reset_fn, step_fn, _ = make_env_fns(cfg)

    # ---- Build renderers (CPU, not JAX) ------------------------------------
    renderer1 = mujoco.Renderer(mj_model, height=render_height, width=render_width)
    renderer2 = mujoco.Renderer(mj_model, height=render_height, width=render_width)
    cam1, cam2 = _make_cameras(
        cam1_azimuth, cam1_elevation, cam1_distance, cam1_lookat_z,
        cam2_azimuth, cam2_elevation, cam2_distance,
    )

    # ---- Async encoder thread (mirrors video_renderer.py) ------------------
    writer = imageio.get_writer(
        save_path,
        fps              = fps,
        codec            = "libx264",
        macro_block_size = 1,
        output_params    = ["-preset", "ultrafast", "-crf", "28"],
    )
    frame_q:  "queue.Queue" = queue.Queue(maxsize=64)
    stop_evt: threading.Event = threading.Event()

    def _encoder_worker():
        while not stop_evt.is_set() or not frame_q.empty():
            try:
                frame = frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break
            writer.append_data(frame)

    enc_thread = threading.Thread(target=_encoder_worker, daemon=True)
    enc_thread.start()

    # ---- Episode rollout ---------------------------------------------------
    n_frames     = 0
    total_reward = 0.0
    terminated   = False
    truncated    = False

    try:
        key = jax.random.PRNGKey(seed)
        state, obs = reset_fn(key)

        # Render initial frame (spawn pose)
        mj_data = mjx_state_to_mj_data(mj_model, state)
        frame = _render_frame(renderer1, renderer2, mj_data, cam1, cam2,
                              camera_track_torso)
        frame_q.put(frame.copy())
        n_frames += 1

        for step_i in range(cap):
            action = policy_fn(obs)
            state, obs, reward, done = step_fn(state, action)
            total_reward += float(reward)

            mj_data = mjx_state_to_mj_data(mj_model, state)
            frame = _render_frame(renderer1, renderer2, mj_data, cam1, cam2,
                                  camera_track_torso)
            frame_q.put(frame.copy())
            n_frames += 1

            if bool(done):
                terminated = bool(state.fell)
                truncated  = not terminated
                break

    finally:
        stop_evt.set()
        frame_q.put(None)    # sentinel to drain the worker
        enc_thread.join()
        writer.close()
        renderer1.close()
        renderer2.close()

    return save_path, {
        "n_frames":     n_frames,
        "terminated":   terminated,
        "truncated":    truncated,
        "total_reward": total_reward,
        "n_steps":      int(state.step_idx),
    }


# ---------------------------------------------------------------------------
# Convenience: render from ExperimentConfig camera settings
# ---------------------------------------------------------------------------

def rollout_to_video_from_exp_config(
    params:    Any,
    cfg:       MJXEnvConfig,
    mj_model:  mujoco.MjModel,
    save_path: str,
    exp_cfg,           # ExperimentConfig instance from config.py
    seed:      int = 0,
    policy_arch: tuple = (256, 256),
    max_steps: Optional[int] = None,
) -> Tuple[str, dict]:
    """
    Thin wrapper that unpacks camera/video settings from an ExperimentConfig.
    Keeps experiment.py clean.
    """
    return rollout_to_video_mjx(
        params             = params,
        cfg                = cfg,
        mj_model           = mj_model,
        save_path          = save_path,
        fps                = exp_cfg.video_fps,
        render_width       = exp_cfg.render_width,
        render_height      = exp_cfg.render_height,
        cam1_azimuth       = exp_cfg.cam1_azimuth,
        cam1_elevation     = exp_cfg.cam1_elevation,
        cam1_distance      = exp_cfg.cam1_distance,
        cam1_lookat_z      = exp_cfg.cam1_lookat_z,
        cam2_azimuth       = exp_cfg.cam2_azimuth,
        cam2_elevation     = exp_cfg.cam2_elevation,
        cam2_distance      = exp_cfg.cam2_distance,
        camera_track_torso = exp_cfg.camera_track_torso,
        seed               = seed,
        policy_arch        = policy_arch,
        max_steps          = max_steps,
    )


# ---------------------------------------------------------------------------
# Debug / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    print("=" * 60)
    print("  video_renderer_mjx.py — debug mode")
    print("=" * 60)

    dev = _pick_mjx_device()
    jax.config.update("jax_default_device", dev)
    print(f"\n  JAX device : {dev}")

    from controller_morph import build_model
    from ppo_trainer_mjx import train_from_scratch_mjx

    print("\n[1] Build env config\n")
    mj_model, _morph = build_model()
    cfg = build_env_config(episode_duration=2.0)
    print(f"  obs_dim={cfg.obs_dim}  n_joints={cfg.n_joints}  max_steps={cfg.max_steps}")

    print("\n[2] Train a tiny policy (2 envs × 16 steps)\n")
    params, fitness = train_from_scratch_mjx(
        cfg           = cfg,
        seed          = 0,
        total_steps   = 2 * 16,
        n_envs        = 2,
        rollout_len   = 16,
        policy_arch   = (64, 64),
        episode_duration = 2.0,
        fitness_episodes = 1,
        verbose       = True,
    )
    print(f"  fitness = {fitness:.4f}")

    print("\n[3] Rollout to MP4\n")
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "smoke.mp4")
        path, info = rollout_to_video_mjx(
            params       = params,
            cfg          = cfg,
            mj_model     = mj_model,
            save_path    = out_path,
            fps          = 20,
            render_width = 64,    # tiny for smoke test speed
            render_height = 64,
            policy_arch  = (64, 64),
            seed         = 0,
            max_steps    = 10,    # short episode for speed
        )
        size = os.path.getsize(path)
        print(f"  saved      : {path}")
        print(f"  size       : {size} B")
        print(f"  frames     : {info['n_frames']}")
        print(f"  total_rwd  : {info['total_reward']:+.3f}")
        print(f"  terminated : {info['terminated']}  truncated={info['truncated']}")
        assert size > 0,             "output file is empty"
        assert info["n_frames"] > 0, "no frames written"
        assert info["n_frames"] <= 11, f"expected ≤11 frames, got {info['n_frames']}"
        print("  Rollout to MP4: OK")

    print("\nAll video_renderer_mjx.py checks passed.")
