"""
video_renderer.py
=================
Roll out a trained PPO policy and write a short MP4 of the episode for
the VLM grader.

The video is the only artefact the VLM sees in stage 3 (the reward weights
and the policy weights live on disk but are never sent to the model), so
this file decides what the grader can perceive about behaviour:

  - libx264 ultrafast preset for fast encoding
  - tracking camera that follows the torso along X so the gait stays
    centred for the full 5 s — VLMs are far more sensitive to motion
    against a static reference frame
  - deterministic action sampling (no exploration noise) so the rendered
    video reflects the policy's mean behaviour, not stochastic outliers

Adapted from `proto/Mujoco/video_render.py` (which renders N robots in
parallel side-by-side); here we render one robot at a time but share the
async-encoder pattern so wall-clock time is dominated by physics, not
imageio.

Debug
-----
Run this file directly to:
  1. Train a tiny PPO policy (~3k steps) against the default reward.
  2. Roll it out and write an MP4 to results/_smoke/rollout.mp4.
  3. Print frame count + file size.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

import imageio
import mujoco
import numpy as np


def rollout_to_video(
    policy,
    env,
    save_path:           str,
    fps:                 int  = 20,
    camera_track_torso:  bool = True,
    deterministic:       bool = True,
    max_steps:           Optional[int] = None,
) -> tuple[str, dict]:
    """
    Roll out one episode of `policy` on `env` and save the rendered frames
    to an MP4 at `save_path`.

    Parameters
    ----------
    policy : object with `.predict(obs, deterministic=...) -> (action, _)`.
             Both `stable_baselines3.PPO` and a plain callable wrapped in
             a small adapter satisfy this interface.
    env    : a `RobotControllerEnv` (or any Gym env exposing `render()`
             returning rgb_array). The caller is responsible for `reset()`-ing
             before calling rollout_to_video — we reset() here too.
    save_path : output MP4 path.
    fps                 : output video framerate.
    camera_track_torso  : if True, the env's render camera follows the torso.
                          (Already the default behaviour of RobotControllerEnv.)
    deterministic       : pass-through to policy.predict.
    max_steps           : cap on env steps; defaults to env's truncation horizon.

    Returns
    -------
    (path, info)  with info = {n_frames, terminated, truncated, total_reward}.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Force the env into rgb_array render mode without dropping its other
    # state. RobotControllerEnv tolerates this — render() will lazily build
    # the renderer on first call.
    env.render_mode = "rgb_array"
    if hasattr(env, "_render_camera") and not camera_track_torso:
        # If the caller wants a static camera, hold lookat at world origin.
        env._render_camera.lookat[:] = [0.0, 0.0, 0.2]

    obs, _ = env.reset()

    # Async encoder thread — render() is fast (<10 ms), but imageio's
    # ffmpeg pipe stalls now and then; pulling encoding off the sim thread
    # keeps the rollout deterministic in wall-clock.
    writer = imageio.get_writer(
        save_path,
        fps             = fps,
        codec           = "libx264",
        macro_block_size = 1,
        output_params   = ["-preset", "ultrafast", "-crf", "28"],
    )
    q:        "queue.Queue" = queue.Queue(maxsize=64)
    stop_evt: threading.Event = threading.Event()

    def _encoder_worker():
        while not stop_evt.is_set() or not q.empty():
            try:
                frame = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break
            writer.append_data(frame)

    thread = threading.Thread(target=_encoder_worker, daemon=True)
    thread.start()

    # Enforce static-camera Z so the torso doesn't disappear out the bottom
    # of the frame if the policy makes the robot bounce.
    if hasattr(env, "_render_camera"):
        env._render_camera.lookat[2] = 0.2

    n_frames     = 0
    total_reward = 0.0
    terminated   = False
    truncated    = False
    step_idx     = 0
    cap          = max_steps if max_steps is not None else getattr(env, "_max_steps", 1_000_000)

    try:
        # Capture the initial frame so the very first second of the MP4
        # shows the spawn pose (helpful for the VLM to compare frame 1 vs end).
        frame = env.render()
        if frame is not None:
            q.put(frame.copy())
            n_frames += 1

        while step_idx < cap and not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, _info = env.step(action)
            total_reward += float(r)
            step_idx += 1

            frame = env.render()
            if frame is not None:
                q.put(frame.copy())
                n_frames += 1
    finally:
        stop_evt.set()
        # Sentinel to make sure the worker exits even if the queue is empty
        q.put(None)
        thread.join()
        writer.close()

    return save_path, {
        "n_frames":     n_frames,
        "terminated":   terminated,
        "truncated":    truncated,
        "total_reward": total_reward,
    }


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    print("=" * 60)
    print("  video_renderer.py — debug mode")
    print("=" * 60)

    out_dir  = Path(__file__).resolve().parent / "results" / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(out_dir / "rollout.mp4")
    policy_path = str(out_dir / "smoke_policy.zip")

    # Train a tiny policy against default weights (or reuse if present)
    from reward import RewardWeights
    if not os.path.exists(policy_path):
        print("\n[1] Training a 5k-step policy (default reward) ...\n")
        from ppo_trainer import train_from_scratch
        train_from_scratch(
            reward_weights   = RewardWeights(),
            seed             = 0,
            total_timesteps  = 5_000,
            n_envs           = 1,
            save_path        = policy_path,
            tensorboard_log  = None,
            episode_duration = 2.0,
            n_steps_per_env  = 256,
            batch_size       = 64,
            use_subproc      = False,
        )
        print(f"  trained.  policy → {policy_path}")
    else:
        print(f"\n[1] Reusing existing policy at {policy_path}")

    # Build env + load policy + render
    print("\n[2] Rolling out and writing MP4 ...\n")
    from stable_baselines3 import PPO
    from mujoco_env import RobotControllerEnv

    env = RobotControllerEnv(
        reward_weights   = RewardWeights(),
        seed             = 0,
        episode_duration = 5.0,
        render_mode      = "rgb_array",
        render_width     = 320,
        render_height    = 240,
    )
    policy = PPO.load(policy_path)
    path, info = rollout_to_video(policy, env, save_path, fps=20)
    env.close()

    print(f"  saved      : {path}")
    print(f"  size       : {os.path.getsize(path)} B")
    print(f"  frames     : {info['n_frames']}")
    print(f"  terminated : {info['terminated']}  truncated : {info['truncated']}")
    print(f"  total_rwd  : {info['total_reward']:+.3f}")

    print("\nAll video_renderer.py checks passed.")
