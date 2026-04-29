"""
_smoke_warm_start.py
====================
End-to-end smoke test for the (mutate → warm-start → render → score)
inner-loop pipeline on a SINGLE individual. This is step 8 of the
build order — once it passes, the per-individual machinery is fully
validated and we can build evolution / archive / experiment on top.

What it does
------------
1. Train a "parent" PPO policy from scratch on the default RewardWeights
   for ~10k steps. (Reuses results/_smoke/smoke_policy_parent.zip if it
   already exists.)
2. Sample one mutated child reward weight vector with σ = 0.3.
3. Warm-start PPO from the parent against the child reward for ~5k steps.
4. Render an MP4 of the child policy.
5. Optionally score it with Gemini (skipped if --no-network).

Usage
-----
    python3 _smoke_warm_start.py            # full pipeline + Gemini call
    python3 _smoke_warm_start.py --no-network   # skip the Gemini step
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from reward import RewardWeights, mutate_weights
from ppo_trainer import train_from_scratch, train_warm_start
from video_renderer import rollout_to_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-network", action="store_true",
                        help="Skip the Gemini scoring step.")
    parser.add_argument("--parent-steps", type=int, default=10_000)
    parser.add_argument("--warm-steps",   type=int, default=5_000)
    parser.add_argument("--episode-duration", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results" / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    parent_zip = out_dir / "smoke_policy_parent.zip"
    child_zip  = out_dir / "smoke_policy_child.zip"
    parent_mp4 = out_dir / "rollout_parent.mp4"
    child_mp4  = out_dir / "rollout_child.mp4"

    print("=" * 60)
    print("  _smoke_warm_start.py — full pipeline smoke test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Parent policy
    # ------------------------------------------------------------------
    parent_rw = RewardWeights()
    print(f"\n[1] Parent reward weights: {parent_rw.to_dict()}")
    if parent_zip.exists():
        print(f"   Reusing parent policy at {parent_zip}")
    else:
        print(f"   Training parent for {args.parent_steps:,} steps ...")
        t0 = time.perf_counter()
        train_from_scratch(
            reward_weights   = parent_rw,
            seed             = 0,
            total_timesteps  = args.parent_steps,
            n_envs           = 2,
            save_path        = str(parent_zip),
            tensorboard_log  = str(out_dir / "tb"),
            tb_run_name      = "smoke_parent",
            episode_duration = args.episode_duration,
            n_steps_per_env  = 512,
            batch_size       = 128,
            use_subproc      = False,
            verbose          = 0,
        )
        print(f"   parent trained in {time.perf_counter() - t0:.1f}s "
              f"→ {parent_zip} ({parent_zip.stat().st_size//1024} KB)")

    # ------------------------------------------------------------------
    # 2. Mutated child weights
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    child_rw = mutate_weights(parent_rw, sigma=0.3, rng=rng)
    print(f"\n[2] Child reward weights : {child_rw.to_dict()}")
    diffs = (child_rw.to_vector() - parent_rw.to_vector()) / parent_rw.to_vector()
    print(f"   relative deltas       : {diffs.round(3).tolist()}")
    assert not np.allclose(child_rw.to_vector(), parent_rw.to_vector()), \
        "child should differ from parent"

    # ------------------------------------------------------------------
    # 3. Warm-start the child from the parent
    # ------------------------------------------------------------------
    print(f"\n[3] Warm-starting child PPO for {args.warm_steps:,} steps ...")
    t0 = time.perf_counter()
    train_warm_start(
        reward_weights     = child_rw,
        parent_policy_path = str(parent_zip),
        seed               = 1,
        n_warm_steps       = args.warm_steps,
        n_envs             = 2,
        save_path          = str(child_zip),
        tensorboard_log    = str(out_dir / "tb"),
        tb_run_name        = "smoke_child",
        episode_duration   = args.episode_duration,
        use_subproc        = False,
        verbose            = 0,
    )
    warm_elapsed = time.perf_counter() - t0
    print(f"   warm-start in {warm_elapsed:.1f}s "
          f"→ {child_zip} ({child_zip.stat().st_size//1024} KB)")

    # ------------------------------------------------------------------
    # 4. Render rollouts
    # ------------------------------------------------------------------
    print("\n[4] Rendering parent + child rollouts ...")
    from stable_baselines3 import PPO
    from mujoco_env import RobotControllerEnv

    for label, zip_path, rw, mp4 in (
        ("parent", parent_zip, parent_rw, parent_mp4),
        ("child",  child_zip,  child_rw,  child_mp4),
    ):
        env = RobotControllerEnv(
            reward_weights   = rw,
            seed             = 7,
            episode_duration = args.episode_duration,
            render_mode      = "rgb_array",
            render_width     = 320,
            render_height    = 240,
        )
        policy = PPO.load(str(zip_path))
        path, info = rollout_to_video(policy, env, str(mp4), fps=20)
        env.close()
        print(f"   {label:6s} → {path}  frames={info['n_frames']}  "
              f"reward={info['total_reward']:+.2f}")

    # ------------------------------------------------------------------
    # 5. Optionally score the child with Gemini
    # ------------------------------------------------------------------
    if args.no_network:
        print("\n[5] Skipping Gemini scoring (--no-network).")
    else:
        print("\n[5] Scoring child with Gemini ...")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from api_keys import APIKEY_GEMINI
        from grader import LocomotionGrader
        from gemini_prompts import WALK_FORWARD

        grader = LocomotionGrader(
            api_key       = APIKEY_GEMINI,
            prompt_config = WALK_FORWARD,
            batch_size    = 2,
            debug         = True,
        )
        out = grader.score(str(child_mp4))
        print(f"\n   child fitness    : {out.fitness:.4f}")
        print(f"   raw_scores       : {out.raw_scores}")
        print(f"   observation      : {out.extra.get('observation', '')[:160]}")
        assert 0.0 <= out.fitness <= 1.0

    print("\nAll _smoke_warm_start checks passed.")


if __name__ == "__main__":
    main()
