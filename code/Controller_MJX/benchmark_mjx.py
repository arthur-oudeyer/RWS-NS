"""
benchmark_mjx.py
================
Wall-clock benchmark for the MJX training pipeline.

Measures time per generation phase:
  - init  : train_from_scratch_mjx  (one individual)
  - warm  : train_warm_start_mjx    (one individual)
  - render: rollout_to_video_mjx    (one episode)

All timings at tiny scale (2 envs × rollout_len steps) so the benchmark
finishes in seconds.  Use `--full` for a realistic scale run.

Output
------
Prints a table with timings and a throughput estimate (steps/s).
Also returns the results dict for programmatic use (used by test_phase4.py).

Usage
-----
    python benchmark_mjx.py           # tiny benchmark
    python benchmark_mjx.py --full    # realistic scale (slow on CPU/M2)
"""

from __future__ import annotations

import argparse
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import numpy as np

from mujoco_env_mjx   import build_env_config, _pick_mjx_device
from ppo_trainer_mjx  import train_from_scratch_mjx, train_warm_start_mjx, PPOConfig
from video_renderer_mjx import rollout_to_video_mjx
from controller_morph  import build_model
from reward           import RewardWeights


@dataclass
class BenchmarkResult:
    n_envs:           int
    rollout_len:      int
    total_init_steps: int
    total_warm_steps: int
    t_build_cfg:      float   # seconds
    t_init_train:     float
    t_warm_train:     float
    t_render:         float
    init_fps:         float   # steps/second during from-scratch
    warm_fps:         float


def run_benchmark(
    n_envs:      int  = 2,
    rollout_len: int  = 8,
    n_init:      int  = 16,
    n_warm:      int  = 16,
    policy_arch: tuple = (64, 64),
    render_width: int = 32,
    render_height: int = 32,
    max_render_steps: int = 5,
    seed:        int  = 0,
    verbose:     bool = True,
) -> BenchmarkResult:
    """
    Run a single benchmark pass and return timings.

    Parameters
    ----------
    n_envs      : parallel environments in vmap.
    rollout_len : steps per rollout (PPO inner scan length).
    n_init      : total timesteps for from-scratch training.
    n_warm      : total timesteps for warm-start training.
    policy_arch : hidden layer sizes.
    render_*    : per-camera resolution for the MP4.
    max_render_steps : cap episode during rendering.
    seed        : PRNG seed.
    verbose     : print progress.
    """
    dev = _pick_mjx_device()
    jax.config.update("jax_default_device", dev)

    if verbose:
        print(f"\n  device: {dev}")
        print(f"  n_envs={n_envs}  rollout_len={rollout_len}  "
              f"n_init={n_init}  n_warm={n_warm}")

    rw_parent = RewardWeights()
    rw_child  = RewardWeights(forward_velocity=1.5, alive_bonus=0.1)

    # ---- Build env config (includes MJX model compile) ---------------------
    t0 = time.perf_counter()
    cfg_parent = build_env_config(
        reward_weights=rw_parent, episode_duration=2.0
    )
    t_build = time.perf_counter() - t0
    if verbose:
        print(f"\n  [build_env_config]  {t_build:.2f}s")

    # ---- from-scratch training ---------------------------------------------
    t0 = time.perf_counter()
    params, fitness_init = train_from_scratch_mjx(
        cfg          = cfg_parent,
        seed         = seed,
        total_steps  = n_init,
        n_envs       = n_envs,
        rollout_len  = rollout_len,
        policy_arch  = policy_arch,
        ppo_cfg      = PPOConfig(n_epochs=1, minibatch_size=max(4, n_envs * rollout_len // 4)),
        fitness_episodes = 1,
        verbose      = False,
    )
    t_init = time.perf_counter() - t0
    init_fps = n_init / t_init if t_init > 0 else 0.0
    if verbose:
        print(f"  [from_scratch]      {t_init:.2f}s  ({init_fps:.0f} steps/s)  "
              f"fitness={fitness_init:.3f}")

    # ---- warm-start training -----------------------------------------------
    from dataclasses import replace as dc_replace
    cfg_child = dc_replace(cfg_parent, reward_weights_vec=rw_child.to_jax_vector())

    t0 = time.perf_counter()
    params2, fitness_warm = train_warm_start_mjx(
        parent_params = params,
        cfg           = cfg_child,
        seed          = seed + 1,
        total_steps   = n_warm,
        n_envs        = n_envs,
        rollout_len   = rollout_len,
        policy_arch   = policy_arch,
        ppo_cfg       = PPOConfig(n_epochs=1, minibatch_size=max(4, n_envs * rollout_len // 4)),
        fitness_episodes = 1,
        verbose       = False,
    )
    t_warm = time.perf_counter() - t0
    warm_fps = n_warm / t_warm if t_warm > 0 else 0.0
    if verbose:
        print(f"  [warm_start]        {t_warm:.2f}s  ({warm_fps:.0f} steps/s)  "
              f"fitness={fitness_warm:.3f}")

    # ---- rendering ---------------------------------------------------------
    mj_model, _ = build_model()
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "bench.mp4")
        t0 = time.perf_counter()
        _, info = rollout_to_video_mjx(
            params       = params2,
            cfg          = cfg_child,
            mj_model     = mj_model,
            save_path    = out,
            render_width = render_width,
            render_height = render_height,
            policy_arch  = policy_arch,
            seed         = seed,
            max_steps    = max_render_steps,
        )
        t_render = time.perf_counter() - t0
    if verbose:
        print(f"  [rollout_to_video]  {t_render:.2f}s  ({info['n_frames']} frames)")

    result = BenchmarkResult(
        n_envs           = n_envs,
        rollout_len      = rollout_len,
        total_init_steps = n_init,
        total_warm_steps = n_warm,
        t_build_cfg      = t_build,
        t_init_train     = t_init,
        t_warm_train     = t_warm,
        t_render         = t_render,
        init_fps         = init_fps,
        warm_fps         = warm_fps,
    )

    if verbose:
        _print_summary(result)

    return result


def _print_summary(r: BenchmarkResult) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Benchmark summary")
    print(f"{'─' * 50}")
    print(f"  n_envs={r.n_envs}  rollout_len={r.rollout_len}")
    print(f"  build_env_cfg  : {r.t_build_cfg:7.2f} s")
    print(f"  from_scratch   : {r.t_init_train:7.2f} s  ({r.init_fps:,.0f} steps/s)")
    print(f"  warm_start     : {r.t_warm_train:7.2f} s  ({r.warm_fps:,.0f} steps/s)")
    print(f"  render_episode : {r.t_render:7.2f} s")
    t_total = r.t_build_cfg + r.t_init_train + r.t_warm_train + r.t_render
    print(f"  ─────────────────────────────")
    print(f"  total          : {t_total:7.2f} s")
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MJX pipeline benchmark.")
    parser.add_argument("--full",    action="store_true",
                        help="Run at realistic scale (n_envs=128, n_init=50k).")
    parser.add_argument("--n_envs",  type=int, default=None)
    parser.add_argument("--n_init",  type=int, default=None)
    parser.add_argument("--n_warm",  type=int, default=None)
    parser.add_argument("--rollout_len", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  benchmark_mjx.py")
    print("=" * 60)

    if args.full:
        n_envs      = args.n_envs  or 128
        n_init      = args.n_init  or 50_000
        n_warm      = args.n_warm  or 10_000
        rollout_len = args.rollout_len or 64
        render_w    = 192
        render_steps = 100
        arch        = (256, 256)
    else:
        n_envs      = args.n_envs  or 2
        n_init      = args.n_init  or 16
        n_warm      = args.n_warm  or 16
        rollout_len = args.rollout_len or 8
        render_w    = 32
        render_steps = 5
        arch        = (64, 64)

    run_benchmark(
        n_envs       = n_envs,
        rollout_len  = rollout_len,
        n_init       = n_init,
        n_warm       = n_warm,
        policy_arch  = arch,
        render_width = render_w,
        render_height = render_w,
        max_render_steps = render_steps,
        verbose      = True,
    )
