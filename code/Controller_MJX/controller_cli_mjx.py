#!/usr/bin/env python3
"""
controller_cli_mjx.py
=====================
SSH-friendly CLI for MJX controller training and rendering.
No GUI — outputs are saved to disk for later review.

Commands
--------
  new     Train from scratch with random reward weights
  mutate  Warm-start from an existing policy (.params file)
  manual  Train from scratch with reward weights from a JSON file
  render  Render a video from an existing policy (no training)
  bench   Quick GPU/CPU benchmark

Common flags
------------
  --steps N         Total PPO training steps       (default: 200 000)
  --envs  N         Parallel environments (vmap)   (default: 512)
  --rollout N       Steps per PPO rollout           (default: 64)
  --seed  N         PRNG seed (default: random)
  --episode FLOAT   Episode duration in seconds     (default: 5.0)
  --fall-height F   Torso-z fall threshold          (default: 0.3)
  --arch  N [N …]   Policy hidden layer sizes       (default: 256 256)
  --out   DIR       Output root directory           (default: cli_output)

GPU selection (shared machine)
------------------------------
  Use the CUDA_VISIBLE_DEVICES env var *before* this script runs:

    CUDA_VISIBLE_DEVICES=0 python controller_cli_mjx.py new --steps 200000 --envs 512
    CUDA_VISIBLE_DEVICES=1 python controller_cli_mjx.py mutate run_0001/policy.params

  If CUDA_VISIBLE_DEVICES is not set, the first visible GPU is used.
  On CPU-only machines the script falls back to CPU automatically.

Output layout
-------------
  cli_output/
    run_0001_new/
      policy.params   Flax policy weights (pickle)
      video.mp4       Rollout video (2-camera side-by-side)
      reward.json     Reward weights used
      info.json       Fitness, steps, timing, device
    run_0002_mutate/
      ...
    log.jsonl         One JSON line per completed run
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GPU selection: set CUDA_VISIBLE_DEVICES *before* JAX is imported.
# We do this at module level so it takes effect regardless of import order.
# ---------------------------------------------------------------------------
import os, sys

# Headless rendering: use EGL (NVIDIA GPU offscreen) instead of GLFW/X11.
# Must be set before mujoco is imported. Override with MUJOCO_GL=osmesa for CPU.
os.environ.setdefault("MUJOCO_GL", "egl")

# XLA GPU optimizations (must be set before jax import).
#   - cublaslt: use the faster cuBLAS-LT path for small GEMMs (always a win
#     for tiny matrices typical of small MLP policies).
#   - autotune_level=4: maximum autotuning during JIT (slower compile, faster
#     runtime). With the persistent cache below, the compile cost is paid once.
os.environ.setdefault("XLA_FLAGS",
    "--xla_gpu_enable_cublaslt=true "
    "--xla_gpu_autotune_level=4"
)

# JAX preallocates 75% of GPU VRAM by default — this is why nvidia-smi shows
# the same number regardless of n_envs. Bump to 0.92 to unlock more headroom
# for larger n_envs scaling. Set XLA_PYTHON_CLIENT_PREALLOCATE=false from the
# shell to see the actual usage instead.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

# ---------------------------------------------------------------------------
# Imports (JAX loads here — CUDA_VISIBLE_DEVICES must already be set)
# ---------------------------------------------------------------------------
import argparse
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# JAX / MJX stack
import jax
import jax.numpy as jnp

from mujoco_env_mjx  import build_env_config, _pick_mjx_device, MJXEnvConfig
from ppo_trainer_mjx import (
    train_from_scratch_mjx, train_warm_start_mjx,
    PPOConfig, make_params,
)
from video_renderer_mjx import rollout_to_video_mjx
from evolution_mjx      import save_params, load_params
from controller_morph   import build_model
from reward             import RewardWeights, mutate_weights, random_initial_weights
from config             import ExperimentConfig

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

_dev = _pick_mjx_device()
jax.config.update("jax_default_device", _dev)

# Persistent JIT compilation cache.
# Saves the ~60-80 s XLA compilation on every run after the first that uses
# matching shapes (n_envs, rollout, policy_arch). The cache is keyed by the
# abstract function signature — changing n_envs from 1024 to 8192 invalidates
# the entry; reusing exact params hits the cache.
# Typical cache size on disk: 50-200 MB per shape combination.
_jax_cache_dir = os.path.expanduser("~/.cache/jax_mjx")
os.makedirs(_jax_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

_cfg = ExperimentConfig()

# ---------------------------------------------------------------------------
# Defaults (GPU-appropriate — much larger than M2 CPU defaults)
# ---------------------------------------------------------------------------

# Defaults tuned for RTX 2080Ti with the nconmax fix (10 contact slots).
# Sweet spot empirically: envs=8192 saturates GPU-Util to ~90%.
# rollout=128 gives ~24 PPO updates per 3M steps (enough to amortize JIT).
_DEFAULT_STEPS   = 3_000_000
_DEFAULT_ENVS    = 8192
_DEFAULT_ROLLOUT = 128
_DEFAULT_ARCH    = (256, 256)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _run_dir(out_root: Path, mode: str) -> Path:
    """Return a new numbered run directory, e.g. cli_output/run_0003_mutate/."""
    out_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name.split("_")[1])
        for p in out_root.iterdir()
        if p.is_dir() and p.name.startswith("run_") and p.name.split("_")[1].isdigit()
    ]
    idx = (max(existing) + 1) if existing else 1
    d = out_root / f"run_{idx:04d}_{mode}"
    d.mkdir(parents=True)
    return d


def _log_run(out_root: Path, entry: dict) -> None:
    with open(out_root / "log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def _banner(title: str) -> None:
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def _print_device_info() -> None:
    devs = jax.devices()
    print(f"  JAX devices : {devs}")
    print(f"  Active dev  : {_dev}")
    if _dev.platform == "cpu":
        print("  [WARNING]  Running on CPU — training will be slow.")
        print("             Set CUDA_VISIBLE_DEVICES=N for GPU acceleration.")


def _ppo_cfg(n_envs: int, rollout_len: int) -> PPOConfig:
    return PPOConfig(n_epochs=4, n_minibatches=32)


def _build_env(rw: RewardWeights, episode: float, fall_height: float,
               mj_model=None) -> tuple:
    """Returns (env_cfg, mj_model). mj_model is built once and reused."""
    env_cfg = build_env_config(
        reward_weights    = rw,
        episode_duration  = episode,
        control_frequency = _cfg.control_frequency,
        fall_height       = fall_height,
    )
    if mj_model is None:
        mj_model, _ = build_model()
    return env_cfg, mj_model


def _render(params: Any, env_cfg: MJXEnvConfig, mj_model, out_dir: Path,
            arch: tuple, seed: int) -> str:
    video_path = str(out_dir / "video.mp4")
    print("  Rendering rollout …")
    t0 = time.perf_counter()
    _, info = rollout_to_video_mjx(
        params               = params,
        cfg                  = env_cfg,
        mj_model             = mj_model,
        save_path            = video_path,
        fps                  = _cfg.video_fps,
        render_width         = _cfg.render_width,
        render_height        = _cfg.render_height,
        cam1_azimuth         = _cfg.cam1_azimuth,
        cam1_elevation       = _cfg.cam1_elevation,
        cam1_distance        = _cfg.cam1_distance,
        cam1_lookat_z        = _cfg.cam1_lookat_z,
        cam2_azimuth         = _cfg.cam2_azimuth,
        cam2_elevation       = _cfg.cam2_elevation,
        cam2_distance        = _cfg.cam2_distance,
        camera_track_torso   = True,
        policy_arch          = arch,
        seed                 = seed,
        deterministic        = True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Rendered {info['n_frames']} frames in {elapsed:.1f}s "
          f"(terminated={info['terminated']}  total_reward={info['total_reward']:.2f})")
    return video_path


def _save_reward(rw: RewardWeights, out_dir: Path) -> str:
    path = out_dir / "reward.json"
    with open(path, "w") as f:
        json.dump(rw.to_dict(), f, indent=2)
    return str(path)


def _save_info(info: dict, out_dir: Path) -> None:
    with open(out_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def _print_result(mode: str, out_dir: Path, fitness: float,
                  steps: int, elapsed: float) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Run complete  [{mode}]")
    print(f"{'─' * 60}")
    print(f"  fitness  : {fitness:+.4f}")
    print(f"  steps    : {steps:,}")
    print(f"  time     : {elapsed:.1f}s  ({steps/elapsed:.0f} steps/s)")
    print(f"  output   : {out_dir}/")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# Load reward weights from JSON (supports partial JSON — missing keys use
# RewardWeights defaults)
# ---------------------------------------------------------------------------

def _load_reward_json(path: str) -> RewardWeights:
    with open(path) as f:
        data = json.load(f)
    # Support both flat {"forward_velocity": 1.0, …} and
    # nested {"reward_weights": {…}} formats
    if "reward_weights" in data:
        data = data["reward_weights"]
    defaults = _cfg.default_reward_weights_dict()
    defaults.update({k: v for k, v in data.items() if k in defaults})
    return RewardWeights(**defaults)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_new(args: argparse.Namespace) -> None:
    _banner("NEW — train from scratch (random reward weights)")
    _print_device_info()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 2**31))
    rng  = np.random.default_rng(seed)
    rw   = random_initial_weights(_cfg.default_reward_weights_dict(),
                                   sigma=args.sigma, rng=rng)
    arch = tuple(args.arch)

    print(f"\n  seed     : {seed}")
    print(f"  steps    : {args.steps:,}")
    print(f"  envs     : {args.envs}")
    print(f"  rollout  : {args.rollout}")
    print(f"  episode  : {args.episode}s")
    print(f"  arch     : {arch}")
    print(f"  sigma    : {args.sigma}")

    out_root = Path(args.out)
    out_dir  = _run_dir(out_root, "new")
    print(f"  output   : {out_dir}/\n")

    env_cfg, mj_model = _build_env(rw, args.episode, args.fall_height)
    ppo = _ppo_cfg(args.envs, args.rollout)

    t0 = time.perf_counter()
    params, fitness = train_from_scratch_mjx(
        cfg             = env_cfg,
        seed            = seed,
        total_steps     = args.steps,
        n_envs          = args.envs,
        rollout_len     = args.rollout,
        policy_arch     = arch,
        ppo_cfg         = ppo,
        fitness_episodes = 20,
        verbose         = True,
    )
    elapsed = time.perf_counter() - t0

    save_params(params, str(out_dir / "policy.params"))
    _save_reward(rw, out_dir)
    _render(params, env_cfg, mj_model, out_dir, arch, seed)

    info = {
        "mode": "new", "fitness": float(fitness),
        "steps": args.steps, "envs": args.envs, "rollout": args.rollout,
        "elapsed_s": round(elapsed, 2), "steps_per_s": round(args.steps / elapsed),
        "seed": seed, "arch": list(arch), "episode": args.episode,
        "sigma": args.sigma, "device": str(_dev),
        "reward_weights": rw.to_dict(),
        "output_dir": str(out_dir),
    }
    _save_info(info, out_dir)
    _log_run(out_root, info)
    _print_result("new", out_dir, fitness, args.steps, elapsed)


def cmd_mutate(args: argparse.Namespace) -> None:
    parent_path = Path(args.parent)
    if not parent_path.exists():
        sys.exit(f"ERROR: policy file not found: {parent_path}")

    _banner(f"MUTATE — warm-start from {parent_path.name}")
    _print_device_info()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 2**31))
    rng  = np.random.default_rng(seed)
    arch = tuple(args.arch)

    # Load parent reward weights from companion JSON if present
    reward_json = parent_path.parent / "reward.json"
    if reward_json.exists():
        parent_rw = _load_reward_json(str(reward_json))
        print(f"  Parent reward : {reward_json}")
    else:
        parent_rw = RewardWeights(**_cfg.default_reward_weights_dict())
        print("  Parent reward : [defaults — no reward.json found]")

    rw = mutate_weights(parent_rw, sigma=args.sigma, rng=rng)

    print(f"\n  seed     : {seed}")
    print(f"  steps    : {args.steps:,}")
    print(f"  envs     : {args.envs}")
    print(f"  rollout  : {args.rollout}")
    print(f"  episode  : {args.episode}s")
    print(f"  sigma    : {args.sigma}")

    out_root = Path(args.out)
    out_dir  = _run_dir(out_root, "mutate")
    print(f"  output   : {out_dir}/\n")

    parent_params = load_params(str(parent_path))
    env_cfg, mj_model = _build_env(rw, args.episode, args.fall_height)
    ppo = _ppo_cfg(args.envs, args.rollout)

    t0 = time.perf_counter()
    params, fitness = train_warm_start_mjx(
        parent_params   = parent_params,
        cfg             = env_cfg,
        seed            = seed,
        total_steps     = args.steps,
        n_envs          = args.envs,
        rollout_len     = args.rollout,
        policy_arch     = arch,
        ppo_cfg         = ppo,
        fitness_episodes = 20,
        verbose         = True,
    )
    elapsed = time.perf_counter() - t0

    save_params(params, str(out_dir / "policy.params"))
    _save_reward(rw, out_dir)
    _render(params, env_cfg, mj_model, out_dir, arch, seed)

    info = {
        "mode": "mutate", "fitness": float(fitness),
        "steps": args.steps, "envs": args.envs, "rollout": args.rollout,
        "elapsed_s": round(elapsed, 2), "steps_per_s": round(args.steps / elapsed),
        "seed": seed, "arch": list(arch), "episode": args.episode,
        "sigma": args.sigma, "device": str(_dev),
        "parent": str(parent_path),
        "reward_weights": rw.to_dict(),
        "output_dir": str(out_dir),
    }
    _save_info(info, out_dir)
    _log_run(out_root, info)
    _print_result("mutate", out_dir, fitness, args.steps, elapsed)


def cmd_manual(args: argparse.Namespace) -> None:
    reward_path = Path(args.reward_json)
    if not reward_path.exists():
        sys.exit(f"ERROR: reward JSON not found: {reward_path}")

    _banner(f"MANUAL — train from scratch with {reward_path.name}")
    _print_device_info()

    rw   = _load_reward_json(str(reward_path))
    seed = args.seed if args.seed is not None else int(np.random.randint(0, 2**31))
    arch = tuple(args.arch)

    print(f"\n  seed     : {seed}")
    print(f"  steps    : {args.steps:,}")
    print(f"  envs     : {args.envs}")
    print(f"  rollout  : {args.rollout}")
    print(f"  episode  : {args.episode}s")
    print(f"  reward   : {reward_path}")

    out_root = Path(args.out)
    out_dir  = _run_dir(out_root, "manual")
    print(f"  output   : {out_dir}/\n")

    env_cfg, mj_model = _build_env(rw, args.episode, args.fall_height)
    ppo = _ppo_cfg(args.envs, args.rollout)

    t0 = time.perf_counter()
    params, fitness = train_from_scratch_mjx(
        cfg             = env_cfg,
        seed            = seed,
        total_steps     = args.steps,
        n_envs          = args.envs,
        rollout_len     = args.rollout,
        policy_arch     = arch,
        ppo_cfg         = ppo,
        fitness_episodes = 20,
        verbose         = True,
    )
    elapsed = time.perf_counter() - t0

    save_params(params, str(out_dir / "policy.params"))
    _save_reward(rw, out_dir)
    _render(params, env_cfg, mj_model, out_dir, arch, seed)

    info = {
        "mode": "manual", "fitness": float(fitness),
        "steps": args.steps, "envs": args.envs, "rollout": args.rollout,
        "elapsed_s": round(elapsed, 2), "steps_per_s": round(args.steps / elapsed),
        "seed": seed, "arch": list(arch), "episode": args.episode,
        "device": str(_dev),
        "reward_json": str(reward_path),
        "reward_weights": rw.to_dict(),
        "output_dir": str(out_dir),
    }
    _save_info(info, out_dir)
    _log_run(out_root, info)
    _print_result("manual", out_dir, fitness, args.steps, elapsed)


def cmd_render(args: argparse.Namespace) -> None:
    policy_path = Path(args.policy)
    if not policy_path.exists():
        sys.exit(f"ERROR: policy file not found: {policy_path}")

    _banner(f"RENDER — {policy_path.name}")
    _print_device_info()

    seed = args.seed if args.seed is not None else 0
    arch = tuple(args.arch)

    # Try to find companion reward.json
    reward_json = policy_path.parent / "reward.json"
    if reward_json.exists():
        rw = _load_reward_json(str(reward_json))
        print(f"  Reward : {reward_json}")
    else:
        rw = RewardWeights(**_cfg.default_reward_weights_dict())
        print("  Reward : [defaults]")

    out_root = Path(args.out)
    out_dir  = _run_dir(out_root, "render")
    print(f"  Output : {out_dir}/\n")

    params = load_params(str(policy_path))
    env_cfg, mj_model = _build_env(rw, args.episode, args.fall_height)

    video_path = _render(params, env_cfg, mj_model, out_dir, arch, seed)
    _save_reward(rw, out_dir)

    info = {
        "mode": "render", "policy": str(policy_path),
        "seed": seed, "arch": list(arch), "episode": args.episode,
        "device": str(_dev), "output_dir": str(out_dir),
    }
    _save_info(info, out_dir)
    _log_run(out_root, info)
    print(f"  Video  : {video_path}")


def cmd_bench(args: argparse.Namespace) -> None:
    _banner("BENCHMARK")
    _print_device_info()
    print()

    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_mjx import run_benchmark

    if args.full:
        result = run_benchmark(
            n_envs=512, rollout_len=64, n_init=50_000, n_warm=10_000,
            policy_arch=(256, 256), render_width=192, render_height=192,
            max_render_steps=100, verbose=True,
        )
    else:
        result = run_benchmark(
            n_envs=32, rollout_len=32, n_init=2_000, n_warm=1_000,
            policy_arch=(64, 64), render_width=64, render_height=64,
            max_render_steps=10, verbose=True,
        )

    print(f"\n  init throughput : {result.init_fps:,.0f} steps/s")
    print(f"  warm throughput : {result.warm_fps:,.0f} steps/s")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--steps",       type=int,   default=_DEFAULT_STEPS,
                   help=f"Total PPO training steps (default: {_DEFAULT_STEPS:,})")
    p.add_argument("--envs",        type=int,   default=_DEFAULT_ENVS,
                   help=f"Parallel environments via vmap (default: {_DEFAULT_ENVS})")
    p.add_argument("--rollout",     type=int,   default=_DEFAULT_ROLLOUT,
                   help=f"Steps per PPO rollout (default: {_DEFAULT_ROLLOUT})")
    p.add_argument("--seed",        type=int,   default=None,
                   help="PRNG seed (default: random)")
    p.add_argument("--episode",     type=float, default=_cfg.episode_duration,
                   help=f"Episode duration in seconds (default: {_cfg.episode_duration})")
    p.add_argument("--fall-height", type=float, default=_cfg.fall_height,
                   help=f"Torso-z fall threshold (default: {_cfg.fall_height})")
    p.add_argument("--arch",        type=int,   nargs="+", default=list(_DEFAULT_ARCH),
                   help=f"Policy hidden layer sizes (default: {list(_DEFAULT_ARCH)})")
    p.add_argument("--out",         type=str,   default="cli_output",
                   help="Output root directory (default: cli_output)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="controller_cli_mjx.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- new ----------------------------------------------------------------
    p_new = sub.add_parser("new", help="Train from scratch with random reward weights")
    _add_common(p_new)
    p_new.add_argument("--sigma", type=float, default=_cfg.reward_init_sigma,
                       help=f"Reward weight init sigma (default: {_cfg.reward_init_sigma})")

    # ---- mutate -------------------------------------------------------------
    p_mut = sub.add_parser("mutate",
        help="Warm-start from an existing policy with mutated reward weights")
    p_mut.add_argument("parent", help="Path to parent policy (.params file)")
    _add_common(p_mut)
    p_mut.add_argument("--sigma", type=float, default=_cfg.reward_mutation_sigma,
                       help=f"Reward mutation sigma (default: {_cfg.reward_mutation_sigma})")

    # ---- manual -------------------------------------------------------------
    p_man = sub.add_parser("manual",
        help="Train from scratch with reward weights from a JSON file")
    p_man.add_argument("reward_json",
                       help="JSON file with reward weights (partial ok, missing keys use defaults)")
    _add_common(p_man)

    # ---- render -------------------------------------------------------------
    p_ren = sub.add_parser("render",
        help="Render a rollout video from an existing policy (no training)")
    p_ren.add_argument("policy", help="Path to policy (.params file)")
    p_ren.add_argument("--seed",        type=int,   default=0)
    p_ren.add_argument("--episode",     type=float, default=_cfg.episode_duration)
    p_ren.add_argument("--fall-height", type=float, default=_cfg.fall_height)
    p_ren.add_argument("--arch",        type=int,   nargs="+", default=list(_DEFAULT_ARCH))
    p_ren.add_argument("--out",         type=str,   default="cli_output")

    # ---- bench --------------------------------------------------------------
    p_ben = sub.add_parser("bench", help="Quick GPU/CPU benchmark")
    p_ben.add_argument("--full", action="store_true",
                       help="Full-scale benchmark (512 envs, 50k steps)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    try:
        {
            "new":    cmd_new,
            "mutate": cmd_mutate,
            "manual": cmd_manual,
            "render": cmd_render,
            "bench":  cmd_bench,
        }[args.command](args)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
