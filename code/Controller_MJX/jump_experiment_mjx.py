#!/usr/bin/env python3
"""
jump_experiment_mjx.py
======================
General (μ+λ) evolutionary experiment to validate the EA machinery BEFORE the
VLM grader is added. The objective is selectable with `--target`:

    jump    make the tripod leave the ground   (peak torso height above spawn)
    walk    travel forward                      (net +x displacement)
    rotate  spin in place                       (accumulated yaw rotation)
    crawl   move while staying low              (path length travelled)

Each individual is scored by a deterministic external performance function of
its rollout (see performance_grader.py) instead of a Gemini call. Both the
reward-weight prior the search starts from AND the fitness metric come from the
selected target's entry in the TARGETS registry below — to add a new objective,
add one TARGETS entry; nothing else changes.

What it exercises
-----------------
  - μ+λ selection / reward-weight mutation / archive   (archive.py, evolution_mjx.py)
  - the shared JIT-compiled PPO inner loop             (one compile, reused)
  - deterministic external-metric grading              (performance_grader.py)
  - per-individual video naming with the score baked in

Each rendered rollout is saved as:
    {out}/{run_id}/videos/gen_{G}_id_{ID}_score_{score:.3f}.mp4

Terminal-only — no UI. Everything is printed and logged to JSONL/archive files
under {out}/{run_id}/ (same layout as experiment_mjx.py).

Usage
-----
    python jump_experiment_mjx.py --target jump      # default
    python jump_experiment_mjx.py --target walk
    python jump_experiment_mjx.py --target rotate --gpu 3

Notes
-----
Each prior leaves only a handful of terms non-zero; mutation is multiplicative
log-normal, so terms left at 0.0 stay disabled — keeping the search in a small
objective-relevant subspace. Selection is on the *measured* metric, so the EA
discovers which weight magnitudes actually produce the behaviour. The priors are
sensible starting points, not tuned optima.
"""

from __future__ import annotations

import os, sys

# Headless EGL rendering + XLA GPU flags — must precede the JAX/mujoco imports.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_FLAGS",
    "--xla_gpu_enable_cublaslt=true --xla_gpu_autotune_level=4")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")


# ---------------------------------------------------------------------------
# Single-GPU pinning — MUST run before JAX is imported.
# ---------------------------------------------------------------------------
# JAX otherwise initialises on EVERY visible GPU and preallocates
# MEM_FRACTION × VRAM on each — which on a shared machine OOMs against GPU 0
# (used by other people). We pin to exactly one GPU. Precedence:
#   1. An explicit CUDA_VISIBLE_DEVICES already in the environment (respected).
#   2. `--gpu N` on the command line.
#   3. Auto-pick the GPU with the most free memory, EXCLUDING GPU 0.

def _pin_single_gpu(exclude=(0,)) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[gpu] using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (from env)")
        return
    if "--gpu" in sys.argv:
        i = sys.argv.index("--gpu")
        if i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            print(f"[gpu] using GPU {sys.argv[i + 1]} (from --gpu)")
            return
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True)
        cands = []
        for line in out.strip().splitlines():
            idx, free = (x.strip() for x in line.split(","))
            idx, free = int(idx), int(free)
            if idx in exclude:
                continue
            cands.append((free, idx))
        if cands:
            free, idx = max(cands)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            print(f"[gpu] auto-selected GPU {idx} ({free} MiB free; excluded {list(exclude)})")
            return
        print(f"[gpu] no eligible GPU outside {list(exclude)} — falling back to JAX default")
    except Exception as e:
        print(f"[gpu] auto-select failed ({e}); using JAX default")


_pin_single_gpu(exclude=(0,))

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import jax

from config             import ExperimentConfig
from archive            import MuLambdaArchive
from data_handler       import result_to_dict
from evolution_mjx      import MuLambdaEvolutionMJX, _videos_dir
from video_renderer_mjx import rollout_to_video_mjx
from performance_grader import (
    PerformanceGrader,
    jump_height_metric,
    max_height_metric,
    forward_distance_metric,
    path_length_metric,
    rotation_metric,
)

# Persistent JIT cache (saves the ~60 s XLA compile between runs of matching shape).
_jax_cache_dir = os.path.expanduser("~/.cache/jax_mjx")
os.makedirs(_jax_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


# ---------------------------------------------------------------------------
# Target registry — reward prior + fitness metric per objective
# ---------------------------------------------------------------------------
# Each Target pairs the reward-weight prior the EA starts mutating from with the
# external performance function it is selected on. Only the terms named in
# `reward_prior` are non-zero; every other RewardWeights term is frozen at 0.0
# (multiplicative mutation keeps zeros at zero), so the search stays in a small
# objective-relevant subspace. `metric_fn` reads a scalar off the renderer's
# info dict (see video_renderer_mjx.rollout_to_video_mjx for available keys).
#
# To add an objective: add one Target to TARGETS. That is the only change.

@dataclass
class Target:
    name:          str
    reward_prior:  dict
    metric_fn:     Callable[[dict], float]
    metric_name:   str
    unit:          str = ""
    description:   str = ""
    descriptor_fn: Optional[Callable[[dict], dict]] = None   # for MAP-Elites later


TARGETS: dict[str, Target] = {
    "jump": Target(
        name        = "jump",
        reward_prior= {
            "vertical_velocity_reward": 1.0,    # reward upward torso velocity
            "torso_height_reward":      1.0,    # reward being high off the ground
            "upright_bonus":            0.3,    # stay roughly upright
            "alive_bonus":              0.1,    # don't terminate early
            "energy_penalty":           0.001,  # mild effort cost
            "fall_penalty":             5.0,    # discourage crashing through the floor
        },
        metric_fn   = jump_height_metric,
        metric_name = "jump_height",
        unit        = "m",
        description = "peak torso height above spawn",
    ),
    "walk": Target(
        name        = "walk",
        reward_prior= {
            "forward_velocity":    1.0,    # reward moving in +x
            "lateral_drift":       0.1,    # discourage sideways drift
            "upright_bonus":       0.5,    # stay upright while moving
            "height_target_reward":0.3,    # keep torso near spawn height
            "alive_bonus":         0.1,
            "energy_penalty":      0.001,
            "fall_penalty":        5.0,
        },
        metric_fn   = forward_distance_metric,
        metric_name = "forward_distance",
        unit        = "m",
        description = "net forward (+x) displacement",
    ),
    "rotate": Target(
        name        = "rotate",
        reward_prior= {
            "torso_rotation_reward": 1.0,    # reward |yaw angular velocity|
            "upright_bonus":         0.5,    # spin while staying upright
            "height_target_reward":  0.3,    # don't collapse
            "alive_bonus":           0.1,
            "energy_penalty":        0.001,
            "fall_penalty":          5.0,
        },
        metric_fn   = rotation_metric,
        metric_name = "abs_yaw",
        unit        = "rad",
        description = "total accumulated yaw rotation",
    ),
    "crawl": Target(
        name        = "crawl",
        reward_prior= {
            "forward_velocity": 1.0,    # reward moving forward
            "contact_reward":   0.2,    # reward keeping feet/body in contact (low)
            "alive_bonus":      0.1,
            "energy_penalty":   0.001,
            "fall_penalty":     5.0,    # low so it can move on its belly without big penalty
        },
        metric_fn   = path_length_metric,
        metric_name = "path_length",
        unit        = "m",
        description = "total distance travelled along the path",
    ),
}


def _reward_defaults(target: Target) -> dict:
    """Full reward-weight dict: the target's prior on top of an all-zero base."""
    from reward import RewardWeights
    base = {name: 0.0 for name in RewardWeights.field_names()}
    base.update(target.reward_prior)
    return base


# ---------------------------------------------------------------------------
# EvoExperimentMJX — μ+λ with custom render naming + metric registration
# ---------------------------------------------------------------------------

class EvoExperimentMJX(MuLambdaEvolutionMJX):
    """
    Same training/selection as MuLambdaEvolutionMJX, but the render step:
      1. rolls out the policy (full physics) and reads its info dict,
      2. computes the performance score via the active grader's metric,
      3. names the MP4  gen_{G}_id_{ID}_score_{score:.3f}.mp4,
      4. registers (video_path -> info) with the grader so score_batch can read it.

    `metric_label` / `metric_unit` are set by main() for human-readable logging;
    they do not affect what is optimised (that is the grader's metric_fn).
    """

    metric_label: str = "score"
    metric_unit:  str = ""

    # initialise/step receive the grader; stash it for _render to use.
    def initialise(self, grader, id_counter: int = 0):
        self._active_grader = grader
        return super().initialise(grader, id_counter)

    def step(self, archive, grader, generation: int, id_counter: int):
        self._active_grader = grader
        return super().step(archive, grader, generation, id_counter)

    def _render(self, params: Any, rw, individual_id: int,
                generation: int, seed: int) -> str:
        env_cfg = self._env_cfg_for(rw)
        videos  = _videos_dir(self.run_dir)
        videos.mkdir(parents=True, exist_ok=True)
        tmp = videos / f"_tmp_gen{generation}_id{individual_id}.mp4"

        _, info = rollout_to_video_mjx(
            params         = params,
            cfg            = env_cfg,
            mj_model       = self._mj_model,
            save_path      = str(tmp),
            fps            = self.cfg.video_fps,
            render_width   = self.cfg.render_width,
            render_height  = self.cfg.render_height,
            cam1_azimuth   = self.cfg.cam1_azimuth,
            cam1_elevation = self.cfg.cam1_elevation,
            cam1_distance  = self.cfg.cam1_distance,
            cam1_lookat_z  = self.cfg.cam1_lookat_z,
            cam2_azimuth   = self.cfg.cam2_azimuth,
            cam2_elevation = self.cfg.cam2_elevation,
            cam2_distance  = self.cfg.cam2_distance,
            camera_track_torso = self.cfg.camera_track_torso,
            seed           = seed,
            policy_arch    = tuple(self.cfg.policy_arch),
            deterministic  = True,
        )

        # Name the file with the score, then register under the FINAL path so
        # score_batch (keyed by spec.video_path == final) finds the info.
        score = self._active_grader.score_of(info)
        final = videos / f"gen_{generation}_id_{individual_id}_score_{score:.3f}.mp4"
        os.replace(tmp, final)
        self._active_grader.register(str(final), info)

        print(f"      rendered id={individual_id}  "
              f"{self.metric_label}={score:.3f}{self.metric_unit}  "
              f"peak={info['max_torso_height']:.3f}m  "
              f"dist={info['horizontal_distance']:.3f}m  "
              f"yaw={info['abs_yaw']:.2f}rad  "
              f"steps={info['n_steps']}  -> {final.name}", flush=True)
        return str(final)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_individuals(path: Path, results) -> None:
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps(result_to_dict(r)) + "\n")


def _print_gen(generation: int, n_gen: int, results, archive, elapsed: float,
               label: str, unit: str) -> None:
    stats = archive.history[-1] if archive.history else None
    best  = archive.best()
    print(f"\n[gen {generation + 1}/{n_gen}]  evaluated={len(results)}  "
          f"best_{label}={best.fitness:.3f}{unit} (id={best.individual_id})  "
          f"mean={stats.mean_fitness:.3f}{unit}  std={stats.std_fitness:.3f}  "
          f"{elapsed:.1f}s", flush=True)
    for r in sorted(results, key=lambda x: x.fitness, reverse=True):
        tag = "★" if r.individual_id == best.individual_id else " "
        print(f"   {tag} id={r.individual_id:>3}  {label}={r.fitness:.3f}{unit}  "
              f"parent={r.parent_id}  {Path(r.video_path).name}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target",      choices=list(TARGETS), default="jump",
                   help="objective to evolve toward (selects reward prior + fitness metric)")
    p.add_argument("--generations", type=int,   default=4)
    p.add_argument("--pop",         type=int,   default=5,  help="initial population size (gen 0)")
    p.add_argument("--mu",          type=int,   default=2,  help="parents kept per generation")
    p.add_argument("--lambda",      type=int,   default=5,  dest="lambda_",
                   help="children per generation")
    p.add_argument("--init-steps",  type=int,   default=5_000_000)
    p.add_argument("--warm-steps",  type=int,   default=2_500_000)
    p.add_argument("--envs",        type=int,   default=2048)
    p.add_argument("--rollout",     type=int,   default=32)
    p.add_argument("--episode",     type=float, default=2.5, help="simulation seconds per episode")
    p.add_argument("--prediction-factor", type=float, default=-30.0, dest="prediction_factor",
                   help="action→joint delta scale = factor/ctrl_freq; smaller |value| = "
                        "slower, smoother, more stable robot (CPU baseline is -60)")
    p.add_argument("--seed",        type=int,   default=11)
    p.add_argument("--out",         type=str,   default="results")
    p.add_argument("--run-id",      type=str,   default=None)
    p.add_argument("--gpu",         type=str, default=None,
                   help="GPU index to pin (sets CUDA_VISIBLE_DEVICES before JAX init; "
                        "default: auto-pick the freest GPU, never GPU 0)")
    args = p.parse_args()

    target = TARGETS[args.target]
    run_id = args.run_id or time.strftime(f"{target.name}_%Y%m%d_%H%M%S")

    cfg = ExperimentConfig(
        run_id               = run_id,
        strategy             = "mu_lambda",
        mu                   = args.mu,
        lambda_              = args.lambda_,
        n_generations        = args.generations,
        init_population_size = args.pop,
        n_init_steps         = args.init_steps,
        n_warm_steps         = args.warm_steps,
        n_envs_mjx           = args.envs,
        n_steps_per_env      = args.rollout,
        episode_duration     = args.episode,
        prediction_factor    = args.prediction_factor,
        seed                 = args.seed,
        output_dir           = args.out,
        **{f"rw_{k}": v for k, v in _reward_defaults(target).items()
           if f"rw_{k}" in ExperimentConfig.__dataclass_fields__},
    )

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(str(run_dir / "config.json"))
    indiv_log = run_dir / "individuals_log.jsonl"

    label, unit = target.metric_name, target.unit
    print("=" * 70)
    print(f"  EVO EXPERIMENT (μ+λ)   target={target.name!r}   run_id={run_id}")
    print("=" * 70)
    print(f"  objective    : {target.description}  (fitness = {label}{', '+unit if unit else ''})")
    print(f"  morphology   : {ExperimentConfig.morphology}")
    print(f"  population   : pop0={args.pop}  μ={args.mu}  λ={args.lambda_}  "
          f"generations={args.generations}")
    print(f"  PPO          : init={args.init_steps:,}  warm={args.warm_steps:,}  "
          f"envs={args.envs}  rollout={args.rollout}")
    print(f"  episode      : {args.episode}s   pred_factor={args.prediction_factor} "
          f"(Δ≈{abs(args.prediction_factor)/cfg.control_frequency:.2f} rad/tick)")
    print(f"  reward prior : {target.reward_prior}")
    print(f"  output       : {run_dir}/")
    print("=" * 70, flush=True)

    rng    = np.random.default_rng(cfg.seed)
    grader = PerformanceGrader(metric_fn=target.metric_fn, metric_name=target.metric_name,
                               descriptor_fn=target.descriptor_fn)
    evo    = EvoExperimentMJX(cfg, run_dir=run_dir, rng=rng)
    evo.verbose_training = True   # print per-update PPO progress for each individual
    evo.metric_label     = label
    evo.metric_unit      = unit
    archive = MuLambdaArchive(mu=cfg.mu)

    # ---- Generation 0 (from scratch) ---------------------------------------
    print(f"\n[gen 1/{args.generations}] training {args.pop} individuals from scratch "
          f"({args.init_steps:,} steps each) …", flush=True)
    t0 = time.perf_counter()
    init_results, id_counter = evo.initialise(grader, id_counter=0)
    archive.update(init_results)
    _log_individuals(indiv_log, init_results)
    _print_gen(0, args.generations, init_results, archive, time.perf_counter() - t0, label, unit)
    archive.save(str(run_dir / "archive_gen0000.json"))

    # ---- Evolution loop (gens 1 .. generations-1) --------------------------
    for generation in range(1, args.generations):
        print(f"\n[gen {generation + 1}/{args.generations}] warm-starting {args.lambda_} children "
              f"({args.warm_steps:,} steps each) …", flush=True)
        t0 = time.perf_counter()
        prev_id = id_counter
        results, id_counter = evo.step(archive, grader, generation, id_counter)
        archive.update(results)
        _log_individuals(indiv_log, [r for r in results if r.individual_id >= prev_id])
        _print_gen(generation, args.generations, results, archive,
                   time.perf_counter() - t0, label, unit)
        archive.save(str(run_dir / f"archive_gen{generation:04d}.json"))

    archive.save(str(run_dir / "archive_final.json"))
    print("\n" + "=" * 70)
    print("  DONE")
    archive.summary()
    best = archive.best()
    if best:
        print(f"\n  Best {target.name}: id={best.individual_id}  "
              f"{label}={best.fitness:.3f}{unit}")
        print(f"  Video      : {best.video_path}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
