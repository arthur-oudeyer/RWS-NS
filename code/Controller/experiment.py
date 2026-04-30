"""
experiment.py
=============
Top-level driver for one controller-study run.

run(cfg)         — start a fresh experiment from an `ExperimentConfig`.
resume(run_dir)  — reload a partially-completed run and continue from the
                   latest archive_gen*.json snapshot.

Per-run artefacts written under `cfg.run_dir`
---------------------------------------------
  config.json                  — frozen config
  archive_gen{N:04d}.json      — periodic archive snapshots
  archive_final.json           — final snapshot
  log.jsonl                    — one line per generation (best/mean/elapsed)
  individuals_log.jsonl        — one line per evaluated individual
  vlm_responses.jsonl          — raw Gemini responses (audit trail)
  policies/id{ID:06d}.zip      — per-individual SB3 policy (.zip)
  videos/gen{G:04d}_id{ID:06d}.mp4 — per-individual rollout MP4
  tb/                          — TensorBoard logs from PPO inner loops

Mirrors `Morphology/experiment.py` line-for-line in shape and side-effects.
The only structural difference is that "evaluation" of one individual now
trains a PPO policy and records an MP4; the batched VLM call still happens
*once per generation*, after every child has been rendered.

Debug
-----
Run this file with `--debug` to execute a tiny end-to-end smoke run with
fake renderer + fake grader (no Gemini).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from config        import ExperimentConfig
from archive       import MuLambdaArchive, MapEliteArchive
from evolution     import BaseEvolution, make_evolution
from data_handler  import ControllerResult, result_to_dict

# ---------------------------------------------------------------------------
# Grader / archive factories
# ---------------------------------------------------------------------------

def _make_grader(cfg: ExperimentConfig):
    """Build a `LocomotionGrader` from the config."""
    if cfg.grader_type != "gemini":
        raise NotImplementedError(f"grader_type={cfg.grader_type!r} not supported "
                                  "for the controller study (only 'gemini').")
    from gemini_prompts import get_locomotion_prompt_set
    from grader import LocomotionGrader

    # Optional descriptor config (reuse Morphology/descriptor.py)
    desc_cfg = None
    try:
        if cfg.descriptor_config_name:
            from descriptor import get_descriptor_config
            desc_cfg = get_descriptor_config(cfg.descriptor_config_name)
    except Exception:
        desc_cfg = None

    # API key from code/api_keys.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from api_keys import APIKEY_GEMINI

    prompt_cfg = get_locomotion_prompt_set(cfg.prompt_name)
    return LocomotionGrader(
        api_key            = APIKEY_GEMINI,
        prompt_config      = prompt_cfg,
        model_name         = cfg.gemini_model,
        batch_size         = cfg.batching,
        descriptor_config  = desc_cfg,
        response_log_path  = str(cfg.run_dir / "vlm_responses.jsonl"),
        debug              = False,
    )


def _make_archive(cfg: ExperimentConfig):
    if cfg.strategy == "mu_lambda":
        return MuLambdaArchive(mu=cfg.mu)
    if cfg.strategy == "map_elite":
        feature_dims = []
        feature_bins: dict = {}
        dim_labels:   dict = {}
        if cfg.descriptor_config_name:
            try:
                from descriptor import get_descriptor_config
                d = get_descriptor_config(cfg.descriptor_config_name)
                feature_dims = list(d.feature_dims)
                feature_bins = {item.name: item.bins for item in d.items if item.bins}
                dim_labels   = {item.name: item.bin_labels for item in d.items if item.bin_labels}
            except Exception:
                pass
        return MapEliteArchive(
            feature_dims = feature_dims,
            feature_bins = feature_bins,
            dim_labels   = dim_labels,
        )
    raise ValueError(f"Unknown strategy: {cfg.strategy!r}")


def _archive_path(run_dir: Path, generation: int) -> Path:
    return run_dir / f"archive_gen{generation:04d}.json"


def _save_archive(archive, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive.save(str(path))


def _load_archive(cfg: ExperimentConfig, path: Path):
    if cfg.strategy == "mu_lambda":
        return MuLambdaArchive.load(str(path))
    return MapEliteArchive.load(str(path))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_individuals(log_path: Path, individuals: "list[ControllerResult]") -> None:
    with open(log_path, "a") as f:
        for r in individuals:
            f.write(json.dumps(result_to_dict(r)) + "\n")


def _log_generation(
    log_path:      Path,
    generation:    int,
    phase:         str,
    results:       "list[ControllerResult]",
    archive,
    elapsed_s:     float,
) -> None:
    best = archive.best()
    entry = {
        "generation":   generation,
        "phase":        phase,
        "n_evaluated":  len(results),
        "best_fitness": best.fitness if best else None,
        "best_id":      best.individual_id if best else None,
        "elapsed_s":    round(elapsed_s, 2),
    }
    if hasattr(archive, "population"):
        entry["population_size"] = len(archive.population)
    if hasattr(archive, "grid"):
        entry["cells_filled"] = len(archive.grid)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _print_progress(
    generation:    int,
    n_generations: int,
    phase:         str,
    results:       "list[ControllerResult]",
    archive,
    elapsed_s:     float,
) -> None:
    best  = archive.best()
    stats = archive.history[-1] if archive.history else None
    extra = f"  cells={len(archive.grid)}" if hasattr(archive, "grid") else ""
    best_s = f"({best.fitness:+.4f}, {best.individual_id})" if best else "N/A"
    mean_s = f"{stats.mean_fitness:+.4f}" if stats else "N/A"
    print(
        f"[gen {generation:>3} / {n_generations}]  "
        f"{phase:<8}  pool_n={len(results):<3}  "
        f"best={best_s}  mean={mean_s}  "
        f"{elapsed_s:.1f}s{extra}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(
    cfg:    ExperimentConfig,
    grader = None,
):
    """
    Run a full experiment.

    Parameters
    ----------
    cfg    : ExperimentConfig.
    grader : optional pre-built grader (used by tests with a fake grader).
    """
    run_dir        = cfg.run_dir
    log_path       = run_dir / "log.jsonl"
    indiv_log_path = run_dir / "individuals_log.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.save(str(run_dir / "config.json"))
    print(f"\n{'=' * 60}")
    cfg.describe()
    print(f"{'=' * 60}\n")

    if grader is None:
        print("[experiment] Building grader ...")
        grader = _make_grader(cfg)
        print(f"[experiment] Grader ready ({type(grader).__name__}).")

    archive = _make_archive(cfg)
    rng     = np.random.default_rng(cfg.seed)
    evo: BaseEvolution = make_evolution(cfg, run_dir=run_dir, rng=rng)
    print(f"[experiment] Archive + evolution manager ready ({cfg.strategy}).")

    # ---- Generation 0 --------------------------------------------------------
    print(f"\n[experiment] Initial population — {cfg.init_population_size or '(default)'} individuals "
          f"× {cfg.n_init_steps:,} PPO steps each.")
    t0 = time.perf_counter()
    init_results, id_counter = evo.initialise(grader, id_counter=0)
    archive.update(init_results)
    elapsed = time.perf_counter() - t0
    _log_individuals(indiv_log_path, init_results)
    _print_progress(0, cfg.n_generations, "init", init_results, archive, elapsed)
    _log_generation(log_path, 0, "init", init_results, archive, elapsed)

    if 0 % cfg.save_every_n_gen == 0:
        _save_archive(archive, _archive_path(run_dir, 0))

    # ---- Evolution loop ------------------------------------------------------
    for generation in range(1, cfg.n_generations + 1):
        t0 = time.perf_counter()
        prev_id = id_counter
        results, id_counter = evo.step(
            archive    = archive,
            grader     = grader,
            generation = generation,
            id_counter = id_counter,
        )
        archive.update(results)
        elapsed = time.perf_counter() - t0
        # Only log truly new individuals (children), not re-tagged parents
        _log_individuals(indiv_log_path, [r for r in results if r.individual_id >= prev_id])
        _print_progress(generation, cfg.n_generations, "step", results, archive, elapsed)
        _log_generation(log_path, generation, "step", results, archive, elapsed)

        if generation % cfg.save_every_n_gen == 0:
            _save_archive(archive, _archive_path(run_dir, generation))

    # ---- Final save ----------------------------------------------------------
    final_path = run_dir / "archive_final.json"
    _save_archive(archive, final_path)
    print(f"\n[experiment] Done. Final archive → {final_path}")
    archive.summary()
    return archive


# ---------------------------------------------------------------------------
# resume()
# ---------------------------------------------------------------------------

def resume(
    run_dir: Union[str, Path],
    grader  = None,
):
    """
    Resume an interrupted run from the latest archive_gen*.json snapshot.
    """
    run_dir        = Path(run_dir)
    cfg            = ExperimentConfig.load(str(run_dir / "config.json"))
    log_path       = run_dir / "log.jsonl"
    indiv_log_path = run_dir / "individuals_log.jsonl"

    snapshots = sorted(run_dir.glob("archive_gen*.json"))
    if not snapshots:
        raise FileNotFoundError(f"No archive snapshots found in {run_dir}.")
    latest    = snapshots[-1]
    start_gen = int(latest.stem.replace("archive_gen", "")) + 1
    print(f"\n[experiment] Resuming from gen {start_gen - 1}  ({latest.name})")
    cfg.describe()

    archive = _load_archive(cfg, latest)
    if hasattr(archive, "population") and archive.population:
        id_counter = max(r.individual_id for r in archive.population) + 1
    elif hasattr(archive, "grid") and archive.grid:
        id_counter = max(r.individual_id for r in archive.grid.values()) + 1
    else:
        id_counter = 0

    if grader is None:
        grader = _make_grader(cfg)

    rng = np.random.default_rng(cfg.seed + start_gen)
    evo = make_evolution(cfg, run_dir=run_dir, rng=rng)

    for generation in range(start_gen, cfg.n_generations + 1):
        t0 = time.perf_counter()
        prev_id = id_counter
        results, id_counter = evo.step(archive, grader, generation, id_counter)
        archive.update(results)
        elapsed = time.perf_counter() - t0
        _log_individuals(indiv_log_path, [r for r in results if r.individual_id >= prev_id])
        _print_progress(generation, cfg.n_generations, "step", results, archive, elapsed)
        _log_generation(log_path, generation, "step", results, archive, elapsed)
        if generation % cfg.save_every_n_gen == 0:
            _save_archive(archive, _archive_path(run_dir, generation))

    final_path = run_dir / "archive_final.json"
    _save_archive(archive, final_path)
    print(f"\n[experiment] Done. Final archive → {final_path}")
    archive.summary()
    return archive


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Run a controller-study experiment.")
    parser.add_argument("--strategy",  default=None, choices=["mu_lambda", "map_elite"])
    parser.add_argument("--mu",        type=int, default=None)
    parser.add_argument("--lambda_",   type=int, default=None)
    parser.add_argument("--n_gen",     type=int, default=None)
    parser.add_argument("--n_init_steps", type=int, default=None)
    parser.add_argument("--n_warm_steps", type=int, default=None)
    parser.add_argument("--n_envs",    type=int, default=None)
    parser.add_argument("--prompt",    default=None)
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume",    default=None, metavar="RUN_DIR")
    parser.add_argument("--debug",     action="store_true",
                        help="Run a tiny smoke run with fake grader.")
    args = parser.parse_args()

    if args.debug:
        return _debug_smoke()

    if args.resume:
        resume(args.resume)
        return

    cfg = ExperimentConfig()
    if args.strategy is not None:    cfg.strategy = args.strategy
    if args.mu is not None:          cfg.mu = args.mu
    if args.lambda_ is not None:     cfg.lambda_ = args.lambda_
    if args.n_gen is not None:       cfg.n_generations = args.n_gen
    if args.n_init_steps is not None: cfg.n_init_steps = args.n_init_steps
    if args.n_warm_steps is not None: cfg.n_warm_steps = args.n_warm_steps
    if args.n_envs is not None:      cfg.n_envs = args.n_envs
    if args.prompt is not None:      cfg.prompt_name = args.prompt
    if args.seed is not None:        cfg.seed = args.seed
    if args.output_dir is not None:  cfg.output_dir = args.output_dir

    run(cfg)


# ---------------------------------------------------------------------------
# Debug — fake grader
# ---------------------------------------------------------------------------

def _debug_smoke():
    """
    Tiny end-to-end smoke run with a fake grader, no Gemini calls.
    Trains real PPO policies and records real MP4s, but fitness is sampled
    uniformly. Useful as a final wiring test.
    """
    import tempfile

    print("=" * 60)
    print("  experiment.py — debug smoke run (fake grader)")
    print("=" * 60)

    class _FakeOut:
        def __init__(self, fit):
            self.fitness = fit
            self.raw_scores = {"coherence": fit, "progress": fit, "interest": fit}
            self.method = "fake"
            self.prompt_set = "fake"
            self.extra = {"vlm_descriptors": {}}

    class _FakeGrader:
        def __init__(self): self.rng = np.random.default_rng(0)
        def score_batch(self, videos, debug=False, reference_video=None):
            return {vid: _FakeOut(float(self.rng.uniform(0.0, 1.0))) for vid, _ in videos}

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ExperimentConfig(
            run_id            = "debug_smoke",
            strategy          = "mu_lambda",
            mu                = 2,
            lambda_           = 2,
            n_generations     = 2,
            init_population_size = 3,
            n_init_steps      = 2_000,
            n_warm_steps      = 1_000,
            n_envs            = 1,
            n_steps_per_env   = 256,
            batch_size        = 64,
            episode_duration  = 1.5,
            output_dir        = tmp,
        )
        archive = run(cfg, grader=_FakeGrader())
        assert archive.best() is not None
        run_dir = Path(tmp) / cfg.run_id
        assert (run_dir / "config.json").exists()
        assert (run_dir / "log.jsonl").exists()
        assert (run_dir / "individuals_log.jsonl").exists()
        assert (run_dir / "archive_final.json").exists()
        print(f"\n  smoke run produced {len(list((run_dir / 'policies').glob('*.zip')))} policies"
              f" and {len(list((run_dir / 'videos').glob('*.mp4')))} videos.")

    print("\nAll experiment.py smoke checks passed.")


if __name__ == "__main__":
    _cli()
