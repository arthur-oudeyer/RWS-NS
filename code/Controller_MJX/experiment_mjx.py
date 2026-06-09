"""
experiment_mjx.py
=================
Top-level driver for one MJX-backend controller-study run.

Mirrors Controller/experiment.py exactly in output layout:
  config.json
  archive_gen{N:04d}.json
  archive_final.json
  log.jsonl
  individuals_log.jsonl
  policies/id{ID:06d}.params   ← Flax params pickle (was .zip in SB3 version)
  videos/gen{G:04d}_id{ID:06d}.mp4

Usage
-----
    from experiment_mjx import run_mjx
    archive = run_mjx(cfg, grader=fake_grader)

    # Or as a standalone CLI:
    python experiment_mjx.py --debug
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
from evolution_mjx import BaseEvolutionMJX, make_evolution_mjx
from data_handler  import ControllerResult, result_to_dict


# ---------------------------------------------------------------------------
# Archive / grader factories (identical to experiment.py)
# ---------------------------------------------------------------------------

def _resolve_target_file(cfg: ExperimentConfig) -> Path:
    """Path to the natural-language target file (relative → package dir)."""
    p = Path(cfg.target_file)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _read_target_behaviour(cfg: ExperimentConfig) -> str:
    """Read the one-line target behaviour the VLM grades against."""
    path = _resolve_target_file(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Target file not found: {path}\n"
            f"Create it (e.g. `nano {path}`) with a natural-language description "
            f"of the behaviour to evolve toward, e.g.:\n"
            f"  a forced and awkward gait"
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Target file is empty: {path}  (edit it with `nano`).")
    return text


def _make_grader(cfg: ExperimentConfig):
    if cfg.grader_type != "gemini":
        raise NotImplementedError(f"grader_type={cfg.grader_type!r} not supported (VLM only).")

    from gemini_prompts import make_prompt_config, LocomotionScoringWeights
    from vlm_grader     import LocomotionGrader

    target  = _read_target_behaviour(cfg)
    weights = LocomotionScoringWeights(
        coherence   = cfg.vlm_weight_coherence,
        originality = cfg.vlm_weight_originality,
        potential    = cfg.vlm_weight_potential,
    )
    prompt_cfg = make_prompt_config(
        name             = cfg.prompt_name or "target",
        target_behaviour = target,
        weights          = weights,
    )
    print(f"[experiment_mjx] Target ({_resolve_target_file(cfg).name}): {target!r}")

    log_path = str(cfg.run_dir / "vlm_responses.jsonl")

    if cfg.use_fake_grader:
        return LocomotionGrader(
            api_key           = "",
            prompt_config     = prompt_cfg,
            model_name        = cfg.gemini_model,
            batch_size        = cfg.batching,
            fake              = True,
            response_log_path = log_path,
            debug             = False,
        )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from api_keys import APIKEY_GEMINI
    return LocomotionGrader(
        api_key           = APIKEY_GEMINI,
        prompt_config     = prompt_cfg,
        model_name        = cfg.gemini_model,
        batch_size        = cfg.batching,
        fake              = False,
        response_log_path = log_path,
        debug             = False,
    )


def _make_archive(cfg: ExperimentConfig):
    if cfg.strategy == "mu_lambda":
        return MuLambdaArchive(mu=cfg.mu)
    if cfg.strategy == "map_elite":
        feature_dims = []; feature_bins: dict = {}; dim_labels: dict = {}
        if cfg.descriptor_config_name:
            try:
                from descriptor import get_descriptor_config
                d = get_descriptor_config(cfg.descriptor_config_name)
                feature_dims = list(d.feature_dims)
                feature_bins = {item.name: item.bins for item in d.items if item.bins}
                dim_labels   = {item.name: item.bin_labels for item in d.items if item.bin_labels}
            except Exception:
                pass
        return MapEliteArchive(feature_dims=feature_dims, feature_bins=feature_bins, dim_labels=dim_labels)
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
# Logging helpers (identical to experiment.py)
# ---------------------------------------------------------------------------

def _log_individuals(log_path: Path, individuals: "list[ControllerResult]") -> None:
    with open(log_path, "a") as f:
        for r in individuals:
            f.write(json.dumps(result_to_dict(r)) + "\n")


def _log_generation(log_path, generation, phase, results, archive, elapsed_s) -> None:
    best  = archive.best()
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


def _print_progress(generation, n_generations, phase, results, archive, elapsed_s) -> None:
    best  = archive.best()
    stats = archive.history[-1] if archive.history else None
    extra = f"  cells={len(archive.grid)}" if hasattr(archive, "grid") else ""
    best_s = f"({best.fitness:+.4f}, {best.individual_id})" if best else "N/A"
    mean_s = f"{stats.mean_fitness:+.4f}" if stats else "N/A"
    print(
        f"[gen {generation:>3} / {n_generations}]  {phase:<8}  "
        f"pool_n={len(results):<3}  best={best_s}  mean={mean_s}  "
        f"{elapsed_s:.1f}s{extra}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# run_mjx()
# ---------------------------------------------------------------------------

def run_mjx(
    cfg:    ExperimentConfig,
    grader = None,
):
    """
    Run a full MJX-backend experiment.

    Parameters
    ----------
    cfg    : ExperimentConfig (from Controller_MJX/config.py).
    grader : optional pre-built grader (fake grader for tests).

    Returns
    -------
    archive : final MuLambdaArchive or MapEliteArchive.
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
        print("[experiment_mjx] Building grader …")
        grader = _make_grader(cfg)
        print(f"[experiment_mjx] Grader ready ({type(grader).__name__}).")

    archive = _make_archive(cfg)
    rng     = np.random.default_rng(cfg.seed)
    evo: BaseEvolutionMJX = make_evolution_mjx(cfg, run_dir=run_dir, rng=rng)
    print(f"[experiment_mjx] Archive + evolution ready ({cfg.strategy}).")

    # ---- Generation 0 --------------------------------------------------------
    print(f"\n[experiment_mjx] Initial population — "
          f"{cfg.init_population_size or '(default)'} individuals.")
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
            archive=archive, grader=grader,
            generation=generation, id_counter=id_counter,
        )
        archive.update(results)
        elapsed = time.perf_counter() - t0
        _log_individuals(indiv_log_path, [r for r in results if r.individual_id >= prev_id])
        _print_progress(generation, cfg.n_generations, "step", results, archive, elapsed)
        _log_generation(log_path, generation, "step", results, archive, elapsed)
        if generation % cfg.save_every_n_gen == 0:
            _save_archive(archive, _archive_path(run_dir, generation))

    # ---- Final save ----------------------------------------------------------
    final_path = run_dir / "archive_final.json"
    _save_archive(archive, final_path)
    print(f"\n[experiment_mjx] Done. Final archive → {final_path}")
    archive.summary()
    return archive


# ---------------------------------------------------------------------------
# resume_mjx()
# ---------------------------------------------------------------------------

def resume_mjx(run_dir: Union[str, Path], grader=None):
    """Resume an interrupted MJX run from the latest archive snapshot."""
    run_dir        = Path(run_dir)
    cfg            = ExperimentConfig.load(str(run_dir / "config.json"))
    log_path       = run_dir / "log.jsonl"
    indiv_log_path = run_dir / "individuals_log.jsonl"

    snapshots = sorted(run_dir.glob("archive_gen*.json"))
    if not snapshots:
        raise FileNotFoundError(f"No archive snapshots found in {run_dir}.")
    latest    = snapshots[-1]
    start_gen = int(latest.stem.replace("archive_gen", "")) + 1
    print(f"\n[experiment_mjx] Resuming from gen {start_gen - 1}  ({latest.name})")

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
    evo = make_evolution_mjx(cfg, run_dir=run_dir, rng=rng)

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
    archive.summary()
    return archive


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="MJX controller-study experiment.")
    parser.add_argument("--strategy",    default=None, choices=["mu_lambda", "map_elite"])
    parser.add_argument("--mu",          type=int, default=None)
    parser.add_argument("--lambda_",     type=int, default=None)
    parser.add_argument("--n_gen",       type=int, default=None)
    parser.add_argument("--n_init_steps",type=int, default=None)
    parser.add_argument("--n_warm_steps",type=int, default=None)
    parser.add_argument("--n_envs_mjx",  type=int, default=None)
    parser.add_argument("--prompt",      default=None, help="label for this target (GraderOutput.prompt_set)")
    parser.add_argument("--target-file", dest="target_file", default=None,
                        help="path to natural-language target.txt (default: package target.txt)")
    parser.add_argument("--fake-grader", action="store_true",
                        help="synthetic VLM responses — no network / no API cost")
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--resume",      default=None, metavar="RUN_DIR")
    parser.add_argument("--debug",       action="store_true")
    args = parser.parse_args()

    if args.debug:
        return _debug_smoke()

    if args.resume:
        resume_mjx(args.resume)
        return

    cfg = ExperimentConfig()
    if args.strategy is not None:      cfg.strategy = args.strategy
    if args.mu is not None:            cfg.mu = args.mu
    if args.lambda_ is not None:       cfg.lambda_ = args.lambda_
    if args.n_gen is not None:         cfg.n_generations = args.n_gen
    if args.n_init_steps is not None:  cfg.n_init_steps = args.n_init_steps
    if args.n_warm_steps is not None:  cfg.n_warm_steps = args.n_warm_steps
    if args.n_envs_mjx is not None:    cfg.n_envs_mjx = args.n_envs_mjx
    if args.prompt is not None:        cfg.prompt_name = args.prompt
    if args.target_file is not None:   cfg.target_file = args.target_file
    if args.fake_grader:               cfg.use_fake_grader = True
    if args.seed is not None:          cfg.seed = args.seed
    if args.output_dir is not None:    cfg.output_dir = args.output_dir

    run_mjx(cfg)


# ---------------------------------------------------------------------------
# Debug smoke run
# ---------------------------------------------------------------------------

def _debug_smoke():
    import tempfile

    print("=" * 60)
    print("  experiment_mjx.py — debug smoke run (fake grader)")
    print("=" * 60)

    class _FakeOut:
        def __init__(self, fit):
            self.fitness = fit
            self.raw_scores = {"coherence": fit}
            self.method = "fake"
            self.prompt_set = "fake"
            self.extra = {"vlm_descriptors": {}}

    class _FakeGrader:
        def __init__(self): self.rng = np.random.default_rng(0)
        def score_batch(self, videos, debug=False, reference_video=None):
            return {vid: _FakeOut(float(self.rng.uniform(0.0, 1.0))) for vid, _ in videos}

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ExperimentConfig(
            run_id               = "debug_smoke_mjx",
            strategy             = "mu_lambda",
            mu                   = 2,
            lambda_              = 2,
            n_generations        = 1,
            init_population_size = 2,
            n_init_steps         = 2 * 8,
            n_warm_steps         = 2 * 8,
            n_envs_mjx           = 2,
            n_steps_per_env      = 8,
            batch_size           = 8,
            episode_duration     = 1.5,
            output_dir           = tmp,
        )
        archive = run_mjx(cfg, grader=_FakeGrader())
        assert archive.best() is not None

        run_dir = Path(tmp) / cfg.run_id
        assert (run_dir / "config.json").exists()
        assert (run_dir / "log.jsonl").exists()
        assert (run_dir / "individuals_log.jsonl").exists()
        assert (run_dir / "archive_final.json").exists()

        n_policies = len(list((run_dir / "policies").glob("*.params")))
        n_videos   = len(list((run_dir / "videos").glob("*.mp4")))
        print(f"\n  smoke run produced {n_policies} policies and {n_videos} videos.")
        assert n_policies > 0
        assert n_videos > 0

    print("\nAll experiment_mjx.py smoke checks passed.")


if __name__ == "__main__":
    _cli()
