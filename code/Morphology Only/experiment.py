"""
experiment.py
=============
Main experiment loop: ties config, rendering, grading, archive and
evolution together into a single reproducible run.

Entry points
------------
run(cfg)          — run a full experiment from an ExperimentConfig.
                    Returns the final archive.
resume(run_dir)   — reload a partially-completed run and continue from
                    the last saved snapshot.

Directory layout (inside cfg.run_dir)
--------------------------------------
  config.json                  — frozen copy of ExperimentConfig
  archive_gen{N:04d}.json      — archive snapshot every save_every_n_gen
  archive_final.json           — archive at end of run
  log.jsonl                    — one JSON line per generation
  renders/gen{N:04d}/          — rendered PNGs (only when save_renders=True)

Progress output (printed to stdout)
------------------------------------
  [gen  0 / 50]  init     n=10   best=+0.31415  mean=+0.11234
  [gen  1 / 50]  step     n=30   best=+0.41234  mean=+0.15678
  ...

Debug
-----
Run this file to execute a short smoke-test experiment using fake
renderer and grader (no CLIP or MuJoCo required).
"""

from __future__ import annotations

import json
import shutil
import sys, os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config       import ExperimentConfig
from archive      import MuLambdaArchive, MapEliteArchive
from evolution    import BaseEvolution, make_evolution
from rendering    import MorphologyRenderer, RenderConfig, CameraView
from grader       import CLIPGrader, GeminiGrader, MorphologyGrader
from CLIP_prompts import get_clip_prompt_set
from gemini_prompts import get_gemini_prompt_set
from data_handler import MorphologyResult, result_to_dict
from report       import generate_report
try:
    from descriptor import get_descriptor_config as _get_descriptor_config
    _DESCRIPTOR_AVAILABLE = True
except ImportError:
    _DESCRIPTOR_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api_keys import APIKEY_GEMINI

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_renderer(cfg: ExperimentConfig) -> MorphologyRenderer:
    """Build a MorphologyRenderer from config."""
    views = [
        CameraView(
            azimuth   = v["azimuth"],
            elevation = v["elevation"],
            distance  = v["distance"],
            lookat    = tuple(v["lookat"]),
        )
        for v in cfg.camera_views
    ]
    render_cfg = RenderConfig(
        width           = cfg.render_width,
        height          = cfg.render_height,
        camera_views    = views,
        floor_clearance = cfg.floor_clearance,
    )
    return MorphologyRenderer(render_cfg)


def _make_grader(cfg: ExperimentConfig) -> MorphologyGrader:
    """Build a CLIPGrader from config."""
    if cfg.grader_type == "clip":
        prompt_set = get_clip_prompt_set(cfg.prompt_name)
        return CLIPGrader(
            model_name     = cfg.clip_model,
            pretrained     = cfg.clip_pretrained,
            cache_dir      = cfg.clip_cache_dir,
            prompt_set     = prompt_set,
            scoring_method = cfg.scoring_method,
        )
    elif cfg.grader_type == "gemini":
        prompt_set = get_gemini_prompt_set(cfg.prompt_name)
        descriptor_config = None
        if _DESCRIPTOR_AVAILABLE and getattr(cfg, "descriptor_config_name", ""):
            descriptor_config = _get_descriptor_config(cfg.descriptor_config_name)
        return GeminiGrader(
            api_key=APIKEY_GEMINI,
            prompt_config=prompt_set,
            model_name=cfg.gemini_model,
            batch_size=cfg.batching,
            descriptor_config=descriptor_config,
        )
    else:
        raise AttributeError(f"{cfg.grader_type} not recognised as grader.")


def _make_archive(cfg: ExperimentConfig) -> Union[MuLambdaArchive, MapEliteArchive]:
    """Build the appropriate archive from config."""
    if cfg.strategy == "mu_lambda":
        return MuLambdaArchive(mu=cfg.mu)
    elif cfg.strategy == "map_elite":
        feature_dims = None
        feature_bins = None
        dim_labels   = None
        if _DESCRIPTOR_AVAILABLE and getattr(cfg, "descriptor_config_name", ""):
            try:
                desc_cfg     = _get_descriptor_config(cfg.descriptor_config_name)
                feature_dims = desc_cfg.feature_dims
                feature_bins = {
                    item.name: item.bins
                    for item in desc_cfg.items
                    if item.bins
                }
                dim_labels = {
                    item.name: item.bin_labels
                    for item in desc_cfg.items
                    if item.bin_labels
                }
            except KeyError:
                pass
        return MapEliteArchive(
            feature_dims  = feature_dims,
            feature_bins  = feature_bins,
            dim_labels    = dim_labels,
            symmetry_bins = getattr(cfg, "symmetry_bins", [0.5, 0.8]),
        )
    else:
        raise ValueError(f"Unknown strategy '{cfg.strategy}'.")


def _archive_path(run_dir: Path, generation: int) -> Path:
    return run_dir / f"archive_gen{generation:04d}.json"


def _save_archive(archive, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive.save(str(path))


def _load_archive(cfg: ExperimentConfig, path: Path) -> Union[MuLambdaArchive, MapEliteArchive]:
    if cfg.strategy == "mu_lambda":
        return MuLambdaArchive.load(str(path))
    else:
        return MapEliteArchive.load(str(path))


def _save_best_render(archive, renderer, run_dir: Path, label: str) -> Optional[str]:
    """
    Render the current best individual and save it to
      run_dir/renders/<label>_id{individual_id:06d}.png

    The individual_id is embedded in the filename so the PNG can always be
    matched to the corresponding entry in the archive JSON.

    Also writes the path back to best.render_path so the archive record
    stays in sync (useful for the final best snapshot).

    Returns the saved path, or None if the archive is empty.
    """
    best = archive.best()
    if best is None:
        return None
    filename  = f"{label}_id{best.individual_id:06d}.png"
    save_path = run_dir / "renders" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    renderer.render(best.morphology, save_path=str(save_path))
    best.render_path = str(save_path)
    return str(save_path)


def _log_individuals(log_path: Path, individuals: list[MorphologyResult]) -> None:
    """Append one JSON line per individual to individuals_log.jsonl."""
    with open(log_path, "a") as f:
        for r in individuals:
            f.write(json.dumps(result_to_dict(r)) + "\n")


def _log_generation(
    log_path:   Path,
    generation: int,
    phase:      str,
    results:    list[MorphologyResult],
    archive,
    elapsed_s:  float,
) -> None:
    """Append one JSON line to log.jsonl."""
    best = archive.best()
    entry = {
        "generation":    generation,
        "phase":         phase,
        "n_evaluated":   len(results),
        "best_fitness":  best.fitness if best else None,
        "best_id":       best.individual_id if best else None,
        "elapsed_s":     round(elapsed_s, 2),
    }
    if hasattr(archive, "population"):          # MuLambda
        entry["population_size"] = len(archive.population)
    if hasattr(archive, "grid"):                # MapElite
        entry["cells_filled"] = len(archive.grid)

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _print_progress(
    generation:   int,
    n_generations: int,
    phase:        str,
    results:      list[MorphologyResult],
    archive,
    elapsed_s:    float,
) -> None:
    best  = archive.best()
    stats = archive.history[-1] if archive.history else None
    extra = ""
    if hasattr(archive, "grid"):
        extra = f"  cells={len(archive.grid)}"
    print(
        f"[gen {generation:>3} / {n_generations}]  "
        f"{phase:<8}  n={len(results):<4}  "
        f"best={best.fitness:+.5f}  "
        f"mean={stats.mean_fitness:+.5f}" if stats else
        f"[gen {generation:>3} / {n_generations}]  "
        f"{phase:<8}  n={len(results):<4}  "
        f"best={best.fitness if best else '?':+.5f}",
        f"  {elapsed_s:.1f}s{extra}",
        sep="",
    )


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(
    cfg:          ExperimentConfig,
    save_renders: bool = False,
    renderer      = None,
    grader        = None,
) -> Union[MuLambdaArchive, MapEliteArchive]:
    """
    Run a complete experiment from scratch.

    Parameters
    ----------
    cfg          : ExperimentConfig with all parameters.
    save_renders : if True, save rendered PNGs to run_dir/renders/.
    renderer     : optional pre-built MorphologyRenderer (for testing).
    grader       : optional pre-built grader (for testing).

    Returns
    -------
    The final archive after n_generations steps.
    """
    run_dir       = cfg.run_dir
    log_path      = run_dir / "log.jsonl"
    indiv_log_path = run_dir / "individuals_log.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reset tmp render folder
    if cfg.save_all_render_tmp:
        tmp_render_dir = Path(cfg.output_dir) / "last_render"
        if tmp_render_dir.exists():
            shutil.rmtree(tmp_render_dir)
        tmp_render_dir.mkdir(parents=True)
        print(f"[experiment] Tmp render dir reset: {tmp_render_dir}")

    # Save frozen config
    cfg.save(str(run_dir / "config.json"))
    print(f"\n{'='*60}")
    cfg.describe()
    print(f"{'='*60}\n")

    # Build components (use injected instances if provided)
    if renderer is None:
        print("[experiment] Building renderer...")
        renderer = _make_renderer(cfg)
        print(f"Renderer initialized : {len(renderer.config.camera_views)} * {renderer.config.width}x{renderer.config.height}")

    if grader is None:
        print("[experiment] Building grader...")
        grader = _make_grader(cfg)
        print("Grader Initialized.")

    archive  = _make_archive(cfg)
    rng      = np.random.default_rng(cfg.seed)
    evo: BaseEvolution = make_evolution(cfg, rng)
    print(f"[experiment] Archive, Random generator, Evolution manager initialized.")

    print(f"[experiment] Starting experiment with random individuals...")
    # ---- Generation 0: initialise -------------------------------------------
    t0 = time.perf_counter()
    init_results, id_counter = evo.initialise(
        renderer     = renderer,
        grader       = grader,
        id_counter   = 0,
        save_renders = save_renders,
        render_dir   = str(run_dir / "renders" / "gen0000") if save_renders else None,
    )
    archive.update(init_results)
    elapsed = time.perf_counter() - t0
    _log_individuals(indiv_log_path, init_results)

    _print_progress(0, cfg.n_generations, "init", init_results, archive, elapsed)
    _log_generation(log_path, 0, "init", init_results, archive, elapsed)

    if 0 % cfg.save_every_n_gen == 0:
        _save_archive(archive, _archive_path(run_dir, 0))
    if cfg.save_best_every_n_gen > 0 and 0 % cfg.save_best_every_n_gen == 0:
        _save_best_render(archive, renderer, run_dir, f"best/gen{0:04d}")

    # ---- Evolution loop ------------------------------------------------------
    for generation in range(1, cfg.n_generations + 1):
        t0 = time.perf_counter()

        gen_render_dir = (
            str(run_dir / "renders" / f"gen{generation:04d}")
            if save_renders else None
        )
        prev_id = id_counter
        results, id_counter = evo.step(
            archive      = archive,
            renderer     = renderer,
            grader       = grader,
            generation   = generation,
            id_counter   = id_counter,
            save_renders = save_renders,
            render_dir   = gen_render_dir,
        )
        archive.update(results)
        elapsed = time.perf_counter() - t0
        # Only log truly new individuals (offspring), not re-tagged parents
        _log_individuals(indiv_log_path, [r for r in results if r.individual_id >= prev_id])

        _print_progress(generation, cfg.n_generations, "step", results, archive, elapsed)
        _log_generation(log_path, generation, "step", results, archive, elapsed)

        if generation % cfg.save_every_n_gen == 0:
            _save_archive(archive, _archive_path(run_dir, generation))
        if cfg.save_best_every_n_gen > 0 and generation % cfg.save_best_every_n_gen == 0:
            _save_best_render(archive, renderer, run_dir, f"best/gen{generation:04d}")

    # ---- Final save ----------------------------------------------------------
    final_path = run_dir / "archive_final.json"
    _save_archive(archive, final_path)
    if cfg.save_final_best:
        _save_best_render(archive, renderer, run_dir, "best_final")
    print(f"\n[experiment] Done.  Final archive → {final_path}")
    archive.summary()

    generate_report(run_dir, print_report=False)

    renderer.close()
    return archive


# ---------------------------------------------------------------------------
# resume()
# ---------------------------------------------------------------------------

def resume(
    run_dir:      Union[str, Path],
    save_renders: bool = False,
    renderer      = None,
    grader        = None,
) -> Union[MuLambdaArchive, MapEliteArchive]:
    """
    Resume an interrupted experiment from the latest snapshot.

    Loads config.json and the most recent archive_gen*.json snapshot,
    then continues the evolution loop from where it left off.

    Parameters
    ----------
    run_dir      : directory of the interrupted run (must contain config.json
                   and at least one archive_gen*.json).
    save_renders : if True, continue saving PNGs.
    renderer     : optional pre-built MorphologyRenderer (for testing).
    grader       : optional pre-built grader (for testing).

    Returns
    -------
    The final archive.
    """
    run_dir        = Path(run_dir)
    cfg            = ExperimentConfig.load(str(run_dir / "config.json"))
    log_path       = run_dir / "log.jsonl"
    indiv_log_path = run_dir / "individuals_log.jsonl"

    # Ensure tmp render folder exists (not wiped on resume — keeps previous renders)
    if cfg.save_all_render_tmp:
        tmp_render_dir = Path(cfg.output_dir) / "last_render"
        tmp_render_dir.mkdir(parents=True, exist_ok=True)
        print(f"[experiment] Tmp render dir: {tmp_render_dir}")

    # Find latest snapshot
    snapshots = sorted(run_dir.glob("archive_gen*.json"))
    if not snapshots:
        raise FileNotFoundError(f"No archive snapshots found in {run_dir}.")
    latest    = snapshots[-1]
    # Parse generation number from filename (archive_gen0010.json → 10)
    start_gen = int(latest.stem.replace("archive_gen", "")) + 1
    print(f"\n[experiment] Resuming from gen {start_gen - 1}  ({latest.name})")
    cfg.describe()

    # Reload archive
    archive = _load_archive(cfg, latest)

    # Infer id_counter from archive history (max individual_id seen so far + 1)
    if hasattr(archive, "population") and archive.population:
        id_counter = max(r.individual_id for r in archive.population) + 1
    elif hasattr(archive, "grid") and archive.grid:
        id_counter = max(r.individual_id for r in archive.grid.values()) + 1
    else:
        id_counter = 0

    # Rebuild renderer, grader, evolution (use injected if provided)
    if renderer is None:
        print("[experiment] Rebuilding renderer and grader...")
        renderer = _make_renderer(cfg)
    if grader is None:
        grader = _make_grader(cfg)
    rng      = np.random.default_rng(cfg.seed + start_gen)   # advance seed
    evo      = make_evolution(cfg, rng)

    print("[experiment] Done, loop restarted.")
    # Resume loop
    for generation in range(start_gen, cfg.n_generations + 1):
        t0 = time.perf_counter()

        gen_render_dir = (
            str(run_dir / "renders" / f"gen{generation:04d}")
            if save_renders else None
        )
        prev_id = id_counter
        results, id_counter = evo.step(
            archive      = archive,
            renderer     = renderer,
            grader       = grader,
            generation   = generation,
            id_counter   = id_counter,
            save_renders = save_renders,
            render_dir   = gen_render_dir,
        )
        archive.update(results)
        elapsed = time.perf_counter() - t0
        _log_individuals(indiv_log_path, [r for r in results if r.individual_id >= prev_id])

        _print_progress(generation, cfg.n_generations, "step", results, archive, elapsed)
        _log_generation(log_path, generation, "step", results, archive, elapsed)

        if generation % cfg.save_every_n_gen == 0:
            _save_archive(archive, _archive_path(run_dir, generation))
        if cfg.save_best_every_n_gen > 0 and generation % cfg.save_best_every_n_gen == 0:
            _save_best_render(archive, renderer, run_dir, f"best/gen{generation:04d}")

    final_path = run_dir / "archive_final.json"
    _save_archive(archive, final_path)
    if cfg.save_final_best:
        _save_best_render(archive, renderer, run_dir, "best_final")
    print(f"\n[experiment] Done.  Final archive → {final_path}")
    archive.summary()

    generate_report(run_dir, print_report=False)

    renderer.close()
    return archive


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    """
    Minimal CLI.  Usage:
      python experiment.py                          → run default config
      python experiment.py --strategy map_elite     → override strategy
      python experiment.py --resume results/run_X   → resume a run
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run a morphology evolution experiment.")
    parser.add_argument("--strategy", default=None,
                        choices=["mu_lambda", "map_elite"],
                        help="Evolution strategy")
    parser.add_argument("--mu",         type=int,   default=None)
    parser.add_argument("--lambda_",    type=int,   default=None)
    parser.add_argument("--n_gen",      type=int,   default=None)
    parser.add_argument("--prompt_set", default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--save_renders", action="store_true")
    parser.add_argument("--resume",     default=None,
                        metavar="RUN_DIR", help="Resume from a saved run directory.")
    args = parser.parse_args()

    if args.resume:
        resume(args.resume, save_renders=args.save_renders)
        return

    cfg = ExperimentConfig()

    if args.strategy is not None: cfg.strategy = args.strategy
    if args.mu        is not None: cfg.mu              = args.mu
    if args.lambda_   is not None: cfg.lambda_         = args.lambda_
    if args.n_gen     is not None: cfg.n_generations   = args.n_gen
    if args.prompt_set is not None: cfg.prompt_set_name = args.prompt_set
    if args.seed is not None: cfg.seed = args.seed
    if args.output_dir is not None: cfg.output_dir = args.output_dir

    run(cfg, save_renders=args.save_renders)


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os
    sys.path.insert(0, str(Path(__file__).parent))

    # If any arguments are passed, treat this as a normal CLI run.
    # Use --debug explicitly to run the self-test suite.
    print(sys.argv)
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] != "--debug") or (len(sys.argv) > 2):
        _cli()
        sys.exit(0)

    print("=" * 60)
    print("  experiment.py — debug mode (fake renderer + grader)")
    print("=" * 60)

    # ---- Fake renderer and grader (no CLIP / MuJoCo) -----------------------
    import numpy as _np
    from PIL import Image as _PILImage
    from grader import GraderOutput

    _fake_rng = _np.random.default_rng(99)

    class _FakeRenderer:
        def render(self, morph, save_path=None, debug=False):
            img = _PILImage.new("RGB", (32, 32), color=(100, 150, 200))
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                img.save(save_path)
            return img
        def close(self):
            pass

    class _FakeGrader:
        def score(self, image, debug=False):
            return GraderOutput(
                fitness    = float(_fake_rng.uniform(-1, 1)),
                raw_scores = {},
                method     = "fake",
                prompt_set = "fake",
            )

    fake_renderer = _FakeRenderer()
    fake_grader   = _FakeGrader()

    # ---- Run mu_lambda experiment -------------------------------------------
    print("\n[1] mu_lambda experiment (3 generations)\n")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ExperimentConfig(
            run_id        = "debug_mu",
            strategy      = "mu_lambda",
            mu            = 4,
            lambda_       = 6,
            n_generations = 3,
            seed          = 0,
            output_dir    = tmp,
            save_every_n_gen = 2,
        )
        archive = run(cfg, renderer=fake_renderer, grader=fake_grader)
        assert archive.best() is not None
        assert len(archive.history) == 4   # gen 0 + gens 1-3
        log_lines = (Path(tmp) / cfg.run_id / "log.jsonl").read_text().strip().split("\n")
        assert len(log_lines) == 4, f"Expected 4 log lines, got {len(log_lines)}"
        print(f"\n  mu_lambda: OK  best={archive.best().fitness:+.5f}  "
              f"log_lines={len(log_lines)}")

    # ---- Run map_elite experiment -------------------------------------------
    print("\n[2] map_elite experiment (3 generations)\n")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ExperimentConfig(
            run_id        = "debug_me",
            strategy      = "map_elite",
            mu            = 4,
            lambda_       = 6,
            n_generations = 3,
            seed          = 0,
            output_dir    = tmp,
            save_every_n_gen = 2,
        )
        archive = run(cfg, renderer=fake_renderer, grader=fake_grader)
        assert archive.best() is not None
        assert len(archive.grid) > 0
        print(f"\n  map_elite: OK  cells={len(archive.grid)}  "
              f"best={archive.best().fitness:+.5f}")

    # ---- Resume test --------------------------------------------------------
    print("\n[3] resume test (interrupt after gen 1, resume to gen 3)\n")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ExperimentConfig(
            run_id        = "debug_resume",
            strategy      = "mu_lambda",
            mu            = 4,
            lambda_       = 6,
            n_generations = 3,
            seed          = 0,
            output_dir    = tmp,
            save_every_n_gen = 1,
        )
        # Run only gen 0 + gen 1 manually, save snapshots
        run_dir = cfg.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(str(run_dir / "config.json"))

        from archive   import MuLambdaArchive
        from evolution import MuLambdaEvolution

        _evo = MuLambdaEvolution(cfg, rng=_np.random.default_rng(0))
        _arc = MuLambdaArchive(mu=cfg.mu)

        _init, _id = _evo.initialise(fake_renderer, fake_grader)
        _arc.update(_init)
        _save_archive(_arc, _archive_path(run_dir, 0))

        _res1, _id = _evo.step(_arc, fake_renderer, fake_grader, generation=1, id_counter=_id)
        _arc.update(_res1)
        _save_archive(_arc, _archive_path(run_dir, 1))

        print(f"  Simulated run interrupted at gen 1.")

        # Resume from snapshot, injecting fake renderer/grader
        final = resume(str(run_dir), renderer=fake_renderer, grader=fake_grader)
        assert final.best() is not None
        assert len(final.history) >= 2    # gen 1 + resumed gens
        print(f"\n  resume: OK  best={final.best().fitness:+.5f}  "
              f"history={len(final.history)}")

    print("\nAll experiment checks passed.")
