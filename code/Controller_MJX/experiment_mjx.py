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

import os
import sys

# Headless EGL rendering + XLA GPU flags — must precede the JAX/mujoco imports.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_FLAGS",
    "--xla_gpu_enable_cublaslt=true --xla_gpu_autotune_level=4")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")


# ---------------------------------------------------------------------------
# Single-GPU pinning — MUST run before JAX is imported (evolution_mjx pulls it).
# ---------------------------------------------------------------------------
# JAX otherwise initialises on EVERY visible GPU and preallocates
# MEM_FRACTION × VRAM on each — which on this shared machine OOMs against GPU 0
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

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from config        import ExperimentConfig
from archive       import MuLambdaArchive, MapEliteArchive
from evolution_mjx import BaseEvolutionMJX, make_evolution_mjx
from data_handler  import ControllerResult, result_to_dict


# ---------------------------------------------------------------------------
# Full terminal logging  (everything printed → run_dir/log.txt)
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror writes to the original stream AND a log file, flushing each write.

    Flushing every write keeps log.txt complete even if the process crashes
    mid-run, so the whole history can be retraced. Carriage-return progress
    lines (printed with end="" + leading "\\r") are written to the file as
    newlines so each update becomes its own readable line in the log.
    """

    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file   = file_handle

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._file.write(data.replace("\r", "\n") if "\r" in data else data)
        self._file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()


class _capture_terminal:
    """Context manager: tee stdout+stderr into ``path`` for the whole run."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        self._fh.write(f"\n{'=' * 70}\n[log started {time.strftime('%Y-%m-%d %H:%M:%S')}]\n{'=' * 70}\n")
        self._fh.flush()
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._old_out, self._fh)
        sys.stderr = _Tee(self._old_err, self._fh)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            import traceback
            self._fh.write("\n[UNCAUGHT EXCEPTION]\n")
            traceback.print_exception(exc_type, exc, tb, file=self._fh)
            self._fh.flush()
        sys.stdout, sys.stderr = self._old_out, self._old_err
        self._fh.close()
        return False   # never swallow the exception


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


def _load_descriptor_config(cfg: ExperimentConfig):
    """Load the MAP-Elites descriptor config for this run, or None.

    Only meaningful for strategy="map_elite". Raises a clear error if the
    configured name is unknown — a silent fallback would collapse the grid to a
    single cell and waste hours of GPU time.
    """
    if cfg.strategy != "map_elite" or not cfg.descriptor_config_name:
        return None
    from descriptor import get_descriptor_config
    return get_descriptor_config(cfg.descriptor_config_name)


def _snapshot_target(cfg: ExperimentConfig, run_dir: Path) -> None:
    """Persist the resolved target behaviour into the run dir.

    The target lives in an editable text file (cfg.target_file), so config.json
    only records the file path + a short label (prompt_name). Snapshotting the
    actual text makes each run self-describing — the analyser shows the real
    target instead of just the label, even if target.txt later changes.
    """
    try:
        text = _read_target_behaviour(cfg)
    except Exception:
        return
    try:
        (run_dir / "target.txt").write_text(text + "\n", encoding="utf-8")
    except Exception:
        pass


def _make_grader(cfg: ExperimentConfig):
    if cfg.grader_type != "gemini":
        raise NotImplementedError(f"grader_type={cfg.grader_type!r} not supported (VLM only).")

    from gemini_prompts import make_prompt_config, LocomotionScoringWeights
    from vlm_grader     import LocomotionGrader

    descriptor_config = _load_descriptor_config(cfg)
    if descriptor_config is not None:
        print(f"[experiment_mjx] MAP-Elites descriptors: {cfg.descriptor_config_name} "
              f"→ {descriptor_config.feature_dims}")

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
            n_score_request   = cfg.n_score_request,
            descriptor_config = descriptor_config,
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
        n_score_request   = cfg.n_score_request,
        descriptor_config = descriptor_config,
        debug             = False,
    )


def _make_archive(cfg: ExperimentConfig):
    if cfg.strategy == "mu_lambda":
        return MuLambdaArchive(mu=cfg.mu)
    if cfg.strategy == "map_elite":
        d = _load_descriptor_config(cfg)
        if d is None:
            raise ValueError(
                "strategy='map_elite' requires a valid descriptor_config_name "
                f"(got {cfg.descriptor_config_name!r}). Without it the grid collapses "
                "to a single cell. See descriptor.py for available configs."
            )
        feature_dims = list(d.feature_dims)
        feature_bins = {item.name: item.bins for item in d.items if item.bins}
        dim_labels   = {item.name: item.bin_labels for item in d.items if item.bin_labels}
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
# Human-readable run summary  (run_dir/SUMMARY.txt)
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _read_individuals_log(run_dir: Path) -> "list[dict]":
    """Load every evaluated individual from individuals_log.jsonl."""
    path = run_dir / "individuals_log.jsonl"
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_summary(
    run_dir:   Path,
    cfg:       ExperimentConfig,
    run_start: float,
    status:    str = "running",
    top_k:     int = 5,
    archive          = None,
) -> None:
    """Write a synthetic, human-readable SUMMARY.txt for the run.

    Refreshed after every generation so a readable digest exists even if the
    run later crashes. Built from individuals_log.jsonl (the full record of all
    evaluated individuals), so it covers the whole history, not just the
    surviving archive population.
    """
    rows    = _read_individuals_log(run_dir)
    elapsed = time.time() - run_start

    try:
        target = _read_target_behaviour(cfg)
    except Exception:
        target = "(target file unavailable)"

    # ---- run health --------------------------------------------------------
    def _failed(r: dict) -> bool:
        method = str(r.get("grader_method", ""))
        err    = (r.get("grader_extra") or {}).get("error")
        return method.endswith("_failed") or err is not None

    n_total   = len(rows)
    failed    = [r for r in rows if _failed(r)]
    zero_fit  = [r for r in rows if float(r.get("fitness", 0.0)) == 0.0]

    ranked = sorted(rows, key=lambda r: float(r.get("fitness", 0.0)), reverse=True)

    L: list[str] = []
    L.append("=" * 72)
    L.append("  RUN SUMMARY")
    L.append("=" * 72)
    L.append(f"run_id          : {cfg.run_id}")
    L.append(f"status          : {status}")
    L.append(f"target          : {target!r}")
    L.append(f"strategy        : {cfg.strategy}   (mu={cfg.mu}, lambda={cfg.lambda_})")
    L.append(f"init population : {cfg.resolved_init_population_size()}")
    L.append(f"generations     : {cfg.n_generations}")
    L.append(f"steps / train   : init(gen0)={cfg.n_init_steps:,}   warm(children)={cfg.n_warm_steps:,}")
    L.append(f"VLM grader      : {cfg.gemini_model}  "
             f"n_score_request={cfg.n_score_request}  batch={cfg.batching}")
    L.append(f"individuals eval: {n_total}")
    L.append(f"total run time  : {_fmt_duration(elapsed)}")
    L.append("")
    L.append("-" * 72)
    L.append("  RUN HEALTH")
    L.append("-" * 72)
    L.append(f"failed / wrong VLM answers : {len(failed)} / {n_total}")
    L.append(f"zero-fitness individuals   : {len(zero_fit)} / {n_total}")
    if failed:
        L.append("  failed ids: " + ", ".join(
            str(r.get("individual_id")) for r in failed[:20])
            + (" …" if len(failed) > 20 else ""))
    L.append("")

    # ---- MAP-Elites grid ---------------------------------------------------
    if archive is not None and getattr(archive, "grid", None) is not None:
        L.append("-" * 72)
        L.append("  MAP-ELITES GRID")
        L.append("-" * 72)
        n_buckets = 1
        for dim in archive.feature_dims:
            n_buckets *= len(archive.feature_bins.get(dim, [])) + 1
        L.append(f"descriptor cfg  : {cfg.descriptor_config_name}  "
                 f"axes={archive.feature_dims}")
        L.append(f"cells filled    : {len(archive.grid)} / {n_buckets}")
        for key in sorted(archive.grid):
            r = archive.grid[key]
            L.append(f"  {archive.feature_label(key):<44} "
                     f"fitness={r.fitness:+.4f}  id={r.individual_id}")
        L.append("")

    if not rows:
        L.append("(no individuals evaluated yet)")
        (run_dir / "SUMMARY.txt").write_text("\n".join(L) + "\n")
        return

    best = ranked[0]
    L.append("-" * 72)
    L.append("  BEST INDIVIDUAL")
    L.append("-" * 72)
    L.append(_format_individual(best))
    L.append("")

    L.append("-" * 72)
    L.append(f"  TOP {min(top_k, len(ranked))} INDIVIDUALS")
    L.append("-" * 72)
    for rank, r in enumerate(ranked[:top_k], 1):
        rs = r.get("raw_scores", {})
        L.append(
            f"  #{rank}  id={r.get('individual_id'):<4} gen={r.get('generation'):<3} "
            f"parent={r.get('parent_id')}  fitness={float(r.get('fitness', 0.0)):+.4f}  "
            f"(coh={rs.get('coherence', 0):.2f} orig={rs.get('originality', 0):.2f} "
            f"pot={rs.get('potential', 0):.2f})")
    L.append("")
    L.append(f"(full machine-readable record: individuals_log.jsonl  ·  "
             f"raw VLM responses: vlm_responses.jsonl  ·  full terminal: log.txt)")

    (run_dir / "SUMMARY.txt").write_text("\n".join(L) + "\n")


def _format_individual(r: dict) -> str:
    rs    = r.get("raw_scores", {})
    extra = r.get("grader_extra", {}) or {}
    lines = [
        f"id            : {r.get('individual_id')}",
        f"generation    : {r.get('generation')}",
        f"parent_id     : {r.get('parent_id')}",
        f"fitness       : {float(r.get('fitness', 0.0)):+.4f}",
        f"raw scores    : coherence={rs.get('coherence', 0):.3f}  "
        f"originality={rs.get('originality', 0):.3f}  potential={rs.get('potential', 0):.3f}",
    ]
    if "fitness_std" in extra:
        lines.append(f"score spread  : std={extra.get('fitness_std')} over "
                     f"{extra.get('n_scored_ok')} requests  "
                     f"samples={extra.get('fitness_samples')}")
    lines.append(f"policy params : {r.get('policy_path')}")
    lines.append(f"video (mp4)   : {r.get('video_path')}")
    # Reasons from the VLM
    if extra.get("observation"):
        lines.append(f"observation   : {extra['observation']}")
    if extra.get("interpretation"):
        lines.append(f"interpretation: {extra['interpretation']}")
    for dim in ("coherence", "originality", "potential"):
        reason = extra.get(f"{dim}_reason")
        if reason:
            lines.append(f"  {dim:<11} : {reason}")
    return "\n".join(lines)


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
    run_start      = time.time()

    cfg.save(str(run_dir / "config.json"))
    _snapshot_target(cfg, run_dir)
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
    evo.verbose_training = cfg.verbose_training   # per-update PPO progress (fps, rw, losses)
    print(f"[experiment_mjx] Archive + evolution ready ({cfg.strategy}).")

    # ---- Generation 0 --------------------------------------------------------
    print(f"\n[experiment_mjx] Initial population — "
          f"{cfg.resolved_init_population_size()} individuals.")
    t0 = time.perf_counter()
    init_results, id_counter = evo.initialise(grader, id_counter=0)
    archive.update(init_results)
    elapsed = time.perf_counter() - t0
    _log_individuals(indiv_log_path, init_results)
    _print_progress(0, cfg.n_generations, "init", init_results, archive, elapsed)
    _log_generation(log_path, 0, "init", init_results, archive, elapsed)
    if 0 % cfg.save_every_n_gen == 0:
        _save_archive(archive, _archive_path(run_dir, 0))
    _write_summary(run_dir, cfg, run_start, status="running (gen 0 done)", archive=archive)

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
        _write_summary(run_dir, cfg, run_start,
                       status=f"running (gen {generation}/{cfg.n_generations} done)",
                       archive=archive)

    # ---- Final save ----------------------------------------------------------
    final_path = run_dir / "archive_final.json"
    _save_archive(archive, final_path)
    _write_summary(run_dir, cfg, run_start, status="completed", archive=archive)
    print(f"\n[experiment_mjx] Done. Final archive → {final_path}")
    print(f"[experiment_mjx] Summary → {run_dir / 'SUMMARY.txt'}")
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
    _snapshot_target(cfg, run_dir)

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
    evo.verbose_training = cfg.verbose_training   # per-update PPO progress (fps, rw, losses)
    run_start = time.time()

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
        _write_summary(run_dir, cfg, run_start,
                       status=f"resumed · running (gen {generation}/{cfg.n_generations} done)",
                       archive=archive)

    final_path = run_dir / "archive_final.json"
    _save_archive(archive, final_path)
    _write_summary(run_dir, cfg, run_start, status="completed (resumed)", archive=archive)
    archive.summary()
    return archive


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="MJX controller-study experiment.")
    parser.add_argument("--strategy",    default=None, choices=["mu_lambda", "map_elite"])
    parser.add_argument("--descriptor",  default=None,
                        help="MAP-Elites descriptor config name (see descriptor.py)")
    parser.add_argument("--mu",          type=int, default=None)
    parser.add_argument("--lambda_",     type=int, default=None)
    parser.add_argument("--n_gen",       type=int, default=None)
    parser.add_argument("--init_ind",    type=int, default=None,
                        help="number of gen-0 individuals trained from scratch "
                             "(overrides init_population_size; 0 = strategy default)")
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
    parser.add_argument("--gpu",         default=None,
                        help="pin to this GPU index (already applied at import; here for --help)")
    parser.add_argument("--debug",       action="store_true")
    args = parser.parse_args()

    if args.debug:
        return _debug_smoke()

    if args.resume:
        with _capture_terminal(Path(args.resume) / "log.txt"):
            resume_mjx(args.resume)
        return

    cfg = ExperimentConfig()
    if args.strategy is not None:      cfg.strategy = args.strategy
    if args.descriptor is not None:    cfg.descriptor_config_name = args.descriptor
    if args.mu is not None:            cfg.mu = args.mu
    if args.lambda_ is not None:       cfg.lambda_ = args.lambda_
    if args.n_gen is not None:         cfg.n_generations = args.n_gen
    if args.init_ind is not None:      cfg.init_population_size = args.init_ind
    if args.n_init_steps is not None:  cfg.n_init_steps = args.n_init_steps
    if args.n_warm_steps is not None:  cfg.n_warm_steps = args.n_warm_steps
    if args.n_envs_mjx is not None:    cfg.n_envs_mjx = args.n_envs_mjx
    if args.prompt is not None:        cfg.prompt_name = args.prompt
    if args.target_file is not None:   cfg.target_file = args.target_file
    if args.fake_grader:               cfg.use_fake_grader = True
    if args.seed is not None:          cfg.seed = args.seed
    if args.output_dir is not None:    cfg.output_dir = args.output_dir

    with _capture_terminal(cfg.run_dir / "log.txt"):
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
