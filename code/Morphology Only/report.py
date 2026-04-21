"""
report.py
=========
Generate a human-readable report from a completed (or in-progress) run.

Reads the latest archive snapshot in a run directory and writes a
text file sorted by fitness (ascending — worst first, best last) so the
most interesting individuals are at the bottom.

Only scores and grader analysis are shown — the full morphology JSON is
omitted for readability.

Usage
-----
    # From Python:
    from report import generate_report
    generate_report("results/run_20260417_151235")

    # From the CLI:
    python report.py results/run_20260417_151235
    python report.py results/run_20260417_151235 --archive archive_gen0010.json
    python report.py results/run_20260417_151235 --no-save   # print only

Output
------
    results/run_20260417_151235/report.txt   (overwritten each run)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _load_population(archive_path: Path) -> list[dict]:
    """Load the population list from an archive JSON file."""
    with open(archive_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "population" in data and data["population"] is not None:
        return data["population"]
    if isinstance(data, dict) and "grid" in data:
        return list(data["grid"].values())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognised archive format in {archive_path}")


def _find_latest_archive(run_dir: Path) -> Path:
    """Return the most informative archive file in run_dir."""
    # Prefer archive_final if it exists
    final = run_dir / "archive_final.json"
    if final.exists():
        return final
    # Otherwise use the highest-numbered snapshot
    snapshots = sorted(run_dir.glob("archive_gen*.json"))
    if snapshots:
        return snapshots[-1]
    raise FileNotFoundError(f"No archive file found in {run_dir}")


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    """Simple word-wrap with a leading indent on continuation lines."""
    if not text:
        return ""
    words  = text.split()
    lines  = []
    line   = []
    length = 0
    for w in words:
        if length + len(w) + (1 if line else 0) > width:
            lines.append(" ".join(line))
            line   = [w]
            length = len(w)
        else:
            line.append(w)
            length += len(w) + (1 if len(line) > 1 else 0)
    if line:
        lines.append(" ".join(line))
    return ("\n" + indent).join(lines)


def _format_entry(rank: int, entry: dict, total: int) -> str:
    """Format one population entry as a text block."""
    fitness   = entry.get("fitness", 0.0)
    ind_id    = entry.get("individual_id", "?")
    gen       = entry.get("generation", "?")
    method    = entry.get("grader_method", "?")
    ps        = entry.get("prompt_set", "?")
    raw       = entry.get("raw_scores", {})
    desc      = entry.get("descriptors", {})
    extra     = entry.get("grader_extra", {})
    rpath     = entry.get("render_path")

    bar    = "─" * 72
    lines  = [bar]
    lines.append(
        f"Rank {rank:>3} / {total}  │  ID {ind_id:>5}  │  gen {gen:>3}  │  "
        f"fitness = {fitness:+.5f}  ({method})"
    )
    lines.append(bar)

    # ---- Scores ----
    if raw:
        score_parts = "  ".join(f"{k}={v}" for k, v in raw.items())
        lines.append(f"  Scores   : {score_parts}")

    # ---- Structural descriptors ----
    n_legs  = desc.get("n_legs", "?")
    n_root  = desc.get("n_root_legs", "?")
    n_br    = desc.get("n_branch_legs", "?")
    n_bp    = desc.get("n_body_parts", "?")
    sym     = desc.get("symmetry_score", "?")
    seg     = desc.get("mean_segment_length", "?")
    tr      = desc.get("torso_radius", "?")
    th      = desc.get("torso_height", "?")
    lines.append(
        f"  Morpho   : legs={n_legs} (root={n_root} branch={n_br} bparts={n_bp})  "
        f"sym={sym:.3f}  seg_len={seg:.3f} m"
        if isinstance(sym, float) and isinstance(seg, float)
        else f"  Morpho   : legs={n_legs}  root={n_root}  branch={n_br}  bparts={n_bp}"
    )
    if isinstance(tr, float) and isinstance(th, float):
        lines.append(f"  Torso    : radius={tr:.3f} m  half-height={th:.3f} m")

    if rpath:
        lines.append(f"  Render   : {rpath}")

    # ---- Grader analysis (Gemini) ----
    obs  = extra.get("observation", "")
    intp = extra.get("interpretation", "")
    c_r  = extra.get("coherence_reason", "")
    o_r  = extra.get("originality_reason", "")
    i_r  = extra.get("interest_reason", "")

    if obs or intp or c_r or o_r or i_r:
        lines.append("")
    if obs:
        lines.append(f"  Observation   : {_wrap(obs,  width=90, indent='                    ')}")
    if intp:
        lines.append(f"  Interpretation: {_wrap(intp, width=90, indent='                    ')}")
    if c_r:
        lines.append(f"  Coherence     : {_wrap(c_r,  width=90, indent='                    ')}")
    if o_r:
        lines.append(f"  Originality   : {_wrap(o_r,  width=90, indent='                    ')}")
    if i_r:
        lines.append(f"  Interest      : {_wrap(i_r,  width=90, indent='                    ')}")

    return "\n".join(lines)


def generate_report(
    run_dir:      str | Path,
    archive_name: Optional[str] = None,
    save:         bool          = True,
    print_report: bool          = True,
) -> str:
    """
    Generate a human-readable report for a run.

    Parameters
    ----------
    run_dir      : path to the run directory (contains config.json, archive_*.json).
    archive_name : specific archive filename to read (e.g. "archive_gen0010.json").
                   If None, uses archive_final.json or the latest snapshot.
    save         : if True, saves the report to run_dir/report.txt.
    print_report : if True, prints the report to stdout.

    Returns
    -------
    The report as a string.
    """
    run_dir = Path(run_dir)

    if archive_name:
        archive_path = run_dir / archive_name
    else:
        archive_path = _find_latest_archive(run_dir)

    population = _load_population(archive_path)

    # Sort ascending by fitness (worst first, best last)
    population = sorted(population, key=lambda e: e.get("fitness", 0.0))

    # Load config for header info
    config_path = run_dir / "config.json"
    cfg: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)

    run_id   = cfg.get("run_id",        run_dir.name)
    strategy = cfg.get("strategy",      "?")
    mu       = cfg.get("mu",            "?")
    lam      = cfg.get("lambda_",       "?")
    n_gen    = cfg.get("n_generations", "?")
    # prompt_name / grader_type are class-level attrs in config.py — not
    # serialised to JSON.  Fall back to inferring from the archive entries.
    prompt   = cfg.get("prompt_name",   None)
    grader   = cfg.get("grader_type",   None)
    if (prompt is None or grader is None) and population:
        sample = population[0]
        if prompt  is None: prompt  = sample.get("prompt_set",    "?")
        if grader  is None: grader  = sample.get("grader_method", "?")

    # Header
    sep   = "═" * 72
    lines = [
        sep,
        f"  RUN REPORT — {run_id}",
        sep,
        f"  Archive  : {archive_path.name}",
        f"  Strategy : {strategy}  (μ={mu}, λ={lam}, generations={n_gen})",
        f"  Grader   : {grader}   Prompt: {prompt}",
        f"  Entries  : {len(population)} individuals (sorted ascending by fitness)",
        sep,
        "",
    ]

    total = len(population)
    for rank, entry in enumerate(population, start=1):
        lines.append(_format_entry(rank, entry, total))
        lines.append("")

    report = "\n".join(lines)

    if save:
        out_path = run_dir / "report.txt"
        out_path.write_text(report, encoding="utf-8")
        if print_report:
            print(f"[report] Saved → {out_path}")

    if print_report:
        print(report)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a human-readable report from a run directory."
    )
    parser.add_argument("run_dir", help="Path to the run directory.")
    parser.add_argument(
        "--archive", default=None,
        help="Specific archive filename (e.g. archive_gen0010.json). "
             "Defaults to archive_final.json or latest snapshot.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print report to stdout without saving to report.txt.",
    )
    args = parser.parse_args()

    generate_report(
        run_dir      = args.run_dir,
        archive_name = args.archive,
        save         = not args.no_save,
        print_report = True,
    )
