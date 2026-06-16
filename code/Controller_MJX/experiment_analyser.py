"""
experiment_analyser.py  —  Interactive result explorer for MJX controller runs
===============================================================================
Standalone Tk explorer for a Controller_MJX experiment (μ+λ or MAP-Elites).
Modelled on Morphology/results/data_analyser.py, adapted for controllers:
  - the per-individual artefact is an MP4 rollout (not a PNG render), shown via
    embedded looped video playback (same handler idea as
    Controller/utils/controller_generator_renderer.py);
  - the MAP-Elites grid is drawn from the archive's feature_dims / dim_labels.

Run from anywhere:
    python experiment_analyser.py                       # auto-pick most recent run
    python experiment_analyser.py results/run_XXXX      # a specific run dir
    python experiment_analyser.py results               # a results dir to browse

Requires: matplotlib, Pillow (PIL), imageio.
Self-contained — no imports from Controller/ or Morphology/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import matplotlib
import matplotlib.patches
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from PIL import Image as PILImage, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ── Quick config ───────────────────────────────────────────────────────────────
DEFAULT_GRAPH    = "Map-Elites Grid Coverage"
WINDOW_TITLE     = "MJX Controller Evolution Analyser"
WINDOW_GEOMETRY  = "1500x900"
FONT_MONO        = ("Courier", 13)
FONT_BOLD        = ("Helvetica", 14, "bold")
FONT_HDR         = ("Helvetica", 16, "bold")
COLOR_G_HDR      = "#8FBF8F"
COLOR_VID_HDR    = "#F0D080"
COLOR_INFO_BG    = "#D8D4F2"
COLOR_CTRL_BG    = "#C8DEFF"
COLOR_RUN_HDR    = "#C8C8C8"
PLAYBACK_MS      = 50            # ~20 fps looped playback
MAX_FRAMES       = 400

GRAPH_OPTIONS = [
    "Map-Elites Grid Coverage",
    "Best Fitness × Generation",
    "Mean Fitness ± Std × Generation",
    "Score Details (Best) × Generation",
    "All Fitnesses (scatter)",
    "Genealogy Path",
    "Individual Descriptors",
]
# ──────────────────────────────────────────────────────────────────────────────


# ── Data loading ───────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _read_target_text(run_dir: Path, config: dict) -> str:
    """Resolve the actual target behaviour text for a run.

    Prefers the per-run snapshot (run_dir/target.txt). Falls back to the
    configured target_file resolved next to this package (for runs predating the
    snapshot). Returns "" if neither is available.
    """
    snap = run_dir / "target.txt"
    if snap.exists():
        t = snap.read_text(encoding="utf-8").strip()
        if t:
            return t
    # Older runs (no snapshot): the resolved target was printed into log.txt as
    #   [experiment_mjx] Target (target.txt): 'a high jumping robot'
    # This is run-specific and accurate even if the package target.txt later changed.
    log_txt = run_dir / "log.txt"
    if log_txt.exists():
        try:
            for line in log_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "] Target (" in line and "): " in line:
                    t = line.split("): ", 1)[1].strip().strip("'\"").strip()
                    if t:
                        return t
        except Exception:
            pass
    target_file = config.get("target_file", "target.txt")
    p = Path(target_file)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def load_run(run_dir: Path) -> dict:
    """Load all available data from a run directory."""
    run_dir = Path(run_dir)
    data: dict = {
        "run_id":          run_dir.name,
        "run_dir":         run_dir,
        "config":          {},
        "log":             [],
        "history":         [],
        "population":      [],
        "grid":            {},
        "all_individuals": {},
    }

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        data["config"] = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Actual target behaviour: prefer the per-run snapshot (target.txt written by
    # experiment_mjx); fall back to resolving the configured target_file next to
    # this package for older runs that predate the snapshot.
    data["config"]["target_behaviour"] = _read_target_text(run_dir, data["config"])

    data["log"] = _read_jsonl(run_dir / "log.jsonl")

    arc_path = run_dir / "archive_final.json"
    if not arc_path.exists():
        snaps = sorted(run_dir.glob("archive_gen*.json"))
        arc_path = snaps[-1] if snaps else None

    if arc_path and arc_path.exists():
        arc = json.loads(arc_path.read_text(encoding="utf-8"))
        data["history"] = arc.get("history", [])
        if arc.get("type") == "map_elite" or "grid" in arc:
            data["population"] = list(arc.get("grid", {}).values())
            data["grid"]       = arc.get("grid", {})
            data["config"].setdefault("map_elite_feature_dims", arc.get("feature_dims", []))
            data["config"].setdefault("map_elite_dim_labels", arc.get("dim_labels", {}))
        elif "population" in arc and arc["population"] is not None:
            data["population"] = arc["population"]

    indiv_records = _read_jsonl(run_dir / "individuals_log.jsonl")
    if indiv_records:
        data["all_individuals"] = {r["individual_id"]: r for r in indiv_records}
    else:
        data["all_individuals"] = {r["individual_id"]: r for r in data["population"]}

    return data


def _find_video(run_dir: Path, individual: dict) -> Optional[Path]:
    """Locate the rollout MP4 for an individual."""
    vp = individual.get("video_path")
    if vp:
        p = Path(vp)
        if p.exists():
            return p
        # video_path may be stored as an absolute path from another machine —
        # fall back to matching the basename inside this run's videos/ dir.
        cand = run_dir / "videos" / p.name
        if cand.exists():
            return cand

    gen = individual.get("generation", 0)
    iid = individual.get("individual_id", 0)
    cand = run_dir / "videos" / f"gen{gen:04d}_id{iid:06d}.mp4"
    if cand.exists():
        return cand

    vids = run_dir / "videos"
    if vids.exists():
        for found in vids.rglob(f"*id{iid:06d}.mp4"):
            return found
    return None


def _extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> list:
    """Read an MP4 into a list of PIL.Image frames (for looped playback)."""
    if not _PIL_OK:
        return []
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        frames = [PILImage.fromarray(f) for i, f in enumerate(reader) if i < max_frames]
        reader.close()
        return frames
    except Exception:
        return []


def _get_ancestors(individual_id: int, all_individuals: dict) -> list:
    chain = []
    current_id: Optional[int] = individual_id
    seen: set = set()
    while current_id is not None and current_id not in seen:
        seen.add(current_id)
        indiv = all_individuals.get(current_id)
        if indiv is None:
            break
        chain.append(indiv)
        current_id = indiv.get("parent_id")
    chain.reverse()
    return chain


# ── Colour helper ──────────────────────────────────────────────────────────────

def _tab_colors(n: int) -> list:
    try:
        cmap = matplotlib.colormaps["tab10"]
    except AttributeError:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab10")
    return [cmap(i % 10) for i in range(max(n, 1))]


# ── Graph rendering functions ──────────────────────────────────────────────────

def _no_data(ax, msg: str = "No data available") -> None:
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="gray", style="italic",
            wrap=True)
    ax.set_axis_off()


def draw_best_fitness(ax, run_data: dict, _sel_id) -> None:
    history = run_data.get("history", [])
    if not history:
        return _no_data(ax)
    gens  = [h["generation"]   for h in history]
    bests = [h["best_fitness"] for h in history]
    ax.plot(gens, bests, "o-", color="#2E86AB", lw=2, ms=5, label="best fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Best Fitness per Generation")
    ax.set_xlim(min(gens) - 0.5, max(gens) + 0.5)
    lo, hi = min(bests), max(bests)
    pad = max(abs(hi - lo) * 0.1, 0.02)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def draw_mean_fitness(ax, run_data: dict, _sel_id) -> None:
    history = run_data.get("history", [])
    if not history:
        return _no_data(ax)
    gens  = [h["generation"]   for h in history]
    means = [h["mean_fitness"] for h in history]
    stds  = [h.get("std_fitness", 0) for h in history]
    ax.plot(gens, means, "o-", color="#E76F51", lw=2, ms=5, label="mean fitness")
    ax.fill_between(gens,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.25, color="#E76F51", label="± 1 std")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Mean Fitness per Generation")
    ax.set_xlim(min(gens) - 0.5, max(gens) + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def draw_score_details(ax, run_data: dict, _sel_id) -> None:
    history = run_data.get("history", [])
    if not history:
        return _no_data(ax)
    if not any(h.get("best_raw_scores") for h in history):
        return _no_data(ax, "No per-dimension score history for this run.")
    gens = [h["generation"] for h in history]
    score_keys = sorted({k for h in history for k in h.get("best_raw_scores", {})})
    colors = _tab_colors(len(score_keys))
    for i, key in enumerate(score_keys):
        vals = [h.get("best_raw_scores", {}).get(key, float("nan")) for h in history]
        ax.plot(gens, vals, "o-", color=colors[i], lw=2, ms=5, label=key)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    ax.set_title("Best Individual — Score Details per Generation")
    ax.set_xlim(min(gens) - 0.5, max(gens) + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def draw_all_fitnesses(ax, run_data: dict, sel_id) -> None:
    all_ind = run_data.get("all_individuals", {})
    if not all_ind:
        return _no_data(ax)
    records   = list(all_ind.values())
    gens      = [r["generation"] for r in records]
    fitnesses = [r["fitness"]    for r in records]
    sc = ax.scatter(gens, fitnesses, c=gens, cmap="viridis", s=25, alpha=0.65, zorder=3)
    ax.figure.colorbar(sc, ax=ax, label="Generation", pad=0.02)
    history = run_data.get("history", [])
    if history:
        ax.plot([h["generation"] for h in history],
                [h["best_fitness"] for h in history],
                "k--", lw=1.5, alpha=0.5, label="best per gen")
    if sel_id is not None and sel_id in all_ind:
        r = all_ind[sel_id]
        ax.scatter([r["generation"]], [r["fitness"]], s=130, c="red",
                   zorder=5, label=f"selected (id={sel_id})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("All Evaluated Individuals")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def draw_genealogy(ax, run_data: dict, sel_id) -> None:
    if sel_id is None:
        return _no_data(ax, "Select an individual")
    all_ind = run_data.get("all_individuals", {})
    chain   = _get_ancestors(sel_id, all_ind)
    if len(chain) <= 1:
        if not any(r.get("parent_id") is not None for r in all_ind.values()):
            return _no_data(ax, "No genealogy recorded for this run.")
        return _no_data(ax, f"No ancestor chain for id={sel_id}")
    gens      = [r["generation"]    for r in chain]
    fitnesses = [r["fitness"]       for r in chain]
    ids       = [r["individual_id"] for r in chain]
    ax.plot(gens, fitnesses, "o-", color="#9B59B6", lw=2, ms=7)
    for g, f, iid in zip(gens, fitnesses, ids):
        ax.annotate(f"id={iid}", (g, f), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    ax.scatter([gens[-1]], [fitnesses[-1]], s=130, color="#E74C3C",
               zorder=5, label=f"id={sel_id} (selected)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Genealogy Path → id={sel_id}  (depth={len(chain)})")
    ax.set_xlim(min(gens) - 0.5, max(gens) + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def draw_descriptors(ax, run_data: dict, sel_id) -> None:
    if sel_id is None:
        return _no_data(ax, "Select an individual")
    indiv = run_data.get("all_individuals", {}).get(sel_id)
    if indiv is None:
        return _no_data(ax, f"id={sel_id} not found")
    # MAP-Elites descriptors are stored on `descriptors` (0–100 per axis).
    desc = {k: v for k, v in (indiv.get("descriptors") or {}).items()
            if isinstance(v, (int, float))}
    if not desc:
        return _no_data(ax, "No numeric descriptors for this individual")
    keys = list(desc.keys())
    vals = [desc[k] for k in keys]
    ax.barh(range(len(keys)), vals, color="#3498DB")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=9)
    ax.set_xlim(0, 100)
    for i, v in enumerate(vals):
        ax.text(min(v + 2, 96), i, f"{v:.0f}", va="center", fontsize=8)
    ax.set_title(f"VLM Behavioural Descriptors  —  id={sel_id}")
    ax.set_xlabel("Score (0–100)")
    ax.grid(True, axis="x", alpha=0.3)


def draw_grid_coverage(ax, run_data: dict, sel_id) -> None:
    grid_raw = run_data.get("grid", {})
    if not grid_raw:
        return _no_data(ax, "No Map-Elites grid found.\n(Only for map_elite runs.)")

    def _parse_key(s):
        s = s.strip()
        if s.startswith("["):
            return tuple(json.loads(s))
        return tuple(int(x) for x in s.strip("()").split(",") if x.strip())

    parsed = {}
    for k, v in grid_raw.items():
        try:
            parsed[_parse_key(k)] = v
        except Exception:
            pass
    if not parsed:
        return _no_data(ax, "Grid keys could not be parsed.")

    keys = list(parsed.keys())
    xs = sorted(set(k[0] for k in keys))
    ys = sorted(set(k[1] for k in keys)) if all(len(k) > 1 for k in keys) else [0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    import numpy as _np
    from matplotlib.colors import LinearSegmentedColormap

    grid_fit = _np.full((ny, nx), _np.nan)
    cell_to_id = {}
    for key, indiv in parsed.items():
        xi = key[0]
        yi = key[1] if len(key) > 1 else 0
        col, row = xi - x_min, yi - y_min
        if 0 <= row < ny and 0 <= col < nx:
            grid_fit[row, col] = indiv.get("fitness", _np.nan)
            cell_to_id[(col, row)] = indiv.get("individual_id")

    rg_cmap = LinearSegmentedColormap.from_list("rg", ["#CC2222", "#22AA44"])
    rg_cmap.set_bad(color="#1A1A1A")
    im = ax.imshow(grid_fit, origin="lower", aspect="auto",
                   cmap=rg_cmap, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, label="Fitness", pad=0.02)

    for key, indiv in parsed.items():
        xi = key[0]
        yi = key[1] if len(key) > 1 else 0
        col, row = xi - x_min, yi - y_min
        ax.text(col, row + 0.12, f"{indiv.get('fitness', 0):.3f}", ha="center",
                va="center", fontsize=10, color="white", fontweight="bold")
        ax.text(col, row - 0.16, f"id={indiv.get('individual_id', '?')}", ha="center",
                va="center", fontsize=10, color="#DDDDDD")

    if sel_id is not None:
        for key, indiv in parsed.items():
            if indiv.get("individual_id") == sel_id:
                xi = key[0]
                yi = key[1] if len(key) > 1 else 0
                ax.add_patch(matplotlib.patches.Rectangle(
                    (xi - x_min - 0.5, yi - y_min - 0.5), 1, 1,
                    fill=False, edgecolor="#4488FF", lw=2.5, zorder=5))

    cfg        = run_data.get("config", {})
    dims       = cfg.get("map_elite_feature_dims") or ["dim0", "dim1"]
    dim_labels = cfg.get("map_elite_dim_labels") or {}
    dim_x = dims[0] if len(dims) > 0 else "dim0"
    dim_y = dims[1] if len(dims) > 1 else "dim1"

    def _labels(dim, n_bins, offset):
        labels = dim_labels.get(dim, [])
        return [labels[offset + i] if labels and offset + i < len(labels) else str(offset + i)
                for i in range(n_bins)]

    ax.set_xticks(range(nx))
    ax.set_xticklabels(_labels(dim_x, nx, x_min), fontsize=7, rotation=15, ha="right")
    ax.set_yticks(range(ny))
    ax.set_yticklabels(_labels(dim_y, ny, y_min), fontsize=7)
    ax.set_xlabel(dim_x.replace("_", " "), fontsize=9)
    ax.set_ylabel(dim_y.replace("_", " "), fontsize=9)
    ax.set_title(f"Map-Elites Grid — {len(parsed)} cells filled", fontsize=10)

    ax._grid_cell_to_id = cell_to_id


GRAPH_RENDERERS: dict = {
    "Best Fitness × Generation":         draw_best_fitness,
    "Mean Fitness ± Std × Generation":   draw_mean_fitness,
    "Score Details (Best) × Generation": draw_score_details,
    "All Fitnesses (scatter)":           draw_all_fitnesses,
    "Genealogy Path":                    draw_genealogy,
    "Individual Descriptors":            draw_descriptors,
    "Map-Elites Grid Coverage":          draw_grid_coverage,
}

_INDIVIDUAL_GRAPHS = {"Genealogy Path", "Individual Descriptors",
                      "All Fitnesses (scatter)", "Map-Elites Grid Coverage"}


# ── Text panel helpers ─────────────────────────────────────────────────────────

def _set_text(widget: tk.Text, text: str) -> None:
    widget.config(state=tk.NORMAL)
    widget.delete("1.0", tk.END)
    widget.insert("1.0", text)
    widget.config(state=tk.DISABLED)


def _build_general_info(run_data: dict) -> str:
    cfg     = run_data.get("config", {})
    log     = run_data.get("log", [])
    history = run_data.get("history", [])
    all_ind = run_data.get("all_individuals", {})

    total_el = sum(e.get("elapsed_s", 0) for e in log)
    n_done   = len(log)
    mean_el  = total_el / n_done if n_done else 0

    best_fit, best_id = None, None
    if history:
        best_h   = max(history, key=lambda h: h["best_fitness"])
        best_fit = best_h["best_fitness"]
        best_id  = best_h.get("best_individual_id")

    strategy = cfg.get("strategy", "?")
    lines = []
    if cfg.get("description"):
        lines += [f"Desc:     {cfg['description']}", ""]
    lines += [
        f"Strategy:  {strategy}",
    ]
    if strategy == "map_elite":
        dims = cfg.get("map_elite_feature_dims") or []
        lines += [
            f"Descriptor: {cfg.get('descriptor_config_name', '?')}",
            f"Axes:      {dims}",
            f"λ={cfg.get('lambda_','?')}  gens={cfg.get('n_generations','?')}",
        ]
    else:
        lines += [
            f"μ={cfg.get('mu','?')}  λ={cfg.get('lambda_','?')}  "
            f"gens={cfg.get('n_generations','?')}",
        ]
    target = cfg.get("target_behaviour") or cfg.get("prompt_name", "?")
    lines += [
        f"Grader:    {cfg.get('gemini_model','?')}  "
        f"n_score={cfg.get('n_score_request','?')}",
        f"Target:    {target}",
        f"Seed:      {cfg.get('seed','?')}",
        "",
        f"Gens done:     {n_done}",
        f"Indiv tracked: {len(all_ind)}",
        f"Cells filled:  {len(run_data.get('grid', {}))}",
        f"Total time:    {total_el:.1f}s",
        f"Mean / gen:    {mean_el:.1f}s",
        "",
    ]
    if best_fit is not None:
        lines += [f"Best fitness:  {best_fit:.5f}", f"Best id:       {best_id}"]
    return "\n".join(lines)


def _build_individual_info(individual: Optional[dict]) -> str:
    if individual is None:
        return "(no individual selected)"
    pid = individual.get("parent_id")
    parent_str = f"id={pid}" if pid is not None else "root (no parent)"
    lines = [
        f"ID:      {individual.get('individual_id', '?')}",
        f"Gen:     {individual.get('generation', '?')}",
        f"Fitness: {individual.get('fitness', 0):.5f}",
        f"Parent:  {parent_str}",
        f"Method:  {individual.get('grader_method', '?')}",
        "",
        "Scores:",
    ]
    for k, v in (individual.get("raw_scores") or {}).items():
        lines.append(f"  {k:<14}  {v:.3f}")

    desc = individual.get("descriptors") or {}
    if desc:
        lines += ["", "Descriptors (0–100):"]
        for k, v in desc.items():
            if isinstance(v, (int, float)):
                lines.append(f"  {k:<14}  {v:.1f}")

    extra = individual.get("grader_extra") or {}
    for key in ["observation", "interpretation",
                "coherence_reason", "originality_reason", "potential_reason"]:
        val = extra.get(key, "")
        if val:
            lines += ["", f"[{key}]", f"  {str(val)[:320]}"]
    for dim, reason in (extra.get("descriptor_reasons") or {}).items():
        if reason:
            lines += ["", f"[{dim}_reason]", f"  {str(reason)[:280]}"]

    vp = individual.get("video_path") or ""
    if vp:
        lines += ["", f"Video: …{vp[-55:]}"]
    return "\n".join(lines)


# ── Main App ───────────────────────────────────────────────────────────────────

class ExperimentAnalyser:

    _CANVAS_W = 520
    _CANVAS_H = 300

    def __init__(self, root: tk.Tk, initial: Optional[Path] = None):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)

        self._run_data:    Optional[dict] = None
        self._selected_id: Optional[int]  = None
        self._indiv_list:  list           = []
        self._results_dir: Path           = Path("results")

        # Video playback state
        self._photo                       = None
        self._play_frames: list           = []
        self._play_idx:    int            = 0
        self._play_after:  Optional[str]  = None
        self._current_video: Optional[Path] = None

        self._build_ui()

        # Resolve what was passed: a run dir, a results dir, or nothing.
        start_run = None
        if initial is not None:
            initial = Path(initial)
            if (initial / "config.json").exists() or list(initial.glob("archive_*.json")):
                self._results_dir = initial.parent
                start_run = initial
            elif initial.is_dir():
                self._results_dir = initial
        else:
            for cand in (Path("results"),
                         Path(__file__).resolve().parent / "results"):
                if cand.exists():
                    self._results_dir = cand
                    break

        self._refresh_run_list()
        if start_run is None:
            runs = self._get_available_runs()
            start_run = runs[-1] if runs else None
        if start_run is not None:
            self._load_run(start_run)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                            sashwidth=6, sashrelief=tk.RAISED, bg="#CCCCCC")
        pw.pack(fill=tk.BOTH, expand=True)

        left  = tk.Frame(pw, bg="#E8E8E8")
        right = tk.Frame(pw, bg=COLOR_INFO_BG)
        pw.add(left,  minsize=520, width=940)
        pw.add(right, minsize=300, width=560)

        self._build_left(left)
        self._build_info(right)

    def _build_left(self, parent: tk.Frame):
        vpw = tk.PanedWindow(parent, orient=tk.VERTICAL,
                             sashwidth=6, sashrelief=tk.RAISED, bg="#BBBBBB")
        vpw.pack(fill=tk.BOTH, expand=True)

        # ── Graph (top) ──
        g = tk.Frame(vpw, bg="#E8E8E8")
        self._g_hdr = tk.Label(g, text=DEFAULT_GRAPH, bg=COLOR_G_HDR,
                               font=FONT_BOLD, pady=3, padx=6)
        self._g_hdr.pack(fill=tk.X)
        self._fig = Figure(tight_layout=True)
        self._ax  = self._fig.add_subplot(111)
        self._cvs = FigureCanvasTkAgg(self._fig, master=g)
        self._cvs.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        vpw.add(g, minsize=320, height=480)

        # ── Video (bottom) ──
        v = tk.Frame(vpw, bg="#1A1A1A")
        vhdr = tk.Frame(v, bg=COLOR_VID_HDR)
        vhdr.pack(fill=tk.X)
        self._vid_hdr = tk.Label(vhdr, text="Rollout Video", bg=COLOR_VID_HDR,
                                 font=FONT_BOLD, pady=3, padx=6)
        self._vid_hdr.pack(side=tk.LEFT)
        tk.Button(vhdr, text="▶ Play",  command=self._replay,
                  width=7).pack(side=tk.RIGHT, padx=2, pady=2)
        tk.Button(vhdr, text="■ Stop",  command=self._stop_playback,
                  width=7).pack(side=tk.RIGHT, padx=2, pady=2)
        tk.Button(vhdr, text="Open ↗", command=self._open_external,
                  width=7).pack(side=tk.RIGHT, padx=2, pady=2)
        self._canvas = tk.Canvas(v, bg="#1A1A1A", highlightthickness=0,
                                 width=self._CANVAS_W, height=self._CANVAS_H)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        vpw.add(v, minsize=220, height=340)

    def _build_info(self, parent: tk.Frame):
        self._run_id_lbl = tk.Label(parent, text="── no run loaded ──",
                                    bg=COLOR_RUN_HDR, font=FONT_HDR,
                                    relief=tk.GROOVE, pady=5, padx=8)
        self._run_id_lbl.pack(fill=tk.X, padx=2, pady=(2, 6))

        tk.Label(parent, text="General info :", bg=COLOR_INFO_BG,
                 font=FONT_BOLD, anchor=tk.W, padx=6).pack(fill=tk.X)
        gf = tk.Frame(parent, bg=COLOR_INFO_BG)
        gf.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 6))
        g_sb = ttk.Scrollbar(gf)
        g_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._gen_txt = tk.Text(gf, font=FONT_MONO, bg=COLOR_INFO_BG, fg="black",
                                relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD,
                                yscrollcommand=g_sb.set)
        self._gen_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        g_sb.config(command=self._gen_txt.yview)

        tk.Label(parent, text="Individual selected info :", bg=COLOR_INFO_BG,
                 font=FONT_BOLD, anchor=tk.W, padx=6).pack(fill=tk.X)
        if_ = tk.Frame(parent, bg=COLOR_INFO_BG)
        if_.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 6))
        i_sb = ttk.Scrollbar(if_)
        i_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._ind_txt = tk.Text(if_, font=FONT_MONO, bg=COLOR_INFO_BG, fg="black",
                                relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD,
                                yscrollcommand=i_sb.set)
        self._ind_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        i_sb.config(command=self._ind_txt.yview)

        ctrl = tk.Frame(parent, bg=COLOR_CTRL_BG, relief=tk.GROOVE, bd=2)
        ctrl.pack(fill=tk.X, padx=2, pady=(0, 4))

        nav = tk.Frame(ctrl, bg=COLOR_CTRL_BG)
        nav.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Label(nav, text="Individual :", bg=COLOR_CTRL_BG,
                 font=FONT_BOLD).pack(side=tk.LEFT)
        tk.Button(nav, text="◀ Prev", command=self._prev_individual,
                  width=7).pack(side=tk.LEFT, padx=(6, 2))
        tk.Button(nav, text="Next ▶", command=self._next_individual,
                  width=7).pack(side=tk.LEFT)
        self._indiv_var   = tk.StringVar()
        self._indiv_combo = ttk.Combobox(ctrl, textvariable=self._indiv_var,
                                          state="readonly")
        self._indiv_combo.pack(fill=tk.X, padx=6, pady=(0, 4))
        self._indiv_combo.bind("<<ComboboxSelected>>", self._on_indiv_select)

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=2)

        row = tk.Frame(ctrl, bg=COLOR_CTRL_BG)
        row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(row, text="Graph :", bg=COLOR_CTRL_BG,
                 font=FONT_BOLD, width=10, anchor=tk.W).pack(side=tk.LEFT)
        self._g_var = tk.StringVar(value=DEFAULT_GRAPH)
        combo = ttk.Combobox(row, textvariable=self._g_var,
                             values=GRAPH_OPTIONS, state="readonly")
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_graph())

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=2)

        run_row = tk.Frame(ctrl, bg=COLOR_CTRL_BG)
        run_row.pack(fill=tk.X, padx=6, pady=(2, 6))
        tk.Label(run_row, text="Run :", bg=COLOR_CTRL_BG,
                 font=FONT_BOLD, width=10, anchor=tk.W).pack(side=tk.LEFT)
        tk.Button(run_row, text="Browse…",
                  command=self._browse_run).pack(side=tk.RIGHT)
        self._run_var   = tk.StringVar()
        self._run_combo = ttk.Combobox(run_row, textvariable=self._run_var,
                                        state="readonly")
        self._run_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._run_combo.bind("<<ComboboxSelected>>", self._on_run_select)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _get_available_runs(self) -> list:
        if not self._results_dir.exists():
            return []
        return sorted(p for p in self._results_dir.iterdir()
                      if p.is_dir() and (p / "config.json").exists())

    def _refresh_run_list(self):
        names = [r.name for r in self._get_available_runs()]
        self._run_combo["values"] = names
        if self._run_data and self._run_data["run_id"] in names:
            self._run_var.set(self._run_data["run_id"])

    def _load_run(self, run_dir: Path):
        try:
            self._run_data = load_run(run_dir)
        except Exception as exc:
            messagebox.showerror("Load error", f"Failed to load run:\n{exc}")
            return
        self._indiv_list = sorted(self._run_data["all_individuals"].keys())

        default_id = None
        history = self._run_data.get("history", [])
        if history:
            best_id = max(history, key=lambda h: h["best_fitness"]).get("best_individual_id")
            if best_id in self._run_data["all_individuals"]:
                default_id = best_id
        if default_id is None and self._indiv_list:
            default_id = self._indiv_list[-1]
        self._selected_id = default_id

        self._update_indiv_combo()
        self._refresh_run_list()
        self._run_var.set(self._run_data["run_id"])
        self._refresh_all()

    def _update_indiv_combo(self):
        all_ind = (self._run_data or {}).get("all_individuals", {})
        self._indiv_combo["values"] = [
            f"id={iid:>5}  gen={all_ind[iid]['generation']:>3}  fit={all_ind[iid]['fitness']:.4f}"
            for iid in self._indiv_list
        ]
        if self._selected_id in self._indiv_list:
            self._indiv_combo.current(self._indiv_list.index(self._selected_id))

    # ── Refresh ────────────────────────────────────────────────────────────────

    def _refresh_all(self):
        if not self._run_data:
            return
        self._run_id_lbl.config(text=f"run  {self._run_data['run_id']}")
        self._refresh_info()
        self._refresh_graph()
        self._refresh_video()

    def _refresh_info(self):
        if not self._run_data:
            return
        _set_text(self._gen_txt, _build_general_info(self._run_data))
        indiv = (self._run_data["all_individuals"].get(self._selected_id)
                 if self._selected_id is not None else None)
        _set_text(self._ind_txt, _build_individual_info(indiv))

    def _refresh_graph(self):
        if not self._run_data:
            return
        name = self._g_var.get()
        self._g_hdr.config(text=name)
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._ax = ax
        fn = GRAPH_RENDERERS.get(name)
        if fn:
            try:
                fn(ax, self._run_data, self._selected_id)
            except Exception as exc:
                _no_data(ax, f"Render error:\n{exc}")
        if hasattr(ax, "_grid_cell_to_id") and ax._grid_cell_to_id:
            cell_map = ax._grid_cell_to_id

            def _on_grid_click(event, _ax=ax, _cell=cell_map):
                if event.inaxes != _ax or event.xdata is None:
                    return
                iid = _cell.get((int(round(event.xdata)), int(round(event.ydata))))
                if iid is not None and iid in self._indiv_list:
                    self._select_individual(iid)

            self._cvs.mpl_connect("button_press_event", _on_grid_click)
        self._cvs.draw()

    # ── Video playback ───────────────────────────────────────────────────────

    def _refresh_video(self):
        self._stop_playback()
        self._canvas.delete("all")
        if self._selected_id is None or not self._run_data:
            return self._show_canvas_message("Select an individual")
        indiv = self._run_data["all_individuals"].get(self._selected_id)
        if indiv is None:
            return self._show_canvas_message("Individual not found")
        video = _find_video(self._run_data["run_dir"], indiv)
        self._current_video = video
        if video is None:
            return self._show_canvas_message(f"No video found for id={self._selected_id}")
        self._vid_hdr.config(text=f"Rollout — id={self._selected_id}  ({video.name})")
        frames = _extract_frames(str(video))
        if not frames:
            return self._show_canvas_message("Could not read video\n(use Open ↗)")
        self._start_playback(frames)

    def _start_playback(self, frames: list):
        self._stop_playback()
        self._play_frames = frames
        self._play_idx = 0
        if frames:
            self._play_next_frame()

    def _stop_playback(self):
        if self._play_after is not None:
            self._canvas.after_cancel(self._play_after)
            self._play_after = None
        self._play_frames = []
        self._play_idx = 0

    def _replay(self):
        self._refresh_video()

    def _play_next_frame(self):
        if not self._play_frames:
            return
        self._show_image(self._play_frames[self._play_idx])
        self._play_idx = (self._play_idx + 1) % len(self._play_frames)
        self._play_after = self._canvas.after(PLAYBACK_MS, self._play_next_frame)

    def _show_image(self, image) -> None:
        if not _PIL_OK:
            return
        cw = max(self._canvas.winfo_width(),  self._CANVAS_W)
        ch = max(self._canvas.winfo_height(), self._CANVAS_H)
        scale = min(cw / image.width, ch / image.height)
        nw = max(1, int(image.width * scale))
        nh = max(1, int(image.height * scale))
        self._photo = ImageTk.PhotoImage(image.resize((nw, nh), PILImage.LANCZOS))
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo)

    def _show_canvas_message(self, msg: str):
        cw = max(self._canvas.winfo_width(),  self._CANVAS_W)
        ch = max(self._canvas.winfo_height(), self._CANVAS_H)
        self._canvas.delete("all")
        self._canvas.create_text(cw // 2, ch // 2, text=msg, fill="#AAAAAA",
                                 font=("Helvetica", 13), justify=tk.CENTER)

    def _open_external(self):
        if not self._current_video or not self._current_video.exists():
            return
        path = str(self._current_video)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("win"):
                import os
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("Open video", f"Could not open:\n{exc}")

    # ── Selection / navigation ─────────────────────────────────────────────────

    def _select_individual(self, iid: int):
        self._selected_id = iid
        if iid in self._indiv_list:
            self._indiv_combo.current(self._indiv_list.index(iid))
        self._refresh_info()
        self._refresh_video()
        if self._g_var.get() in _INDIVIDUAL_GRAPHS:
            self._refresh_graph()

    def _prev_individual(self):
        if not self._indiv_list:
            return
        idx = (self._indiv_list.index(self._selected_id)
               if self._selected_id in self._indiv_list else 0)
        self._select_individual(self._indiv_list[max(0, idx - 1)])

    def _next_individual(self):
        if not self._indiv_list:
            return
        idx = (self._indiv_list.index(self._selected_id)
               if self._selected_id in self._indiv_list else -1)
        self._select_individual(self._indiv_list[min(len(self._indiv_list) - 1, idx + 1)])

    def _on_indiv_select(self, _event=None):
        idx = self._indiv_combo.current()
        if 0 <= idx < len(self._indiv_list):
            self._select_individual(self._indiv_list[idx])

    def _on_run_select(self, _event=None):
        name = self._run_var.get()
        if name:
            self._load_run(self._results_dir / name)

    def _browse_run(self):
        d = filedialog.askdirectory(title="Select a run directory",
                                    initialdir=str(self._results_dir))
        if d:
            d = Path(d)
            if (d / "config.json").exists() or list(d.glob("archive_*.json")):
                self._results_dir = d.parent
                self._refresh_run_list()
                self._load_run(d)
            else:
                self._results_dir = d
                self._refresh_run_list()


def main():
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    root = tk.Tk()
    ExperimentAnalyser(root, initial=initial)
    root.mainloop()


if __name__ == "__main__":
    main()
