"""
data_analyser.py  —  Interactive result explorer for morphology evolution runs
===============================================================================
Run from anywhere:
    python results/data_analyser.py
    python results/data_analyser.py results/run_XXXXXX

Requires: matplotlib, Pillow (PIL)
"""

from __future__ import annotations

import json
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

# ── Quick config ───────────────────────────────────────────────────────────────
DEFAULT_RUN_DIR  = None           # None = auto-select most-recent run on startup
                                  # or set to e.g. "run_20260417_151235"
DEFAULT_GRAPH_1  = "Map-Elites Grid Coverage"
DEFAULT_GRAPH_2  = "Individual Render"
WINDOW_TITLE     = "Morphology Evolution Analyser"
WINDOW_GEOMETRY  = "1400x860"
FONT_MONO        = ("Courier", 14)
FONT_BOLD        = ("Helvetica", 15, "bold")
FONT_HDR         = ("Helvetica", 16, "bold")
COLOR_G1_HDR     = "#8FBF8F"     # green header for graph 1
COLOR_G2_HDR     = "#F0D080"     # yellow header for graph 2
COLOR_INFO_BG    = "#D8D4F2"     # lavender for info panels
COLOR_CTRL_BG    = "#C8DEFF"     # blue for controls panel
COLOR_RUN_HDR    = "#C8C8C8"     # grey for run ID header

GRAPH_OPTIONS = [
    "Best Fitness × Generation",
    "Mean Fitness ± Std × Generation",
    "Score Details (Best) × Generation",
    "All Fitnesses (scatter)",
    "Genealogy Path",
    "Individual Render",
    "Individual Descriptors",
    "Map-Elites Grid Coverage",
]
# ──────────────────────────────────────────────────────────────────────────────


# ── Data loading ───────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
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
        "all_individuals": {},
    }

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        data["config"] = json.loads(cfg_path.read_text(encoding="utf-8"))

    data["log"]  = _read_jsonl(run_dir / "log.jsonl")
    data["grid"] = {}

    arc_path = run_dir / "archive_final.json"
    if not arc_path.exists():
        snaps = sorted(run_dir.glob("archive_gen*.json"))
        arc_path = snaps[-1] if snaps else None

    if arc_path and arc_path.exists():
        arc = json.loads(arc_path.read_text(encoding="utf-8"))
        data["history"] = arc.get("history", [])
        if "population" in arc and arc["population"] is not None:
            data["population"] = arc["population"]
            data["grid"] = {}
        elif "grid" in arc:
            data["population"] = list(arc["grid"].values())
            data["grid"] = arc["grid"]  # raw {key_str: individual_dict}
            # feature_dims / dim_labels stored in the archive (set by MapEliteArchive)
            if "feature_dims" in arc:
                data["config"].setdefault("map_elite_feature_dims", arc["feature_dims"])
            if "dim_labels" in arc:
                data["config"].setdefault("map_elite_dim_labels", arc["dim_labels"])

    # all_individuals: prefer streaming log (has genealogy), fall back to population
    indiv_records = _read_jsonl(run_dir / "individuals_log.jsonl")
    if indiv_records:
        data["all_individuals"] = {r["individual_id"]: r for r in indiv_records}
    else:
        data["all_individuals"] = {r["individual_id"]: r for r in data["population"]}

    return data


def _find_render(run_dir: Path, individual: dict) -> Optional[Path]:
    rp = individual.get("render_path")
    if rp:
        p = Path(rp)
        if p.exists():
            return p

    gen = individual.get("generation", 0)
    iid = individual.get("individual_id", 0)
    candidates = [
        run_dir / "renders" / f"gen{gen:04d}" / f"gen{gen:04d}_id{iid:06d}.png",
        run_dir / "renders" / "best" / f"gen{gen:04d}_id{iid:06d}.png",
        run_dir / "renders" / f"gen{gen:04d}_id{iid:06d}.png",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Broad search inside run renders
    renders_dir = run_dir / "renders"
    if renders_dir.exists():
        for found in renders_dir.rglob(f"*id{iid:06d}.png"):
            return found

    # Fallback: last_render temp folder (sibling of run_dir, reset each experiment)
    last_render = run_dir.parent / "last_render"
    if last_render.exists():
        candidate = last_render / f"id{iid:06d}.png"
        if candidate.exists():
            return candidate

    return None


def _get_ancestors(individual_id: int, all_individuals: dict) -> list[dict]:
    """Trace genealogy from root to individual. Returns list ordered root→individual."""
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
    gens  = [h["generation"]  for h in history]
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
    has_scores = any(h.get("best_raw_scores") for h in history)
    if not has_scores:
        return _no_data(ax,
            "Score detail data not available for this run.\n"
            "(Re-run with updated experiment code to enable.)")

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
    records  = list(all_ind.values())
    gens     = [r["generation"] for r in records]
    fitnesses = [r["fitness"]   for r in records]

    sc = ax.scatter(gens, fitnesses, c=gens, cmap="viridis", s=25, alpha=0.65, zorder=3)
    ax.figure.colorbar(sc, ax=ax, label="Generation", pad=0.02)

    history = run_data.get("history", [])
    if history:
        ax.plot([h["generation"] for h in history],
                [h["best_fitness"] for h in history],
                "k--", lw=1.5, alpha=0.5, label="best per gen")

    if sel_id and sel_id in all_ind:
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
        has_any_parent = any(r.get("parent_id") is not None for r in all_ind.values())
        if not has_any_parent:
            return _no_data(ax,
                "Genealogy data not recorded for this run.\n"
                "(Re-run with updated experiment code to enable.)")
        return _no_data(ax, f"No ancestor chain available for id={sel_id}")

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


def draw_render(ax, run_data: dict, sel_id) -> None:
    if sel_id is None:
        return _no_data(ax, "Select an individual")
    all_ind = run_data.get("all_individuals", {})
    indiv   = all_ind.get(sel_id)
    if indiv is None:
        return _no_data(ax, f"Individual id={sel_id} not found in loaded data")

    render_path = _find_render(run_data["run_dir"], indiv)
    if render_path is None:
        return _no_data(ax, f"No render image found for id={sel_id}")

    try:
        from PIL import Image
        img = Image.open(render_path)
        ax.imshow(img)
        ax.set_title(f"Render  —  id={sel_id}   gen={indiv['generation']}   "
                     f"fitness={indiv['fitness']:.4f}")
        ax.axis("off")
    except Exception as exc:
        _no_data(ax, f"Failed to load image:\n{exc}")


def draw_descriptors(ax, run_data: dict, sel_id) -> None:
    if sel_id is None:
        return _no_data(ax, "Select an individual")
    all_ind = run_data.get("all_individuals", {})
    indiv   = all_ind.get(sel_id)
    if indiv is None:
        return _no_data(ax, f"Individual id={sel_id} not found")

    desc = {k: v for k, v in (indiv.get("descriptors") or {}).items()
            if isinstance(v, (int, float))}
    if not desc:
        return _no_data(ax, "No numeric descriptors available")

    keys   = list(desc.keys())
    vals   = [desc[k] for k in keys]
    colors = ["#3498DB" if v >= 0 else "#E74C3C" for v in vals]

    ax.barh(range(len(keys)), vals, color=colors)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Structural Descriptors  —  id={sel_id}")
    ax.set_xlabel("Value")
    ax.grid(True, axis="x", alpha=0.3)


def draw_grid_coverage(ax, run_data: dict, sel_id) -> None:
    grid_raw = run_data.get("grid", {})
    if not grid_raw:
        return _no_data(ax, "No Map-Elites grid found.\n(Only available for map_elite strategy runs.)")

    def _parse_key(s):
        s = s.strip()
        if s.startswith("["):
            return tuple(json.loads(s))
        return tuple(int(x) for x in s.strip("()").split(","))

    parsed = {}
    for k, v in grid_raw.items():
        try:
            parsed[_parse_key(k)] = v
        except Exception:
            pass

    if not parsed:
        return _no_data(ax, "Grid keys could not be parsed.")

    keys  = list(parsed.keys())
    xs    = sorted(set(k[0] for k in keys))
    ys    = sorted(set(k[1] for k in keys))
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    import numpy as _np
    from matplotlib.colors import LinearSegmentedColormap

    grid_fit  = _np.full((ny, nx), _np.nan)
    # cell_to_id maps (col, row) in imshow coords → individual_id
    cell_to_id = {}
    for (xi, yi), indiv in parsed.items():
        col = xi - x_min   # x axis = col
        row = yi - y_min   # y axis = row
        if 0 <= row < ny and 0 <= col < nx:
            grid_fit[row, col] = indiv.get("fitness", _np.nan)
            cell_to_id[(col, row)] = indiv.get("individual_id")

    # 2-color map: red (low) → green (high); NaN shown as near-black
    rg_cmap = LinearSegmentedColormap.from_list("rg", ["#CC2222", "#22AA44"])
    rg_cmap.set_bad(color="#1A1A1A")

    im = ax.imshow(
        grid_fit, origin="lower", aspect="auto",
        cmap=rg_cmap, interpolation="nearest",
    )
    ax.figure.colorbar(im, ax=ax, label="Fitness", pad=0.02)

    # Cell annotations: fitness + id
    for (xi, yi), indiv in parsed.items():
        col = xi - x_min
        row = yi - y_min
        fit = indiv.get("fitness", 0)
        iid = indiv.get("individual_id", "?")
        ax.text(col, row + 0.1,    f"{fit:.3f}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(col, row - 0.15, f"id={iid}", ha="center", va="center",
                fontsize=10, color="#DDDDDD")

    # Blue border around the selected individual's cell
    if sel_id is not None:
        for (xi, yi), indiv in parsed.items():
            if indiv.get("individual_id") == sel_id:
                col = xi - x_min
                row = yi - y_min
                ax.add_patch(matplotlib.patches.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    fill=False, edgecolor="#4488FF", lw=2.5, zorder=5,
                ))

    cfg        = run_data.get("config", {})
    dims       = cfg.get("map_elite_feature_dims") or ["dim0", "dim1"]
    dim_labels = cfg.get("map_elite_dim_labels") or {}
    dim_x = dims[0] if len(dims) > 0 else "dim0"
    dim_y = dims[1] if len(dims) > 1 else "dim1"

    def _bin_labels(dim: str, n_bins: int, offset: int) -> list[str]:
        """Return tick labels for a dimension: bin_labels if available, else indices."""
        labels = dim_labels.get(dim, [])
        result = []
        for i in range(n_bins):
            idx = offset + i
            if labels and idx < len(labels):
                result.append(labels[idx])
            else:
                result.append(str(idx))
        return result

    ax.set_xticks(range(nx))
    ax.set_xticklabels(_bin_labels(dim_x, nx, x_min), fontsize=7, rotation=15, ha="right")
    ax.set_yticks(range(ny))
    ax.set_yticklabels(_bin_labels(dim_y, ny, y_min), fontsize=7)
    ax.set_xlabel(dim_x.replace("_", " "), fontsize=9)
    ax.set_ylabel(dim_y.replace("_", " "), fontsize=9)
    ax.set_title(
        f"Map-Elites Grid  —  {len(parsed)} / {nx * ny} cells filled", fontsize=10
    )

    # Store cell map on the axes for the click handler (registered in _refresh_graph)
    ax._grid_cell_to_id   = cell_to_id
    ax._grid_x_min        = x_min
    ax._grid_y_min        = y_min


GRAPH_RENDERERS: dict = {
    "Best Fitness × Generation":        draw_best_fitness,
    "Mean Fitness ± Std × Generation":  draw_mean_fitness,
    "Score Details (Best) × Generation": draw_score_details,
    "All Fitnesses (scatter)":          draw_all_fitnesses,
    "Genealogy Path":                   draw_genealogy,
    "Individual Render":                draw_render,
    "Individual Descriptors":           draw_descriptors,
    "Map-Elites Grid Coverage":         draw_grid_coverage,
}

# graphs that need a refresh when the selected individual changes
_INDIVIDUAL_GRAPHS = {"Genealogy Path", "Individual Render", "Individual Descriptors",
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

    total_el  = sum(e.get("elapsed_s", 0) for e in log)
    n_done    = len(log)
    mean_el   = total_el / n_done if n_done else 0
    n_total   = len(all_ind)

    best_fit, best_id = None, None
    if history:
        best_h  = max(history, key=lambda h: h["best_fitness"])
        best_fit = best_h["best_fitness"]
        best_id  = best_h.get("best_individual_id")

    # Fall back to individual-level fields for old runs where config didn't save these
    first_indiv = next(iter(all_ind.values()), {}) if all_ind else {}
    grader_type = cfg.get("grader_type") or first_indiv.get("grader_method") or ""
    prompt_name_cfg = cfg.get("prompt_name") or first_indiv.get("prompt_set") or ""
    grader_str = grader_type or cfg.get("scoring_method") or "?"
    prompt_str = prompt_name_cfg or cfg.get("clip_pretrained") or "?"

    lines = []
    if cfg.get("description"):
        lines += [f"Desc:     {cfg['description']}", ""]
    lines += [
        f"Strategy:  {cfg.get('strategy', '?')}",
        f"μ={cfg.get('mu','?')}  λ={cfg.get('lambda_','?')}  "
        f"gens={cfg.get('n_generations','?')}",
        f"Grader:    {grader_str} / {prompt_str}",
        f"Seed:      {cfg.get('seed','?')}",
        "",
        f"Gens done:     {n_done}",
        f"Indiv tracked: {n_total}",
        f"Total time:    {total_el:.1f}s",
        f"Mean / gen:    {mean_el:.1f}s",
        "",
    ]
    if best_fit is not None:
        lines += [
            f"Best fitness:  {best_fit:.5f}",
            f"Best id:       {best_id}",
        ]

    # Prompt content
    if prompt_name_cfg:
        lines += ["", f"── Prompt: {prompt_name_cfg} ──"]
        code_dir = Path(__file__).parent.parent
        try:
            import importlib, sys as _sys
            _sys.path.insert(0, str(code_dir))
            if grader_type == "gemini":
                mod = importlib.import_module("gemini_prompts")
                pc  = mod.get_gemini_prompt_set(prompt_name_cfg)
                lines += [
                    f"Target:  {pc.target}",
                    f"Weights: coherence={pc.weights.coherence}  "
                    f"originality={pc.weights.originality}  "
                    f"interest={pc.weights.interest}",
                    "",
                    pc.prompt[:600] + ("…" if len(pc.prompt) > 600 else ""),
                ]
            elif grader_type in ("clip", "cosine", "softmax"):
                mod = importlib.import_module("CLIP_prompts")
                pc  = mod.get_clip_prompt_set(prompt_name_cfg)
                pos = [f"  + {p.text} (w={p.weight:.2f})" for p in pc.positive]
                neg = [f"  - {p.text} (w={p.weight:.2f})" for p in pc.negative]
                lines += ["Positive:"] + pos + ["Negative:"] + neg
        except Exception:
            pass

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
        f"Prompts: {individual.get('prompt_set', '?')}",
        "",
        "Scores:",
    ]
    for k, v in (individual.get("raw_scores") or {}).items():
        lines.append(f"  {k:<20}  {v:.3f}")

    lines += ["", "Descriptors:"]
    for k, v in (individual.get("descriptors") or {}).items():
        if isinstance(v, (int, float)):
            lines.append(f"  {k:<28}  {v:.4f}")

    extra = individual.get("grader_extra") or {}
    for key in ["observation", "interpretation",
                "coherence_reason", "originality_reason", "interest_reason"]:
        val = extra.get(key, "")
        if val:
            lines += ["", f"[{key}]", f"  {str(val)[:320]}"]

    rp = individual.get("render_path") or ""
    if rp:
        lines += ["", f"Render: …{rp[-55:]}"]

    return "\n".join(lines)


# ── Main App ───────────────────────────────────────────────────────────────────

class DataAnalyser:

    def __init__(self, root: tk.Tk, initial_run: Optional[Path] = None):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)

        self._run_data:    Optional[dict] = None
        self._selected_id: Optional[int]  = None
        self._indiv_list:  list[int]       = []
        self._results_dir: Path = Path(__file__).parent

        self._build_ui()
        self._refresh_run_list()

        if initial_run:
            self._load_run(Path(initial_run))
        elif DEFAULT_RUN_DIR:
            p = self._results_dir / DEFAULT_RUN_DIR
            if p.exists():
                self._load_run(p)
        else:
            runs = self._get_available_runs()
            if runs:
                self._load_run(runs[-1])

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                             sashwidth=6, sashrelief=tk.RAISED, bg="#CCCCCC")
        pw.pack(fill=tk.BOTH, expand=True)

        left  = tk.Frame(pw, bg="#E8E8E8")
        right = tk.Frame(pw, bg=COLOR_INFO_BG)
        pw.add(left,  minsize=400, width=860)
        pw.add(right, minsize=280, width=540)

        self._build_graphs(left)
        self._build_info(right)

    def _build_graphs(self, parent: tk.Frame):
        # ── Graph 1 ──
        g1 = tk.Frame(parent, bg="#E8E8E8")
        g1.pack(fill=tk.BOTH, expand=True, padx=2, pady=(2, 1))

        self._g1_hdr = tk.Label(g1, text=DEFAULT_GRAPH_1,
                                  bg=COLOR_G1_HDR, font=FONT_BOLD, pady=3, padx=6)
        self._g1_hdr.pack(fill=tk.X)

        self._fig1 = Figure(tight_layout=True)
        self._ax1  = self._fig1.add_subplot(111)
        self._cvs1 = FigureCanvasTkAgg(self._fig1, master=g1)
        self._cvs1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Divider ──
        tk.Frame(parent, height=2, bg="#AAAAAA").pack(fill=tk.X, pady=1)

        # ── Graph 2 ──
        g2 = tk.Frame(parent, bg="#E8E8E8")
        g2.pack(fill=tk.BOTH, expand=True, padx=2, pady=(1, 2))

        self._g2_hdr = tk.Label(g2, text=DEFAULT_GRAPH_2,
                                  bg=COLOR_G2_HDR, font=FONT_BOLD, pady=3, padx=6)
        self._g2_hdr.pack(fill=tk.X)

        self._fig2 = Figure(tight_layout=True)
        self._ax2  = self._fig2.add_subplot(111)
        self._cvs2 = FigureCanvasTkAgg(self._fig2, master=g2)
        self._cvs2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_info(self, parent: tk.Frame):
        # ── Run ID header ──
        self._run_id_lbl = tk.Label(
            parent, text="── no run loaded ──",
            bg=COLOR_RUN_HDR, font=FONT_HDR,
            relief=tk.GROOVE, pady=5, padx=8)
        self._run_id_lbl.pack(fill=tk.X, padx=2, pady=(2, 6))

        # ── General info ──
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

        # ── Individual info ──
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

        # ── Controls (blue) ──
        ctrl = tk.Frame(parent, bg=COLOR_CTRL_BG, relief=tk.GROOVE, bd=2)
        ctrl.pack(fill=tk.X, padx=2, pady=(0, 4))

        # Individual nav row
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

        # Graph selectors
        for label, attr, default, n in [
            ("Graph 1 :", "_g1_var", DEFAULT_GRAPH_1, 1),
            ("Graph 2 :", "_g2_var", DEFAULT_GRAPH_2, 2),
        ]:
            row = tk.Frame(ctrl, bg=COLOR_CTRL_BG)
            row.pack(fill=tk.X, padx=6, pady=2)
            tk.Label(row, text=label, bg=COLOR_CTRL_BG,
                     font=FONT_BOLD, width=10, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            combo = ttk.Combobox(row, textvariable=var,
                                  values=GRAPH_OPTIONS, state="readonly")
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            combo.bind("<<ComboboxSelected>>",
                       lambda e, n_=n: self._refresh_graph(n_))

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=2)

        # Run selector
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

    # ── Data loading ───────────────────────────────────────────────────────────

    def _get_available_runs(self) -> list[Path]:
        return sorted(p for p in self._results_dir.iterdir()
                      if p.is_dir() and p.name.startswith("run_"))

    def _refresh_run_list(self):
        runs  = self._get_available_runs()
        names = [r.name for r in runs]
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

        # Default-select the overall best individual
        history = self._run_data.get("history", [])
        default_id = None
        if history:
            best_h  = max(history, key=lambda h: h["best_fitness"])
            best_id = best_h.get("best_individual_id")
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
        entries = [
            f"id={iid:>5}  gen={all_ind[iid]['generation']:>3}  "
            f"fit={all_ind[iid]['fitness']:.4f}"
            for iid in self._indiv_list
        ]
        self._indiv_combo["values"] = entries
        if self._selected_id in self._indiv_list:
            self._indiv_combo.current(self._indiv_list.index(self._selected_id))

    # ── Refresh ────────────────────────────────────────────────────────────────

    def _refresh_all(self):
        if not self._run_data:
            return
        self._run_id_lbl.config(text=f"run  {self._run_data['run_id']}")
        self._refresh_info()
        self._refresh_graph(1)
        self._refresh_graph(2)

    def _refresh_info(self):
        if not self._run_data:
            return
        _set_text(self._gen_txt, _build_general_info(self._run_data))
        indiv = (self._run_data["all_individuals"].get(self._selected_id)
                 if self._selected_id is not None else None)
        _set_text(self._ind_txt, _build_individual_info(indiv))

    def _refresh_graph(self, n: int):
        if not self._run_data:
            return
        if n == 1:
            fig, cvs = self._fig1, self._cvs1
            var, hdr = self._g1_var, self._g1_hdr
        else:
            fig, cvs = self._fig2, self._cvs2
            var, hdr = self._g2_var, self._g2_hdr

        name = var.get()
        hdr.config(text=name)

        # Clear the whole figure (removes colorbars and any extra axes)
        # then recreate a fresh subplot so repeated graph changes don't shrink the area
        fig.clear()
        ax = fig.add_subplot(111)
        if n == 1:
            self._ax1 = ax
        else:
            self._ax2 = ax

        fn = GRAPH_RENDERERS.get(name)
        if fn:
            try:
                fn(ax, self._run_data, self._selected_id)
            except Exception as exc:
                _no_data(ax, f"Render error:\n{exc}")

        # Grid click-to-select: register handler if the renderer stored a cell map
        if hasattr(ax, "_grid_cell_to_id") and ax._grid_cell_to_id:
            cell_map = ax._grid_cell_to_id

            def _on_grid_click(event, _ax=ax, _cvs=cvs, _cell=cell_map):
                if event.inaxes != _ax or event.xdata is None:
                    return
                col = int(round(event.xdata))
                row = int(round(event.ydata))
                iid = _cell.get((col, row))
                if iid is not None and iid in self._indiv_list:
                    self._select_individual(iid)

            cvs.mpl_connect("button_press_event", _on_grid_click)

        cvs.draw()

    # ── Individual navigation ──────────────────────────────────────────────────

    def _select_individual(self, iid: int):
        self._selected_id = iid
        if iid in self._indiv_list:
            self._indiv_combo.current(self._indiv_list.index(iid))
        self._refresh_info()
        for n, var in [(1, self._g1_var), (2, self._g2_var)]:
            if var.get() in _INDIVIDUAL_GRAPHS:
                self._refresh_graph(n)

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
        self._select_individual(
            self._indiv_list[min(len(self._indiv_list) - 1, idx + 1)])

    def _on_indiv_select(self, _event=None):
        idx = self._indiv_combo.current()
        if 0 <= idx < len(self._indiv_list):
            self._select_individual(self._indiv_list[idx])

    # ── Run selection ──────────────────────────────────────────────────────────

    def _on_run_select(self, _event=None):
        name = self._run_var.get()
        if name:
            self._load_run(self._results_dir / name)

    def _browse_run(self):
        d = filedialog.askdirectory(
            title="Select a run directory",
            initialdir=str(self._results_dir),
        )
        if d:
            self._load_run(Path(d))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    root = tk.Tk()
    DataAnalyser(root, initial_run=initial)
    root.mainloop()


if __name__ == "__main__":
    main()
