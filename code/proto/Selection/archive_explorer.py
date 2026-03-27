"""
archive_explorer.py
===================
Analyse and visualise the Map-Elites elite archive.

The archive is a 2-D grid:
  - X axis : number of legs  (discrete integer)
  - Y axis : symmetry bin    (asymmetric / semi-sym / symmetric)
  - Color  : displacement_xy (performance score, higher = brighter)

Usage (standalone):
    cd code/proto/Selection
    python archive_explorer.py               # plots last_best
    python archive_explorer.py my_archive    # plots a named save
    python archive_explorer.py --list        # list all available saves

Usage from code:
    from archive_explorer import plot_archive, print_archive, load_archive

    data = load_archive("last_best")      # list of elite dicts
    print_archive("last_best")
    plot_archive("last_best")             # opens a matplotlib window
    plot_archive("last_best", save_path="grid.png")  # save to file instead
"""

from __future__ import annotations
import os
import sys

# ------------------------------------------------------------------
# Path setup — allow running as a standalone script or imported from
# anywhere in the project tree
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'Brain'))
sys.path.insert(0, _HERE)

from saver import load_controller, list_saves
from selector import SYMMETRY_BIN_LABELS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_archive(name: str = "last_best") -> list[dict]:
    """
    Load the elite archive from saves/<name>.pkl.

    Returns a list of elite dicts, each with:
        feature_key    : [nb_legs, symmetry_bin]
        feature_label  : "4legs/symmetric"
        displacement_m : float
        avg_speed_ms   : float
        fell_at        : float | None
        robot_index    : int  (index in the original simulation)
        saved_at       : str  (ISO timestamp)
    """
    payload  = load_controller(name)
    raw_list = payload["context"].get("elites", [])
    saved_at = payload.get("saved_at", "unknown")
    for entry in raw_list:
        entry["saved_at"] = saved_at
    return raw_list


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_archive(name: str = "last_best") -> None:
    """Print a tabular summary of the archive to stdout."""
    try:
        elites = load_archive(name)
    except FileNotFoundError:
        print(f"[archive_explorer] No save found: '{name}'")
        return

    if not elites:
        print(f"[archive_explorer] Archive '{name}' is empty.")
        return

    print(f"\n── Archive: {name}  ({len(elites)} cell(s)) ──")
    print(f"  {'Cell':<20}  {'disp (m)':>9}  {'speed (m/s)':>11}  {'fell':>6}  {'robot':>5}")
    print(f"  {'─'*20}  {'─'*9}  {'─'*11}  {'─'*6}  {'─'*5}")
    for e in sorted(elites, key=lambda x: tuple(x["feature_key"])):
        fell = f"{e['fell_at']:.1f}s" if e.get("fell_at") is not None else "never"
        print(
            f"  {e['feature_label']:<20}  "
            f"{e['displacement_m']:>9.3f}  "
            f"{e['avg_speed_ms']:>11.3f}  "
            f"{fell:>6}  "
            f"R{e['robot_index']:>4}"
        )
    best = max(elites, key=lambda x: x["displacement_m"])
    print(f"\n  Best: {best['feature_label']}  →  {best['displacement_m']:.3f} m\n")


# ---------------------------------------------------------------------------
# Grid plot
# ---------------------------------------------------------------------------

def plot_archive(
    name:      str        = "last_best",
    save_path: str | None = None,
    leg_range: tuple      = (2, 8),
) -> None:
    """
    Plot the Map-Elites archive as a 2-D grid.

    Parameters
    ----------
    name       : save name to load (default "last_best")
    save_path  : if given, save the figure to this path instead of showing it
    leg_range  : (min_legs, max_legs) inclusive range for the X axis
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("[archive_explorer] matplotlib is required for plot_archive(). Install with: pip install matplotlib")
        return

    try:
        elites = load_archive(name)
        payload = load_controller(name)
        saved_at = payload.get("saved_at", "")
    except FileNotFoundError:
        print(f"[archive_explorer] No save found: '{name}'")
        return

    # ------------------------------------------------------------------
    # Build grid arrays
    # ------------------------------------------------------------------
    min_legs, max_legs = leg_range
    x_labels = list(range(min_legs, max_legs + 1))          # nb_legs
    y_labels = SYMMETRY_BIN_LABELS                           # symmetry bins
    n_x, n_y = len(x_labels), len(y_labels)

    score_grid  = np.full((n_y, n_x), np.nan)    # displacement (NaN = empty)
    elite_map   = {}                              # (row, col) → elite dict

    for e in elites:
        nb_legs, sym_bin = e["feature_key"]
        if nb_legs < min_legs or nb_legs > max_legs:
            continue
        col = nb_legs - min_legs
        row = sym_bin
        score_grid[row, col] = e["displacement_m"]
        elite_map[(row, col)] = e

    filled      = int(np.sum(~np.isnan(score_grid)))
    total_cells = n_x * n_y
    vmax        = np.nanmax(score_grid) if filled > 0 else 1.0
    vmin        = 0.0

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(7, n_x * 1.4), max(4, n_y * 1.4)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    cmap_filled = plt.cm.plasma
    color_empty = "#2e2e4a"
    color_text  = "white"

    # ------------------------------------------------------------------
    # Draw cells
    # ------------------------------------------------------------------
    for row in range(n_y):
        for col in range(n_x):
            val = score_grid[row, col]
            if np.isnan(val):
                face = color_empty
                alpha = 1.0
                label_lines = ["—"]
            else:
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                face  = cmap_filled(0.2 + 0.78 * norm_val)
                alpha = 1.0
                e = elite_map[(row, col)]
                label_lines = [
                    f"{val:.3f} m",
                    f"R{e['robot_index']}",
                ]

            rect = plt.Rectangle(
                (col, row), 1, 1,
                facecolor=face, edgecolor="#0d0d1a", linewidth=1.5, alpha=alpha,
            )
            ax.add_patch(rect)

            for k, line in enumerate(label_lines):
                font_size = 9 if len(label_lines) == 1 else 8
                y_offset  = 0.5 + (len(label_lines) / 2 - k - 0.5) * 0.22
                ax.text(
                    col + 0.5, row + y_offset, line,
                    ha="center", va="center",
                    fontsize=font_size, color=color_text, fontweight="bold",
                )

    # ------------------------------------------------------------------
    # Axes, labels, colorbar
    # ------------------------------------------------------------------
    ax.set_xlim(0, n_x)
    ax.set_ylim(0, n_y)
    ax.set_xticks([i + 0.5 for i in range(n_x)])
    ax.set_xticklabels([f"{l} legs" for l in x_labels], color=color_text, fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(n_y)])
    ax.set_yticklabels(y_labels, color=color_text, fontsize=10)
    ax.tick_params(length=0)

    ax.set_xlabel("Number of legs", color=color_text, fontsize=11, labelpad=8)
    ax.set_ylabel("Symmetry", color=color_text, fontsize=11, labelpad=8)

    title = f"Map-Elites Archive — {name}"
    if saved_at:
        title += f"\n{saved_at}   |   {filled}/{total_cells} cells filled"
    ax.set_title(title, color=color_text, fontsize=12, pad=12)

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap_filled,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Displacement (m)", color=color_text, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=color_text)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=color_text)
    cbar.outline.set_edgecolor(color_text)

    for spine in ax.spines.values():
        spine.set_edgecolor("#0d0d1a")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[archive_explorer] Saved figure → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--list" in args:
        saves = list_saves()
        if saves:
            print("Available saves:")
            for s in sorted(saves):
                print(f"  {s}")
        else:
            print("No saves found.")
    else:
        archive_name = args[0] if args else "last_best"
        print_archive(archive_name)
        plot_archive(archive_name)
