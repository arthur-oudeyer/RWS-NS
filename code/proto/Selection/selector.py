"""
selector.py
===========
Map-Elites-style selection for heterogeneous robot populations.

Each robot is mapped to a cell in a discrete feature space via feature_descriptor().
selection() keeps the best-performing robot (by displacement_xy) per cell,
producing the elite archive that is saved and used as mutation seeds.

Feature space (edit here to change dimensions):
  - nb_legs:        discrete integer (2, 3, 4, ...)
  - symmetry_bin:   0 = asymmetric (<0.5), 1 = semi (0.5–0.8), 2 = symmetric (≥0.8)

Score used to rank robots within a cell: displacement_xy (only standing robots qualify).

Usage:
    from selector import feature_descriptor, selection

    elites = selection(all_metrics)          # dict: feature_key → RobotMetrics
    key    = feature_descriptor(metrics[i])  # tuple, e.g. (4, 2)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature space configuration
# ---------------------------------------------------------------------------

# Bin edges for symmetry_score (produces len(SYMMETRY_BINS)+1 buckets)
SYMMETRY_BINS = [0.5, 0.8]   # → buckets: [0, 0.5) = 0 | [0.5, 0.8) = 1 | [0.8, 1] = 2

SYMMETRY_BIN_LABELS = ["asymmetric", "semi-sym", "symmetric"]


def _symmetry_bin(score: float) -> int:
    for i, edge in enumerate(SYMMETRY_BINS):
        if score < edge:
            return i
    return len(SYMMETRY_BINS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def feature_descriptor(metrics) -> tuple:
    """
    Map a RobotMetrics to a hashable feature cell key.

    Parameters
    ----------
    metrics : RobotMetrics
        Must have .nb_legs (int) and .symmetry_score (float).

    Returns
    -------
    tuple — e.g. (4, 2) for a symmetric quadruped.
    """
    return (metrics.nb_legs, _symmetry_bin(metrics.symmetry_score))


def feature_label(key: tuple) -> str:
    """Human-readable label for a feature key."""
    nb_legs, sym_bin = key
    sym_label = SYMMETRY_BIN_LABELS[sym_bin] if sym_bin < len(SYMMETRY_BIN_LABELS) else str(sym_bin)
    return f"{nb_legs}legs/{sym_label}"


def selection(all_metrics: list) -> dict:
    """
    Map-Elites selection: keep the best robot per feature cell.

    Only robots standing at the end of the simulation are considered.
    Within each cell the robot with the highest displacement_xy wins.

    Parameters
    ----------
    all_metrics : list[RobotMetrics]

    Returns
    -------
    dict mapping feature_key (tuple) → RobotMetrics.
    Empty dict if no robot is standing.
    """
    elites: dict[tuple, object] = {}
    for m in all_metrics:
        if not m.is_standing_end:
            continue
        key = feature_descriptor(m)
        if key not in elites or m.displacement_xy > elites[key].displacement_xy:
            elites[key] = m
    return elites
