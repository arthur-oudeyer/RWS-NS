"""
correlation_experiment.py
==========================
Compares human morphology evaluations to Gemini VLM scores and produces
correlation graphs.

Workflow
--------
1. Load the human-annotated dataset from
       utils/human_eval_dataset/dataset.json
   (built with utils/morph_human_evaluation.py).
2. For each morphology, load its saved PNG and score it with GeminiGrader
   using the exact static/dynamic targets that the human annotator defined.
   Entries that share the same target pair are batched in a single API call.
3. Compute Pearson correlations (and p-values) between human and VLM scores
   on three dimensions: coherence, originality, interest, plus an overall average.
4. Save per-morphology VLM scores and correlation stats to JSON.
5. Generate scatter plots with regression lines (requires matplotlib).

Usage
-----
    python correlation_experiment.py
    python correlation_experiment.py --output results/corr_01
    python correlation_experiment.py --vlm-scores results/corr_01/vlm_scores.json
    python correlation_experiment.py --api-key YOUR_GEMINI_KEY   # override key

Arguments
---------
    --api-key       Gemini API key (defaults to APIKEY_GEMINI from code/api_keys.py)
    --dataset       Path to dataset.json  (default: utils/human_eval_dataset/dataset.json)
    --output        Output directory       (default: results/correlation)
    --model         Gemini model ID        (default: gemini-3-flash-preview)
    --batch-size    Images per API call    (default: 10)
    --vlm-scores    Path to a pre-computed vlm_scores.json to skip re-scoring
    --debug         Verbose grader output

Output in <output>/
-------------------
    vlm_scores.json          — per-morphology VLM scores + reasoning
    correlation_results.json — Pearson r, p-value and n per dimension
    plot_coherence.png
    plot_originality.png
    plot_interest.png
    plot_overall.png
    plot_all.png             — 4-panel summary figure
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── project imports ──────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent))  # code/ — for api_keys.py

try:
    from api_keys import APIKEY_GEMINI as _DEFAULT_API_KEY
except ImportError:
    _DEFAULT_API_KEY = None

from gemini_prompts import GeminiPromptConfig, GeminiScoringWeights, build_morphology_prompt
from grader import GeminiGrader

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
_DATASET_DIR  = _SCRIPT_DIR / "utils" / "human_eval_dataset"
_DATASET_FILE = _DATASET_DIR / "dataset.json"
DIMENSIONS    = ("coherence", "originality", "interest")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: Dataset not found: {path}")
        print("Run utils/morph_human_evaluation.py first to build the dataset.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if not data:
        print("ERROR: Dataset is empty — annotate some morphologies first.")
        sys.exit(1)
    print(f"Loaded {len(data)} annotated morphologies from {path}")
    return data


# ---------------------------------------------------------------------------
# VLM evaluation
# ---------------------------------------------------------------------------

def run_vlm_evaluation(
    dataset:    list[dict],
    api_key:    str,
    model_name: str,
    batch_size: int,
    debug:      bool,
) -> dict[str, dict]:
    """
    Score each morphology with GeminiGrader using the human-defined targets.

    Entries are grouped by (static_target, dynamic_target) so each unique
    target pair shares one GeminiGrader instance and one batch API call.

    VLM raw scores (coherence / originality / interest) are stored in the
    0–10 range: the Gemini prompt asks for 0–100 and the batch parser
    divides by 10, matching the human 0–10 annotation scale.

    Returns
    -------
    dict[morph_id] -> {coherence, originality, interest, fitness,
                       observation, interpretation,
                       coherence_reason, originality_reason, interest_reason}
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for entry in dataset:
        key = (entry["static_target"], entry["dynamic_target"])
        groups[key].append(entry)

    results: dict[str, dict] = {}

    for (static_t, dynamic_t), entries in groups.items():
        print(f"\n[VLM] {len(entries)} morphologies  "
              f"[static='{static_t}'  dynamic='{dynamic_t}']")

        prompt_config = GeminiPromptConfig(
            name    = f"corr_{static_t.replace(' ', '_')[:20]}",
            target  = static_t,
            prompt  = build_morphology_prompt(static_t, dynamic_t),
            weights = GeminiScoringWeights(coherence=1.0, originality=1.0, interest=1.0),
        )
        grader = GeminiGrader(
            api_key       = api_key,
            prompt_config = prompt_config,
            model_name    = model_name,
            batch_size    = batch_size,
            debug         = debug,
        )

        images: list[tuple[str, PILImage.Image]] = []
        for entry in entries:
            img_path = _DATASET_DIR / entry["image_path"]
            if not img_path.exists():
                print(f"  WARNING: image not found: {img_path} — skipping {entry['id']}")
                continue
            img = PILImage.open(img_path).convert("RGB")
            images.append((entry["id"], img))

        if not images:
            print("  No images found for this group — skipping.")
            continue

        try:
            batch_out = grader.score_batch(images, debug=debug)
        except Exception as exc:
            print(f"  ERROR during batch scoring: {exc}")
            continue

        for morph_id, output in batch_out.items():
            results[morph_id] = {
                "coherence":          output.raw_scores.get("coherence",   0.0),
                "originality":        output.raw_scores.get("originality", 0.0),
                "interest":           output.raw_scores.get("interest",    0.0),
                "fitness":            output.fitness,
                "observation":        output.extra.get("observation",        ""),
                "interpretation":     output.extra.get("interpretation",     ""),
                "coherence_reason":   output.extra.get("coherence_reason",   ""),
                "originality_reason": output.extra.get("originality_reason", ""),
                "interest_reason":    output.extra.get("interest_reason",    ""),
            }
            if debug:
                v = results[morph_id]
                print(f"  {morph_id}: coh={v['coherence']:.2f}  "
                      f"orig={v['originality']:.2f}  int={v['interest']:.2f}")

    return results


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_correlations(
    dataset:    list[dict],
    vlm_scores: dict[str, dict],
) -> dict[str, dict]:
    """
    Pair human and VLM scores and compute Pearson correlations.

    Both human scores (0–10 integers) and VLM raw scores (0–10 floats)
    are on the same scale and directly comparable.

    Returns
    -------
    dict[dimension] -> {n, pearson_r, p_value, human: [...], vlm: [...]}
    """
    paired: dict[str, dict[str, list]] = {
        dim: {"human": [], "vlm": []}
        for dim in (*DIMENSIONS, "overall")
    }

    for entry in dataset:
        mid = entry["id"]
        if mid not in vlm_scores:
            continue
        h = entry["human_scores"]
        v = vlm_scores[mid]

        for dim in DIMENSIONS:
            paired[dim]["human"].append(float(h[dim]))
            paired[dim]["vlm"].append(float(v[dim]))

        paired["overall"]["human"].append(sum(h[d] for d in DIMENSIONS) / len(DIMENSIONS))
        paired["overall"]["vlm"].append(  sum(v[d] for d in DIMENSIONS) / len(DIMENSIONS))

    correlations: dict[str, dict] = {}
    for dim, data in paired.items():
        h_arr = np.array(data["human"])
        v_arr = np.array(data["vlm"])
        n     = len(h_arr)

        if n < 3:
            correlations[dim] = {
                "n": n, "pearson_r": None, "p_value": None,
                "human": data["human"], "vlm": data["vlm"],
                "note": "Too few samples (need ≥ 3) to compute correlation.",
            }
            continue

        if _SCIPY_AVAILABLE:
            r, p = _scipy_stats.pearsonr(h_arr, v_arr)
            r, p = float(r), float(p)
        else:
            hm, vm = h_arr.mean(), v_arr.mean()
            denom  = (np.sqrt(np.sum((h_arr - hm) ** 2)) *
                      np.sqrt(np.sum((v_arr - vm) ** 2)) + 1e-12)
            r = float(np.sum((h_arr - hm) * (v_arr - vm)) / denom)
            p = None

        correlations[dim] = {
            "n":         n,
            "pearson_r": round(r, 4),
            "p_value":   round(p, 6) if p is not None else None,
            "human":     data["human"],
            "vlm":       data["vlm"],
        }

    return correlations


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_DARK_BG  = "#1e1e1e"
_DARK_BG2 = "#252525"
_GRID     = "#333333"
_SCATTER  = "#66aaff"
_REGLINE  = "#ff8844"
_TEXT_DIM = ("coherence", "#88aaff"), ("originality", "#ffaa44"), ("interest", "#88ff88"), ("overall", "#cccccc")
_DIM_COLOR = {d: c for d, c in _TEXT_DIM}


def _make_axes(ax, dim: str):
    ax.set_facecolor(_DARK_BG2)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.grid(True, color=_GRID, linewidth=0.5, zorder=0)
    ax.set_xlabel("Human score (0–10)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("VLM score (0–10)",   color="#aaaaaa", fontsize=9)


def _scatter_and_fit(ax, h: np.ndarray, v: np.ndarray, dim: str):
    color = _DIM_COLOR.get(dim, "#66aaff")
    ax.scatter(h, v, color=color, alpha=0.75, s=60, zorder=3,
               edgecolors="#ffffff22", linewidths=0.4)
    if len(h) >= 3:
        m, b = np.polyfit(h, v, 1)
        xs = np.linspace(h.min(), h.max(), 100)
        ax.plot(xs, m * xs + b, color=_REGLINE, linewidth=1.8, zorder=4)


def plot_dimension(corr: dict, dim: str, output: Path) -> None:
    h = np.array(corr["human"])
    v = np.array(corr["vlm"])
    r = corr.get("pearson_r")
    p = corr.get("p_value")
    n = corr["n"]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(_DARK_BG)

    _make_axes(ax, dim)
    _scatter_and_fit(ax, h, v, dim)

    r_str = f"r = {r:.3f}" if r is not None else "r = n/a"
    p_str = f"  p = {p:.4f}" if p is not None else ""
    ax.set_title(
        f"{dim.capitalize()}  ({r_str}{p_str}  n = {n})",
        color="#cccccc", fontsize=11, pad=10,
    )

    fig.tight_layout()
    out = output / f"plot_{dim}.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_all(correlations: dict, output: Path) -> None:
    dims = list(DIMENSIONS) + ["overall"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor(_DARK_BG)
    axes = axes.flatten()

    for ax, dim in zip(axes, dims):
        corr = correlations.get(dim, {})
        h = np.array(corr.get("human", []))
        v = np.array(corr.get("vlm",   []))
        r = corr.get("pearson_r")
        p = corr.get("p_value")
        n = corr.get("n", 0)

        _make_axes(ax, dim)
        if len(h) > 0:
            _scatter_and_fit(ax, h, v, dim)

        r_str = f"r={r:.3f}" if r is not None else "r=n/a"
        p_str = f"  p={p:.4f}" if p is not None else ""
        ax.set_title(
            f"{dim.capitalize()}  ({r_str}{p_str}  n={n})",
            color="#cccccc", fontsize=10,
        )

    fig.suptitle(
        "Human vs VLM Morphology Evaluation — Correlation",
        color="#dddddd", fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out = output / "plot_all.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Correlation experiment: human vs VLM morphology evaluation"
    )
    parser.add_argument("--api-key",    default=None,
                        help="Gemini API key (defaults to APIKEY_GEMINI from code/api_keys.py)")
    parser.add_argument("--dataset",    default=str(_DATASET_FILE),
                        help="Path to dataset.json")
    parser.add_argument("--output",     default="results/correlation",
                        help="Output directory")
    parser.add_argument("--model",      default="gemini-3-flash-preview",
                        help="Gemini model ID")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Images per batch API call")
    parser.add_argument("--vlm-scores", default=None,
                        help="Path to pre-computed vlm_scores.json (skips VLM re-scoring)")
    parser.add_argument("--debug",      action="store_true",
                        help="Verbose grader output")
    args = parser.parse_args()

    if not _PIL_AVAILABLE:
        print("ERROR: Pillow is required.  pip install Pillow")
        sys.exit(1)
    if not _MPL_AVAILABLE:
        print("WARNING: matplotlib not available — plots will not be generated.  pip install matplotlib")
    if not _SCIPY_AVAILABLE:
        print("WARNING: scipy not available — p-values will not be computed.  pip install scipy")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load human dataset ────────────────────────────────────────────
    dataset = load_dataset(Path(args.dataset))

    # ── 2. VLM evaluation ───────────────────────────────────────────────
    if args.vlm_scores:
        print(f"\nLoading pre-computed VLM scores from: {args.vlm_scores}")
        with open(args.vlm_scores) as f:
            vlm_scores = json.load(f)
        print(f"Loaded {len(vlm_scores)} VLM entries.")
    else:
        api_key = args.api_key or _DEFAULT_API_KEY
        if not api_key:
            print("ERROR: no API key found.  Pass --api-key or add APIKEY_GEMINI to code/api_keys.py")
            sys.exit(1)
        print("\n─── VLM Evaluation ───")
        vlm_scores = run_vlm_evaluation(
            dataset    = dataset,
            api_key    = api_key,
            model_name = args.model,
            batch_size = args.batch_size,
            debug      = args.debug,
        )
        print(f"\nScored {len(vlm_scores)} / {len(dataset)} morphologies.")

        vlm_out = output_dir / "vlm_scores.json"
        with open(vlm_out, "w") as f:
            json.dump(vlm_scores, f, indent=2)
        print(f"VLM scores saved to: {vlm_out}")

    # ── 3. Correlation analysis ──────────────────────────────────────────
    print("\n─── Correlation Analysis ───")
    correlations = compute_correlations(dataset, vlm_scores)

    header = f"{'Dimension':<14}  {'n':>4}  {'Pearson r':>10}  {'p-value':>10}"
    print(header)
    print("─" * len(header))
    for dim in (*DIMENSIONS, "overall"):
        c = correlations.get(dim, {})
        r = c.get("pearson_r")
        p = c.get("p_value")
        n = c.get("n", 0)
        r_str = f"{r:+.4f}" if r is not None else "     n/a"
        p_str = f"{p:.6f}" if p is not None else "       n/a"
        print(f"  {dim:<12}  {n:4d}  {r_str:>10}  {p_str:>10}")

    # Save summary (exclude raw arrays for file size)
    corr_summary = {
        dim: {k: v for k, v in c.items() if k not in ("human", "vlm")}
        for dim, c in correlations.items()
    }
    corr_out = output_dir / "correlation_results.json"
    with open(corr_out, "w") as f:
        json.dump({
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "n_dataset":    len(dataset),
            "n_scored":     len(vlm_scores),
            "model":        args.model,
            "correlations": corr_summary,
        }, f, indent=2)
    print(f"\nCorrelation results saved to: {corr_out}")

    # ── 4. Plots ─────────────────────────────────────────────────────────
    if _MPL_AVAILABLE:
        print("\n─── Generating Plots ───")
        for dim in (*DIMENSIONS, "overall"):
            c = correlations.get(dim, {})
            if c.get("n", 0) >= 2:
                plot_dimension(c, dim, output_dir)
        plot_all(correlations, output_dir)
    else:
        print("\nSkipping plots (matplotlib not installed).")

    print(f"\nDone.  Results in: {output_dir}")


if __name__ == "__main__":
    main()
