"""
performance_grader.py
=====================
Non-VLM, deterministic fitness grader for the MJX evolution loop.

Drop-in replacement for `LocomotionGrader` that scores each individual with
an *external performance function* of the rollout instead of a Gemini call.
Used to validate the evolutionary machinery (μ+λ, MAP-Elites) cheaply, before
the VLM layer is wired in.

How it plugs into the existing loop
-----------------------------------
`data_handler.evaluate_batch` only ever touches the grader through
`grader.score_batch(videos, debug, reference_video) -> {id: GraderOutput}`,
where each `videos` entry is `(individual_id_str, mp4_path)`. It then reads
`.fitness / .raw_scores / .method / .prompt_set / .extra` off each output.

The physical performance metric (e.g. max jump height) is a *physics* quantity,
not something visible in the encoded MP4. So the evolution loop computes it at
render time (the renderer already runs a full physics rollout) and hands the
per-rollout `info` dict to this grader via `register(video_path, info)`. At
`score_batch` time the grader looks each video up and applies `metric_fn(info)`.

`metric_fn` is the pluggable "external function": `info -> float`. The default
scores jump height (peak torso z above spawn). Swap it for any other scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Minimal GraderOutput (duck-typed match for grader.GraderOutput)
# ---------------------------------------------------------------------------
# Defined locally rather than imported from grader.py so this module has no
# Gemini / google-genai dependency.

@dataclass
class GraderOutput:
    fitness:    float
    raw_scores: dict
    method:     str
    prompt_set: str
    extra:      dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric functions (info dict -> float). The "external" objective.
# ---------------------------------------------------------------------------

def jump_height_metric(info: dict) -> float:
    """Peak torso height above the spawn pose (metres). 0 if it never rose."""
    return float(info.get("jump_height", 0.0))


def max_height_metric(info: dict) -> float:
    """Absolute peak torso z reached during the episode (metres)."""
    return float(info.get("max_torso_height", 0.0))


# ---------------------------------------------------------------------------
# PerformanceGrader
# ---------------------------------------------------------------------------

class PerformanceGrader:
    """
    Score rollouts with a deterministic external metric function.

    Parameters
    ----------
    metric_fn   : info_dict -> float. The performance score (higher = better).
                  Defaults to jump height (peak torso z above spawn).
    metric_name : label used in raw_scores / method (default "jump_height").
    descriptor_fn : optional info_dict -> dict, returns MAP-Elites descriptors
                  (e.g. {"height_bin": 2}). Stored under extra["vlm_descriptors"]
                  so the MapEliteArchive can read it unchanged. Default: {}.
    """

    def __init__(
        self,
        metric_fn:     Callable[[dict], float] = jump_height_metric,
        metric_name:   str = "jump_height",
        descriptor_fn: Optional[Callable[[dict], dict]] = None,
    ):
        self._metric_fn     = metric_fn
        self._metric_name   = metric_name
        self._descriptor_fn = descriptor_fn
        # video_path -> rollout info dict, filled by the evolution loop at render
        self._registry: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Registration (called by the evolution loop after each render)
    # ------------------------------------------------------------------

    def score_of(self, info: dict) -> float:
        """Apply the metric function without registering (e.g. to name a file)."""
        return float(self._metric_fn(info))

    def register(self, video_path: str, info: dict) -> float:
        """Record a rollout's info dict and return its metric score."""
        self._registry[video_path] = dict(info)
        return self._metric_fn(info)

    # ------------------------------------------------------------------
    # Grader interface (duck-typed for data_handler.evaluate_batch)
    # ------------------------------------------------------------------

    def score_batch(
        self,
        videos:          "list[tuple[str, str]]",
        debug:           Optional[bool] = None,
        reference_video: Optional[str]  = None,
    ) -> "dict[str, GraderOutput]":
        results: dict[str, GraderOutput] = {}
        for vid, mp4 in videos:
            info = self._registry.get(mp4)
            if info is None:
                raise KeyError(
                    f"[PerformanceGrader] no rollout info registered for "
                    f"{vid} ({mp4}). Call register(video_path, info) at render time."
                )
            score = float(self._metric_fn(info))
            descriptors = self._descriptor_fn(info) if self._descriptor_fn else {}
            results[vid] = GraderOutput(
                fitness    = score,
                raw_scores = {
                    self._metric_name: round(score, 4),
                    "max_torso_height": round(float(info.get("max_torso_height", 0.0)), 4),
                    "spawn_height":     round(float(info.get("spawn_height", 0.0)), 4),
                    "total_reward":     round(float(info.get("total_reward", 0.0)), 4),
                },
                method     = f"performance_{self._metric_name}",
                prompt_set = self._metric_name,
                extra      = {"vlm_descriptors": descriptors, "rollout_info": info},
            )
        return results


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  performance_grader.py — debug mode")
    print("=" * 60)

    g = PerformanceGrader()
    infos = {
        "/tmp/a.mp4": {"spawn_height": 0.30, "max_torso_height": 0.55, "jump_height": 0.25,
                       "total_reward": 12.0},
        "/tmp/b.mp4": {"spawn_height": 0.30, "max_torso_height": 0.31, "jump_height": 0.01,
                       "total_reward": 3.0},
    }
    for path, info in infos.items():
        g.register(path, info)

    out = g.score_batch([("robot_0", "/tmp/a.mp4"), ("robot_1", "/tmp/b.mp4")])
    for vid, go in out.items():
        print(f"  {vid}: fitness={go.fitness:.4f}  raw={go.raw_scores}  method={go.method}")
    assert out["robot_0"].fitness > out["robot_1"].fitness
    print("\nAll performance_grader.py checks passed.")
