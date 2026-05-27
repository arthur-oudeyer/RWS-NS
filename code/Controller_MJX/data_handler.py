"""
data_handler.py
===============
The unit of data stored in the controller archive — `ControllerResult` —
plus `evaluate_batch` which ties (mutate → warm-start → rollout → score)
together for a generation of children.

ControllerResult
----------------
Everything about one trained-and-graded individual:
  - the `reward_weights` (the EA-evolved variable), stored as a plain dict
    so the result is JSON-serialisable for `individuals_log.jsonl`
  - `policy_path` to the SB3 .zip the trainer produced
  - `video_path` to the rendered MP4 the grader saw
  - `fitness`, `raw_scores`, `descriptors` from the VLM
  - `parent_id` for lineage analysis (None for from-scratch initial individuals)
  - `n_train_steps` to track the cumulative inner-loop compute spent

evaluate_batch
--------------
Given a list of `(reward_weights, policy_path, video_path, parent_id)`
specs (already trained and rendered by the evolution loop), call the
grader once for the whole batch and assemble `ControllerResult` records.

Reconstruct a `RewardWeights` from a result via `RewardWeights(**r.reward_weights)`.

Debug
-----
Run this file to round-trip a `ControllerResult` to/from a dict and to
exercise `evaluate_batch` against a fake grader (no Gemini call).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as dc_field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ControllerResult
# ---------------------------------------------------------------------------

@dataclass
class ControllerResult:
    """
    Full record of one evaluated controller individual.

    Attributes
    ----------
    generation     : generation index (0 = initial population).
    individual_id  : unique int across the run (assigned by experiment loop).
    parent_id      : individual_id of the parent it was warm-started from,
                     or None for initial-population (from-scratch) individuals.
    reward_weights : the evolved variable, stored as plain dict for JSONL.
                     Reconstruct via `RewardWeights(**result.reward_weights)`.
    policy_path    : path to the SB3 .zip the trainer produced.
    video_path     : path to the rendered MP4 the grader saw.
    n_train_steps  : PPO timesteps spent on this individual
                     (n_init_steps for from-scratch, n_warm_steps otherwise).
    fitness        : scalar in [0, 1] from the VLM. Drives selection.
    raw_scores     : per-dimension VLM scores (coherence, progress, interest …).
    descriptors    : MAP-Elites descriptors (VLM-returned + structural).
    grader_method  : "gemini_video" | "gemini_video_batch" | "fake".
    prompt_set     : name of the LocomotionPromptConfig.
    grader_extra   : VLM observation/interpretation/per-dim reasons.
    """
    generation:     int
    individual_id:  int
    parent_id:      Optional[int]
    reward_weights: dict
    policy_path:    Optional[str]
    video_path:     Optional[str]
    n_train_steps:  int
    fitness:        float
    raw_scores:     dict
    descriptors:    dict
    grader_method:  str
    prompt_set:     str
    grader_extra:   dict = dc_field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[gen={self.generation:>3}  id={self.individual_id:>5}]  "
            f"fitness={self.fitness:+.4f}  "
            f"parent={self.parent_id}  "
            f"steps={self.n_train_steps:,}  "
            f"prompt={self.prompt_set}"
        )


def result_to_dict(r: ControllerResult) -> dict:
    return asdict(r)


def dict_to_result(d: dict) -> ControllerResult:
    return ControllerResult(
        generation     = d["generation"],
        individual_id  = d["individual_id"],
        parent_id      = d.get("parent_id"),
        reward_weights = d["reward_weights"],
        policy_path    = d.get("policy_path"),
        video_path     = d.get("video_path"),
        n_train_steps  = d.get("n_train_steps", 0),
        fitness        = d["fitness"],
        raw_scores     = d.get("raw_scores", {}),
        descriptors    = d.get("descriptors", {}),
        grader_method  = d.get("grader_method", "unknown"),
        prompt_set     = d.get("prompt_set", "unknown"),
        grader_extra   = d.get("grader_extra", {}),
    )


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------

@dataclass
class _IndividualSpec:
    """Minimal description of a child to grade — produced by the evolution loop."""
    reward_weights: dict           # plain dict (JSON-serialisable)
    policy_path:    Optional[str]
    video_path:     str            # MP4 path is required for grading
    parent_id:      Optional[int]
    n_train_steps:  int


def evaluate_batch(
    specs:           "list[_IndividualSpec]",
    grader,
    generation:      int,
    id_counter:      int,
    reference_video: Optional[str] = None,
    debug:           bool = False,
) -> "tuple[list[ControllerResult], int]":
    """
    Score a generation of children that have already been trained and
    rendered. Calls `grader.score_batch` once for the whole batch.

    Parameters
    ----------
    specs           : list of _IndividualSpec, one per child.
    grader          : LocomotionGrader (or any duck-typed equivalent
                      exposing `.score_batch(videos, reference_video=...)`).
    generation      : current generation index.
    id_counter      : first individual_id to assign; incremented per spec.
    reference_video : optional MP4 path of the current best individual,
                      passed through to the grader.
    debug           : passed through to the grader.

    Returns
    -------
    (results, new_id_counter)
    """
    n   = len(specs)
    ids = list(range(id_counter, id_counter + n))

    # Pair each spec with its assigned individual id (string key for the grader)
    labeled: list[tuple[str, str]] = [
        (f"robot_{ids[i]}", specs[i].video_path)
        for i in range(n)
    ]
    grader_outputs = grader.score_batch(
        videos          = labeled,
        debug           = debug,
        reference_video = reference_video,
    )

    results: list[ControllerResult] = []
    for i, spec in enumerate(specs):
        key = f"robot_{ids[i]}"
        go  = grader_outputs.get(key)
        if go is None:
            print(f"[evaluate_batch] WARNING: no grader output for {key}, skipping.")
            continue
        descriptors = dict(go.extra.get("vlm_descriptors", {}))
        results.append(ControllerResult(
            generation     = generation,
            individual_id  = ids[i],
            parent_id      = spec.parent_id,
            reward_weights = dict(spec.reward_weights),
            policy_path    = spec.policy_path,
            video_path     = spec.video_path,
            n_train_steps  = int(spec.n_train_steps),
            fitness        = float(go.fitness),
            raw_scores     = dict(go.raw_scores),
            descriptors    = descriptors,
            grader_method  = go.method,
            prompt_set     = go.prompt_set,
            grader_extra   = dict(go.extra),
        ))
    return results, id_counter + n


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  data_handler.py — debug mode")
    print("=" * 60)

    # 1. Round-trip
    r = ControllerResult(
        generation     = 0,
        individual_id  = 7,
        parent_id      = None,
        reward_weights = {"forward_velocity": 1.0},
        policy_path    = "/tmp/policy.zip",
        video_path     = "/tmp/rollout.mp4",
        n_train_steps  = 50_000,
        fitness        = 0.42,
        raw_scores     = {"coherence": 0.5, "progress": 0.4, "interest": 0.3},
        descriptors    = {},
        grader_method  = "fake",
        prompt_set     = "walk_forward",
    )
    back = dict_to_result(result_to_dict(r))
    assert back == r
    print(f"\n[1] round-trip: OK — {r}")

    # 2. evaluate_batch with a fake grader
    print("\n[2] evaluate_batch with fake grader\n")

    class _FakeOut:
        def __init__(self, fit, robot_id):
            self.fitness = fit
            self.raw_scores = {"coherence": fit, "progress": fit, "interest": fit}
            self.method = "fake"
            self.prompt_set = "fake"
            self.extra = {"vlm_descriptors": {"limb_count": 4}}

    class _FakeGrader:
        def score_batch(self, videos, debug=False, reference_video=None):
            return {vid: _FakeOut(0.5 + 0.1 * i, vid) for i, (vid, _) in enumerate(videos)}

    specs = [
        _IndividualSpec(
            reward_weights = {"forward_velocity": 1.0 + 0.1*i},
            policy_path    = f"/tmp/p{i}.zip",
            video_path     = f"/tmp/v{i}.mp4",
            parent_id      = None,
            n_train_steps  = 25_000,
        )
        for i in range(3)
    ]
    results, new_id = evaluate_batch(specs, _FakeGrader(), generation=1, id_counter=10)
    assert new_id == 13
    assert len(results) == 3
    for r in results:
        print(f"  {r}")
    assert results[0].descriptors == {"limb_count": 4}
    print("  evaluate_batch: OK")

    print("\nAll data_handler.py checks passed.")
