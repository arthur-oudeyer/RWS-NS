"""
gemini_prompts.py  (MJX edition — standalone)
=============================================
Prompt configurations for `vlm_grader.LocomotionGrader`.

Standalone copy of Controller/gemini_prompts.py (the Controller_MJX package must
not import from Controller/). The VLM sees an MP4 of one episode and scores three
dimensions; fitness ∈ [0, 1] is a weighted mean computed by the grader.

Output JSON schema expected from Gemini (per video)
---------------------------------------------------
    {
      "observation":     "factual key-step description",
      "interpretation":  "behavioural interpretation relative to the target",
      "coherence":       { "score": <int 0-100>, "reason": "..." },
      "originality":     { "score": <int 0-100>, "reason": "..." },
      "potential":        { "score": <int 0-100>, "reason": "..." }
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LocomotionScoringWeights:
    """How the three locomotion scores combine into one fitness value."""
    coherence:   float = 1.0
    originality: float = 0.5
    potential:   float = 1.5


@dataclass
class LocomotionPromptConfig:
    """A named evaluation configuration for the locomotion grader.

    Attributes
    ----------
    name    : short identifier (used in GraderOutput.prompt_set).
    target  : target behaviour ("walk forward", "jump high", "crawl", …).
    prompt  : full prompt sent to Gemini alongside the MP4.
    weights : per-dimension weights for fitness aggregation.
    """
    name:    str
    target:  str
    prompt:  str
    weights: LocomotionScoringWeights = field(default_factory=LocomotionScoringWeights)


# Marker the grader splits on to swap the single-video schema for the batch one.
OUTPUT_MARKER = "═══ OUTPUT FORMAT ═══"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_locomotion_prompt(target_behaviour: str) -> str:
    """
    Build a strict, behaviourally-scoped Gemini prompt for grading one rollout
    video against a target behaviour description.
    """
    output_format = """\
    {
      "observation":    "factual description",
      "interpretation": "behavioural interpretation",
      "coherence":      { "score": <int 0-100>, "reason": "..." },
      "originality":    { "score": <int 0-100>, "reason": "..." },
      "potential":       { "score": <int 0-100>, "reason": "..." }
    }"""

    return f"""
    ═══ CONTEXT ═══

    You are looking at video of a short simulation showing two side-by-side view of a simulated robot composed of a white torso and colored legs. It stands on a green checkered floor, and the background is blue.

    Target behavior : {target_behaviour}

    ═══ ANALYSIS ═══

    Step 1 — factual observation
    Describe the robot morphology and behavior. Don't hold back when it comes to wordiness.
    
    Step 2 — Behavioural interpretation
    - Did the robot make consistent actions relevant to the target behaviour ?
    - Was the gait coherent (periodic, balanced, repeatable) or random ? What type of gait (smooth, energetic, nervous, wide, brutal, efficient, small, homogeneous, ...) ?
    - Is there anything novel or interesting about the motion pattern even if the robot did not perform well for the target behaviour ? (e.g. is a limb doing a movement with great potential ?)

    Step 3 — Scoring (each dimension 0–100)

    coherence — Is the gait relevant for the target behaviour ?
      0–29   = chaotic thrashing, immediate collapse, fully static or no recognisable pattern
      30–49  = unstable, sporadic; one or two coherent moments only that have a link to the target
      50–69  = partial coherence; clear periodic pattern or specific movement but with wobble or stalls. The target can be identified.
      70–89  = coherent, repeatable gait or target well reached ; minor instabilities only. The intention toward target is obvious.
      90–100 = clean, stable, periodic locomotion throughout; the target is perfectly depicted through this video.

    originality — Did the robot achieve something toward the behavioural target in an original way ?
      0–29   = no movement or movement very basic with no progress toward the target
      30–49  = one basic movement, not very original
      50–69  = novel movements that provide new ability for the robot
      70–89  = clear and unexpected movement that somehow helps the robot progress toward the target behaviour
      90–100 = very unexpected but very efficient way to reach the wanted behaviour

    potential — Is the gait pattern interesting, biologically plausible and leading to real evolutionary potential ?
      0–29   = uninteresting (random, fallen) or obviously broken
      30–49  = generic, predictable motion with no notable features
      50–69  = one notable element (unusual gait phase, rhythm, recovery) that has potential
      70–89  = clearly interesting motion: reminiscent of an animal gait, coordinated pattern, or creative body usage to reach the target. Great potential.
      90–100 = highly interesting; novel and biologically convincing locomotion, great abilities and great potential for further evolution.

    ═══ OUTPUT FORMAT ═══
    Respond ONLY with valid JSON, no text before or after:

    {output_format}
    """


def get_fake_answer() -> str:
    raw = {
        "observation":    "fake observation",
        "interpretation": "fake interpretation",
        "coherence":      {"score": random.randint(0, 100), "reason": "coherence reason"},
        "originality":    {"score": random.randint(0, 100), "reason": "originality reason"},
        "potential":      {"score": random.randint(0, 100), "reason": "potential reason"},
    }
    return json.dumps(raw, indent=2)


def generate_fake_vlm_batch_response(robot_ids) -> str:
    """Fake batch VLM response keyed by robot id (for wiring tests, no network)."""
    robots_data = {}
    for robot_id in robot_ids:
        robots_data[robot_id] = {
            "observation":    "fake observation",
            "interpretation": "fake interpretation",
            "coherence":      {"score": random.randint(0, 100), "reason": f"Coherence reason for {robot_id}."},
            "originality":    {"score": random.randint(0, 100), "reason": f"Originality reason for {robot_id}."},
            "potential":       {"score": random.randint(0, 100), "reason": f"potential reason for {robot_id}."},
        }
    return json.dumps(robots_data, indent=2)


# ---------------------------------------------------------------------------
# Pre-built configurations
# ---------------------------------------------------------------------------

def make_prompt_config(name: str, target_behaviour: str,
                       weights: "LocomotionScoringWeights | None" = None) -> LocomotionPromptConfig:
    """Build a LocomotionPromptConfig from a behaviour description."""
    return LocomotionPromptConfig(
        name    = name,
        target  = target_behaviour,
        prompt  = build_locomotion_prompt(target_behaviour),
        weights = weights or LocomotionScoringWeights(),
    )


WALK_FORWARD = make_prompt_config(
    "walk_forward", "walk forward fast and continuously while staying upright")
JUMP_HIGH = make_prompt_config(
    "jump_high", "jump as high as possible using all four legs")
ROTATE = make_prompt_config(
    "rotate", "spin/rotate in place continuously about the vertical axis while staying upright")
CRAWL = make_prompt_config(
    "crawl", "crawl forward with the torso low to the ground")


ALL_LOCOMOTION_PROMPT_CONFIGS: dict[str, LocomotionPromptConfig] = {
    c.name: c for c in (WALK_FORWARD, JUMP_HIGH, ROTATE, CRAWL)
}


def get_locomotion_prompt_set(name: str) -> LocomotionPromptConfig:
    if name not in ALL_LOCOMOTION_PROMPT_CONFIGS:
        raise KeyError(
            f"Unknown locomotion prompt config '{name}'. "
            f"Available: {list(ALL_LOCOMOTION_PROMPT_CONFIGS.keys())}"
        )
    return ALL_LOCOMOTION_PROMPT_CONFIGS[name]


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for c in ALL_LOCOMOTION_PROMPT_CONFIGS.values():
        print("\n" + "=" * 60)
        print(f"  {c.name}  (target = {c.target})")
        print(f"  weights : coherence={c.weights.coherence} "
              f"originality={c.weights.originality} potential={c.weights.potential}")
        print()
        print(c.prompt.strip()[:600], "…")
