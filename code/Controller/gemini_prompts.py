"""
gemini_prompts.py
=================
Prompt configurations for `LocomotionGrader`.

Mirrors `Morphology/gemini_prompts.py` but is targeted at *behaviour*, not
appearance — the artefact the VLM sees is an MP4 of one episode (5 s by
default) rather than a still PNG. Three scored dimensions, fitness in
[0, 1].

Output JSON schema expected from Gemini
---------------------------------------
    {
      "observation":     "factual frame-by-frame description",
      "interpretation":  "behavioural interpretation",
      "coherence":       { "score": <int 0-100>, "reason": "..." },
      "progress":        { "score": <int 0-100>, "reason": "..." },
      "interest":        { "score": <int 0-100>, "reason": "..." }
    }

Fitness formula (computed by LocomotionGrader)
    fitness = (w_c * coherence + w_p * progress + w_i * interest)
              / (10 * (w_c + w_p + w_i))
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
    coherence: float = 1.0
    originality:  float = 0.5
    interest:  float = 1.5


@dataclass
class LocomotionPromptConfig:
    """A named evaluation configuration for the locomotion grader.

    Attributes
    ----------
    name    : short identifier (used in filenames, GraderOutput.prompt_set).
    target  : target behaviour ("walk forward", "jump high", "crawl", …).
    prompt  : full prompt sent to Gemini alongside the MP4.
    weights : per-dimension weights for fitness aggregation.
    """
    name:    str
    target:  str
    prompt:  str
    weights: LocomotionScoringWeights = field(default_factory=LocomotionScoringWeights)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_locomotion_prompt(target_behaviour: str) -> str:
    """
    Build a strict, behaviourally-scoped Gemini prompt for grading one
    rollout video against a target behaviour description.

    Sections
    --------
    CONTEXT  : describes the scene the VLM is looking at.
    ANALYSIS : forces frame-by-frame factual observation, then stagnation
               / fall checks, then conservative scoring.
    OUTPUT   : strict JSON-only schema. The single-video and batch graders
               both split the prompt at the OUTPUT FORMAT marker so we
               only define the schema once.
    """
    output_format = """\
    {
      "observation":    "key-time steps factual description (timestamps, floor-tile reference, posture)",
      "interpretation": "behavioural interpretation relative to the target",
      "coherence":      { "score": <int 0-100>, "reason": "..." },
      "originality":    { "score": <int 0-100>, "reason": "..." },
      "interest":       { "score": <int 0-100>, "reason": "..." }
    }"""

    return f"""
    ═══ CONTEXT ═══

    You are a strict and skeptical evaluator analysing a 5-second physics-simulation
    video of a robot in MuJoCo. Your job is to be PRECISE and
    CONSERVATIVE — do not give benefit of the doubt and do not assume movement
    if you are unsure.

    The scene:
    - Fixed-pose camera that pans laterally to keep the torso in view, the floor
      is a grassy checkerboard (use floor tiles as a position reference grid)
    - The robot has a white ellipsoidal body parts and colored legs
    - Background is plain blue (no distractors)
    
    Target behaviour : {target_behaviour}

    ═══ ANALYSIS ═══

    Step 1 — Frame-by-frame factual observation
    Be specific. Count tiles to estimate displacement between frames. Note the moment of any event, 
    example :
    - Frame 1 (0.0 s): posture, position, contact pattern
    - ~2.5 s: posture, did it advance / stagnate / fall?
    - Final frame (~5.0 s): final posture, final position relative to start
    - Key events: fall at X s, stagnation at Y s, gait change at Z s, …

    Step 2 — Behavioural interpretation
    - Did the robot make consistent forward progress, intermittent progress, or none toward the target behavior ?
    - Was the gait coherent (periodic, balanced, repeatable) or random thrashing ?
    - Is there anything novel or interesting about the motion pattern even if the robot did not perform well for the target behavior ?

    Step 3 — Conservative scoring (each dimension 0–100)

    coherence — Is the gait stable, periodic and well-controlled ?
      0–29   = chaotic thrashing, immediate collapse, fully static or no recognisable pattern
      30–49  = unstable, sporadic; one or two coherent moments only
      50–69  = partial coherence; clear periodic pattern but with wobble or stalls
      70–89  = coherent, repeatable gait; minor instabilities only
      90–100 = clean, stable, periodic locomotion throughout

    originality — Did the robot achieve something toward the behavioral target in an original way ?
      0–29   = no movement or movement very basic with no progress toward the target
      30–49  = one basic movement, not very original
      50–69  = novel movements that provide new ability for the robot
      70–89  = clear and unexpected movement that somehow help the robot progress toward the target behavior
      90–100 = very unexpected but very efficient way to reach the behavior wanted

    interest — Is the gait pattern interesting or biologically plausible?
      0–29   = uninteresting (random, fallen) or obviously broken
      30–49  = generic, predictable motion with no notable features
      50–69  = one notable element (unusual gait phase, rhythm, recovery)
      70–89  = clearly interesting motion: reminiscent of an animal gait,
               coordinated pattern, or creative body usage
      90–100 = highly interesting; novel and biologically convincing locomotion, great abilities

    Be conservative. Do not infer behaviour you did not or barely see. A frame that
    "looks like it could be moving" but with no actual displacement is NOT movement — call it stagnation.

    ═══ OUTPUT FORMAT ═══
    Respond ONLY with valid JSON, no text before or after:
    
    {output_format}
    """

def get_fake_answer():
    raw = {
      "observation":    "fake observation",
      "interpretation": "fake interpretation",
      "coherence":      { "score": random.randint(0, 100), "reason": "coherence reason" },
      "originality":    { "score": random.randint(0, 100), "reason": "originality reason" },
      "interest":       { "score": random.randint(0, 100), "reason": "interest reason" }
    }
    return json.dumps(raw, indent=2)

def generate_fake_vlm_batch_response(robot_ids):
    """
    Generate a fake VLM response for a list of robot IDs.
    Args:
        robot_ids: List of robot IDs (e.g., ["robot_0", "robot_1"]).
    Returns:
        A dictionary matching the structure of your example JSON.
    """
    import json

    # Build the raw JSON string
    robots_data = {}
    for robot_id in robot_ids:
        robots_data[robot_id] = {
            "observation": "fake observation",
            "interpretation": "fake interpretation",
            "coherence": {
                "score": random.randint(0, 100),
                "reason": f"Coherence reason for {robot_id}."
            },
            "originality": {
                "score": random.randint(0, 100),
                "reason": f"Originality reason for {robot_id}."
            },
            "interest": {
                "score": random.randint(0, 100),
                "reason": f"Interest reason for {robot_id}."
            }
        }

    # Convert to a JSON string (like in your example)
    raw_json = json.dumps(robots_data, indent=2)

    # Return the full response
    return raw_json

# ---------------------------------------------------------------------------
# Pre-built configurations
# ---------------------------------------------------------------------------

WALK_FORWARD = LocomotionPromptConfig(
    name    = "walk_forward",
    target  = "walk forward fast and continuously while staying upright",
    prompt  = build_locomotion_prompt("walk forward fast and continuously while staying upright"),
    weights = LocomotionScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

JUMP_HIGH = LocomotionPromptConfig(
    name    = "jump_high",
    target  = "jump as high as possible using all four legs",
    prompt  = build_locomotion_prompt("jump as high as possible using all four legs"),
    weights = LocomotionScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

CRAWL = LocomotionPromptConfig(
    name    = "crawl",
    target  = "crawl forward with the torso low to the ground",
    prompt  = build_locomotion_prompt("crawl forward with the torso low to the ground"),
    weights = LocomotionScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)


ALL_LOCOMOTION_PROMPT_CONFIGS: dict[str, LocomotionPromptConfig] = {
    cfg.name: cfg
    for cfg in (WALK_FORWARD, JUMP_HIGH, CRAWL)
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
    for cfg in ALL_LOCOMOTION_PROMPT_CONFIGS.values():
        print("\n" + "=" * 60)
        print(f"  {cfg.name}  (target = {cfg.target})")
        print(f"  weights : coherence={cfg.weights.coherence} "
              f"progress={cfg.weights.originality} interest={cfg.weights.interest}")
        print(f"\n--- Prompt preview (first 240 chars) ---")
        print(cfg.prompt[:240].strip(), "...")
