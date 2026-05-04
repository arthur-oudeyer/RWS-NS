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
      "observation":    "key-time steps factual description (timestamps, floor-tile reference, posture of each limb, ...)",
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
      is a grassy green checkerboard (use floor tiles as a position reference grid)
    - The robot has white ellipsoidal body parts and colored legs
    - Background is plain blue (no distractors)
    
    Target behaviour : {target_behaviour}

    ═══ ANALYSIS ═══

    Step 1 — Frame-by-frame factual observation
    Be specific. Note the moment of any event, the position of the limb, de displacement of the robot between frames.
    example :
    - Frame 1 (0.0 s): the robot is standing upright, 3 limbs are touching the floor and one (the blue one) is pointing toward the sky.
    - ~2.0 s: the robot have just started a movement toward the left, it's limbs are moving in a periodic but nervous pattern but slip a lot on the ground. The robot stays balanced but tilts a lot in some positions.
    - ~2.5 s: the robot have moved by around 3 ground tiles from its starting point toward the left, his gait looks like he is nervous due to hesitating but brutal movements. The limbs are all touching the ground in a stable position. The torso is slightly tilted but the overall balance is good.
    - Final frame (~5.0 s): the robot has completely crashed on the ground, he has 3 out of 4 limbs toward the sky and is laying on the side. He is not moving anymore.
    - Key events: fall at 4s s, stagnation at 4s s, gait change at 2s, …

    Step 2 — Behavioural interpretation
    - Did the robot make consistent consistent action relevant with the target behavior ?
    - Was the gait coherent (periodic, balanced, repeatable) or random ? What was the type of the gait (smooth, energetic, nervous, wide, brutal, efficient, small, homogenous, ...) ?
    - Is there anything novel or interesting about the motion pattern even if the robot did not perform well for the target behavior ? (ex: is a limb doing a movement with great potential ?)

    Step 3 — Conservative scoring (each dimension 0–100)

    coherence — Is the gait relevant for the target behavior ?
      0–29   = chaotic thrashing, immediate collapse, fully static or no recognisable pattern
      30–49  = unstable, sporadic; one or two coherent moments only that have a link to the target
      50–69  = partial coherence; clear periodic pattern or specific movement but with wobble or stalls. The target can be identified.
      70–89  = coherent, repeatable gait or target well reached ; minor instabilities only. The intention toward target is obvious.
      90–100 = clean, stable, periodic locomotion throughout, the target is perfectly depict through this video.

    originality — Did the robot achieve something toward the behavioral target in an original way ?
      0–29   = no movement or movement very basic with no progress toward the target
      30–49  = one basic movement, not very original
      50–69  = novel movements that provide new ability for the robot
      70–89  = clear and unexpected movement that somehow help the robot progress toward the target behavior
      90–100 = very unexpected but very efficient way to reach the behavior wanted

    interest — Is the gait pattern interesting, biologically plausible and leads to a real evolutionary potential ?
      0–29   = uninteresting (random, fallen) or obviously broken
      30–49  = generic, predictable motion with no notable features
      50–69  = one notable element (unusual gait phase, rhythm, recovery) that have potential
      70–89  = clearly interesting motion: reminiscent of an animal gait,
               coordinated pattern, or creative body usage to reached the target. There is a great potential.
      90–100 = highly interesting; novel and biologically convincing locomotion, great abilities and great potential for further evolution.

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
        print()
        print(cfg.prompt.strip())
