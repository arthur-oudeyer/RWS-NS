"""
gemini_prompts.py
=================
Prompt configurations for GeminiGrader.

Each GeminiPromptConfig bundles:
  - name    : short identifier used in logs and GraderOutput.
  - target  : target organism / shape (e.g. "insect", "spider").
  - prompt  : full evaluation prompt sent to Gemini.
  - weights : how coherence / originality / interest contribute to fitness.

Fitness formula (computed in GeminiGrader):
    fitness = (w_coherence * coherence
               + w_originality * originality
               + w_interest * interest)
              / (10 * (w_coherence + w_originality + w_interest))
    → always in [0, 1]

Gemini response schema expected by GeminiGrader
------------------------------------------------
    {
      "observation":  "...",
      "interpretation": "...",
      "coherence":   { "score": <int 0-10>, "reason": "..." },
      "originality": { "score": <int 0-10>, "reason": "..." },
      "interest":    { "score": <int 0-10>, "reason": "..." }
    }

Usage
-----
    from gemini_prompts import INSECT_MORPH

    grader = GeminiGrader(
        api_key      = "...",
        prompt_config = INSECT_MORPH,
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GeminiScoringWeights:
    """
    Per-dimension weights that control how the three Gemini scores
    (each 0-10) combine into a single fitness value.

    Defaults give equal weight to coherence and interest, and half
    weight to originality — the latter is a stylistic bonus rather than
    a functional requirement.
    """
    coherence:   float = 1.0
    originality: float = 0.5
    interest:    float = 1.5


@dataclass
class GeminiPromptConfig:
    """
    A named evaluation configuration for GeminiGrader.

    Attributes
    ----------
    name    : short identifier used in filenames and logs.
    target  : target organism / shape used in the prompt (e.g. "insect").
    prompt  : full text prompt sent to Gemini alongside the image.
    weights : scoring weights for fitness computation.
    """
    name:    str
    target:  str
    prompt:  str
    weights: GeminiScoringWeights = field(default_factory=GeminiScoringWeights)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_morphology_prompt(static_target: str, dynamic_target: str) -> str:
    """
    Build the standard morphology evaluation prompt for a given target.

    The prompt instructs Gemini to:
      1. Observe the robot's structure from two simultaneous views.
      2. Interpret the design relative to the target organism.
      3. Return a JSON object with observation, interpretation, and three
         scored dimensions: coherence, originality, interest.

    Parameters
    ----------
    target : target organism / shape (e.g. "insect", "spider", "crab").

    Returns
    -------
    str — the complete prompt text.
    """
    output_format = """\
    {
      "observation":    "factual description",
      "interpretation": "interpretation description and explanation",
      "coherence":      { "score": <int 0-100>, "reason": "..." },
      "originality":    { "score": <int 0-100>, "reason": "..." },
      "interest":       { "score": <int 0-100>, "reason": "..." }
    }"""

    return f"""
    ═══ CONTEXT ═══
    
    You are a strict and skeptical evaluator analyzing a static image of a MuJoCo robot morphology.
    Your job is to be PRECISE and reproduce human-like feedback on the robot's structural design.
    
    The scene:
    - 2 simultaneous views of the same morphology: left = front/side angle, right = 3/4 perspective
    - checkerboard floor and sky in background
    - Robot has a white cylindrical torso and body part, and colored limbs (red, yellow, green, purple...)
    - The robot's locomotion objective: {dynamic_target} (= dynamic target)
    - The robot's morphology objective: looking like a {static_target} (= static target)
    
    ═══ ANALYSIS ═══
    
    Step 1 — Factual observation
    Describe precisely what you see in both views:
    - Overall shape, size and position relative to the ground (use the left view to see if the legs touch the ground)
    - Number of limbs, their attachment points, articulations, segment lengths and approximate angles
    - Overall stance: is the robot upright, crouching, sprawled, collapsed?
    - Any asymmetry, specificity or unusual structural feature across the two views (shapes, connections, ..)
    
    Step 2 — Morphology interpretation
    You are evaluating structural design.
    Based on the static pose and limb layout:
    
    - Does the morphology resemble a {static_target}? Identify which features do or do not match.
    - Does the structure suggest the dynamic target ({dynamic_target}) is even physically plausible?
      Consider: center of mass, ground contact points, limb symmetry, limb articulation, joint range of motion (~90°).
    - If the morphology shows originality or promising structural traits, state what they are
      and how they could support the static target and dynamic target.
    - If the morphology is poorly designed, state specifically why
      (e.g. too few contact points, limbs too short to reach ground, torso too high).
    
    Step 3 — Score
    Score each dimension using only the static image evidence.
    Be conservative. Do not infer runtime behavior from a single frame. 
    Avoid OVERGRADING, let space for better morphology further improvement (A medium morphology cannot have more the ~60/100 in overall).
    
    SCORING RULES:
    
    coherence  — How well does the morphology match the static target ({static_target}) ?
      0–29   = no recognizable similarity to a {static_target}
      30–49  = vague resemblance, one weak matching feature
      50–69  = partial match, 1–2 clear {static_target}-like features present
      70–89  = strong resemblance, most key features identifiable
      90–100 = unmistakable likeness, structurally faithful to a {static_target}
    
    originality  — Is the structural design novel or inventive ?
      0–29   = generic, indistinguishable from a randomly generated MuJoCo morphology
      30–49  = basic organisation and minor variation on a standard body plan
      50–69  = one interesting structural choice (unusual limb count, asymmetry, etc.)
      70–89  = clearly novel design with multiple inventive features
      90–100 = highly creative, unexpected combination of structures
    
    interest  — Evolutionary potential from structural analysis alone
      0–29   = structurally implausible: cannot control movement, no viable contact points
      30–49  = poor design but not hopeless; major locomotion issues likely
      50–69  = plausible but inefficient; gait would be limited or unstable, or contains many useless limb
      70–89  = solid design; structure suggests stable and potentially efficient gait
      90–100 = excellent design; high control movement potential, well-suited to target morphology
    
    ═══ OUTPUT FORMAT ═══
    Respond ONLY with valid JSON, no text before or after:
    {output_format}
    """


# ---------------------------------------------------------------------------
# Pre-built prompt configurations
# ---------------------------------------------------------------------------

INSECT_MORPH = GeminiPromptConfig(
    name    = "insect_morph",
    target  = "insect",
    prompt  = build_morphology_prompt("insect", "move forward continuously while staying upright"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

SPIDER_MORPH = GeminiPromptConfig(
    name    = "spider_morph",
    target  = "spider",
    prompt  = build_morphology_prompt("spider", "move forward continuously while staying upright"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

CRAB_MORPH = GeminiPromptConfig(
    name    = "crab_morph",
    target  = "crab",
    prompt  = build_morphology_prompt("crab", "move forward continuously while staying upright"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

CENTIPEDE_MORPH = GeminiPromptConfig(
    name    = "centipede_morph",
    target  = "centipede",
    prompt  = build_morphology_prompt("centipede", "move forward continuously like a long insect (centipede, caterpillar..)"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

KANGAROO_MORPH = GeminiPromptConfig(
    name    = "kangaroo_morph",
    target  = "kangaroo",
    prompt  = build_morphology_prompt("kangaroo", "jumping very high with it's legs"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

ELEPHANT_MORPH = GeminiPromptConfig(
    name    = "elephant_morph",
    target  = "elephant",
    prompt  = build_morphology_prompt("elephant", "grabbing and carrying small object"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

GOALKEEPER_MORPH = GeminiPromptConfig(
    name    = "goal_keeper_morph",
    target  = "humanoid",
    prompt  = build_morphology_prompt("humanoid", "playing soccer as goal keeper"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

LAMP_MORPH = GeminiPromptConfig(
    name    = "lamp_morph",
    target  = "lamp",
    prompt  = build_morphology_prompt("lamp", "ergonomic and stylish lamp"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_GEMINI_PROMPT_CONFIGS: dict[str, GeminiPromptConfig] = {
    cfg.name: cfg
    for cfg in (INSECT_MORPH, SPIDER_MORPH, CRAB_MORPH, KANGAROO_MORPH, GOALKEEPER_MORPH, CENTIPEDE_MORPH, LAMP_MORPH, ELEPHANT_MORPH)
}


def get_gemini_prompt_set(name: str) -> GeminiPromptConfig:
    """Retrieve a GeminiPromptConfig by name.  Raises KeyError if not found."""
    if name not in ALL_GEMINI_PROMPT_CONFIGS:
        raise KeyError(
            f"Unknown Gemini prompt config '{name}'. "
            f"Available: {list(ALL_GEMINI_PROMPT_CONFIGS.keys())}"
        )
    return ALL_GEMINI_PROMPT_CONFIGS[name]


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for cfg in ALL_GEMINI_PROMPT_CONFIGS.values():
        print(f"\n{'='*60}")
        print(f"  {cfg.name}  (target={cfg.target})")
        print(f"  weights: coherence={cfg.weights.coherence}  "
              f"originality={cfg.weights.originality}  "
              f"interest={cfg.weights.interest}")
        print(f"\n--- Prompt preview (first 200 chars) ---")
        print(cfg.prompt[:200].strip(), "...")