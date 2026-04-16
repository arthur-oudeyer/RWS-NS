"""
prompts.py
==========
CLIP prompt sets for morphology evaluation.

A PromptSet groups positive and negative WeightedPrompts under a named
objective.  The grader uses these to compute a fitness score.

Scoring logic (in grader.py):
    fitness = Σ (pos_weight * pos_score) - Σ (neg_weight * neg_score)

With cosine / multi-label scoring each prompt is scored independently,
so positive prompts do not compete with each other — a robot image can
score high on "many legs" AND "symmetric body" simultaneously.

How to write good prompts for CLIP
-----------------------------------
- Be concrete about visual appearance ("a white cylinder connected to
  colored rods") rather than abstract ("a good robot").
- Describe what you actually see in the rendered image — MuJoCo outputs
  a 3D scene with a gray floor, capsule-shaped limbs, a cylindrical torso.
- Use scene-level cues ("a 3D simulation screenshot of a robot with...").
- Positive prompts describe what you want to REWARD.
- Negative prompts describe what you want to PENALISE.
- Weights default to 1.0; increase a weight to make one prompt dominate.

Usage
-----
    from prompts import SPIDER_BODY, MANY_LEGS, COMPACT_STABLE

    # In CLIPGrader:
    grader = CLIPGrader(config, prompt_set=SPIDER_BODY)

    # Print contents:
    SPIDER_BODY.describe()
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WeightedPrompt:
    """One text prompt and its contribution weight in the fitness formula."""
    text:   str
    weight: float = 1.0


@dataclass
class PromptSet:
    """
    A named collection of positive and negative prompts for one objective.

    Attributes
    ----------
    name        : short identifier used in filenames and logs.
    description : human-readable explanation of the design objective.
    positive    : prompts that increase fitness when matched.
    negative    : prompts that decrease fitness when matched.
    """
    name:        str
    description: str
    positive:    list[WeightedPrompt]
    negative:    list[WeightedPrompt] = field(default_factory=list)

    def all_texts(self) -> list[str]:
        """Flat list of all texts (positive first, then negative)."""
        return [p.text for p in self.positive] + [p.text for p in self.negative]

    def describe(self) -> None:
        """Print the prompt set in a readable format."""
        print(f"\nPromptSet: {self.name}")
        print(f"  Objective: {self.description}")
        print(f"  Positive ({len(self.positive)}):")
        for p in self.positive:
            print(f"    [{p.weight:+.1f}]  {p.text}")
        if self.negative:
            print(f"  Negative ({len(self.negative)}):")
            for p in self.negative:
                print(f"    [-{p.weight:.1f}]  {p.text}")


# ---------------------------------------------------------------------------
# Prompt sets
# ---------------------------------------------------------------------------

SPIDER_BODY = PromptSet(
    name        = "spider_body",
    description = "Reward spider-like morphologies: many legs spread radially, "
                  "compact torso, limbs pointing outward.",
    positive = [
        WeightedPrompt("a 3D simulation render of a spider-like robot with many legs spread outward",    weight=2.0),
        WeightedPrompt("a symmetric robot body with multiple limbs radiating from a central cylinder",   weight=1.5),
        WeightedPrompt("a robot with 6 or more legs distributed evenly around its body",                 weight=1.5),
        WeightedPrompt("a MuJoCo simulation screenshot showing a multi-legged creature on a flat floor", weight=1.0),
        WeightedPrompt("a robot torso with colored capsule-shaped limbs arranged radially",              weight=1.0),
    ],
    negative = [
        WeightedPrompt("a robot with only 2 or 3 legs",       weight=1.0),
        WeightedPrompt("a biped or humanoid robot shape",      weight=1.5),
        WeightedPrompt("a robot fallen over or lying flat",    weight=1.0),
    ],
)

COMPACT_STABLE = PromptSet(
    name        = "compact_stable",
    description = "Reward compact, low-profile morphologies that look stable "
                  "and grounded. Short legs, wide stance.",
    positive = [
        WeightedPrompt("a compact robot close to the ground with short sturdy legs",                    weight=2.0),
        WeightedPrompt("a low-profile robot body with legs that spread wide for stability",             weight=1.5),
        WeightedPrompt("a 3D simulation render of a stable robot with a wide base of support",         weight=1.0),
        WeightedPrompt("a robot with legs bent outward keeping its torso near the floor",              weight=1.0),
    ],
    negative = [
        WeightedPrompt("a robot with very long thin legs",              weight=1.0),
        WeightedPrompt("a tall unstable-looking robot",                 weight=1.5),
        WeightedPrompt("a robot with legs pointing straight downward",  weight=1.0),
    ],
)


MANY_LEGS = PromptSet(
    name        = "many_legs",
    description = "Reward morphologies with a high number of limbs regardless "
                  "of their arrangement.",
    positive = [
        WeightedPrompt("a robot with many legs, more than six limbs",                                   weight=2.0),
        WeightedPrompt("a multi-legged simulation robot resembling a centipede or millipede",           weight=1.5),
        WeightedPrompt("a 3D render of a robot body covered with numerous colored rod-like limbs",     weight=1.5),
        WeightedPrompt("a densely-limbed robotic creature in a physics simulation",                    weight=1.0),
    ],
    negative = [
        WeightedPrompt("a robot with four legs or fewer",  weight=2.0),
        WeightedPrompt("a bipedal or quadrupedal robot",   weight=1.5),
    ],
)

# ---------------------------------------------------------------------------
# Registry — all available prompt sets in one dict
# ---------------------------------------------------------------------------

ALL_PROMPT_SETS: dict[str, PromptSet] = {
    ps.name: ps
    for ps in (SPIDER_BODY, COMPACT_STABLE, MANY_LEGS)
}


def get_prompt_set(name: str) -> PromptSet:
    """Retrieve a prompt set by name.  Raises KeyError if not found."""
    if name not in ALL_PROMPT_SETS:
        raise KeyError(
            f"Unknown prompt set '{name}'. "
            f"Available: {list(ALL_PROMPT_SETS.keys())}"
        )
    return ALL_PROMPT_SETS[name]


# ---------------------------------------------------------------------------
# Debug — run directly to inspect all prompt sets
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  prompts.py — available prompt sets")
    print("=" * 60)

    SPIDER_BODY.describe()

    print(f"\nTotal prompt sets : {len(ALL_PROMPT_SETS)}")
    total_prompts = sum(
        len(ps.positive) + len(ps.negative) for ps in ALL_PROMPT_SETS.values()
    )
    print(f"Total prompts     : {total_prompts}")
