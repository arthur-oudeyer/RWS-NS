"""
descriptor.py
=============
VLM-based feature descriptor configurations for Map-Elites.

Each DescriptorItem defines one feature the VLM is asked to rate (0-10).
DescriptorConfig bundles several items and names the two that serve as
Map-Elites feature axes.

The GeminiGrader appends the descriptor prompt section to its main scoring
prompt so that a single API call returns both fitness scores and feature
descriptor values.  The values are stored in MorphologyResult.descriptors
alongside the structural descriptors from morphology.encoding(), and
MapEliteArchive uses them to place each individual in the 2-D grid.

Why VLM over pure math?
-----------------------
Mathematical descriptors (n_legs, symmetry_score) only capture simple
structure.  When morphologies have multiple body parts, branched limbs,
and varying torso orientations, a visual assessment captures "bilateral
symmetry", "body complexity", and "limb reach" far more faithfully than
angle-gap statistics.

Usage
-----
    from descriptor import INSECT_DESCRIPTORS

    grader = GeminiGrader(
        api_key           = "...",
        prompt_config     = INSECT_MORPH,
        descriptor_config = INSECT_DESCRIPTORS,
    )

    # In config:
    cfg.descriptor_config_name  = "insect_descriptors"
    cfg.map_elite_feature_dims  = ["n_legs", "bilateral_symmetry"]
    cfg.map_elite_feature_bins  = {"bilateral_symmetry": [4.0, 7.0]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DescriptorItem:
    """
    One assessed feature dimension.

    Attributes
    ----------
    name       : key used in the JSON response and in descriptor dicts.
    question   : instruction shown verbatim to the VLM evaluator.
    bins       : bin edges for Map-Elites discretization.
                 Empty list → raw integer used as-is (for discrete counts).
                 E.g. [4.0, 7.0] produces 3 buckets: [0,4), [4,7), [7,10].
    bin_labels : human-readable label per bucket (optional, for display).
    """
    name:       str
    question:   str
    bins:       list[float] = field(default_factory=list)
    bin_labels: list[str]   = field(default_factory=list)


@dataclass
class DescriptorConfig:
    """
    A named set of feature descriptors for VLM-based assessment.

    Attributes
    ----------
    name         : identifier used in config files and logs.
    items        : all descriptors the VLM is asked to rate.
    feature_dims : exactly 2 item names that serve as Map-Elites axes.
                   These names must appear in items.
    """
    name:         str
    items:        list[DescriptorItem]
    feature_dims: list[str]

    def get_item(self, name: str) -> Optional[DescriptorItem]:
        for item in self.items:
            if item.name == name:
                return item
        return None


# ---------------------------------------------------------------------------
# Prompt section builder
# ---------------------------------------------------------------------------

def build_descriptor_prompt_section(config: DescriptorConfig) -> str:
    """
    Build the descriptor-evaluation block appended to a Gemini prompt.

    Instructs the model to add a top-level "descriptors" key to the JSON
    it is already going to produce, with one integer (0-10) per item.
    """
    lines = [
        "\n\n    ═══ FEATURE DESCRIPTORS ═══",
        "",
        "    These structural features are used to CLASSIFY the morphology along",
        "    diversity axes in the evolutionary archive (MAP-Elites).",
        "    Each score places the robot in a distinct region of the design space,",
        "    so rate them accurately and independently of the fitness scores above.",
        "    A robot can score high on fitness but low on a descriptor (or vice-versa).",
        "",
        f"    Rate each of the following {len(config.items)} features (integer 0-10):",
        "    Scoring guide: 0 = not present / impossible to assess from this image,",
        "                   5 = moderately present, medium amount",
        "                   10 = strongly and unambiguously present / high amount",
        "",
    ]
    for item in config.items:
        lines.append(f"    {item.name}:")
        lines.append(f"      {item.question}")

    lines += ["", "    Include these ratings as a top-level \"descriptors\" key in your output JSON."]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Pre-built descriptor configurations
# ---------------------------------------------------------------------------

GENERIC_DESCRIPTORS = DescriptorConfig(
    name  = "generic_descriptors",
    feature_dims = ["bilateral_symmetry", "limb_count_category"],
    items = [
        DescriptorItem(
            name       = "bilateral_symmetry",
            question   = (
                "How bilaterally symmetric is the robot's overall structure? "
                "Consider both root limbs AND any branched limbs and body parts. "
                "0 = completely asymmetric, 10 = perfect mirror on any axis."
            ),
            bins       = [4.0, 7.0],
            bin_labels = ["asymmetric", "partial", "symmetric"],
        ),
        DescriptorItem(
            name       = "limb_count_category",
            question   = (
                "How many total limbs (counting both root and branched limbs) does the robot have? "
                "0-2 = very few (1-2 limbs), 3-4 = few (3-4), 5-6 = moderate (5-6), "
                "7-8 = many (7-8), 9-10 = very many (9+)."
            ),
            bins       = [2.5, 5.0, 7.5],
            bin_labels = ["very few limbs", "few limbs", "different limbs", "many limbs"],
        ),
    ],
)

LAMP_DESCRIPTORS = DescriptorConfig(
    name  = "lamp_descriptors",
    feature_dims = ["design", "adjustability"],
    items = [
        DescriptorItem(
            name       = "design",
            question   = (
                "How to classify the design of this lamp ? "
                "Consider the base, the arm and the lampshade. "
                "0-1 = small bedside lamp on stand, 2-3 = small furniture/decorating lamp, "
                "4-5 = large furniture/decorating lamp, 6-7 = reading adjustable lamp, "
                "8 = flashlight, 9 = experimental design, 10 = huge powerful spotlight"
            ),
            bins       = [1.5, 3.5, 5.5, 7.5, 8.5, 9.5],
            bin_labels = ["bedside", "small deco", "large deco", "reading lamp", "flashlight", "experimental", "spotlight"],
        ),
        DescriptorItem(
            name       = "adjustability",
            question   = (
                "How much can the robot-lamp adjust its position or light orientation? "
                "0-2 = totally fixed, no articulation, 3-5 = basic adjustability (poor articulation), "
                "6-7 = clear articulation that allows adjustment, "
                "8-10 = good or clever articulation to adjust the lamp"
            ),
            bins       = [2.5, 5.5, 7.5],
            bin_labels = ["fixed", "basic adjustment", "good mechanism", "great mechanism"],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_DESCRIPTOR_CONFIGS: dict[str, DescriptorConfig] = {
    cfg.name: cfg
    for cfg in (GENERIC_DESCRIPTORS, LAMP_DESCRIPTORS,)
}


def get_descriptor_config(name: str) -> DescriptorConfig:
    """Retrieve a DescriptorConfig by name.  Raises KeyError if not found."""
    if name not in ALL_DESCRIPTOR_CONFIGS:
        raise KeyError(
            f"Unknown descriptor config '{name}'. "
            f"Available: {list(ALL_DESCRIPTOR_CONFIGS.keys())}"
        )
    return ALL_DESCRIPTOR_CONFIGS[name]


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for cfg in ALL_DESCRIPTOR_CONFIGS.values():
        print(f"\n{'='*60}")
        print(f"  {cfg.name}")
        print(f"  feature_dims: {cfg.feature_dims}")
        for item in cfg.items:
            print(f"  - {item.name}  bins={item.bins}  labels={item.bin_labels}")
            print(f"      {item.question[:100]}")
        print("\n  Prompt section preview:")
        print(build_descriptor_prompt_section(cfg)[:500])
