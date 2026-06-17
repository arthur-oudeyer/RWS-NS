"""
descriptor.py
=============
Behavioural feature space for the MAP-Elites controller study.

A `DescriptorConfig` defines the 2-D feature map MAP-Elites diversifies over.
Each axis (`DescriptorDim`) is a behavioural property the VLM scores on a
0–100 scale; the score is then placed in a bucket via `bins` (edges on the
0–100 score) and named via `bin_labels`.

Flow
----
    vlm_grader  → asks the VLM for an int 0–100 + reason per dim
                → ControllerResult.descriptors = {dim_name: score 0–100}
    MapEliteArchive(feature_dims, feature_bins, dim_labels)
                → bins each score into a grid cell (archive.py:_bin)

Score convention
----------------
Descriptor scores stay on the **0–100** scale (the same range `bins` are
expressed in). This is intentional and differs from coherence/originality/
potential, which the grader divides by 100 into [0, 1] for the fitness mean.

Usage
-----
    from descriptor import get_descriptor_config
    dcfg = get_descriptor_config("coordination_amplitude")
    feature_dims = dcfg.feature_dims                     # ["coordination", "amplitude"]
    feature_bins = {it.name: it.bins for it in dcfg.items}
    dim_labels   = {it.name: it.bin_labels for it in dcfg.items}

To add a new feature space, append a `DescriptorConfig` to `DESCRIPTOR_CONFIGS`.
Keep it 2-D (exactly two `items`) with 3–6 buckets per axis.

Debug
-----
Run this file directly to print every config and validate bin/label shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DescriptorDim:
    """One axis of the MAP-Elites feature map.

    Attributes
    ----------
    name       : grid-axis name AND the key the VLM returns in `descriptors`.
    description: what the VLM should assess (injected verbatim into the prompt).
    low_desc   : what a score near 0 means.
    high_desc  : what a score near 100 means.
    bins       : ascending edges on the 0–100 score. N edges → N+1 buckets.
    bin_labels : human-readable name per bucket. len == len(bins) + 1.
    """
    name:        str
    description: str
    low_desc:    str
    high_desc:   str
    bins:        list = field(default_factory=list)
    bin_labels:  list = field(default_factory=list)

    def __post_init__(self):
        if len(self.bin_labels) != len(self.bins) + 1:
            raise ValueError(
                f"DescriptorDim '{self.name}': bin_labels must have "
                f"len(bins)+1 = {len(self.bins) + 1} entries, got {len(self.bin_labels)}."
            )

    @property
    def n_buckets(self) -> int:
        return len(self.bins) + 1


@dataclass
class DescriptorConfig:
    """A named 2-D behavioural feature space."""
    name:  str
    items: list  # exactly two DescriptorDim

    def __post_init__(self):
        if len(self.items) != 2:
            raise ValueError(
                f"DescriptorConfig '{self.name}' must have exactly 2 dims "
                f"(2-D map), got {len(self.items)}."
            )

    @property
    def feature_dims(self) -> list:
        return [it.name for it in self.items]


# ---------------------------------------------------------------------------
# Built-in configurations
# ---------------------------------------------------------------------------

COORDINATION_AMPLITUDE = DescriptorConfig(
    name  = "coordination_amplitude",
    items = [
        DescriptorDim(
            name        = "coordination",
            description = "How coordinated and synchronised the limb movements are "
                          "across the whole rollout (do the legs work together in a "
                          "consistent, organised pattern, or independently / randomly?).",
            low_desc    = "uncoordinated — limbs thrash independently or randomly, "
                          "no organisation between them",
            high_desc   = "fully coordinated — ALL limbs move together in a clear, "
                          "synchronised, organised pattern",
            bins        = [33, 66],
            bin_labels  = ["uncoordinated", "partial", "coordinated"],
        ),
        DescriptorDim(
            name        = "amplitude",
            description = "The overall amplitude / energy of the motion: how large and "
                          "wide-ranging the body and limb movements are over the rollout.",
            low_desc    = "minimal — tiny, subtle, almost-static movements",
            high_desc   = "extreme — very large, wide, energetic full-range movements",
            bins        = [25, 50, 75],
            bin_labels  = ["minimal", "small", "large", "extreme"],
        ),
    ],
)

SIMILITUDE_FEELINGS = DescriptorConfig(
    name  = "similitude_feeling",
    items = [
        DescriptorDim(
            name        = "similitude",
            description = "Does it move like something real ? animal / machine / nothing",
            low_desc    = "0-33 -> Animal, 34-66 -> Machine, ",
            high_desc   = "67-100 -> nothing special",
            bins        = [33, 66],
            bin_labels  = ["animal", "machine", "nothing"],
        ),
        DescriptorDim(
            name        = "feeling",
            description = "What feelings comes from this behavior ? sadness / neutral / happyness",
            low_desc    = "sadness",
            high_desc   = "happyness",
            bins        = [33, 66],
            bin_labels  = ["sadness", "neutral", "happyness"],
        ),
    ],
)

ENERGY_ABSTRACTION = DescriptorConfig(
    name  = "energy_abstraction",
    items = [
        DescriptorDim(
            name        = "energy",
            description = "How energetic the limb movements are ?",
            low_desc    = "almost static, slow and smooth movements",
            high_desc   = "very excited, dynamic, energetic and intensive movements",
            bins        = [20, 40, 60, 75],
            bin_labels  = ["almost static", "slow", "medium", "dynamic", "extremely energetic"],
        ),
        DescriptorDim(
            name        = "abstraction",
            description = "How abstract is this dance ?",
            low_desc    = "Very expressive and interpretable, can easily be described",
            high_desc   = "Low interpretability, high creativity, hard to put words describing the movements",
            bins        = [33, 66],
            bin_labels  = ["abstract", "medium", "down-to-earth"],
        ),
    ],
)

DESCRIPTOR_CONFIGS: dict = {
    c.name: c for c in (COORDINATION_AMPLITUDE, SIMILITUDE_FEELINGS, ENERGY_ABSTRACTION)
}


def get_descriptor_config(name: str) -> DescriptorConfig:
    """Return the named descriptor config, or raise KeyError listing the options."""
    if name not in DESCRIPTOR_CONFIGS:
        raise KeyError(
            f"Unknown descriptor config '{name}'. "
            f"Available: {list(DESCRIPTOR_CONFIGS.keys())}"
        )
    return DESCRIPTOR_CONFIGS[name]


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  descriptor.py — debug mode")
    print("=" * 60)

    for cfg in DESCRIPTOR_CONFIGS.values():
        print(f"\nDescriptorConfig: {cfg.name}   (dims={cfg.feature_dims})")
        for it in cfg.items:
            assert len(it.bin_labels) == len(it.bins) + 1
            print(f"  • {it.name}  ({it.n_buckets} buckets)")
            print(f"      0   = {it.low_desc}")
            print(f"      100 = {it.high_desc}")
            print(f"      bins   : {it.bins}")
            print(f"      labels : {it.bin_labels}")

    # round-trip the lookup
    assert get_descriptor_config("coordination_amplitude") is COORDINATION_AMPLITUDE
    try:
        get_descriptor_config("does_not_exist")
    except KeyError as e:
        print(f"\n  unknown-name guard OK → {e}")

    print("\nAll descriptor.py checks passed.")
