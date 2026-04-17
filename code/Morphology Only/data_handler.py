"""
data_handler.py
===============
Ties rendering and grading together into a single evaluation step,
and defines MorphologyResult — the unit of data stored in the archive.

MorphologyResult
----------------
Everything about one evaluated individual:
  - its morphology (structure)
  - its fitness score (from CLIP)
  - its raw per-prompt scores (for full traceability)
  - its structural descriptors (from morphology.encoding(), used by MapElite)
  - metadata: generation, individual id, which grader/prompts were used

evaluate()
----------
Given a morphology, a renderer and a grader, produces a MorphologyResult:
  1. Render the morphology → PIL Image
  2. Grade the image → GraderOutput
  3. Compute structural descriptors → encoding()
  4. Wrap everything into a MorphologyResult and return it

Serialisation
-------------
result_to_dict / dict_to_result — plain dicts safe for JSON.
The morphology is stored via morphology_to_dict.

Debug
-----
Run this file to test the full evaluation pipeline end-to-end
(requires mujoco and, if scoring, CLIP).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from morphology import RobotMorphology, morphology_to_dict, dict_to_morphology


# ---------------------------------------------------------------------------
# MorphologyResult
# ---------------------------------------------------------------------------

@dataclass
class MorphologyResult:
    """
    Full record of one evaluated morphology.

    Attributes
    ----------
    generation    : which generation this individual was created in.
    individual_id : unique integer across the entire run (monotonically
                    increasing, assigned by the experiment loop).
    morphology    : the robot's body descriptor.
    fitness       : scalar used for selection (higher = better).
    raw_scores    : every prompt's CLIP score, keyed by prompt text.
    descriptors   : structural descriptors from morphology.encoding().
                    Used by MapElite as feature dimensions.
    grader_method : "cosine" or "softmax".
    prompt_set    : name of the PromptSet that was applied.
    render_path   : path to the saved PNG render, or None.
    """
    generation:    int
    individual_id: int
    morphology:    RobotMorphology
    fitness:       float
    raw_scores:    dict[str, float]
    descriptors:   dict[str, float]
    grader_method: str
    prompt_set:    str
    render_path:   Optional[str] = None
    grader_extra:  dict = dataclass_field(default_factory=dict)
    # grader_extra stores backend-specific metadata for analysis:
    # - CLIP  : {} (empty)
    # - Gemini: {"observation", "interpretation",
    #             "coherence_reason", "originality_reason", "interest_reason"}

    def __str__(self) -> str:
        enc = self.descriptors
        return (
            f"[gen={self.generation:>3}  id={self.individual_id:>5}]  "
            f"fitness={self.fitness:+.5f}  "
            f"legs={enc.get('n_legs', '?')}  "
            f"sym={enc.get('symmetry_score', 0):.3f}  "
            f"prompts={self.prompt_set}"
        )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def result_to_dict(r: MorphologyResult) -> dict:
    """Convert a MorphologyResult to a JSON-safe plain dict."""
    return {
        "generation":    r.generation,
        "individual_id": r.individual_id,
        "morphology":    morphology_to_dict(r.morphology),
        "fitness":       r.fitness,
        "raw_scores":    r.raw_scores,
        "descriptors":   r.descriptors,
        "grader_method": r.grader_method,
        "prompt_set":    r.prompt_set,
        "render_path":   r.render_path,
        "grader_extra":  r.grader_extra,
    }


def dict_to_result(d: dict) -> MorphologyResult:
    return MorphologyResult(
        generation    = d["generation"],
        individual_id = d["individual_id"],
        morphology    = dict_to_morphology(d["morphology"]),
        fitness       = d["fitness"],
        raw_scores    = d["raw_scores"],
        descriptors   = d["descriptors"],
        grader_method = d["grader_method"],
        prompt_set    = d["prompt_set"],
        render_path   = d.get("render_path"),
        grader_extra  = d.get("grader_extra", {}),
    )


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def evaluate(
    morph:            RobotMorphology,
    renderer,         # MorphologyRenderer from rendering.py
    grader,           # MorphologyGrader from grader.py
    generation:       int,
    individual_id:    int,
    render_save_path: Optional[str] = None,
    debug:            bool = False,
) -> MorphologyResult:
    """
    Evaluate one morphology and return a MorphologyResult.

    Steps
    -----
    1. Render the morphology to a PIL Image (using rest_angles for pose).
    2. Optionally save the image to render_save_path.
    3. Grade the image with CLIP → GraderOutput.
    4. Compute structural descriptors via morphology.encoding().
    5. Return a MorphologyResult with all data.

    Parameters
    ----------
    morph            : morphology to evaluate.
    renderer         : MorphologyRenderer instance (already configured).
    grader           : MorphologyGrader instance (CLIP already loaded).
    generation       : current generation index.
    individual_id    : unique id for this individual in the run.
    render_save_path : if given, the rendered image is saved here.
    debug            : passed through to renderer and grader for verbose output.
    """
    # Step 1–2: render
    image = renderer.render(morph, save_path=render_save_path, debug=debug)

    # Step 3: grade
    grader_output = grader.score(image, debug=debug)

    # Step 4: structural descriptors (no simulation, pure structure)
    descriptors = morph.encoding()

    return MorphologyResult(
        generation    = generation,
        individual_id = individual_id,
        morphology    = morph,
        fitness       = grader_output.fitness,
        raw_scores    = grader_output.raw_scores,
        descriptors   = descriptors,
        grader_method = grader_output.method,
        prompt_set    = grader_output.prompt_set,
        render_path   = render_save_path,
        grader_extra  = grader_output.extra,
    )


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from morphology import QUADRIPOD, TRIPOD, HEXAPOD, NewMorph
    from rendering  import MorphologyRenderer, RenderConfig
    from grader     import CLIPGrader
    from prompts    import SPIDER_BODY

    print("=" * 60)
    print("  data_handler.py — debug mode")
    print("=" * 60)

    # --- 1. Serialisation round-trip (no CLIP needed) ---
    print("\n[1] Serialisation round-trip\n")
    dummy_result = MorphologyResult(
        generation    = 0,
        individual_id = 42,
        morphology    = QUADRIPOD,
        fitness       = 0.314,
        raw_scores    = {"a spider robot": 0.8, "a fallen robot": 0.2},
        descriptors   = QUADRIPOD.encoding(),
        grader_method = "cosine",
        prompt_set    = "spider_body",
        render_path   = "renders/gen0_id42.png",
    )
    d    = result_to_dict(dummy_result)
    back = dict_to_result(d)
    assert back.fitness       == dummy_result.fitness
    assert back.individual_id == dummy_result.individual_id
    assert back.morphology.name == dummy_result.morphology.name
    print(f"  {dummy_result}")
    print(f"  Serialisation: OK")

    # --- 2. Full evaluation (rendering + CLIP) ---
    print("\n[2] Full evaluation pipeline (requires mujoco + CLIP)\n")
    try:
        render_cfg = RenderConfig(width=256, height=256, debug=False)
        renderer   = MorphologyRenderer(render_cfg)

        grader = CLIPGrader(
            prompt_set     = SPIDER_BODY,
            scoring_method = "cosine",
            debug          = False,
        )

        print("  Evaluating TRIPOD, QUADRIPOD, HEXAPOD ...\n")
        results = []
        for i, morph in enumerate((TRIPOD, QUADRIPOD, HEXAPOD)):
            r = evaluate(
                morph         = morph,
                renderer      = renderer,
                grader        = grader,
                generation    = 0,
                individual_id = i,
                debug         = False,
            )
            results.append(r)
            print(f"  {r}")

        best = max(results, key=lambda r: r.fitness)
        print(f"\n  Best: {best.morphology.name}  fitness={best.fitness:+.5f}")

        renderer.close()

    except ImportError as e:
        print(f"  Skipped (missing dependency): {e}")

    print("\nAll data_handler checks passed.")
