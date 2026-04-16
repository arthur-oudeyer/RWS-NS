"""
grader.py
=========
CLIP-based morphology fitness grader.

Architecture
------------
MorphologyGrader (abstract base)
    └─ CLIPGrader   — current implementation using OpenCLIP

The abstract base makes it easy to swap in a different backend later
(e.g. a Gemini or GPT-4V grader) without changing any calling code.

Scoring methods
---------------
"cosine"  (default — multi-label):
    Each prompt is scored independently via its cosine similarity to the
    image embedding.  Scores are in [-1, 1].  Positive prompts do not
    compete with each other: a robot can score high on "many legs" AND
    "symmetric body" simultaneously.  This is the right choice when
    prompts measure orthogonal qualities.

"softmax"  (relative):
    Scores are a probability distribution over all prompts (sum = 1).
    Use this when you want prompts to compete, i.e. to measure which
    description fits best.

Fitness formula (both methods):
    fitness = mean(Σ (pos_weight * pos_score)) − mean(Σ (neg_weight * neg_score))

GraderOutput
------------
    fitness      : float — scalar used by the evolution loop for selection.
    raw_scores   : dict[text → float] — every prompt's score, for analysis.
    method       : "cosine" or "softmax"
    prompt_set   : name of the PromptSet used.

Debug mode
----------
    Set debug=True in CLIPGrader or pass debug=True to score() to:
      - print per-prompt scores sorted by value
      - print the final fitness

Usage
-----
    from grader import CLIPGrader
    from prompts import SPIDER_BODY

    grader = CLIPGrader(
        model_name   = "ViT-B-32",
        pretrained   = "openai",
        cache_dir    = "/Volumes/T7_AO/clip-models",
        prompt_set   = SPIDER_BODY,
        scoring_method = "cosine",
        debug        = False,
    )

    image   = ...          # PIL.Image
    result  = grader.score(image)
    print(result.fitness)
    print(result.raw_scores)
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — only loaded when CLIPGrader is instantiated
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import open_clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from prompts import PromptSet


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class GraderOutput:
    """
    Result of scoring one rendered morphology image.

    Attributes
    ----------
    fitness      : weighted combination of positive minus negative scores.
                   This is the number the evolution loop uses for selection.
    raw_scores   : dict mapping each prompt text to its score.
                   Stored for full traceability and analysis.
    method       : "cosine" or "softmax" — how raw_scores were computed.
    prompt_set   : name of the PromptSet that was applied.
    """
    fitness:    float
    raw_scores: dict[str, float]
    method:     str
    prompt_set: str

    def __str__(self) -> str:
        lines = [f"fitness={self.fitness:.4f}  method={self.method}  set={self.prompt_set}"]
        for text, score in sorted(self.raw_scores.items(), key=lambda x: -x[1]):
            lines.append(f"  {score:+.4f}  {text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MorphologyGrader(ABC):
    """
    Abstract base for morphology graders.

    Subclasses must implement score().  The prompt_set and debug flag
    are stored on the instance so callers do not need to pass them every time.
    """

    def __init__(self, prompt_set: PromptSet, debug: bool = False):
        self.prompt_set = prompt_set
        self.debug      = debug

    @abstractmethod
    def score(
        self,
        image: "PILImage.Image",
        debug: Optional[bool] = None,
    ) -> GraderOutput:
        """
        Score one rendered morphology image.

        Parameters
        ----------
        image : PIL Image in RGB mode (output of rendering.py).
        debug : overrides self.debug for this call if not None.

        Returns
        -------
        GraderOutput with fitness, raw_scores, method, prompt_set name.
        """
        ...


# ---------------------------------------------------------------------------
# CLIPGrader
# ---------------------------------------------------------------------------

class CLIPGrader(MorphologyGrader):
    """
    Scores morphology images using OpenCLIP.

    The model is loaded once in __init__.  score() is stateless and
    can be called for many images without reloading.

    Parameters
    ----------
    model_name     : OpenCLIP model architecture, e.g. "ViT-B-32".
    pretrained     : pretrained weights, e.g. "openai".
    cache_dir      : directory where model weights are cached.
    prompt_set     : PromptSet to use for scoring.
    scoring_method : "cosine" (multi-label, default) or "softmax" (relative).
    device         : "mps" / "cuda" / "cpu".  Auto-detected if None.
    debug          : global debug flag.
    """

    def __init__(
        self,
        model_name:     str       = "ViT-B-32",
        pretrained:     str       = "openai",
        cache_dir:      str       = "/Volumes/T7_AO/clip-models",
        prompt_set:     PromptSet = None,
        scoring_method: str       = "cosine",
        device:         Optional[str] = None,
        debug:          bool      = False,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for CLIPGrader.")
        if not _CLIP_AVAILABLE:
            raise ImportError("open_clip is required for CLIPGrader.")
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow (PIL) is required for CLIPGrader.")
        if prompt_set is None:
            raise ValueError("A PromptSet must be provided to CLIPGrader.")
        if scoring_method not in ("cosine", "softmax"):
            raise ValueError(f"scoring_method must be 'cosine' or 'softmax', got '{scoring_method}'.")

        super().__init__(prompt_set=prompt_set, debug=debug)
        self.model_name     = model_name
        self.scoring_method = scoring_method

        # Device selection
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Load model
        import os
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        print(f"[grader] Loading CLIP {model_name} ({pretrained}) on {device}...")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained  = pretrained,
            cache_dir   = cache_dir,
            quick_gelu  = True,
        )
        self._model.eval().to(device)
        self._tokenizer = open_clip.get_tokenizer(model_name)
        print(f"[grader] CLIP ready.")

        # Pre-tokenise all prompts once (they don't change between calls)
        self._all_texts      = prompt_set.all_texts()
        self._n_positive     = len(prompt_set.positive)
        self._text_tokens    = self._tokenizer(self._all_texts).to(device)
        self._pos_weights    = [p.weight for p in prompt_set.positive]
        self._neg_weights    = [p.weight for p in prompt_set.negative]

        if debug:
            print(f"[grader] Prompt set: {prompt_set.name}  "
                  f"({len(prompt_set.positive)} positive, "
                  f"{len(prompt_set.negative)} negative)  "
                  f"scoring={scoring_method}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        image: "PILImage.Image",
        debug: Optional[bool] = None,
    ) -> GraderOutput:
        """
        Score one PIL Image and return a GraderOutput.

        The image must be in RGB mode.  It is preprocessed internally —
        do not resize or normalise it before passing.
        """
        import torch

        dbg = self.debug if debug is None else debug

        # Preprocess and encode image
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat  = self._model.encode_image(img_tensor)
            txt_feat  = self._model.encode_text(self._text_tokens)

            # L2-normalise both (required for cosine similarity)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            # Raw dot products = cosine similarities after normalisation
            raw_sims = (img_feat @ txt_feat.T)[0]  # shape: (n_texts,)

            if self.scoring_method == "softmax":
                scores_tensor = raw_sims.softmax(dim=-1)
            else:  # cosine
                scores_tensor = raw_sims

        scores_list = [float(s) for s in scores_tensor]

        # Map text → score
        raw_scores = {text: round(score, 5)
                      for text, score in zip(self._all_texts, scores_list)}

        # Fitness = weighted positive − weighted negative
        pos_scores = scores_list[:self._n_positive]
        neg_scores = scores_list[self._n_positive:]

        fitness = (
                (sum(w * s for w, s in zip(self._pos_weights, pos_scores))/len(self._pos_weights))
            - (sum(w * s for w, s in zip(self._neg_weights, neg_scores))/len(self._neg_weights))
        )
        fitness = round(fitness, 6)

        result = GraderOutput(
            fitness    = fitness,
            raw_scores = raw_scores,
            method     = self.scoring_method,
            prompt_set = self.prompt_set.name,
        )

        if dbg:
            print(f"\n  [grader] {self.prompt_set.name}  method={self.scoring_method}")
            print(f"  Positive scores:")
            for p in self.prompt_set.positive:
                print(f"    {raw_scores[p.text]:+.4f} (w={p.weight})  {p.text[:70]}")
            if self.prompt_set.negative:
                print(f"  Negative scores:")
                for p in self.prompt_set.negative:
                    print(f"    {raw_scores[p.text]:+.4f} (w={p.weight})  {p.text[:70]}")
            print(f"  → fitness = {fitness:.5f}")

        return result


# ---------------------------------------------------------------------------
# Debug — run directly to test the full grader pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from morphology import QUADRIPOD, TRIPOD, HEXAPOD
    from rendering  import MorphologyRenderer, RenderConfig
    from prompts    import SPIDER_BODY, MANY_LEGS, COMPACT_STABLE, ALL_PROMPT_SETS

    print("=" * 60)
    print("  grader.py — debug mode")
    print("=" * 60)

    if not _TORCH_AVAILABLE or not _CLIP_AVAILABLE:
        print("ERROR: torch and open_clip must be installed.")
        sys.exit(1)

    # --- Render test morphologies ---
    print("\n[1] Rendering test morphologies...")
    render_cfg = RenderConfig(width=256, height=256, debug=False)
    renderer   = MorphologyRenderer(render_cfg)

    images = {}
    for morph in (TRIPOD, QUADRIPOD, HEXAPOD):
        images[morph.name] = renderer.render(morph)
        print(f"  rendered {morph.name}")

    # --- Score with multiple prompt sets ---
    print("\n[2] Scoring with SPIDER_BODY (cosine)\n")
    grader = CLIPGrader(
        prompt_set     = SPIDER_BODY,
        scoring_method = "cosine",
        debug          = True,
    )

    results = {}
    for name, img in images.items():
        result = grader.score(img, debug=True)
        results[name] = result

    # --- Compare fitness across morphologies ---
    print("\n[3] Fitness comparison (SPIDER_BODY)\n")
    for name, res in sorted(results.items(), key=lambda x: -x[1].fitness):
        print(f"  {name:<14}  fitness={res.fitness:+.5f}")

    # --- Compare softmax vs cosine on QUADRIPOD ---
    print("\n[4] Softmax vs Cosine on QUADRIPOD\n")
    for method in ("cosine", "softmax"):
        g = CLIPGrader(prompt_set=MANY_LEGS, scoring_method=method, debug=False)
        r = g.score(images["quadripod"])
        print(f"  {method:<10}  fitness={r.fitness:+.5f}")

    renderer.close()
    print("\nAll grader tests done.")
