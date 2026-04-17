"""
grader.py
=========
Morphology fitness graders.

Architecture
------------
MorphologyGrader (abstract base)
    ├─ CLIPGrader    — local scoring via OpenCLIP
    └─ GeminiGrader  — VLM scoring via Google Gemini

CLIPGrader — scoring methods
-----------------------------
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

CLIPGrader fitness formula (both methods):
    fitness = mean(Σ (pos_weight * pos_score)) − mean(Σ (neg_weight * neg_score))

GeminiGrader — scoring
-----------------------
    Sends the image to Gemini and parses a JSON response with three
    dimensions: coherence, originality, interest (each 0-10).
    Fitness formula:
        fitness = (w_coherence * coherence
                   + w_originality * originality
                   + w_interest * interest)
                  / (10 * (w_coherence + w_originality + w_interest))
    → always in [0, 1].

GraderOutput
------------
    fitness      : float — scalar used by the evolution loop for selection.
    raw_scores   : dict[text → float] — every prompt's score, for analysis.
    method       : "cosine", "softmax", or "gemini"
    prompt_set   : name of the PromptSet / GeminiPromptConfig used.

Debug mode
----------
    Set debug=True in any grader or pass debug=True to score() to print
    per-prompt / per-dimension scores and the final fitness.

Usage — CLIPGrader
------------------
    from grader import CLIPGrader
    from CLIP_prompts import SPIDER_BODY

    grader = CLIPGrader(
        model_name     = "ViT-B-32",
        pretrained     = "openai",
        cache_dir      = "/Volumes/T7_AO/clip-models",
        prompt_set     = SPIDER_BODY,
        scoring_method = "cosine",
        debug          = False,
    )

    result = grader.score(image)

Usage — GeminiGrader
--------------------
    from grader import GeminiGrader
    from gemini_prompts import INSECT_MORPH

    grader = GeminiGrader(
        api_key       = "YOUR_KEY",
        prompt_config = INSECT_MORPH,
        model_name    = "gemini-2.0-flash",
        debug         = False,
    )

    result = grader.score(image)
    print(result.fitness)
    print(result.raw_scores)
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
from CLIP_prompts import PromptSet


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
    raw_scores   : dict mapping each prompt / dimension to its numeric score.
                   Stored for full traceability and analysis.
    method       : "cosine", "softmax", or "gemini".
    prompt_set   : name of the PromptSet / GeminiPromptConfig that was applied.
    extra        : backend-specific metadata, empty for CLIP.
                   For GeminiGrader this contains:
                     "observation"        — factual description of the image
                     "interpretation"     — structural interpretation
                     "coherence_reason"   — why Gemini gave that coherence score
                     "originality_reason" — why Gemini gave that originality score
                     "interest_reason"    — why Gemini gave that interest score
    """
    fitness:    float
    raw_scores: dict[str, float]
    method:     str
    prompt_set: str
    extra:      dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"fitness={self.fitness:.4f}  method={self.method}  set={self.prompt_set}"]
        for text, score in sorted(self.raw_scores.items(), key=lambda x: -x[1]):
            lines.append(f"  {score:+.4f}  {text}")
        if self.extra.get("observation"):
            lines.append(f"  observation: {self.extra['observation'][:100]}")
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
# GeminiGrader
# ---------------------------------------------------------------------------

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


class GeminiGrader(MorphologyGrader):
    """
    Scores morphology images using Google Gemini (VLM).

    The image is uploaded to the Gemini Files API, scored by the model,
    then the remote file is deleted.  score() is stateless and can be
    called for many images.

    Parameters
    ----------
    api_key       : Gemini API key.
    prompt_config : GeminiPromptConfig (from gemini_prompts.py).
    model_name    : Gemini model ID.  Defaults to "gemini-2.0-flash".
    debug         : global debug flag.

    Fitness
    -------
    Gemini returns coherence / originality / interest scores (each 0-10).
    Fitness is their weighted average normalised to [0, 1]:
        fitness = (w_c * coherence + w_o * originality + w_i * interest)
                  / (10 * (w_c + w_o + w_i))
    """

    def __init__(
        self,
        api_key:       str,
        prompt_config: "GeminiPromptConfig",  # noqa: F821 — imported lazily below
        model_name:    str  = "gemini-3-flash-preview",
        debug:         bool = False,
    ):
        if not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai is required for GeminiGrader.  "
                "Install with: pip install google-genai"
            )

        # Store prompt_config as self.prompt_set so the base-class interface
        # (and GraderOutput.prompt_set = self.prompt_set.name) works unchanged.
        super().__init__(prompt_set=prompt_config, debug=debug)  # type: ignore[arg-type]

        """
        Gemini 3.1 Flash-Lite -> gemini-3.1-flash-lite-preview
        Gemini 3 Flash        -> gemini-3-flash-preview
        Gemini 3.1 Pro        -> gemini-3.1-pro-preview
        """
        self._model_name    = model_name
        self._prompt_config = prompt_config
        self._client        = _genai.Client(api_key=api_key)

        if debug:
            print(f"[grader] GeminiGrader ready — model={model_name}  "
                  f"config={prompt_config.name}  target={prompt_config.target}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        image: "PILImage.Image",
        debug: Optional[bool] = None,
    ) -> GraderOutput:
        """
        Score one PIL Image using Gemini and return a GraderOutput.

        The image is uploaded as a PNG, scored, and the remote file is
        deleted in a finally block.  The image must be in RGB mode.
        """
        import io
        import json
        import time

        dbg = self.debug if debug is None else debug

        # --- Encode image to PNG bytes ---
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # --- Upload to Gemini Files API ---
        if dbg:
            print(f"  [grader/gemini] Uploading image ({len(png_bytes)//1024} KB)...")

        img_file = self._client.files.upload(
            file   = io.BytesIO(png_bytes),
            config = _genai_types.UploadFileConfig(mime_type="image/png"),
        )

        # Wait for processing
        while img_file.state.name == "PROCESSING":
            time.sleep(0.2)
            img_file = self._client.files.get(name=img_file.name)

        if img_file.state.name == "FAILED":
            raise RuntimeError("[grader/gemini] Image processing failed on Gemini's side.")

        # --- Query model ---
        try:
            response = self._client.models.generate_content(
                model    = self._model_name,
                contents = [
                    _genai_types.Part.from_uri(
                        file_uri  = img_file.uri,
                        mime_type = "image/png",
                    ),
                    self._prompt_config.prompt,
                ],
            )
            text = response.text
        finally:
            self._client.files.delete(name=img_file.name)
            if dbg:
                print("  [grader/gemini] Remote file deleted.")

        # --- Parse JSON ---
        # Strip optional markdown code fences (```json ... ```)
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]

        start = stripped.find("{")
        end   = stripped.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(
                f"[grader/gemini] No JSON found in Gemini response.\n"
                f"Raw response:\n{text}"
            )

        parsed = json.loads(stripped[start:end])

        # --- Extract scores ---
        def _extract_score(key: str) -> float:
            val = parsed.get(key, {})
            if isinstance(val, dict):
                return float(val.get("score", 0))
            return float(val)

        coherence   = _extract_score("coherence")
        originality = _extract_score("originality")
        interest    = _extract_score("interest")

        w = self._prompt_config.weights
        total_w = w.coherence + w.originality + w.interest
        fitness = (
            w.coherence   * coherence
            + w.originality * originality
            + w.interest    * interest
        ) / (10.0 * total_w)
        fitness = round(fitness, 6)

        raw_scores = {
            "coherence":   round(coherence,   4),
            "originality": round(originality, 4),
            "interest":    round(interest,    4),
        }

        def _reason(key: str) -> str:
            val = parsed.get(key, {})
            if isinstance(val, dict):
                return val.get("reason", "")
            return ""

        extra = {
            "observation":        parsed.get("observation", ""),
            "interpretation":     parsed.get("interpretation", ""),
            "coherence_reason":   _reason("coherence"),
            "originality_reason": _reason("originality"),
            "interest_reason":    _reason("interest"),
        }

        result = GraderOutput(
            fitness    = fitness,
            raw_scores = raw_scores,
            method     = "gemini",
            prompt_set = self._prompt_config.name,
            extra      = extra,
        )

        if dbg:
            print(f"\n  [grader/gemini] {self._prompt_config.name}  "
                  f"target={self._prompt_config.target}")
            print(f"    coherence   = {coherence:.1f}  (w={w.coherence})")
            print(f"    originality = {originality:.1f}  (w={w.originality})")
            print(f"    interest    = {interest:.1f}  (w={w.interest})")
            print(f"  → fitness = {fitness:.5f}")
            if extra.get("observation"):
                print(f"  observation: {extra['observation'][:120]}")
            if extra.get("interpretation"):
                print(f"  interpretation: {extra['interpretation'][:120]}")

        return result


# ---------------------------------------------------------------------------
# Debug — run directly to test the full grader pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from morphology    import QUADRIPOD, TRIPOD, HEXAPOD
    from rendering     import MorphologyRenderer, RenderConfig
    from CLIP_prompts  import SPIDER_BODY, MANY_LEGS, COMPACT_STABLE, ALL_PROMPT_SETS

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
