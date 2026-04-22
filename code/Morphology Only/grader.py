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

    def score_batch(
        self,
        images: "list[tuple[str, PILImage.Image]]",
        debug: Optional[bool] = None,
        reference_image: "Optional[PILImage.Image]" = None,
    ) -> "dict[str, GraderOutput]":
        """
        Score multiple images, returning a dict keyed by robot_id.

        Default implementation calls score() in a loop.
        GeminiGrader overrides this to send all images in one API call.

        Parameters
        ----------
        images          : list of (robot_id, PIL Image) pairs.
        debug           : overrides self.debug for this call if not None.
        reference_image : optional reference image (e.g. current best).
                          Ignored by the default loop implementation; only
                          GeminiGrader uses it to anchor the batch prompt.
        """
        return {robot_id: self.score(img, debug=debug) for robot_id, img in images}


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

try:
    from descriptor import DescriptorConfig as _DescriptorConfig
    from descriptor import build_descriptor_prompt_section as _build_desc_section
    _DESCRIPTOR_AVAILABLE = True
except ImportError:
    _DESCRIPTOR_AVAILABLE = False


class GeminiGrader(MorphologyGrader):
    """
    Scores morphology images using Google Gemini (VLM).

    The image is uploaded to the Gemini Files API, scored by the model,
    then the remote file is deleted.  score() is stateless and can be
    called for many images.

    Parameters
    ----------
    api_key           : Gemini API key.
    prompt_config     : GeminiPromptConfig (from gemini_prompts.py).
    model_name        : Gemini model ID.
    batch_size        : max images per batch call (default 10).
    descriptor_config : optional DescriptorConfig (from descriptor.py).
                        When set, the VLM also rates each descriptor item
                        and the scores are stored in extra["vlm_descriptors"].
    debug             : global debug flag.

    Fitness
    -------
    Gemini returns coherence / originality / interest scores (each 0-10).
    Fitness is their weighted average normalised to [0, 1]:
        fitness = (w_c * coherence + w_o * originality + w_i * interest)
                  / (10 * (w_c + w_o + w_i))
    """

    def __init__(
        self,
        api_key:           str,
        prompt_config:     "GeminiPromptConfig",  # noqa: F821
        model_name:        str  = "gemini-3-flash-preview",
        batch_size:        int  = 10,
        descriptor_config: "Optional[_DescriptorConfig]" = None,
        response_log_path: "Optional[str]" = None,
        debug:             bool = False,
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
        self._model_name        = model_name
        self._prompt_config     = prompt_config
        self._client            = _genai.Client(api_key=api_key)
        self._batch_size        = batch_size
        self._descriptor_config = descriptor_config
        self._response_log_path = response_log_path

        if debug:
            desc_str = descriptor_config.name if descriptor_config else "none"
            print(f"[grader] GeminiGrader ready — model={model_name}  "
                  f"config={prompt_config.name}  target={prompt_config.target}  "
                  f"batch_size={batch_size}  descriptors={desc_str}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_full_prompt(self, base_prompt: str) -> str:
        """Append the descriptor section to a prompt when configured."""
        if self._descriptor_config is None:
            return base_prompt
        return base_prompt + _build_desc_section(self._descriptor_config)

    def _extract_vlm_descriptors(self, parsed: dict) -> dict:
        """Extract VLM descriptor scores from a parsed JSON response dict."""
        if self._descriptor_config is None or "descriptors" not in parsed:
            return {}
        block = parsed["descriptors"]
        result = {}
        for item in self._descriptor_config.items:
            if item.name in block:
                try:
                    result[item.name] = float(block[item.name])
                except (TypeError, ValueError):
                    result[item.name] = 0.0
        return result

    def _log_response(self, mode: str, robot_ids: list, raw_text: str) -> None:
        """Append the raw VLM response to the response log file (one JSON line)."""
        if not self._response_log_path:
            return
        import json as _json
        from datetime import datetime as _dt
        entry = {
            "ts":        _dt.now().isoformat(timespec="seconds"),
            "mode":      mode,
            "robot_ids": robot_ids,
            "raw":       raw_text,
        }
        with open(self._response_log_path, "a") as f:
            f.write(_json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        images: "list[tuple[str, PILImage.Image]]",
        debug: Optional[bool] = None,
        reference_image: "Optional[PILImage.Image]" = None,
    ) -> "dict[str, GraderOutput]":
        """
        Score multiple images in a single Gemini API call.

        Sends all images as labeled Parts in one generate_content request.
        Automatically splits into chunks of self._batch_size if needed.

        Parameters
        ----------
        images          : list of (robot_id, PIL Image) pairs to evaluate.
        debug           : overrides self.debug for this call if not None.
        reference_image : optional image of the current best individual.
                          When provided it is prepended to every chunk as a
                          labelled "reference" Part and the prompt is updated
                          to instruct Gemini not to score it but to use it as
                          a baseline for judging novelty and improvement.
                          The reference is never included in the returned dict.
        """
        import io
        import json
        import time

        dbg = self.debug if debug is None else debug
        results: dict[str, GraderOutput] = {}

        # Pre-encode reference image once (reused for every chunk)
        ref_bytes: Optional[bytes] = None
        if reference_image is not None:
            ref_buf = io.BytesIO()
            reference_image.save(ref_buf, format="PNG")
            ref_bytes = ref_buf.getvalue()

        for chunk_start in range(0, len(images), self._batch_size):
            chunk = images[chunk_start : chunk_start + self._batch_size]
            robot_ids = [rid for rid, _ in chunk]
            uploaded = []
            ref_file = None

            try:
                # Upload reference first (if provided)
                if ref_bytes is not None:
                    if dbg:
                        print(f"  [grader/gemini/batch] Uploading reference ({len(ref_bytes)//1024} KB)...")
                    ref_file = self._client.files.upload(
                        file=io.BytesIO(ref_bytes),
                        config=_genai_types.UploadFileConfig(mime_type="image/png"),
                    )
                    while ref_file.state.name == "PROCESSING":
                        time.sleep(0.2)
                        ref_file = self._client.files.get(name=ref_file.name)
                    if ref_file.state.name == "FAILED":
                        raise RuntimeError("[grader/gemini/batch] Reference image upload failed")

                # Upload candidate robots
                for robot_id, img in chunk:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    if dbg:
                        print(f"  [grader/gemini/batch] Uploading {robot_id} ({len(buf.getvalue())//1024} KB)...")
                    img_file = self._client.files.upload(
                        file=buf,
                        config=_genai_types.UploadFileConfig(mime_type="image/png"),
                    )
                    while img_file.state.name == "PROCESSING":
                        time.sleep(0.2)
                        img_file = self._client.files.get(name=img_file.name)
                    if img_file.state.name == "FAILED":
                        raise RuntimeError(f"[grader/gemini/batch] Upload failed for {robot_id}")
                    uploaded.append(img_file)

                # Build contents: [reference?,] label, image, label, image, ..., prompt
                contents = []
                if ref_file is not None:
                    contents.append("reference:")
                    contents.append(
                        _genai_types.Part.from_uri(file_uri=ref_file.uri, mime_type="image/png")
                    )
                for robot_id, img_file in zip(robot_ids, uploaded):
                    contents.append(f"{robot_id}:")
                    contents.append(
                        _genai_types.Part.from_uri(file_uri=img_file.uri, mime_type="image/png")
                    )
                contents.append(self._build_batch_prompt(robot_ids, has_reference=ref_file is not None))

                if dbg:
                    ref_str = " + reference" if ref_file else ""
                    print(f"  [grader/gemini/batch] Sending {len(chunk)} images{ref_str}...")

                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                )
                text = response.text
                self._log_response("batch", robot_ids, text)

            finally:
                for img_file in uploaded:
                    self._client.files.delete(name=img_file.name)
                if ref_file is not None:
                    self._client.files.delete(name=ref_file.name)
                if dbg:
                    n_del = len(uploaded) + (1 if ref_file else 0)
                    print(f"  [grader/gemini/batch] Deleted {n_del} remote files.")

            # Parse JSON dict keyed by robot_id
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1]
                stripped = stripped.rsplit("```", 1)[0]
            start = stripped.find("{")
            end = stripped.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError(f"[grader/gemini/batch] No JSON in response.\nRaw:\n{text}")

            parsed = json.loads(stripped[start:end])
            # Drop any stray "reference" key Gemini may have included
            parsed.pop("reference", None)
            for robot_id in robot_ids:
                if robot_id not in parsed:
                    raise ValueError(f"[grader/gemini/batch] Missing robot_id '{robot_id}' in response.")
                results[robot_id] = self._parse_batch_entry(parsed[robot_id], dbg)

        return results

    def _build_batch_prompt(self, robot_ids: list, has_reference: bool = False) -> str:
        """
        Build the batch evaluation prompt.

        The CONTEXT, ANALYSIS, and scoring-criteria sections are taken verbatim
        from self._prompt_config.prompt (defined in gemini_prompts.py) — split
        at the OUTPUT FORMAT marker so that updating gemini_prompts.py
        automatically applies to batch mode.  Only the batch-specific framing
        (robot IDs, reference section, multi-robot JSON schema) is built here.
        """
        id_list = ", ".join(robot_ids)

        # --- Reuse CONTEXT + ANALYSIS from the config prompt ---
        # Everything before "═══ OUTPUT FORMAT ═══" is the evaluation body.
        _OUTPUT_MARKER = "═══ OUTPUT FORMAT ═══"
        base = self._prompt_config.prompt
        if _OUTPUT_MARKER in base:
            body = base[:base.index(_OUTPUT_MARKER)].rstrip()
        else:
            body = base.rstrip()

        # Append descriptor instructions if configured (mirrors _build_full_prompt)
        if self._descriptor_config and _DESCRIPTOR_AVAILABLE:
            body = body + _build_desc_section(self._descriptor_config)

        # --- Per-robot JSON output schema ---
        desc_schema = ""
        if self._descriptor_config:
            desc_items = "\n".join(
                f'        "{item.name}": <int 0-10>,'
                for item in self._descriptor_config.items
            )
            desc_schema = (
                '      "descriptors": {\n'
                + desc_items
                + '\n      },'
            )

        single_schema = (
            "{\n"
            '      "observation":    "factual description of what you see",\n'
            '      "interpretation": "structural interpretation relative to the target",\n'
            '      "coherence":      { "score": <int 0-10>, "reason": "..." },\n'
            '      "originality":    { "score": <int 0-10>, "reason": "..." },\n'
            '      "interest":       { "score": <int 0-10>, "reason": "..." }'
            + (f',\n{desc_schema}' if desc_schema else "")
            + "\n    }"
        )

        # --- Reference section ---
        reference_section = ""
        if has_reference:
            reference_section = f"""
    ═══ REFERENCE IMAGE ═══

    The first image labeled "reference" shows the CURRENT BEST-PERFORMING robot from the previous generation.
    It is provided as a contextual baseline ONLY.
    — Do NOT score the reference. Do NOT include "reference" as a key in your JSON output.
    — Your goal is NOT to reward morphologies that merely look like the reference.
    Instead, use the reference to better identify and reward:
      • Genuine structural novelty: designs clearly different from the reference in limb count,
        body plan, attachment points, or overall stance.
      • Real improvement: morphologies that address visible weaknesses of the reference
        (e.g. better ground contact, more coherent body plan, more {self._prompt_config.target}-like structure).
      • Interesting new traits the reference does not have.
    If a candidate robot is merely a minor variant of the reference, do not inflate its scores.
            """

        return f"""
    ═══ BATCH EVALUATION ═══

    You will evaluate {len(robot_ids)} robot morphologies in one pass.
    Each image was labeled before being sent: {id_list}.
    Evaluate each one independently.
    {reference_section}
    {body}

    ═══ OUTPUT FORMAT ═══
    Respond ONLY with valid JSON, no text before or after.
    The top-level keys must be exactly the robot IDs: {id_list}
    Each value follows this schema:
    {single_schema}
    """

    def _parse_batch_entry(self, parsed: dict, dbg: bool) -> GraderOutput:
        """Convert one dict entry from a batch response into a GraderOutput."""
        def _score(key):
            val = parsed.get(key, {})
            return float(val.get("score", 0) if isinstance(val, dict) else val)

        def _reason(key):
            val = parsed.get(key, {})
            return val.get("reason", "") if isinstance(val, dict) else ""

        coherence   = _score("coherence")
        originality = _score("originality")
        interest    = _score("interest")

        w = self._prompt_config.weights
        total_w = w.coherence + w.originality + w.interest
        fitness = (
            w.coherence * coherence + w.originality * originality + w.interest * interest
        ) / (10.0 * total_w)

        if dbg:
            print(f"    coherence={coherence:.1f}  originality={originality:.1f}  "
                  f"interest={interest:.1f}  → fitness={fitness:.4f}")

        return GraderOutput(
            fitness=round(fitness, 6),
            raw_scores={
                "coherence":   round(coherence, 4),
                "originality": round(originality, 4),
                "interest":    round(interest, 4),
            },
            method="gemini_batch",
            prompt_set=self._prompt_config.name,
            extra={
                "observation":        parsed.get("observation", ""),
                "interpretation":     parsed.get("interpretation", ""),
                "coherence_reason":   _reason("coherence"),
                "originality_reason": _reason("originality"),
                "interest_reason":    _reason("interest"),
                "vlm_descriptors":    self._extract_vlm_descriptors(parsed),
            },
        )

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
                    self._build_full_prompt(self._prompt_config.prompt),
                ],
            )
            text = response.text
            self._log_response("single", [], text)
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

        vlm_descriptors = self._extract_vlm_descriptors(parsed)
        extra = {
            "observation":        parsed.get("observation", ""),
            "interpretation":     parsed.get("interpretation", ""),
            "coherence_reason":   _reason("coherence"),
            "originality_reason": _reason("originality"),
            "interest_reason":    _reason("interest"),
            "vlm_descriptors":    vlm_descriptors,
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
            if vlm_descriptors:
                print(f"  descriptors: {vlm_descriptors}")

        return result


# ---------------------------------------------------------------------------
# Debug — print the exact prompts that will be sent to the VLM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from config          import ExperimentConfig
    from gemini_prompts  import get_gemini_prompt_set
    from descriptor      import get_descriptor_config

    SEP  = "═" * 72
    SEP2 = "─" * 72

    print(SEP)
    print("  grader.py — prompt preview")
    print(SEP)

    if not _GEMINI_AVAILABLE:
        print("ERROR: google-genai is not installed — cannot build GeminiGrader.")
        sys.exit(1)

    # --- Load config and build grader (dummy key — no network call for printing) ---
    cfg           = ExperimentConfig()
    prompt_config = get_gemini_prompt_set(cfg.prompt_name)
    desc_config   = None
    if _DESCRIPTOR_AVAILABLE and getattr(cfg, "descriptor_config_name", ""):
        try:
            desc_config = get_descriptor_config(cfg.descriptor_config_name)
        except KeyError:
            pass

    grader = GeminiGrader(
        api_key           = "DUMMY_KEY",
        prompt_config     = prompt_config,
        model_name        = cfg.gemini_model,
        batch_size        = cfg.batching,
        descriptor_config = desc_config,
    )

    print(f"\n  Config         : {cfg.run_id}")
    print(f"  Prompt name    : {cfg.prompt_name}  (target: {prompt_config.target})")
    print(f"  Gemini model   : {cfg.gemini_model}")
    print(f"  Descriptor cfg : {desc_config.name if desc_config else 'none'}")
    print(f"  Reference best : {getattr(cfg, 'reference_best_in_batch', False)}")
    print(f"  Batch size     : {cfg.batching}")

    # --- [1] Single-image prompt ---
    print(f"\n{SEP}")
    print("  [1] SINGLE-IMAGE PROMPT  (used by score())")
    print(SEP)
    print(grader._build_full_prompt(prompt_config.prompt))

    # --- [2] Batch prompt — no reference ---
    fake_ids = [f"robot_{i:03d}" for i in range(1, min(cfg.batching, 4) + 1)]
    print(f"\n{SEP}")
    print(f"  [2] BATCH PROMPT — no reference  ({len(fake_ids)} robots: {', '.join(fake_ids)})")
    print(SEP)
    print(grader._build_batch_prompt(fake_ids, has_reference=False))

    # --- [3] Batch prompt — with reference ---
    print(f"\n{SEP}")
    print(f"  [3] BATCH PROMPT — with reference  (reference_best_in_batch=True)")
    print(SEP)
    print(grader._build_batch_prompt(fake_ids, has_reference=True))

    print(f"\n{SEP2}")
    print("  Done — no API calls made.")
