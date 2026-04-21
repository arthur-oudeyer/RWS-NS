"""
gemini_batch.py
===============
Prototype: batch scoring of multiple robot morphology images in a single
Gemini API call, vs. the current one-image-per-call approach.

Strategy
--------
Instead of building a grid (lossy, hard to attribute scores back), we send
all N images as separate Parts in one generate_content call, each preceded
by a short "Robot_<id>:" label so the model knows which image is which.
The prompt asks for a JSON dict keyed by robot ID.

Usage
-----
    python gemini_batch.py                   # runs benchmark on img/batch/
    python gemini_batch.py img/batch/ 5      # batch size 5
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as genai_types
from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from api_keys import APIKEY_GEMINI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../Morphology Only"))
from gemini_prompts import GeminiPromptConfig, SPIDER_MORPH


MODEL = "gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Output dataclass (mirrors grader.GraderOutput for drop-in compatibility)
# ---------------------------------------------------------------------------

@dataclass
class GraderOutput:
    fitness:    float
    raw_scores: dict[str, float]
    method:     str
    prompt_set: str
    extra:      dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"fitness={self.fitness:.4f}  method={self.method}  set={self.prompt_set}"]
        for k, v in self.raw_scores.items():
            lines.append(f"  {k}: {v}")
        if self.extra.get("observation"):
            lines.append(f"  obs: {self.extra['observation'][:100]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch prompt builder
# ---------------------------------------------------------------------------

def build_batch_prompt(prompt_config: GeminiPromptConfig, robot_ids: list[str]) -> str:
    """
    Wraps the standard single-image prompt into a multi-robot batch variant.
    The model receives N labeled images and must return a JSON dict keyed by
    robot ID.
    """
    id_list = ", ".join(robot_ids)
    single_schema = """\
    {
      "observation":    "factual description",
      "interpretation": "interpretation description and explanation",
      "coherence":      { "score": <int 0-10>, "reason": "..." },
      "originality":    { "score": <int 0-10>, "reason": "..." },
      "interest":       { "score": <int 0-10>, "reason": "..." }
    }"""

    static_target  = prompt_config.target
    dynamic_target = "move forward continuously while staying upright"

    return f"""
    ═══ BATCH EVALUATION ═══

    You will evaluate {len(robot_ids)} robot morphologies in one pass.
    Each image was labeled before being sent: {id_list}.
    Evaluate each one independently. Do NOT compare robots to each other.

    ═══ CONTEXT (same for all robots) ═══

    You are a strict and skeptical evaluator analyzing static images of MuJoCo robot morphologies.
    Your job is to be PRECISE and reproduce human-like feedback on each robot's structural design.

    The scene (applies to every image):
    - 2 simultaneous views of the same morphology: left = front/side angle, right = 3/4 perspective
    - dark/grey checkerboard floor
    - Robot has a white cylindrical torso and colored limbs (red, yellow, green, purple...)
    - The robot's locomotion objective: {dynamic_target}
    - The robot's morphology objective: looking like a {static_target} (= target)

    ═══ ANALYSIS (repeat for every robot) ═══

    Step 1 — Factual observation
    - Torso shape, size and position relative to the ground
    - Number of limbs, attachment points, segment lengths and approximate angles
    - Overall stance: upright, crouching, sprawled, collapsed?
    - Any asymmetry or unusual structural feature

    Step 2 — Morphology interpretation
    - Does it resemble a {static_target}? Which features match or not?
    - Is stable locomotion physically plausible? (center of mass, ground contacts, symmetry)
    - Originality or structural issues?

    Step 3 — Score (conservative, static image only)

    coherence  — How well does the morphology match a {static_target}?
      0–2 = no similarity | 3–4 = vague | 5–6 = partial | 7–8 = strong | 9–10 = unmistakable

    originality  — Is the structural design novel?
      0–2 = generic | 3–4 = basic | 5–6 = one interesting choice | 7–8 = novel | 9–10 = highly creative

    interest  — Evolutionary/locomotion potential
      0–2 = implausible | 3–4 = poor | 5–6 = plausible but inefficient | 7–8 = solid | 9–10 = excellent

    ═══ OUTPUT FORMAT ═══
    Respond ONLY with valid JSON, no text before or after.
    The top-level keys must be exactly the robot IDs: {id_list}
    Each value follows this schema:
    {single_schema}

    Example (2 robots):
    {{
      "Robot_0": {{ "observation": "...", "interpretation": "...", "coherence": {{"score": 5, "reason": "..."}}, "originality": {{"score": 4, "reason": "..."}}, "interest": {{"score": 6, "reason": "..."}} }},
      "Robot_1": {{ "observation": "...", "interpretation": "...", "coherence": {{"score": 7, "reason": "..."}}, "originality": {{"score": 3, "reason": "..."}}, "interest": {{"score": 8, "reason": "..."}} }}
    }}
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_single(parsed: dict, prompt_config: GeminiPromptConfig) -> GraderOutput:
    """Convert one parsed dict (single-robot schema) into a GraderOutput."""
    def _score(key: str) -> float:
        val = parsed.get(key, {})
        return float(val.get("score", 0) if isinstance(val, dict) else val)

    def _reason(key: str) -> str:
        val = parsed.get(key, {})
        return val.get("reason", "") if isinstance(val, dict) else ""

    coherence   = _score("coherence")
    originality = _score("originality")
    interest    = _score("interest")

    w = prompt_config.weights
    total_w = w.coherence + w.originality + w.interest
    fitness = (w.coherence * coherence + w.originality * originality + w.interest * interest) / (10.0 * total_w)

    return GraderOutput(
        fitness    = round(fitness, 6),
        raw_scores = {
            "coherence":   round(coherence, 4),
            "originality": round(originality, 4),
            "interest":    round(interest, 4),
        },
        method     = "gemini_batch",
        prompt_set = prompt_config.name,
        extra      = {
            "observation":        parsed.get("observation", ""),
            "interpretation":     parsed.get("interpretation", ""),
            "coherence_reason":   _reason("coherence"),
            "originality_reason": _reason("originality"),
            "interest_reason":    _reason("interest"),
        },
    )


def _upload_image(client: genai.Client, image: PILImage.Image, label: str, debug: bool = False) -> genai_types.File:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    if debug:
        print(f"  Uploading {label} ({len(buf.getvalue()) // 1024} KB)...")
    img_file = client.files.upload(
        file   = buf,
        config = genai_types.UploadFileConfig(mime_type="image/png"),
    )
    while img_file.state.name == "PROCESSING":
        time.sleep(0.2)
        img_file = client.files.get(name=img_file.name)
    if img_file.state.name == "FAILED":
        raise RuntimeError(f"Image upload failed for {label}")
    return img_file


# ---------------------------------------------------------------------------
# GeminiBatchGrader
# ---------------------------------------------------------------------------

class GeminiBatchGrader:
    """
    Scores up to `batch_size` robot morphology images in a single API call.

    Parameters
    ----------
    api_key       : Gemini API key.
    prompt_config : GeminiPromptConfig (from gemini_prompts.py).
    model_name    : Gemini model ID.
    batch_size    : max images per request (default 10).
    debug         : print upload/parse details.
    """

    def __init__(
        self,
        api_key:       str,
        prompt_config: GeminiPromptConfig,
        model_name:    str  = MODEL,
        batch_size:    int  = 10,
        debug:         bool = False,
    ):
        self._client        = genai.Client(api_key=api_key)
        self._prompt_config = prompt_config
        self._model_name    = model_name
        self._batch_size    = batch_size
        self._debug         = debug

    def score_batch(
        self,
        images: list[tuple[str, PILImage.Image]],
    ) -> dict[str, GraderOutput]:
        """
        Score a list of (robot_id, PIL Image) pairs.

        If len(images) > batch_size, splits into multiple calls automatically.
        Returns a dict mapping robot_id -> GraderOutput.
        """
        results: dict[str, GraderOutput] = {}
        for chunk_start in range(0, len(images), self._batch_size):
            chunk = images[chunk_start : chunk_start + self._batch_size]
            chunk_results = self._score_chunk(chunk)
            results.update(chunk_results)
        return results

    def _score_chunk(
        self,
        images: list[tuple[str, PILImage.Image]],
    ) -> dict[str, GraderOutput]:
        robot_ids = [rid for rid, _ in images]
        uploaded: list[genai_types.File] = []

        try:
            # Upload all images in parallel (sequentially here; could use threads)
            for robot_id, img in images:
                f = _upload_image(self._client, img, robot_id, self._debug)
                uploaded.append(f)

            # Build contents: [label, image, label, image, ..., prompt]
            contents = []
            for robot_id, file in zip(robot_ids, uploaded):
                contents.append(f"{robot_id}:")
                contents.append(
                    genai_types.Part.from_uri(file_uri=file.uri, mime_type="image/png")
                )
            contents.append(build_batch_prompt(self._prompt_config, robot_ids))

            if self._debug:
                print(f"  Sending batch of {len(images)} images to {self._model_name}...")

            t0 = time.time()
            response = self._client.models.generate_content(
                model    = self._model_name,
                contents = contents,
            )
            elapsed = time.time() - t0
            if self._debug:
                print(f"  Batch response received in {elapsed:.2f}s")

        finally:
            for f in uploaded:
                self._client.files.delete(name=f.name)
            if self._debug:
                print(f"  Deleted {len(uploaded)} remote files.")

        # Parse JSON
        text = response.text
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]

        start = stripped.find("{")
        end   = stripped.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in batch response.\nRaw:\n{text}")

        parsed = json.loads(stripped[start:end])

        results = {}
        for robot_id in robot_ids:
            if robot_id not in parsed:
                print(f"  WARNING: robot_id '{robot_id}' missing from response.")
                continue
            results[robot_id] = _parse_single(parsed[robot_id], self._prompt_config)

        return results


# ---------------------------------------------------------------------------
# Single-image grader (for timing comparison)
# ---------------------------------------------------------------------------

class GeminiSingleGrader:
    """Same interface as GeminiBatchGrader but uses one API call per image."""

    def __init__(
        self,
        api_key:       str,
        prompt_config: GeminiPromptConfig,
        model_name:    str  = MODEL,
        debug:         bool = False,
    ):
        self._client        = genai.Client(api_key=api_key)
        self._prompt_config = prompt_config
        self._model_name    = model_name
        self._debug         = debug

    def score_batch(
        self,
        images: list[tuple[str, PILImage.Image]],
    ) -> dict[str, GraderOutput]:
        results = {}
        for robot_id, img in images:
            results[robot_id] = self._score_one(robot_id, img)
        return results

    def _score_one(self, robot_id: str, image: PILImage.Image) -> GraderOutput:
        img_file = _upload_image(self._client, image, robot_id, self._debug)
        try:
            response = self._client.models.generate_content(
                model    = self._model_name,
                contents = [
                    genai_types.Part.from_uri(file_uri=img_file.uri, mime_type="image/png"),
                    self._prompt_config.prompt,
                ],
            )
            text = response.text
        finally:
            self._client.files.delete(name=img_file.name)

        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]
        start = stripped.find("{")
        end   = stripped.rfind("}") + 1
        parsed = json.loads(stripped[start:end])
        return _parse_single(parsed, self._prompt_config)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(image_dir: str, batch_size: int = 10, debug: bool = True):
    """
    Load all PNG images from `image_dir`, then score them with:
      1. GeminiBatchGrader  (N images in 1 call)
      2. GeminiSingleGrader (N images in N calls)
    Prints timing and score comparison.
    """
    img_paths = sorted(Path(image_dir).glob("*.png"))
    if not img_paths:
        print(f"No PNG images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(img_paths)} images in {image_dir}")

    images: list[tuple[str, PILImage.Image]] = []
    for p in img_paths:
        robot_id = p.stem  # e.g. "gen0000_id000000"
        images.append((robot_id, PILImage.open(p).convert("RGB")))

    print(f"\n{'='*60}")
    print(f"  BATCH approach  (batch_size={batch_size})")
    print(f"{'='*60}")
    batch_grader = GeminiBatchGrader(
        api_key       = APIKEY_GEMINI,
        prompt_config = SPIDER_MORPH,
        batch_size    = batch_size,
        debug         = debug,
    )
    t_batch_start = time.time()
    batch_results = batch_grader.score_batch(images)
    t_batch = time.time() - t_batch_start

    print(f"\nBatch results ({t_batch:.1f}s total, {t_batch/len(images):.2f}s/robot):")
    for rid, out in batch_results.items():
        print(f"  {rid:<30} fitness={out.fitness:.4f}  "
              f"coh={out.raw_scores['coherence']:.0f}  "
              f"ori={out.raw_scores['originality']:.0f}  "
              f"int={out.raw_scores['interest']:.0f}")

    print(f"\n{'='*60}")
    print(f"  SINGLE approach  (1 call per image)")
    print(f"{'='*60}")
    single_grader = GeminiSingleGrader(
        api_key       = APIKEY_GEMINI,
        prompt_config = SPIDER_MORPH,
        debug         = debug,
    )
    t_single_start = time.time()
    single_results = single_grader.score_batch(images)
    t_single = time.time() - t_single_start

    print(f"\nSingle results ({t_single:.1f}s total, {t_single/len(images):.2f}s/robot):")
    for rid, out in single_results.items():
        print(f"  {rid:<30} fitness={out.fitness:.4f}  "
              f"coh={out.raw_scores['coherence']:.0f}  "
              f"ori={out.raw_scores['originality']:.0f}  "
              f"int={out.raw_scores['interest']:.0f}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Images scored  : {len(images)}")
    print(f"  Batch time     : {t_batch:.1f}s  ({t_batch/len(images):.2f}s/robot)")
    print(f"  Single time    : {t_single:.1f}s  ({t_single/len(images):.2f}s/robot)")
    speedup = t_single / t_batch if t_batch > 0 else 0
    print(f"  Speedup        : {speedup:.2f}x")

    print(f"\n  Score diff (batch - single):")
    for rid in batch_results:
        if rid not in single_results:
            continue
        b = batch_results[rid]
        s = single_results[rid]
        df = b.fitness - s.fitness
        print(f"  {rid:<30} Δfitness={df:+.4f}  "
              f"Δcoh={b.raw_scores['coherence']-s.raw_scores['coherence']:+.0f}  "
              f"Δori={b.raw_scores['originality']-s.raw_scores['originality']:+.0f}  "
              f"Δint={b.raw_scores['interest']-s.raw_scores['interest']:+.0f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    img_dir    = sys.argv[1] if len(sys.argv) > 1 else "./img/batch"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_benchmark(img_dir, batch_size=batch_size, debug=True)