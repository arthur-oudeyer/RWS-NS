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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../Morphology"))
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
# Video batch prompt builder
# ---------------------------------------------------------------------------

def build_batch_video_prompt(target_behaviour: str, video_ids: list[str]) -> str:
    """
    Multi-video variant of score_robot_video()'s prompt.

    The model receives N labeled 5s videos and must return a JSON dict keyed
    by video ID, each value following the single-video scoring schema (0-100).
    """
    id_list = ", ".join(video_ids)
    single_schema = """\
        {
          "observation":    "key-time steps factual description (timestamps, floor-tile reference, posture of each limb, ...)",
          "interpretation": "behavioural interpretation relative to the target",
          "coherence":      { "score": <int 0-100>, "reason": "..." },
          "originality":    { "score": <int 0-100>, "reason": "..." },
          "potential":      { "score": <int 0-100>, "reason": "..." }
        }"""

    return f"""
        ═══ BATCH EVALUATION ═══

        You will evaluate {len(video_ids)} robot videos in one pass.
        Each video was labeled before being sent: {id_list}.
        Evaluate each one independently. Do NOT compare robots to each other.

        ═══ CONTEXT (same for all videos) ═══

        Each video is a 5 second simulation showing two side-by-side view of a simulated robot composed of a white torso and colored legs. It stands on a green checkered floor, and the background is blue.

        Target behavior : {target_behaviour}

        ═══ ANALYSIS (repeat for every video) ═══

        Step 1 — factual observation
        Describe the robot morphology and behavior.

        Step 2 — Behavioural interpretation
        - Did the robot make consistent consistent action relevant with the target behavior ?
        - Was the gait coherent (periodic, balanced, repeatable) or random ? What was the type of the gait (smooth, energetic, nervous, wide, brutal, efficient, small, homogenous, ...) ?
        - Is there anything novel or interesting about the motion pattern even if the robot did not perform well for the target behavior ? (ex: is a limb doing a movement with great potential ?)

        Step 3 — scoring (each dimension 0–100)

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

        potential — Is the gait pattern interesting, biologically plausible and leads to a real evolutionary potential ?
          0–29   = uninteresting (random, fallen) or obviously broken
          30–49  = generic, predictable motion with no notable features
          50–69  = one notable element (unusual gait phase, rhythm, recovery) that have potential
          70–89  = clearly interesting motion: reminiscent of an animal gait,
                   coordinated pattern, or creative body usage to reached the target. There is a great potential.
          90–100 = highly interesting; novel and biologically convincing locomotion, great abilities and great potential for further evolution.

        ═══ OUTPUT FORMAT ═══
        Respond ONLY with valid JSON, no text before or after.
        The top-level keys must be exactly the video IDs: {id_list}
        Each value follows this schema:
        {single_schema}
        """


def _parse_single_video(parsed: dict) -> dict:
    """
    Normalise one parsed single-video dict, adding an `overall` score using
    the same weighting as score_robot_video()'s __main__ block
    (coherence*1.0 + originality*0.5 + potential*1.5) / 3.
    """
    def _score(key: str) -> float:
        val = parsed.get(key, {})
        return float(val.get("score", 0) if isinstance(val, dict) else val)

    coherence   = _score("coherence")
    originality = _score("originality")
    potential   = _score("potential")
    overall     = round((1.0 * coherence + 0.5 * originality + 1.5 * potential) / 3, 1)

    return {**parsed, "overall": overall}


def _upload_video(client: genai.Client, video_path: str, label: str, debug: bool = False) -> genai_types.File:
    if debug:
        print(f"  Uploading {label} ({video_path})...")
    video_file = client.files.upload(
        file   = video_path,
        config = genai_types.UploadFileConfig(mime_type="video/mp4"),
    )
    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)
    if video_file.state.name == "FAILED":
        raise RuntimeError(f"Video upload failed for {label}")
    return video_file


# ---------------------------------------------------------------------------
# GeminiVideoBatchGrader
# ---------------------------------------------------------------------------

class GeminiVideoBatchGrader:
    """
    Scores up to `batch_size` robot videos in a single API call, mirroring
    score_robot_video() but for a whole batch.

    Parameters
    ----------
    api_key          : Gemini API key.
    target_behaviour : behavioural target string (e.g. "move forward").
    model_name       : Gemini model ID.
    batch_size       : max videos per request (default 10).
    debug            : print upload/parse details.
    """

    def __init__(
        self,
        api_key:          str,
        target_behaviour: str,
        model_name:       str  = MODEL,
        batch_size:       int  = 10,
        debug:            bool = False,
    ):
        self._client           = genai.Client(api_key=api_key)
        self._target_behaviour = target_behaviour
        self._model_name       = model_name
        self._batch_size       = batch_size
        self._debug            = debug

    def score_batch(
        self,
        videos: list[tuple[str, str]],
    ) -> dict[str, dict]:
        """
        Score a list of (video_id, video_path) pairs.

        If len(videos) > batch_size, splits into multiple calls automatically.
        Returns a dict mapping video_id -> parsed scoring dict (with `overall`).
        """
        results: dict[str, dict] = {}
        for chunk_start in range(0, len(videos), self._batch_size):
            chunk = videos[chunk_start : chunk_start + self._batch_size]
            results.update(self._score_chunk(chunk))
        return results

    def _score_chunk(
        self,
        videos: list[tuple[str, str]],
    ) -> dict[str, dict]:

        video_ids = [vid for vid, _ in videos]
        uploaded: list[genai_types.File] = []

        try:
            t_start = time.time()
            for video_id, path in videos:
                f = _upload_video(self._client, path, video_id, self._debug)
                uploaded.append(f)

            contents = []
            for video_id, file in zip(video_ids, uploaded):
                contents.append(f"{video_id}:")
                contents.append(
                    genai_types.Part.from_uri(file_uri=file.uri, mime_type="video/mp4")
                )
            contents.append(build_batch_video_prompt(self._target_behaviour, video_ids))
            print(f"Batch uplaoded in {time.time() - t_start} s")
            if self._debug:
                print(f"  Sending batch of {len(videos)} videos to {self._model_name}...")

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
        for video_id in video_ids:
            if video_id not in parsed:
                print(f"  WARNING: video_id '{video_id}' missing from response.")
                continue
            results[video_id] = _parse_single_video(parsed[video_id])

        return results


def score_robot_video_batch(
    video_dir:        str,
    target_behaviour: str,
    batch_size:       int  = 10,
    debug:            bool = True,
) -> dict[str, dict]:
    """
    Convenience wrapper: load all .mp4 from `video_dir` and score them in
    batches. Returns dict mapping video_id (file stem) -> parsed scoring dict.
    """
    video_paths = sorted(Path(video_dir).glob("*.mp4"))
    if not video_paths:
        print(f"No MP4 videos found in {video_dir}")
        sys.exit(1)

    videos = [(p.stem, str(p)) for p in video_paths]
    print(f"Found {len(videos)} videos in {video_dir}")

    grader = GeminiVideoBatchGrader(
        api_key          = APIKEY_GEMINI,
        target_behaviour = target_behaviour,
        batch_size       = batch_size,
        debug            = debug,
    )
    t0 = time.time()
    results = grader.score_batch(videos)
    elapsed = time.time() - t0

    print(f"\nVideo batch results ({elapsed:.1f}s total, {elapsed/len(videos):.2f}s/video):")
    for vid, out in results.items():
        print(f"  {vid:<30} overall={out.get('overall')}  "
              f"coh={out.get('coherence', {}).get('score')}  "
              f"ori={out.get('originality', {}).get('score')}  "
              f"pot={out.get('potential', {}).get('score')}")
    return results


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
    # Image batch benchmark:
    #   python gemini_batch.py [img_dir] [batch_size]
    # Video batch scoring:
    #   python gemini_batch.py video [video_dir] [batch_size]

    video_dir  = "./video/batch"
    batch_size = 10
    score_robot_video_batch(
        video_dir        = video_dir,
        target_behaviour = "move forward continuously while staying upright",
        batch_size       = batch_size,
        debug            = True,
    )

    # img_dir    = sys.argv[1] if len(sys.argv) > 1 else "./img/batch"
    # batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    # run_benchmark(img_dir, batch_size=batch_size, debug=True)