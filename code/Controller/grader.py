"""
grader.py
=========
Locomotion fitness grader — Gemini VLM scores rollout MP4s.

Mirrors `Morphology/grader.py:GeminiGrader.score_batch` but consumes
`(individual_id, mp4_path)` pairs instead of PIL images. The Files API
upload + chunking + reference-video pattern is identical; the only thing
that changes is the mime type and the per-dimension parsing.

Public API
----------
    GraderOutput     : same shape as Morphology/grader.GraderOutput.
    LocomotionGrader : Gemini-only grader. score(path)/score_batch(paths).

Fitness formula
---------------
    fitness = (w_c * coherence + w_p * progress + w_i * interest)
              / (10 * (w_c + w_p + w_i))
    → always in [0, 1]

The grader supports an optional `descriptor_config` argument so that one
batch call can return both fitness scores and MAP-Elites descriptors —
exactly like the morphology grader. We import `DescriptorConfig` from
`Morphology/descriptor.py` to avoid duplication.

Debug
-----
Run this file directly to score a single MP4 with Gemini. The script
expects the smoke-test MP4 created by `video_renderer.py` to exist; if
not, it builds one (5 k-step PPO + rollout). Set `--no-network` to skip
the actual Gemini call and only print the prompt that would be sent.
"""

from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Reuse Morphology/descriptor.py — DescriptorConfig + prompt section builder.
# We must ensure the local Controller dir stays *ahead* of Morphology on
# sys.path, otherwise Morphology/gemini_prompts.py would shadow our own.
_CTRL_DIR  = Path(__file__).resolve().parent
_MORPH_DIR = _CTRL_DIR.parent / "Morphology"
if str(_CTRL_DIR) not in sys.path:
    sys.path.insert(0, str(_CTRL_DIR))
if str(_MORPH_DIR) not in sys.path:
    sys.path.append(str(_MORPH_DIR))
try:
    from descriptor import DescriptorConfig as _DescriptorConfig
    from descriptor import build_descriptor_prompt_section as _build_desc_section
    _DESCRIPTOR_AVAILABLE = True
except Exception:
    _DESCRIPTOR_AVAILABLE = False
    _DescriptorConfig    = None  # type: ignore
    _build_desc_section  = None  # type: ignore

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

import config as cfg
from gemini_prompts import get_fake_answer, generate_fake_vlm_batch_response

# ---------------------------------------------------------------------------
# GraderOutput
# ---------------------------------------------------------------------------

@dataclass
class GraderOutput:
    """
    Result of grading one rollout video.

    Attributes
    ----------
    fitness     : weighted average of the three dimensions, in [0, 1].
    raw_scores  : dict[dim → score] for analysis (each in [0, 1]).
    method      : "gemini_video" | "gemini_video_batch".
    prompt_set  : name of the LocomotionPromptConfig used.
    extra       : observation/interpretation/per-dim reasons + vlm_descriptors.
    """
    fitness:    float
    raw_scores: dict
    method:     str
    prompt_set: str
    extra:      dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LocomotionGrader
# ---------------------------------------------------------------------------

class LocomotionGrader:
    """
    Score MP4 rollouts with Gemini.

    Parameters
    ----------
    api_key             : Gemini API key.
    prompt_config       : LocomotionPromptConfig (from gemini_prompts.py).
    model_name          : Gemini model ID (default same as Morphology).
    batch_size          : max videos per Gemini request (default 6).
    descriptor_config   : optional DescriptorConfig from Morphology/descriptor.py.
    response_log_path   : if set, every raw response is appended to this JSONL.
    upload_poll_seconds : how often to poll a PROCESSING file (default 1s;
                          videos take ~3-10s to process).
    debug               : print upload progress + parsed scores.
    """

    def __init__(
        self,
        api_key:           str,
        prompt_config:     "LocomotionPromptConfig",  # noqa: F821
        model_name:        str = "gemini-3-flash-preview",
        batch_size:        int = 6,
        descriptor_config: "Optional[_DescriptorConfig]" = None,
        response_log_path: Optional[str] = None,
        upload_poll_seconds: float = 1.0,
        debug:             bool = False,
    ):
        if not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai is required for LocomotionGrader. "
                "Install with: pip install google-genai"
            )
        self._api_key            = api_key
        self._prompt_config      = prompt_config
        self._model_name         = model_name
        self._batch_size         = batch_size
        self._descriptor_config  = descriptor_config
        self._response_log_path  = response_log_path
        self._upload_poll        = upload_poll_seconds
        self.debug               = debug

        self._client = _genai.Client(api_key=api_key)

        if debug:
            desc = descriptor_config.name if descriptor_config else "none"
            print(f"[grader] LocomotionGrader ready — model={model_name} "
                  f"prompt={prompt_config.name} target={prompt_config.target} "
                  f"batch={batch_size} descriptors={desc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_full_prompt(self, base_prompt: str) -> str:
        if self._descriptor_config is None or not _DESCRIPTOR_AVAILABLE:
            return base_prompt
        return base_prompt + _build_desc_section(self._descriptor_config)

    def _extract_vlm_descriptors(self, parsed: dict) -> dict:
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

    def _log_response(self, mode: str, ids: list, raw_text: str) -> None:
        if not self._response_log_path:
            return
        from datetime import datetime as _dt
        entry = {
            "ts":   _dt.now().isoformat(timespec="seconds"),
            "mode": mode,
            "ids":  ids,
            "raw":  raw_text,
        }
        Path(self._response_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._response_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _upload_video(self, mp4_path: str):
        """Upload one MP4 file and wait until it leaves PROCESSING state."""
        path = Path(mp4_path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {mp4_path}")
        size_kb = path.stat().st_size // 1024
        if self.debug:
            print(f"  [grader] uploading {path.name} ({size_kb} KB) ...")

        f = self._client.files.upload(
            file   = str(path),
            config = _genai_types.UploadFileConfig(mime_type="video/mp4"),
        )
        while f.state.name == "PROCESSING":
            time.sleep(self._upload_poll)
            f = self._client.files.get(name=f.name)
        if f.state.name == "FAILED":
            raise RuntimeError(f"video upload failed: {mp4_path}")
        return f

    # ------------------------------------------------------------------
    # Single-video scoring (smoke tests / debug)
    # ------------------------------------------------------------------

    def score(self, mp4_path: str, debug: Optional[bool] = None) -> GraderOutput:
        """Score one MP4 rollout. Used by the step-7 smoke test."""
        dbg = self.debug if debug is None else debug
        fake = cfg.ExperimentConfig.use_fake_grader

        if fake:
            text = get_fake_answer()
            self._log_response("single", [], text)
        else:
            video_file = self._upload_video(mp4_path)
            try:
                response = self._client.models.generate_content(
                    model    = self._model_name,
                    contents = [
                        _genai_types.Part.from_uri(
                            file_uri  = video_file.uri,
                            mime_type = "video/mp4",
                        ),
                        self._build_full_prompt(self._prompt_config.prompt),
                    ],
                )
                text = response.text
                self._log_response("single", [], text)
            finally:
                self._client.files.delete(name=video_file.name)
                if dbg:
                    print("  [grader] remote file deleted")

        parsed = self._parse_json(text)
        return self._build_grader_output(parsed, method="gemini_video", dbg=dbg)

    # ------------------------------------------------------------------
    # Batch scoring (the production path called once per generation)
    # ------------------------------------------------------------------

    def score_batch(
        self,
        videos:           "list[tuple[str, str]]",
        debug:            Optional[bool]      = None,
        reference_video:  Optional[str]       = None,
    ) -> "dict[str, GraderOutput]":
        """
        Score multiple MP4s in one (chunked) Gemini call.

        Parameters
        ----------
        videos : list of (individual_id_str, mp4_path) pairs.
        debug  : overrides self.debug for this call.
        reference_video : optional MP4 path of the current best-performing
                          individual. Prepended as a "reference" Part on every
                          chunk so Gemini can judge novelty/improvement, but
                          never scored.

        Returns
        -------
        dict[individual_id_str → GraderOutput].
        """
        dbg = self.debug if debug is None else debug
        results: dict[str, GraderOutput] = {}

        fake = cfg.ExperimentConfig.use_fake_grader

        for chunk_start in range(0, len(videos), self._batch_size):
            chunk = videos[chunk_start : chunk_start + self._batch_size]
            ids   = [vid for vid, _ in chunk]
            uploaded = []
            ref_file = None

            text = ""
            if fake:
                text = generate_fake_vlm_batch_response(ids)
                time.sleep(1)
                self._log_response("batch", ids, text)
            else:
                try:
                    if reference_video is not None:
                        ref_file = self._upload_video(reference_video)
                    for vid, mp4 in chunk:
                        uploaded.append((vid, self._upload_video(mp4)))

                    contents = []
                    if ref_file is not None:
                        contents.append("reference:")
                        contents.append(_genai_types.Part.from_uri(
                            file_uri=ref_file.uri, mime_type="video/mp4"))
                    for vid, vfile in uploaded:
                        contents.append(f"{vid}:")
                        contents.append(_genai_types.Part.from_uri(
                            file_uri=vfile.uri, mime_type="video/mp4"))
                    contents.append(self._build_batch_prompt(ids, has_reference=ref_file is not None))

                    if dbg:
                        ref_str = " + reference" if ref_file else ""
                        print(f"  [grader/batch] sending {len(chunk)} videos{ref_str} ...")

                    response = self._client.models.generate_content(
                        model    = self._model_name,
                        contents = contents,
                    )
                    text = response.text
                    self._log_response("batch", ids, text)
                finally:
                    for _vid, vfile in uploaded:
                        try:
                            self._client.files.delete(name=vfile.name)
                        except Exception:
                            pass
                    if ref_file is not None:
                        try:
                            self._client.files.delete(name=ref_file.name)
                        except Exception:
                            pass

            parsed = self._parse_json(text)
            parsed.pop("reference", None)
            for vid in ids:
                if vid not in parsed:
                    raise ValueError(
                        f"[grader/batch] missing individual id '{vid}' in response.\n"
                        f"Available keys: {list(parsed.keys())}"
                    )
                results[vid] = self._build_grader_output(
                    parsed[vid], method="gemini_video_batch", dbg=dbg
                )

        return results

    # ------------------------------------------------------------------
    # Prompt building (single + batch share the body of the prompt)
    # ------------------------------------------------------------------

    def _build_batch_prompt(self, ids: list, has_reference: bool = False) -> str:
        """
        Build the per-batch evaluation prompt. The CONTEXT + ANALYSIS body is
        taken verbatim from `self._prompt_config.prompt` and the JSON output
        schema is rewritten to require one entry per individual id.
        """
        id_list = ", ".join(ids)
        _OUTPUT_MARKER = "═══ OUTPUT FORMAT ═══"
        base = self._prompt_config.prompt
        body = base[:base.index(_OUTPUT_MARKER)].rstrip() if _OUTPUT_MARKER in base else base.rstrip()

        if self._descriptor_config and _DESCRIPTOR_AVAILABLE:
            body = body + _build_desc_section(self._descriptor_config)

        desc_schema = ""
        if self._descriptor_config and _DESCRIPTOR_AVAILABLE:
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
            '      "observation":    "frame-by-frame factual description",\n'
            '      "interpretation": "behavioural interpretation",\n'
            '      "coherence":      { "score": <int 0-100>, "reason": "..." },\n'
            '      "originality":    { "score": <int 0-100>, "reason": "..." },\n'
            '      "interest":       { "score": <int 0-100>, "reason": "..." }'
            + (f',\n{desc_schema}' if desc_schema else "")
            + "\n    }"
        )

        reference_section = ""
        if has_reference:
            reference_section = f"""
    ═══ REFERENCE VIDEO ═══

    The first video labeled "reference" shows the CURRENT BEST-PERFORMING controller
    from the previous generation. It is provided as a contextual baseline ONLY.
    — Do NOT score the reference. Do NOT include "reference" as a key in your JSON output.
    Use the reference to better identify and reward:
      • Genuine behavioural novelty (different gait pattern, posture, rhythm).
      • Real improvement (better stability, faster progress, less fall risk).
      • Interesting new motion qualities the reference does not have.
    """

        return f"""
    ═══ BATCH EVALUATION ═══

    You will evaluate {len(ids)} robot rollout videos in one pass.
    Each video was labeled before being sent: {id_list}.
    Evaluate each one independently.
    {reference_section}
    {body}

    ═══ OUTPUT FORMAT ═══
    Respond ONLY with valid JSON, no text before or after.
    The top-level keys must be exactly the individual IDs: {id_list}.
    Each value follows this schema:
    {single_schema}
    """

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]
        start = stripped.find("{")
        end   = stripped.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"[grader] no JSON in response.\nRaw:\n{text}")
        return json.loads(stripped[start:end])

    def _build_grader_output(self, parsed: dict, method: str, dbg: bool) -> GraderOutput:
        def _score(key: str) -> float:
            val = parsed.get(key, {})
            if isinstance(val, dict):
                return float(val.get("score", 0)) / 100.0
            try:
                return float(val) / 100.0
            except (TypeError, ValueError):
                return 0.0

        def _reason(key: str) -> str:
            val = parsed.get(key, {})
            return val.get("reason", "") if isinstance(val, dict) else ""

        coherence = _score("coherence")
        progress  = _score("progress")
        interest  = _score("interest")

        w = self._prompt_config.weights
        total_w = w.coherence + w.originality + w.interest
        # Each dim is now in [0,1] (we already divided by 100). The instruction
        # spec asks for fitness in [0,1] so we just take the weighted mean.
        fitness = (w.coherence * coherence + w.originality * progress + w.interest * interest) / total_w
        fitness = round(fitness, 6)

        if dbg:
            print(f"  coherence={coherence:.2f}  progress={progress:.2f}  "
                  f"interest={interest:.2f}  → fitness={fitness:.4f}")

        return GraderOutput(
            fitness    = fitness,
            raw_scores = {
                "coherence": round(coherence, 4),
                "progress":  round(progress,  4),
                "interest":  round(interest,  4),
            },
            method     = method,
            prompt_set = self._prompt_config.name,
            extra={
                "observation":      parsed.get("observation", ""),
                "interpretation":   parsed.get("interpretation", ""),
                "coherence_reason": _reason("coherence"),
                "progress_reason":  _reason("progress"),
                "interest_reason":  _reason("interest"),
                "vlm_descriptors":  self._extract_vlm_descriptors(parsed),
            },
        )


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Single-video smoke test against the real Gemini API.

    Reuses results/_smoke/rollout.mp4 from video_renderer.py. If that file
    does not exist, regenerate it via video_renderer.py first.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-network", action="store_true",
                        help="Print prompt only, do not call Gemini")
    parser.add_argument("--mp4", default=None,
                        help="Path to an MP4 to score (default: results/_smoke/rollout.mp4)")
    args = parser.parse_args()

    print("=" * 60)
    print("  grader.py — debug mode")
    print("=" * 60)

    mp4 = args.mp4 or str(Path(__file__).resolve().parent / "results" / "_smoke" / "rollout.mp4")
    if not Path(mp4).exists():
        print(f"\n  ERROR: {mp4} not found — run video_renderer.py first.")
        sys.exit(1)

    from gemini_prompts import WALK_FORWARD

    if args.no_network:
        # Print the (single-video) prompt that would be sent.
        from gemini_prompts import build_locomotion_prompt
        print("\n[no-network] Single-video prompt preview\n")
        print(WALK_FORWARD.prompt[:1200], "...")
        sys.exit(0)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from api_keys import APIKEY_GEMINI

    grader = LocomotionGrader(
        api_key       = APIKEY_GEMINI,
        prompt_config = WALK_FORWARD,
        batch_size    = 4,
        debug         = True,
    )

    print(f"\n[1] Scoring {mp4}\n")
    out = grader.score(mp4)
    print(f"\n  fitness    : {out.fitness:.4f}")
    print(f"  raw_scores : {out.raw_scores}")
    print(f"  method     : {out.method}")
    print(f"  observation : {out.extra.get('observation', '')[:120]}")
    print(f"  prog reason : {out.extra.get('progress_reason', '')[:120]}")
    assert 0.0 <= out.fitness <= 1.0, f"fitness out of [0,1]: {out.fitness}"

    print("[2] Batch prompt")

    print(grader._build_batch_prompt(["robot_0", "robot_1"], True))

    print("\nAll grader.py checks passed.")
