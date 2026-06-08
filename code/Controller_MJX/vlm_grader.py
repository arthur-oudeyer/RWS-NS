"""
vlm_grader.py  (MJX edition — standalone)
=========================================
Gemini VLM fitness grader for the MJX evolution loop.

Standalone reimplementation of Controller/grader.py:LocomotionGrader (the
Controller_MJX package must not import from Controller/). Drop-in alternative to
performance_grader.PerformanceGrader — same duck-typed interface consumed by
data_handler.evaluate_batch:

    grader.score_batch(videos, debug, reference_video) -> {id: GraderOutput}
        videos : list of (individual_id_str, mp4_path)

Unlike PerformanceGrader (which scores a physics quantity from the rollout info
at render time), the VLM grader scores the *rendered video* itself: it uploads
each MP4 via the Gemini Files API, sends one batched generate_content call, and
parses a per-individual JSON of coherence / originality / interest scores into a
fitness ∈ [0, 1]. No `register()` step is needed.

Set fake=True to exercise the whole pipeline with synthetic responses and no
network / API cost.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from gemini_prompts import (
    LocomotionPromptConfig,
    OUTPUT_MARKER,
    get_fake_answer,
    generate_fake_vlm_batch_response,
)

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ---------------------------------------------------------------------------
# GraderOutput (duck-typed match for performance_grader.GraderOutput)
# ---------------------------------------------------------------------------

@dataclass
class GraderOutput:
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
    api_key       : Gemini API key.
    prompt_config : LocomotionPromptConfig (target behaviour + scoring weights).
    model_name    : Gemini model ID.
    batch_size    : max videos per Gemini request.
    fake          : if True, return synthetic responses (no upload / no API call).
    response_log_path : if set, every raw response is appended to this JSONL.
    upload_poll_seconds : poll interval while an uploaded file is PROCESSING.
    debug         : print upload progress + parsed scores.
    """

    def __init__(
        self,
        api_key:             str,
        prompt_config:       LocomotionPromptConfig,
        model_name:          str = "gemini-3-flash-preview",
        batch_size:          int = 6,
        fake:                bool = False,
        response_log_path:   Optional[str] = None,
        upload_poll_seconds: float = 1.0,
        debug:               bool = False,
    ):
        self._prompt_config     = prompt_config
        self._model_name        = model_name
        self._batch_size        = max(1, batch_size)
        self._fake              = fake
        self._response_log_path = response_log_path
        self._upload_poll       = upload_poll_seconds
        self.debug              = debug

        if not fake:
            if not _GEMINI_AVAILABLE:
                raise ImportError(
                    "google-genai is required for LocomotionGrader (non-fake mode). "
                    "Install with: pip install google-genai"
                )
            self._client = _genai.Client(api_key=api_key)
        else:
            self._client = None

        if debug:
            print(f"[vlm_grader] ready — model={model_name} prompt={prompt_config.name} "
                  f"target={prompt_config.target!r} batch={batch_size} fake={fake}")

    # ------------------------------------------------------------------
    # Files API helpers
    # ------------------------------------------------------------------

    def _upload_video(self, mp4_path: str):
        path = Path(mp4_path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {mp4_path}")
        if self.debug:
            print(f"  [vlm_grader] uploading {path.name} ({path.stat().st_size // 1024} KB) …")
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

    def _log_response(self, mode: str, ids: list, raw_text: str) -> None:
        if not self._response_log_path:
            return
        from datetime import datetime as _dt
        entry = {"ts": _dt.now().isoformat(timespec="seconds"),
                 "mode": mode, "ids": ids, "raw": raw_text}
        Path(self._response_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._response_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Grader interface (consumed by data_handler.evaluate_batch)
    # ------------------------------------------------------------------

    def score_batch(
        self,
        videos:          "list[tuple[str, str]]",
        debug:           Optional[bool] = None,
        reference_video: Optional[str]  = None,
    ) -> "dict[str, GraderOutput]":
        dbg = self.debug if debug is None else debug
        results: dict[str, GraderOutput] = {}

        for start in range(0, len(videos), self._batch_size):
            chunk = videos[start : start + self._batch_size]
            ids   = [vid for vid, _ in chunk]

            if self._fake:
                text = generate_fake_vlm_batch_response(ids)
                self._log_response("batch", ids, text)
            else:
                text = self._score_chunk_remote(chunk, ids, reference_video, dbg)

            parsed = self._parse_json(text)
            parsed.pop("reference", None)
            for vid in ids:
                if vid not in parsed:
                    raise ValueError(
                        f"[vlm_grader] missing individual id '{vid}' in response.\n"
                        f"Available keys: {list(parsed.keys())}"
                    )
                results[vid] = self._build_grader_output(parsed[vid], dbg)

        return results

    def _score_chunk_remote(self, chunk, ids, reference_video, dbg) -> str:
        uploaded = []
        ref_file = None
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
                print(f"  [vlm_grader/batch] sending {len(chunk)} videos{ref_str} …")

            response = self._client.models.generate_content(
                model=self._model_name, contents=contents)
            text = response.text
            self._log_response("batch", ids, text)
            return text
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

    # ------------------------------------------------------------------
    # Prompt building / parsing
    # ------------------------------------------------------------------

    def _build_batch_prompt(self, ids: list, has_reference: bool = False) -> str:
        id_list = ", ".join(ids)
        base = self._prompt_config.prompt
        body = base[:base.index(OUTPUT_MARKER)].rstrip() if OUTPUT_MARKER in base else base.rstrip()

        single_schema = (
            "{\n"
            '      "observation":    "frame-by-frame factual description",\n'
            '      "interpretation": "behavioural interpretation",\n'
            '      "coherence":      { "score": <int 0-100>, "reason": "..." },\n'
            '      "originality":    { "score": <int 0-100>, "reason": "..." },\n'
            '      "interest":       { "score": <int 0-100>, "reason": "..." }\n'
            "    }"
        )

        reference_section = ""
        if has_reference:
            reference_section = """
    ═══ REFERENCE VIDEO ═══

    The first video labeled "reference" shows the CURRENT BEST-PERFORMING controller
    from the previous generation. It is provided as a contextual baseline ONLY.
    — Do NOT score the reference. Do NOT include "reference" as a key in your JSON output.
    Use the reference to better identify and reward genuine behavioural novelty and real
    improvement over it.
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

    @staticmethod
    def _parse_json(text: str) -> dict:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]
        start = stripped.find("{")
        end   = stripped.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"[vlm_grader] no JSON in response.\nRaw:\n{text}")
        return json.loads(stripped[start:end])

    def _build_grader_output(self, parsed: dict, dbg: bool) -> GraderOutput:
        def _score(key: str) -> float:
            val = parsed.get(key, {})
            if isinstance(val, dict):
                s = float(val.get("score", 0))
            else:
                try:
                    s = float(val)
                except (TypeError, ValueError):
                    s = 0.0
            return s / 100.0   # dimensions arrive on a 0–100 scale → [0, 1]

        def _reason(key: str) -> str:
            val = parsed.get(key, {})
            return val.get("reason", "") if isinstance(val, dict) else ""

        coherence   = _score("coherence")
        originality = _score("originality")
        interest    = _score("interest")

        w = self._prompt_config.weights
        total_w = w.coherence + w.originality + w.interest
        fitness = round(
            (w.coherence * coherence + w.originality * originality + w.interest * interest)
            / total_w, 6)

        if dbg:
            print(f"  coherence={coherence:.2f}  originality={originality:.2f}  "
                  f"interest={interest:.2f}  → fitness={fitness:.4f}")

        return GraderOutput(
            fitness    = fitness,
            raw_scores = {
                "coherence":   round(coherence, 4),
                "originality": round(originality, 4),
                "interest":    round(interest, 4),
            },
            method     = "fake" if self._fake else "gemini_video_batch",
            prompt_set = self._prompt_config.name,
            extra = {
                "observation":       parsed.get("observation", ""),
                "interpretation":    parsed.get("interpretation", ""),
                "coherence_reason":  _reason("coherence"),
                "originality_reason":_reason("originality"),
                "interest_reason":   _reason("interest"),
                "vlm_descriptors":   {},   # MAP-Elites descriptors not wired yet
            },
        )


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from gemini_prompts import ROTATE
    print("=" * 60)
    print("  vlm_grader.py — debug mode (fake, no network)")
    print("=" * 60)
    g = LocomotionGrader(api_key="", prompt_config=ROTATE, fake=True, debug=True)
    out = g.score_batch([("robot_0", "/tmp/a.mp4"), ("robot_1", "/tmp/b.mp4")])
    for vid, go in out.items():
        print(f"  {vid}: fitness={go.fitness:.4f}  raw={go.raw_scores}  method={go.method}")
    assert all(0.0 <= o.fitness <= 1.0 for o in out.values())
    print("\nAll vlm_grader.py (fake) checks passed.")
