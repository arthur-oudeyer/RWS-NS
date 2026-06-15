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
parses a per-individual JSON of coherence / originality / potential scores into a
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
    get_reference_section,
    build_descriptor_section,
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
        n_score_request:     int = 1,
        descriptor_config         = None,
        debug:               bool = False,
    ):
        self._prompt_config     = prompt_config
        self._model_name        = model_name
        self._batch_size        = max(1, batch_size)
        self._fake              = fake
        self._response_log_path = response_log_path
        self._upload_poll       = upload_poll_seconds
        self._n_score_request   = max(1, n_score_request)
        # Optional descriptor.DescriptorConfig — when set, the VLM is asked to
        # assign each behavioural feature axis (MAP-Elites). dim names are read
        # from .feature_dims; per-dim prompt text from .items.
        self._descriptor_config = descriptor_config
        self._descriptor_dims   = (
            list(descriptor_config.feature_dims) if descriptor_config else []
        )
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

            # The VLM score is noisy: score the same batch n_score_request times
            # and average the per-dimension scores to cut variance. Each request
            # is fully independent (re-uploads + fresh generate_content) so the
            # samples are uncorrelated.
            per_request: "list[dict[str, GraderOutput]]" = []
            for attempt in range(self._n_score_request):
                if self._n_score_request > 1 and dbg:
                    print(f"  [vlm_grader] scoring attempt {attempt+1}/{self._n_score_request} "
                          f"for {ids}", flush=True)
                per_request.append(
                    self._score_chunk_once(chunk, ids, reference_video, dbg)
                )

            for vid in ids:
                results[vid] = self._average_outputs(
                    [r[vid] for r in per_request], vid
                )

        return results

    def _score_chunk_once(
        self, chunk, ids, reference_video, dbg
    ) -> "dict[str, GraderOutput]":
        """One scoring request for a chunk → {id: GraderOutput}.

        Resilient: a malformed batch response must NOT abort the whole
        experiment (a run can be hours of GPU time). Affected individuals
        degrade to fitness 0.0; the raw response is already logged.
        """
        out: dict[str, GraderOutput] = {}

        if self._fake:
            text = generate_fake_vlm_batch_response(ids, descriptor_dims=self._descriptor_dims)
            self._log_response("batch", ids, text)
        else:
            text = self._score_chunk_remote(chunk, ids, reference_video, dbg)

        try:
            parsed = self._parse_json(text)
        except Exception as e:
            print(f"[vlm_grader] WARNING: could not parse batch response "
                  f"({e}); assigning fitness 0.0 to {ids}. Raw logged.", flush=True)
            for vid in ids:
                out[vid] = self._fallback_output(f"parse_error: {e}")
            return out

        parsed.pop("reference", None)
        for vid in ids:
            entry = parsed.get(vid)
            if entry is None:
                print(f"[vlm_grader] WARNING: id '{vid}' missing from response "
                      f"(keys: {list(parsed.keys())}); fitness 0.0.", flush=True)
                out[vid] = self._fallback_output("missing_in_response")
                continue
            try:
                out[vid] = self._build_grader_output(entry, dbg)
            except Exception as e:
                print(f"[vlm_grader] WARNING: could not score '{vid}' "
                      f"({e}); fitness 0.0.", flush=True)
                out[vid] = self._fallback_output(f"score_error: {e}")
        return out

    def _average_outputs(
        self, outputs: "list[GraderOutput]", vid: str
    ) -> "GraderOutput":
        """Average the per-dimension scores across repeated scoring requests.

        Only successful requests (not the fitness-0.0 fallbacks) are averaged.
        Reasons/observations are taken from the first successful request. If
        every request failed, a single fallback is returned.
        """
        if len(outputs) == 1:
            return outputs[0]

        good = [o for o in outputs if not o.method.endswith("_failed")]
        if not good:
            return outputs[0]   # all failed → keep the (fallback) output

        keys  = ("coherence", "originality", "potential")
        means = {k: sum(o.raw_scores.get(k, 0.0) for o in good) / len(good) for k in keys}

        w = self._prompt_config.weights
        total_w = w.coherence + w.originality + w.potential
        fitness = round(
            (w.coherence * means["coherence"]
             + w.originality * means["originality"]
             + w.potential * means["potential"]) / total_w, 6)

        fitness_samples = [o.fitness for o in good]
        n = len(fitness_samples)
        mean_fit = sum(fitness_samples) / n
        std_fit = (sum((f - mean_fit) ** 2 for f in fitness_samples) / n) ** 0.5

        base = dict(good[0].extra)
        base.update({
            "n_score_request":  self._n_score_request,
            "n_scored_ok":      len(good),
            "fitness_samples":  [round(f, 4) for f in fitness_samples],
            "fitness_std":      round(std_fit, 4),
        })

        # Average the MAP-Elites descriptors per axis across successful requests
        # (each on the 0–100 scale). A dim is averaged only over the requests
        # that actually reported it.
        if self._descriptor_dims:
            desc_mean: dict = {}
            for dim in self._descriptor_dims:
                vals = [o.extra.get("vlm_descriptors", {}).get(dim)
                        for o in good]
                vals = [v for v in vals if v is not None]
                if vals:
                    desc_mean[dim] = round(sum(vals) / len(vals), 4)
            base["vlm_descriptors"] = desc_mean

        return GraderOutput(
            fitness    = fitness,
            raw_scores = {k: round(means[k], 4) for k in keys},
            method     = good[0].method + f"_mean{len(good)}",
            prompt_set = good[0].prompt_set,
            extra      = base,
        )

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

        # Optional MAP-Elites descriptor block appended to each video's schema.
        descriptor_schema = ""
        descriptor_section = ""
        if self._descriptor_dims:
            dim_lines = ",\n".join(
                f'        "{dim}":        {{ "score": <int 0-100>, "reason": "..." }}'
                for dim in self._descriptor_dims
            )
            descriptor_schema = (
                ',\n      "descriptors": {\n' + dim_lines + "\n      }"
            )
            descriptor_section = build_descriptor_section(self._descriptor_config.items)

        single_schema = (
            "{\n"
            '      "observation":    "factual description",\n'
            '      "interpretation": "behavioural interpretation",\n'
            '      "coherence":      { "score": <int 0-100>, "reason": "..." },\n'
            '      "originality":    { "score": <int 0-100>, "reason": "..." },\n'
            '      "potential":      { "score": <int 0-100>, "reason": "..." }'
            + descriptor_schema + "\n"
            "    }"
        )

        reference_section = ""
        if has_reference:
            reference_section = get_reference_section()

        return f"""
    ═══ BATCH EVALUATION ═══

    You will evaluate {len(ids)} robot rollout videos in one pass.
    Each video was labeled before being sent: {id_list}.
    Evaluate each one independently.

    {reference_section}

    {body}

    {descriptor_section}

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
        # strict=False tolerates raw control characters (unescaped newlines /
        # tabs) that Gemini sometimes emits inside string values — these would
        # otherwise raise "Invalid control character" and crash the whole run.
        return json.loads(stripped[start:end], strict=False)

    def _fallback_output(self, note: str) -> "GraderOutput":
        """Neutral result for an individual whose VLM response could not be
        parsed/scored — keeps a long run alive instead of crashing it."""
        return GraderOutput(
            fitness    = 0.0,
            raw_scores = {"coherence": 0.0, "originality": 0.0, "potential": 0.0},
            method     = ("fake" if self._fake else "gemini_video_batch") + "_failed",
            prompt_set = self._prompt_config.name,
            extra      = {"error": note, "vlm_descriptors": {}},
        )

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
        potential    = _score("potential")

        w = self._prompt_config.weights
        total_w = w.coherence + w.originality + w.potential
        fitness = round(
            (w.coherence * coherence + w.originality * originality + w.potential * potential)
            / total_w, 6)

        if dbg:
            print(f"  coherence={coherence:.2f}  originality={originality:.2f}  "
                  f"potential={potential:.2f}  → fitness={fitness:.4f}")

        # MAP-Elites descriptors: per-axis int 0–100 (kept on the 0–100 scale so
        # MapEliteArchive can bin them with edges expressed in the same range).
        vlm_descriptors: dict = {}
        descriptor_reasons: dict = {}
        if self._descriptor_dims:
            desc_block = parsed.get("descriptors", {}) or {}
            for dim in self._descriptor_dims:
                val = desc_block.get(dim)
                if val is None:
                    continue
                if isinstance(val, dict):
                    try:
                        vlm_descriptors[dim] = float(val.get("score", 0))
                    except (TypeError, ValueError):
                        continue
                    descriptor_reasons[dim] = val.get("reason", "")
                else:
                    try:
                        vlm_descriptors[dim] = float(val)
                    except (TypeError, ValueError):
                        continue
            if dbg:
                print(f"  descriptors={vlm_descriptors}")

        return GraderOutput(
            fitness    = fitness,
            raw_scores = {
                "coherence":   round(coherence, 4),
                "originality": round(originality, 4),
                "potential":    round(potential, 4),
            },
            method     = "fake" if self._fake else "gemini_video_batch",
            prompt_set = self._prompt_config.name,
            extra = {
                "observation":       parsed.get("observation", ""),
                "interpretation":    parsed.get("interpretation", ""),
                "coherence_reason":  _reason("coherence"),
                "originality_reason":_reason("originality"),
                "potential_reason":   _reason("potential"),
                "vlm_descriptors":   vlm_descriptors,
                "descriptor_reasons": descriptor_reasons,
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
