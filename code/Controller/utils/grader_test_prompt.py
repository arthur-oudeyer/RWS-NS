# ----------- Grader Tester ---------------- #
# This script allows to test the grader on robot video and prompt.
# The function test_grader(video_folder, grader, prompt_target) uses the Grader
# to evaluate each MP4 in the folder and prints a detailed readable report.
# ------------------------------------------ #
from pathlib import Path
import sys
import textwrap

_UTILS_DIR      = Path(__file__).resolve().parent
_CONTROLLER_DIR = _UTILS_DIR.parent
_CODE_DIR       = _CONTROLLER_DIR.parent

for _p in (_CONTROLLER_DIR, _CODE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from grader import LocomotionGrader, GraderOutput
import gemini_prompts as gp
from config import ExperimentConfig
from api_keys import APIKEY_GEMINI

# ---- Test parameters (edit these) ----------------------------------------
prompt      = gp.WALK_FORWARD
video_folder = "videos"   # relative to this script's directory
# --------------------------------------------------------------------------


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
        for line in text.splitlines()
    ) if text else f"{indent}(empty)"


def _print_result(video_name: str, result: GraderOutput) -> None:
    SEP  = "=" * 70
    DASH = "-" * 70

    print(f"\n{SEP}")
    print(f"  Video      : {video_name}")
    print(f"  Prompt     : {result.prompt_set}")
    print(f"  Method     : {result.method}")
    print(DASH)
    print(f"  FITNESS    : {result.fitness:.4f}")
    print(f"  coherence  : {result.raw_scores.get('coherence', 0):.2f}  "
          f"  originality : {result.raw_scores.get('originality', 0):.2f}  "
          f"  interest : {result.raw_scores.get('interest', 0):.2f}")
    print(DASH)
    print("  OBSERVATION:")
    print(_wrap(result.extra.get("observation", ""), indent="    "))
    print()
    print("  INTERPRETATION:")
    print(_wrap(result.extra.get("interpretation", ""), indent="    "))
    print()
    print("  COHERENCE  reason :", _wrap(result.extra.get("coherence_reason", ""), indent="    ").lstrip())
    print("  ORIGINALITY   reason :", _wrap(result.extra.get("originality_reason",  ""), indent="    ").lstrip())
    print("  INTEREST   reason :", _wrap(result.extra.get("interest_reason",  ""), indent="    ").lstrip())
    print(SEP)


def grader_test(
    folder_path: str,
    grader_instance: LocomotionGrader,
    prompt_target: gp.LocomotionPromptConfig,
    print_prompt: bool = False
) -> None:
    folder = Path(folder_path)
    if not folder.is_absolute():
        folder = _UTILS_DIR / folder_path

    videos = sorted(folder.glob("*.mp4"))
    if not videos:
        print(f"[grader_tester] No MP4 files found in: {folder}")
        return

    print(f"\n[grader_tester] {len(videos)} video(s) found in {folder}")
    print(f"[grader_tester] Prompt : '{prompt_target.name}' — {prompt_target.target}")

    pairs = [(v.stem, str(v)) for v in videos]
    results = grader_instance.score_batch(pairs, print_prompt=print_prompt)

    for vid_id, result in results.items():
        _print_result(vid_id, result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ExperimentConfig()

    grader = LocomotionGrader(
        api_key       = APIKEY_GEMINI,
        prompt_config = prompt,
        model_name    = cfg.gemini_model,
        batch_size    = cfg.batching,
        debug         = True,
    )

    grader_test(video_folder, grader_instance=grader, prompt_target=prompt, print_prompt=True)
