"""
prompt_tester.py
================
Interactive tool for testing CLIP prompt sensitivity before running
a full evolution experiment.

Workflow
--------
1. Run a generation script that saves rendered morphology PNGs into
   ./prompt_testing_source/   (any filename, any subfolder layout).
2. Hand-sort the images into subfolders — each subfolder name is the
   "source label" you pass to this tool:
       prompt_testing_source/
           spider_good/   ← images you want to reward
           spider_bad/    ← images you want to penalise
3. Run the tester:
       python prompt_tester.py --pos spider_good --neg spider_bad
   Or call from a notebook:
       run_sensitivity_test(["spider_good"], ["spider_bad"])

Source resolution
-----------------
A source string is resolved in this order:
  1. Literal path to an existing file  → load that single image.
  2. Literal path to an existing dir   → load all PNG/JPG inside.
  3. Name of a subfolder in SOURCE_ROOT (./prompt_testing_source/)
     → load all PNG/JPG inside that subfolder.
  Raises ValueError if none of the above matches.

Report modes
------------
Full (default)  — three sections:
  • FITNESS BY IMAGE   (all images, sorted worst → best)
  • PER-PROMPT SCORES  (per-image prompt breakdown)
  • PROMPT DISCRIMINATION  (which prompts separate pos/neg)

Compact         — two sections only, ideal when you have many images:
  • FITNESS BY IMAGE   (same, all images sorted)
  • PROMPT DISCRIMINATION  (top N prompts only, default 5)
  The verbose per-image prompt breakdown is skipped.

CLI
---
    python prompt_tester.py \\
        --pos spider_good \\
        --neg spider_bad \\
        --sets spider_body many_legs \\
        --method cosine \\
        --compact \\
        --top-n 5
"""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

os.environ["HUGGINGFACE_HUB_CACHE"] = "/Volumes/T7_AO/clip-models"

# Default root for hand-sorted source images
SOURCE_ROOT = Path(__file__).parent / "prompt_testing_source"

# ---------------------------------------------------------------------------
# Lazy / conditional imports
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    import open_clip
    _CLIP_OK = True
except ImportError:
    _CLIP_OK = False

try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

from prompts import PromptSet, ALL_PROMPT_SETS, get_prompt_set


# ---------------------------------------------------------------------------
# Report configuration
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """
    Controls how much detail the printed report shows.

    Attributes
    ----------
    compact      : if True, skip the per-image per-prompt breakdown section.
                   Useful when you have many source images.
    top_n_prompts: in compact mode, limit the PROMPT DISCRIMINATION section
                   to the N most discriminative prompts (0 = show all).
    """
    compact:       bool = False
    top_n_prompts: int  = 0   # 0 = show all prompts


# ---------------------------------------------------------------------------
# CLIP engine (loaded once, reused across calls)
# ---------------------------------------------------------------------------

_ENGINE: Optional["_CLIPEngine"] = None


class _CLIPEngine:
    """Load CLIP once; encode images and texts on demand."""

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        if not _TORCH_OK:
            raise ImportError("torch is required.")
        if not _CLIP_OK:
            raise ImportError("open_clip is required.")
        if not _PIL_OK:
            raise ImportError("Pillow is required.")

        import torch

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"[prompt_tester] Loading CLIP {model_name} ({pretrained}) on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained,
            cache_dir  = os.environ.get("HUGGINGFACE_HUB_CACHE", "~/.cache"),
            quick_gelu = True,
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"[prompt_tester] CLIP ready.\n")

    def encode_image(self, pil_img: "PILImage.Image"):
        import torch
        t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0]  # (D,)

    def encode_texts(self, texts: list[str]):
        import torch
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat  # (N, D)


def _get_engine() -> _CLIPEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _CLIPEngine()
    return _ENGINE


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")


def _load_dir(folder: Path, label: str) -> list[tuple["PILImage.Image", str]]:
    """Load all images from a directory; label = 'foldername/filestem'."""
    imgs = []
    for ext in _IMG_EXTS:
        imgs.extend(sorted(folder.glob(ext)))
    if not imgs:
        print(f"  [warn] No images found in: {folder}")
    pairs = []
    for p in imgs:
        img = PILImage.open(p).convert("RGB")
        pairs.append((img, f"{label}/{p.stem}"))
    return pairs


def load_sources(
    sources:     list[str],
    source_root: Path = SOURCE_ROOT,
) -> list[tuple["PILImage.Image", str]]:
    """
    Resolve each source string and return (PIL Image, label) pairs.

    Resolution order per source string:
      1. Existing file path  → single image, label = stem.
      2. Existing directory  → all PNG/JPG, label = 'dir/stem'.
      3. Subfolder of source_root → all PNG/JPG, label = 'name/stem'.
      Raises ValueError if none matches.
    """
    results = []
    for src in sources:
        p = Path(src)

        if p.is_file():
            img = PILImage.open(p).convert("RGB")
            results.append((img, p.stem))
            continue

        if p.is_dir():
            results.extend(_load_dir(p, p.name))
            continue

        # Try resolving under source_root
        rooted = source_root / src
        if rooted.is_dir():
            print(f"  Loading from source_root: {rooted} …")
            results.extend(_load_dir(rooted, src))
            continue

        raise ValueError(
            f"Cannot resolve source '{src}'.\n"
            f"  Checked: {p} (file/dir) and {rooted} (in SOURCE_ROOT).\n"
            f"  Available in SOURCE_ROOT: "
            + (", ".join(d.name for d in sorted(source_root.iterdir()) if d.is_dir())
               if source_root.exists() else f"(SOURCE_ROOT not found: {source_root})")
        )

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_images(
    images:         list[tuple["PILImage.Image", str]],
    prompt_set:     PromptSet,
    scoring_method: str = "cosine",
) -> list[dict]:
    """
    Score a list of (image, label) pairs against a PromptSet.

    Returns a list of result dicts:
        label, fitness, raw_scores, per_prompt (sorted by score desc)
    """
    engine    = _get_engine()
    all_texts = prompt_set.all_texts()
    n_pos     = len(prompt_set.positive)

    txt_feats = engine.encode_texts(all_texts)  # (N_texts, D)

    results = []
    for img, label in images:
        img_feat = engine.encode_image(img)
        raw_sims = (img_feat @ txt_feats.T).cpu().float().tolist()

        if scoring_method == "softmax":
            import torch as _torch
            scores = _torch.tensor(raw_sims).softmax(dim=-1).tolist()
        else:
            scores = raw_sims

        raw_scores = {text: round(s, 5) for text, s in zip(all_texts, scores)}

        pos_scores = scores[:n_pos]
        neg_scores = scores[n_pos:]
        pos_w      = [p.weight for p in prompt_set.positive]
        neg_w      = [p.weight for p in prompt_set.negative]

        fitness = (
              sum(w * s for w, s in zip(pos_w, pos_scores)) / max(len(pos_w), 1)
            - sum(w * s for w, s in zip(neg_w, neg_scores)) / max(len(neg_w), 1)
        )

        per_prompt = []
        for p in prompt_set.positive:
            per_prompt.append({"text": p.text, "weight": p.weight,
                                "score": raw_scores[p.text], "role": "pos"})
        for p in prompt_set.negative:
            per_prompt.append({"text": p.text, "weight": p.weight,
                                "score": raw_scores[p.text], "role": "neg"})
        per_prompt.sort(key=lambda x: -x["score"])

        results.append({
            "label":      label,
            "fitness":    round(fitness, 5),
            "raw_scores": raw_scores,
            "per_prompt": per_prompt,
        })

    return results


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

class C:
    """Terminal ANSI colour codes."""
    RESET      = "\033[0m"
    RED        = "\033[31m"
    GREEN      = "\033[32m"
    YELLOW     = "\033[33m"
    BLUE       = "\033[34m"
    WHITE_BOLD = "\033[1;97m"

    @staticmethod
    def red(s):        return f"{C.RED}{s}{C.RESET}"
    @staticmethod
    def green(s):      return f"{C.GREEN}{s}{C.RESET}"
    @staticmethod
    def yellow(s):     return f"{C.YELLOW}{s}{C.RESET}"
    @staticmethod
    def blue(s):       return f"{C.BLUE}{s}{C.RESET}"
    @staticmethod
    def white_bold(s): return f"{C.WHITE_BOLD}{s}{C.RESET}"


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

_BAR_WIDTH = 28


def _bar(score: float, lo: float = -0.1, hi: float = 0.4) -> str:
    frac   = max(0.0, min(1.0, (score - lo) / (hi - lo)))
    filled = int(frac * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def print_report(
    positive_results: list[dict],
    negative_results: list[dict],
    prompt_set:       PromptSet,
    scoring_method:   str         = "cosine",
    cfg:              ReportConfig = None,
) -> None:
    """Pretty-print the coloured sensitivity report to stdout."""
    if cfg is None:
        cfg = ReportConfig()

    pos_ids     = {id(r) for r in positive_results}
    all_results = positive_results + negative_results
    w           = 72

    print("\n" + "═" * w)
    print(f"  PROMPT SENSITIVITY REPORT — {C.white_bold(prompt_set.name)}"
          + ("  [compact]" if cfg.compact else ""))
    print(f"  method={scoring_method}   "
          f"{len(prompt_set.positive)} pos-prompts / {len(prompt_set.negative)} neg-prompts   "
          f"{len(positive_results)} pos-images / {len(negative_results)} neg-images")
    print("═" * w)

    # ── Section 1: fitness table (always shown, sorted worst → best) ─────────
    print(f"\n  {C.white_bold('FITNESS BY IMAGE')}  (best → worst)\n")
    print(f"  {'label':<28}  {'role':<8}  {'fitness':>8}  bar")
    print(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*_BAR_WIDTH}")

    for r in sorted(all_results, key=lambda x: -x["fitness"]):
        is_pos  = id(r) in pos_ids
        col     = C.green if is_pos else C.red
        tag     = "[+pos]" if is_pos else "[-neg]"
        bar_str = _bar(r["fitness"], lo=-0.5, hi=1.5)
        lbl     = "{:<28}".format(r["label"][:28])
        tag_f   = "{:<8}".format(tag)
        fit_f   = "{:>+8.4f}".format(r["fitness"])
        print(f"  {col(lbl)}  {col(tag_f)}  {col(fit_f)}  {col(bar_str)}")

    # ── Sensitivity gap + verdict (always shown) ──────────────────────────────
    if positive_results and negative_results:
        mean_pos = sum(r["fitness"] for r in positive_results) / len(positive_results)
        mean_neg = sum(r["fitness"] for r in negative_results) / len(negative_results)
        gap      = mean_pos - mean_neg
        if gap > 0.1:
            verdict_str = C.green("GOOD  — prompts discriminate well")
        elif gap > 0:
            verdict_str = C.yellow("WEAK  — prompts barely separate pos/neg")
        else:
            verdict_str = C.red("BAD   — prompts score negatives higher than positives!")
        print(f"\n  Sensitivity gap  :  mean_pos={mean_pos:+.4f}  "
              f"mean_neg={mean_neg:+.4f}  gap={gap:+.4f}")
        print(f"  {C.white_bold('Verdict')}          :  {verdict_str}")

    # ── Section 2: per-image prompt breakdown (full mode only) ───────────────
    if not cfg.compact:
        print(f"\n{'─' * w}")
        print(f"  {C.white_bold('PER-PROMPT SCORES')}  (sorted by score, each image)\n")

        for r in all_results:
            is_pos   = id(r) in pos_ids
            tag_col  = C.green if is_pos else C.red
            role_tag = "[+]" if is_pos else "[-]"
            print(f"  {tag_col(role_tag + ' ' + r['label'])}")
            for pp in r["per_prompt"]:
                bar_str  = _bar(pp["score"])
                role_sym = "+" if pp["role"] == "pos" else "-"
                tag      = (C.yellow(f"[{role_sym}]") if pp["role"] == "pos"
                            else C.blue(f"[{role_sym}]"))
                print(f"      {tag} {pp['score']:>+.4f} (w={pp['weight']:.1f})  "
                      f"{bar_str}  {pp['text'][:55]}")
            print()

    # ── Section 3: prompt discrimination (always shown, limited in compact) ──
    if positive_results and negative_results:
        print(f"{'─' * w}")
        n_shown  = cfg.top_n_prompts if cfg.compact and cfg.top_n_prompts > 0 else None
        hdr_note = f"  (top {n_shown})" if n_shown else ""
        print(f"  {C.white_bold('PROMPT DISCRIMINATION')}{hdr_note}"
              f"  — score gap  pos − neg per prompt\n")

        all_texts = [pp["text"] for pp in positive_results[0]["per_prompt"]]
        gaps      = {}
        for text in all_texts:
            pos_s = [r["raw_scores"][text] for r in positive_results]
            neg_s = [r["raw_scores"][text] for r in negative_results]
            gaps[text] = sum(pos_s) / len(pos_s) - sum(neg_s) / len(neg_s)

        role_map = {pp["text"]: pp["role"] for pp in positive_results[0]["per_prompt"]}
        ranked   = sorted(gaps.items(), key=lambda x: -abs(x[1]))
        if n_shown:
            ranked = ranked[:n_shown]

        for text, gap_val in ranked:
            role     = role_map[text]
            role_sym = "+" if role == "pos" else "-"
            tag      = (C.yellow(f"[{role_sym}]") if role == "pos"
                        else C.blue(f"[{role_sym}]"))
            is_good  = (gap_val > 0 and role == "pos") or (gap_val < 0 and role == "neg")
            is_neutral = abs(gap_val) < 0.02
            direction = C.white_bold("~ weak") if is_neutral else (C.green("↑ good") if is_good else C.red("↓ weak"))
            print(f"  {tag}  gap={gap_val:>+.4f}  {direction}  {text[:62]}")

    print("\n" + "═" * w + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sensitivity_test(
    positive_sources: list[str],
    negative_sources: list[str]      = None,
    prompt_sets:      list[PromptSet] = None,
    scoring_method:   str             = "cosine",
    cfg:              ReportConfig    = None,
    source_root:      Path            = SOURCE_ROOT,
) -> None:
    """
    Full sensitivity test: load images → score → print report.

    Parameters
    ----------
    positive_sources : source strings for "good" / target morphologies.
    negative_sources : source strings for "bad" / counter-examples.
    prompt_sets      : PromptSet objects to test (default: SPIDER_BODY).
    scoring_method   : "cosine" or "softmax".
    cfg              : ReportConfig controlling output detail level.
    source_root      : root directory for named source folders
                       (default: ./prompt_testing_source/).
    """
    from prompts import SPIDER_BODY

    if prompt_sets is None:
        prompt_sets = [SPIDER_BODY]
    if negative_sources is None:
        negative_sources = []
    if cfg is None:
        cfg = ReportConfig()

    print("[prompt_tester] Loading positive images …")
    pos_images = load_sources(positive_sources, source_root)
    print(f"  → {len(pos_images)} image(s).")

    neg_images = []
    if negative_sources:
        print("[prompt_tester] Loading negative images …")
        neg_images = load_sources(negative_sources, source_root)
        print(f"  → {len(neg_images)} image(s).")

    for ps in prompt_sets:
        print(f"\n[prompt_tester] Scoring against: {ps.name} …")
        all_images  = pos_images + neg_images
        all_results = score_images(all_images, ps, scoring_method)
        pos_results = all_results[:len(pos_images)]
        neg_results = all_results[len(pos_images):]
        print_report(pos_results, neg_results, ps, scoring_method, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Test CLIP prompt sensitivity on hand-sorted PNG folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Source folders are looked up in ./prompt_testing_source/ by default.\n"
            "Example:\n"
            "  python prompt_tester.py --pos spider_good --neg spider_bad\n"
            "  python prompt_tester.py --pos spider_good --neg spider_bad "
            "--compact --top-n 5"
        ),
    )
    parser.add_argument("--pos", nargs="+", default=[], metavar="SRC",
                        help="Positive source folder names or file paths.")
    parser.add_argument("--neg", nargs="+", default=[], metavar="SRC",
                        help="Negative source folder names or file paths.")
    parser.add_argument("--sets", nargs="+", default=["spider_body"], metavar="SET",
                        help=f"Prompt set names. Available: {list(ALL_PROMPT_SETS.keys())}")
    parser.add_argument("--method", choices=["cosine", "softmax"], default="cosine",
                        help="CLIP scoring method (default: cosine).")
    parser.add_argument("--compact", action="store_true",
                        help="Compact output: skip per-image prompt breakdown.")
    parser.add_argument("--top-n", type=int, default=0, dest="top_n",
                        help="In compact mode, show only top N discrimination prompts "
                             "(0 = all, default 5).")
    parser.add_argument("--source-root", default=str(SOURCE_ROOT), dest="source_root",
                        help=f"Root folder for named source dirs (default: {SOURCE_ROOT}).")
    args = parser.parse_args()

    prompt_sets = [get_prompt_set(name) for name in args.sets]
    cfg         = ReportConfig(compact=args.compact, top_n_prompts=args.top_n)

    if not args.pos:
        print("[prompt_tester] No --pos given.  Pass --pos <folder_name> to get started.\n ex: python3 prompt_tester.py --pos <positive png folder> --neg <negative png folder> --sets <sets to test> --compact --top_n <nb of prompt discrimination shown, 0 for all>")
        print(f"  SOURCE_ROOT = {SOURCE_ROOT}")
        if SOURCE_ROOT.exists():
            subfolders = [d.name for d in sorted(SOURCE_ROOT.iterdir()) if d.is_dir()]
            print(f"  Available folders: {subfolders or '(none yet)'}")
        return

    run_sensitivity_test(
        positive_sources = args.pos,
        negative_sources = args.neg,
        prompt_sets      = prompt_sets,
        scoring_method   = args.method,
        cfg              = cfg,
        source_root      = Path(args.source_root),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _cli()
