"""
utils/morph_human_evaluation.py
================================
Interactive morphology evaluator for building a human-annotated dataset.

For each generated morphology the annotator can:
  - Set a static target  (what it should look like, e.g. "spider")
  - Set a dynamic target (what it should do,  e.g. "move forward")
  - Score 3 criteria on a 0–10 scale:
      coherence   – how well it matches the static target
      originality – structural novelty
      interest    – evolutionary potential
  - Save  [S]  → records image + morphology + targets + scores to the dataset
  - Skip  [Space] → discard and move on
  - Mutate [M] → mutate the current morphology in place
  - Back   [B / ←] → restore the previous morphology
  - Quit   [Q]

Output  (relative to this script's parent directory)
------
    human_eval_dataset/
        images/       ← rendered PNGs (one per saved morphology)
        dataset.json  ← JSON array of annotated entries
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk

try:
    from PIL import Image as PILImage, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---- project imports -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from morphology import (
    RobotMorphology, morphology_to_dict,
    compute_spawn_height, NewMorph, MutateMorphology,
)
from rendering import MorphologyRenderer, RenderConfig, CameraView
from config import ExperimentConfig


# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

_cfg = ExperimentConfig()

_SCRIPT_DIR   = Path(__file__).parent
_DATASET_DIR  = _SCRIPT_DIR / "human_eval_dataset"
_IMAGES_DIR   = _DATASET_DIR / "images"
_DATASET_FILE = _DATASET_DIR / "dataset.json"

_DEFAULT_STATIC_TARGET  = "spider"
_DEFAULT_DYNAMIC_TARGET = "move forward continuously while staying upright"


# ---------------------------------------------------------------------------
# Render / generation parameters
# ---------------------------------------------------------------------------

@dataclass
class GenParams:
    n_legs_min:      int   = _cfg.init_n_legs_min
    n_legs_max:      int   = _cfg.init_n_legs_max
    render_size:     int   = _cfg.render_width
    floor_clearance: float = _cfg.floor_clearance
    photorealistic:  bool  = _cfg.photorealistic


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_dataset() -> list:
    if _DATASET_FILE.exists():
        with open(_DATASET_FILE) as f:
            return json.load(f)
    return []


def _save_dataset(data: list) -> None:
    _DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DATASET_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _next_id(data: list) -> int:
    if not data:
        return 1
    indices = [
        int(e["id"].split("_")[-1])
        for e in data
        if e.get("id", "").split("_")[-1].isdigit()
    ]
    return (max(indices) + 1) if indices else 1


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

_HISTORY_MAX = 20


class MorphEvalApp:
    """
    tkinter application for human annotation of robot morphologies.

    The annotator sets free-text targets and three 0-10 sliders per
    morphology, then saves or skips.  Each saved entry is immediately
    appended to dataset.json (no buffering).
    """

    _CANVAS_W = 700
    _CANVAS_H = 420
    _PANEL_W  = 290

    _BG       = "#1e1e1e"
    _BG2      = "#252525"
    _BG3      = "#2d2d2d"
    _FG       = "#cccccc"
    _FG_DIM   = "#888888"
    _SAVE_BG  = "#2d5a2d"
    _SAVE_FG  = "#ccffcc"
    _SKIP_BG  = "#3a3a3a"
    _MUT_BG   = "#2d4a6a"
    _MUT_FG   = "#cce0ff"
    _PREV_BG  = "#4a3a2d"
    _PREV_FG  = "#ffe0cc"

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.rng    = np.random.default_rng()
        self.params = GenParams()

        self._dataset: list = _load_dataset()
        self._saved_count   = len(self._dataset)
        self._skip_count    = 0

        self._current_morph: Optional[RobotMorphology]    = None
        self._current_image: Optional[PILImage.Image]     = None
        self._photo:         Optional[ImageTk.PhotoImage] = None
        self._history: list[tuple[RobotMorphology, PILImage.Image]] = []

        self._renderer:      Optional[MorphologyRenderer] = None
        self._renderer_size: int = 0

        self._build_ui()
        self._init_renderer()
        self.root.after(100, self._do_generate)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.title("Morphology Human Evaluator")
        self.root.configure(bg=self._BG)
        self.root.resizable(True, True)

        self._status_var = tk.StringVar(value="  Starting…")
        tk.Label(
            self.root, textvariable=self._status_var,
            bg="#2a2a2a", fg=self._FG_DIM, anchor="w", padx=10,
            font=("Courier", 11),
        ).pack(side=tk.TOP, fill=tk.X)

        content = tk.Frame(self.root, bg=self._BG)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── Left: canvas + buttons ──────────────────────────────────────
        left = tk.Frame(content, bg=self._BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._canvas = tk.Canvas(
            left, bg="#2a2a2a",
            width=self._CANVAS_W, height=self._CANVAS_H,
            highlightthickness=1, highlightbackground="#444",
        )
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._canvas.create_text(
            self._CANVAS_W // 2, self._CANVAS_H // 2,
            text="Generating…", fill=self._FG_DIM, font=("Courier", 15),
        )

        btn_cfg = dict(
            font=("Helvetica", 13, "bold"),
            relief=tk.FLAT, cursor="hand2",
            padx=10, pady=9,
        )

        # Navigation row
        nav_frame = tk.Frame(left, bg=self._BG, pady=3)
        nav_frame.pack(side=tk.TOP, fill=tk.X)
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)

        self._btn_prev = tk.Button(
            nav_frame, text="← Back  [B]",
            bg=self._PREV_BG, fg=self._PREV_FG,
            activebackground="#6a5a3a", activeforeground="#ffffff",
            command=self._on_previous, **btn_cfg,
        )
        self._btn_mutate = tk.Button(
            nav_frame, text="⟳ Mutate  [M]",
            bg=self._MUT_BG, fg=self._MUT_FG,
            activebackground="#3a6a9a", activeforeground="#ffffff",
            command=self._on_mutate, **btn_cfg,
        )
        self._btn_prev.grid(  row=0, column=0, sticky="ew", padx=3)
        self._btn_mutate.grid(row=0, column=1, sticky="ew", padx=3)

        # Action row
        action_frame = tk.Frame(left, bg=self._BG, pady=3)
        action_frame.pack(side=tk.TOP, fill=tk.X)
        action_frame.columnconfigure(0, weight=2)
        action_frame.columnconfigure(1, weight=1)

        self._btn_save = tk.Button(
            action_frame, text="✓  Save  [S]",
            bg=self._SAVE_BG, fg=self._SAVE_FG,
            activebackground="#3a8a3a", activeforeground="#ffffff",
            command=self._on_save, **btn_cfg,
        )
        self._btn_skip = tk.Button(
            action_frame, text="↷  Skip  [Space]",
            bg=self._SKIP_BG, fg=self._FG,
            activebackground="#555", activeforeground="#fff",
            command=self._on_skip, **btn_cfg,
        )
        self._btn_save.grid(row=0, column=0, sticky="ew", padx=3)
        self._btn_skip.grid(row=0, column=1, sticky="ew", padx=3)

        # Keyboard shortcuts — suppressed when an Entry widget has focus
        def _guarded(f):
            def handler(e):
                if not isinstance(self.root.focus_get(), tk.Entry):
                    f()
            return handler

        for key, cb in (
            ("<s>", self._on_save),     ("<S>", self._on_save),
            ("<space>", self._on_skip),
            ("<m>", self._on_mutate),   ("<M>", self._on_mutate),
            ("<b>", self._on_previous), ("<B>", self._on_previous),
            ("<Left>", self._on_previous),
            ("<q>", lambda: self.root.destroy()),
            ("<Q>", lambda: self.root.destroy()),
        ):
            self.root.bind(key, _guarded(cb))

        # ── Right: parameter + evaluation panel ────────────────────────
        right = tk.Frame(content, bg=self._BG2, width=self._PANEL_W)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        right.pack_propagate(False)
        self._build_right_panel(right)
        self._set_buttons_enabled(False)

    # ------------------------------------------------------------------

    def _build_right_panel(self, parent: tk.Frame):
        _canvas = tk.Canvas(parent, bg=self._BG2, highlightthickness=0)
        _scroll = tk.Scrollbar(parent, orient=tk.VERTICAL, command=_canvas.yview)
        p = tk.Frame(_canvas, bg=self._BG2)

        p.bind("<Configure>",
               lambda e: _canvas.configure(scrollregion=_canvas.bbox("all")))
        _canvas.create_window((0, 0), window=p, anchor="nw", width=self._PANEL_W - 16)
        _canvas.configure(yscrollcommand=_scroll.set)
        _scroll.pack(side=tk.RIGHT, fill=tk.Y)
        _canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _wheel(e):
            if e.num == 4:   _canvas.yview_scroll(-1, "units")
            elif e.num == 5: _canvas.yview_scroll(1,  "units")
            else:            _canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        for w in (_canvas, p):
            w.bind("<MouseWheel>", _wheel)
            w.bind("<Button-4>",   _wheel)
            w.bind("<Button-5>",   _wheel)

        def section(text):
            tk.Label(p, text=text, bg=self._BG2, fg="#aaaaaa",
                     font=("Helvetica", 11, "bold")).pack(pady=(10, 3))

        def separator():
            tk.Frame(p, bg="#444444", height=1).pack(fill=tk.X, padx=10, pady=6)

        # ── Targets ──
        section("Targets")

        def _unfocus(e):
            self._canvas.focus_set()

        tk.Label(p, text="Static target  (looks like…)", bg=self._BG2, fg="#bbbbbb",
                 font=("Helvetica", 9), anchor="w").pack(fill=tk.X, padx=10)
        self._static_target_var = tk.StringVar(value=_DEFAULT_STATIC_TARGET)
        static_entry = tk.Entry(
            p, textvariable=self._static_target_var,
            bg=self._BG3, fg="#ffffff", insertbackground="#ffffff",
            relief=tk.FLAT, font=("Courier", 10), bd=4,
        )
        static_entry.pack(fill=tk.X, padx=10, pady=(0, 8))
        static_entry.bind("<Return>",  _unfocus)
        static_entry.bind("<Escape>",  _unfocus)

        tk.Label(p, text="Dynamic target  (should do…)", bg=self._BG2, fg="#bbbbbb",
                 font=("Helvetica", 9), anchor="w").pack(fill=tk.X, padx=10)
        self._dynamic_target_var = tk.StringVar(value=_DEFAULT_DYNAMIC_TARGET)
        dynamic_entry = tk.Entry(
            p, textvariable=self._dynamic_target_var,
            bg=self._BG3, fg="#ffffff", insertbackground="#ffffff",
            relief=tk.FLAT, font=("Courier", 10), bd=4,
        )
        dynamic_entry.pack(fill=tk.X, padx=10, pady=(0, 2))
        dynamic_entry.bind("<Return>",  _unfocus)
        dynamic_entry.bind("<Escape>",  _unfocus)

        # ── Human scores ──
        separator()
        section("Human Scores  (0 – 10)")

        self._score_vars: dict[str, tk.IntVar] = {}
        for criterion, label, color in (
            ("coherence",   "Coherence   — matches target",    "#88aaff"),
            ("originality", "Originality — structural novelty", "#ffaa44"),
            ("interest",    "Interest    — evol. potential",   "#88ff88"),
        ):
            var = tk.IntVar(value=5)
            self._score_vars[criterion] = var

            tk.Label(p, text=label, bg=self._BG2, fg=color,
                     font=("Helvetica", 9), anchor="w").pack(fill=tk.X, padx=10, pady=(4, 0))

            score_row = tk.Frame(p, bg=self._BG2)
            score_row.pack(fill=tk.X, padx=8)

            lbl = tk.Label(score_row, text="5", bg=self._BG2, fg="#ffffff",
                           font=("Courier", 12, "bold"), width=3, anchor="e")
            lbl.pack(side=tk.RIGHT, padx=(0, 4))

            def _make_cb(v=var, l=lbl):
                def cb(*_): l.config(text=str(v.get()))
                return cb

            tk.Scale(
                score_row, variable=var, from_=0, to=10, resolution=1,
                orient=tk.HORIZONTAL, showvalue=False,
                bg=self._BG3, troughcolor="#555555",
                highlightthickness=0, bd=0, command=_make_cb(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── Config summary (read-only) ──
        separator()
        section("Config (config.py)")
        summary = (
            f"length_std      {_cfg.length_std}\n"
            f"angle_std       {_cfg.angle_std}°\n"
            f"add_remove_prob {_cfg.add_remove_prob}\n"
            f"branching       {_cfg.allow_branching} ({_cfg.branching_prob})\n"
            f"init_mutation   {_cfg.init_n_mutation}"
        )
        tk.Label(p, text=summary, bg=self._BG2, fg="#666666",
                 justify=tk.LEFT, font=("Courier", 8)).pack(padx=10, anchor="w")

        # ── Session stats ──
        separator()
        section("Session stats")
        self._stats_var = tk.StringVar()
        tk.Label(p, textvariable=self._stats_var,
                 bg=self._BG2, fg="#88ff88",
                 justify=tk.LEFT, font=("Courier", 10)).pack(padx=14, anchor="w")
        self._update_stats()

        # ── Morphology info ──
        separator()
        section("Morphology info")
        self._morph_info_var = tk.StringVar(value="–")
        tk.Label(p, textvariable=self._morph_info_var,
                 bg=self._BG2, fg="#99ccff",
                 justify=tk.LEFT, font=("Courier", 10)).pack(padx=14, anchor="w")

        tk.Frame(p, bg=self._BG2, height=16).pack()

    # ------------------------------------------------------------------
    # Renderer management
    # ------------------------------------------------------------------

    def _init_renderer(self):
        if self._renderer is not None:
            self._renderer.close()
        sz = self.params.render_size
        camera_views = [
            CameraView(
                azimuth   = v["azimuth"],
                elevation = v["elevation"],
                distance  = v["distance"],
                lookat    = tuple(v.get("lookat", (0.0, 0.0, 0.25))),
            )
            for v in _cfg.camera_views
        ]
        self._renderer = MorphologyRenderer(RenderConfig(
            width           = sz,
            height          = sz,
            camera_views    = camera_views,
            floor_clearance = self.params.floor_clearance,
            photorealistic  = self.params.photorealistic,
        ))
        self._renderer_size = sz

    # ------------------------------------------------------------------
    # Generation and rendering
    # ------------------------------------------------------------------

    def _do_generate(self):
        self._status_var.set("  Generating…")
        self.root.update_idletasks()
        self._push_history()

        try:
            morph = NewMorph(
                min_init_legs   = self.params.n_legs_min,
                max_init_legs   = self.params.n_legs_max,
                n_init_mutation = _cfg.init_n_mutation,
            )
            image = self._renderer.render(morph)
        except Exception as exc:
            self._status_var.set(f"  ERROR: {exc}")
            self._set_buttons_enabled(False)
            return

        self._current_morph = morph
        self._current_image = image
        self._show_image(image)
        self._update_morph_info(morph)
        self._set_status(morph)
        self._set_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def _show_image(self, image: PILImage.Image):
        cw = max(self._canvas.winfo_width(),  self._CANVAS_W)
        ch = max(self._canvas.winfo_height(), self._CANVAS_H)
        scale = min(cw / image.width, ch / image.height)
        nw = max(1, int(image.width  * scale))
        nh = max(1, int(image.height * scale))
        self._photo = ImageTk.PhotoImage(image.resize((nw, nh), PILImage.LANCZOS))
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_save(self):
        if self._current_image is None or self._current_morph is None:
            return

        static_target  = self._static_target_var.get().strip()
        dynamic_target = self._dynamic_target_var.get().strip()
        if not static_target or not dynamic_target:
            self._status_var.set(
                "  ERROR: targets cannot be empty — fill in both target fields before saving."
            )
            return

        morph_id = f"morph_{_next_id(self._dataset):04d}"
        img_rel  = f"images/{morph_id}.png"
        img_path = _DATASET_DIR / img_rel

        _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self._current_image.save(img_path)

        entry = {
            "id":             morph_id,
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "static_target":  static_target,
            "dynamic_target": dynamic_target,
            "human_scores": {
                "coherence":   self._score_vars["coherence"].get(),
                "originality": self._score_vars["originality"].get(),
                "interest":    self._score_vars["interest"].get(),
            },
            "morphology":  morphology_to_dict(self._current_morph),
            "image_path":  img_rel,
        }
        self._dataset.append(entry)
        _save_dataset(self._dataset)

        self._saved_count = len(self._dataset)
        self._update_stats()
        self._status_var.set(
            f"  Saved {morph_id}  "
            f"[coh={entry['human_scores']['coherence']} "
            f"orig={entry['human_scores']['originality']} "
            f"int={entry['human_scores']['interest']}]"
        )
        self._set_buttons_enabled(False)
        self.root.after(10, self._do_generate)

    def _on_skip(self):
        if self._current_image is None:
            return
        self._skip_count += 1
        self._update_stats()
        self._set_buttons_enabled(False)
        self.root.after(10, self._do_generate)

    def _on_mutate(self):
        if self._current_morph is None:
            return
        self._status_var.set("  Mutating…")
        self.root.update_idletasks()
        self._push_history()

        try:
            mutated = MutateMorphology(
                base                      = self._current_morph,
                length_std                = _cfg.length_std,
                angle_std                 = _cfg.angle_std,
                rest_angle_std            = _cfg.rest_angle_std,
                add_remove_prob           = _cfg.add_remove_prob,
                allow_branching           = _cfg.allow_branching,
                branching_prob            = _cfg.branching_prob,
                torso_a_std               = _cfg.torso_a_std,
                torso_b_std               = _cfg.torso_b_std,
                torso_c_std               = _cfg.torso_c_std,
                torso_euler_std           = _cfg.torso_euler_std,
                add_remove_body_part_prob = _cfg.add_remove_body_part_prob,
                body_part_a_std           = _cfg.body_part_a_std,
                body_part_b_std           = _cfg.body_part_b_std,
                body_part_c_std           = _cfg.body_part_c_std,
                body_part_euler_std       = _cfg.body_part_euler_std,
                body_part_leg_prob        = _cfg.body_part_leg_prob,
                rng                       = self.rng,
            )
            image = self._renderer.render(mutated)
        except Exception as exc:
            self._status_var.set(f"  ERROR: {exc}")
            if self._history:
                self._history.pop()
            return

        self._current_morph = mutated
        self._current_image = image
        self._show_image(image)
        self._update_morph_info(mutated)
        self._set_status(mutated, tag="mutated")
        self._set_buttons_enabled(True)

    def _on_previous(self):
        if not self._history:
            return
        morph, image = self._history.pop()
        self._current_morph = morph
        self._current_image = image
        self._show_image(image)
        self._update_morph_info(morph)
        self._set_status(morph, tag=f"restored · {len(self._history)} left")
        self._set_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _push_history(self):
        if self._current_morph is None or self._current_image is None:
            return
        if len(self._history) >= _HISTORY_MAX:
            self._history.pop(0)
        self._history.append((self._current_morph, self._current_image))

    def _set_status(self, morph: RobotMorphology, tag: str = ""):
        enc = morph.encoding()
        suffix = f"  [{tag}]" if tag else ""
        self._status_var.set(
            f"  {morph.name}{suffix}   "
            f"legs={enc['n_legs']} (root={enc['n_root_legs']} branch={enc['n_branch_legs']})   "
            f"bparts={enc['n_body_parts']}   "
            f"sym={enc['symmetry_score']:.2f}"
        )

    def _set_buttons_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self._btn_save, self._btn_skip, self._btn_mutate):
            btn.config(state=state)
        prev_state = tk.NORMAL if (enabled and self._history) else tk.DISABLED
        self._btn_prev.config(state=prev_state)

    def _update_stats(self):
        self._stats_var.set(
            f"Saved   : {self._saved_count}\n"
            f"Skipped : {self._skip_count}\n"
            f"Dataset : {len(self._dataset)} entries"
        )

    def _update_morph_info(self, morph: RobotMorphology):
        enc = morph.encoding()
        spawn_h = compute_spawn_height(morph, self.params.floor_clearance)
        self._morph_info_var.set(
            f"Legs      : {enc['n_legs']} total\n"
            f"  root    : {enc['n_root_legs']}\n"
            f"  branch  : {enc['n_branch_legs']}\n"
            f"Body parts: {enc['n_body_parts']}\n"
            f"Symmetry  : {enc['symmetry_score']:.3f}\n"
            f"Seg len   : {enc['mean_segment_length']:.3f} m\n"
            f"Spawn h   : {spawn_h:.3f} m\n"
            f"History   : {len(self._history)}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not _PIL_AVAILABLE:
        print("ERROR: Pillow (PIL) is required.  pip install Pillow")
        sys.exit(1)
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("ERROR: mujoco is required.  pip install mujoco")
        sys.exit(1)

    _DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.geometry("1080x700")
    app = MorphEvalApp(root)  # noqa: F841
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
