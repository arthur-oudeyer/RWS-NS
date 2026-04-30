"""
utils/morph_generator_renderer.py
==================================
Interactive morphology generator and labeller.

Generates random robot morphologies via NewMorph() (morphology.py) and
renders them with MuJoCo.  All generation and mutation parameters come
from ExperimentConfig (config.py) — nothing is hardcoded here.

Output  (relative to this script's parent directory)
------
    prompt_testing_source/
        positive/   ← [P] key or button
        negative/   ← [N] key or button
        (skipped images are discarded)

Controls
--------
    P  /  button  →  save to positive/
    N  /  button  →  save to negative/
    S  /  Space   →  skip (discard)
    M             →  mutate current morphology (uses config mutation params)
    B  /  ←       →  go back to previous morphology
    Q             →  quit

Live parameter sliders (right panel)
--------------------------------------
    Min / Max init legs  : leg count range passed to NewMorph()
    Render size          : output image resolution
    Floor margin         : clearance above ground plane
    Photorealistic       : grass floor + blue-sky skybox toggle

    All other generation/mutation parameters are read directly from
    ExperimentConfig in config.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
    RobotMorphology,
    compute_spawn_height, NewMorph, MutateMorphology,
)
from rendering import MorphologyRenderer, RenderConfig, CameraView
from config import ExperimentConfig


# ---------------------------------------------------------------------------
# Module-level config — single source of truth for all defaults
# ---------------------------------------------------------------------------

_cfg = ExperimentConfig()


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent
_SOURCE_ROOT = _SCRIPT_DIR.parent / "prompt_testing_source"
_POS_DIR     = _SOURCE_ROOT / "positive"
_NEG_DIR     = _SOURCE_ROOT / "negative"


# ---------------------------------------------------------------------------
# Render/generation parameters editable via sliders
#
# Only parameters that make sense to tweak interactively are kept here.
# Everything else (mutation stds, branching probs, body-part probs…) lives
# in ExperimentConfig and is used directly without sliders.
# ---------------------------------------------------------------------------

@dataclass
class GenParams:
    """Slider-controlled parameters for the interactive generator."""
    n_legs_min:     int   = _cfg.init_n_legs_min
    n_legs_max:     int   = _cfg.init_n_legs_max
    render_size:    int   = _cfg.render_width
    floor_clearance: float = _cfg.floor_clearance
    photorealistic:  bool  = _cfg.photorealistic


# ---------------------------------------------------------------------------
# Filename helper — finds the next unused morph_NNNN.png index in a folder
# ---------------------------------------------------------------------------

def _next_idx(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    indices = [
        int(p.stem.split("_")[-1])
        for p in folder.glob("morph_*.png")
        if p.stem.split("_")[-1].isdigit()
    ]
    return (max(indices) + 1) if indices else 1


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

_HISTORY_MAX = 20  # maximum entries kept in the back-stack


class MorphSorterApp:
    """
    tkinter application for interactively labelling robot morphologies.

    Rendering is intentionally done on the main thread (via root.after) to
    avoid OpenGL threading issues on macOS.
    """

    _CANVAS_W = 700
    _CANVAS_H = 420
    _PANEL_W  = 240

    # Colours
    _BG       = "#1e1e1e"
    _BG2      = "#252525"
    _BG3      = "#2d2d2d"
    _FG       = "#cccccc"
    _FG_DIM   = "#888888"
    _GREEN    = "#2d6a2d"
    _GREEN_FG = "#ccffcc"
    _RED      = "#6a2d2d"
    _RED_FG   = "#ffcccc"
    _SKIP_BG  = "#3a3a3a"
    _MUT_BG   = "#2d4a6a"
    _MUT_FG   = "#cce0ff"
    _PREV_BG  = "#4a3a2d"
    _PREV_FG  = "#ffe0cc"

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.rng    = np.random.default_rng()
        self.params = GenParams()

        self._pos_count  = len(list(_POS_DIR.glob("morph_*.png"))) if _POS_DIR.exists() else 0
        self._neg_count  = len(list(_NEG_DIR.glob("morph_*.png"))) if _NEG_DIR.exists() else 0
        self._skip_count = 0

        self._current_morph: Optional[RobotMorphology]    = None
        self._current_image: Optional[PILImage.Image]     = None
        self._photo:         Optional[ImageTk.PhotoImage] = None

        # History stack — each entry is (morph, rendered_image)
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
        self.root.title("Morphology Sorter")
        self.root.configure(bg=self._BG)
        self.root.resizable(True, True)

        # ---- Status bar ----
        self._status_var = tk.StringVar(value="  Starting…")
        tk.Label(
            self.root, textvariable=self._status_var,
            bg="#2a2a2a", fg=self._FG_DIM, anchor="w", padx=10,
            font=("Courier", 11),
        ).pack(side=tk.TOP, fill=tk.X)

        # ---- Main content ----
        content = tk.Frame(self.root, bg=self._BG)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: canvas + buttons
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

        # ---- Navigation row: Back + Mutate ----
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

        # ---- Label row: Positive + Skip + Negative ----
        btn_frame = tk.Frame(left, bg=self._BG, pady=3)
        btn_frame.pack(side=tk.TOP, fill=tk.X)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        self._btn_pos = tk.Button(
            btn_frame, text="✓  Positive  [P]",
            bg=self._GREEN, fg=self._GREEN_FG,
            activebackground="#3a8a3a", activeforeground="#ffffff",
            command=self._on_positive, **btn_cfg,
        )
        self._btn_skip = tk.Button(
            btn_frame, text="↷  Skip  [S]",
            bg=self._SKIP_BG, fg=self._FG,
            activebackground="#555", activeforeground="#fff",
            command=self._on_skip, **btn_cfg,
        )
        self._btn_neg = tk.Button(
            btn_frame, text="✗  Negative  [N]",
            bg=self._RED, fg=self._RED_FG,
            activebackground="#8a3a3a", activeforeground="#ffffff",
            command=self._on_negative, **btn_cfg,
        )
        self._btn_pos.grid( row=0, column=0, sticky="ew", padx=3)
        self._btn_skip.grid(row=0, column=1, sticky="ew", padx=3)
        self._btn_neg.grid( row=0, column=2, sticky="ew", padx=3)

        # Keyboard shortcuts
        for key, cb in (
            ("<p>", self._on_positive), ("<P>", self._on_positive),
            ("<n>", self._on_negative), ("<N>", self._on_negative),
            ("<s>", self._on_skip),     ("<S>", self._on_skip),
            ("<space>", self._on_skip),
            ("<m>", self._on_mutate),   ("<M>", self._on_mutate),
            ("<b>", self._on_previous), ("<B>", self._on_previous),
            ("<Left>", self._on_previous),
            ("<q>", lambda e: self.root.destroy()),
            ("<Q>", lambda e: self.root.destroy()),
        ):
            self.root.bind(key, lambda e, f=cb: f())

        # Right: parameter + info panel
        right = tk.Frame(content, bg=self._BG2, width=self._PANEL_W)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        right.pack_propagate(False)

        self._build_param_panel(right)
        self._set_buttons_enabled(False)

    # ------------------------------------------------------------------

    def _build_param_panel(self, parent: tk.Frame):
        """Scrollable panel with the few sliders that make sense interactively."""
        _canvas = tk.Canvas(parent, bg=self._BG2, highlightthickness=0)
        _scroll = tk.Scrollbar(parent, orient=tk.VERTICAL, command=_canvas.yview)
        p = tk.Frame(_canvas, bg=self._BG2)

        p.bind("<Configure>",
               lambda e: _canvas.configure(scrollregion=_canvas.bbox("all")))
        _canvas.create_window((0, 0), window=p, anchor="nw",
                              width=self._PANEL_W - 16)
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

        def add_slider(attr, label, from_, to, res, is_int, constraint_cb=None):
            row = tk.Frame(p, bg=self._BG2)
            row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(row, text=label, bg=self._BG2, fg="#bbbbbb",
                     font=("Helvetica", 9), width=16, anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(row, bg=self._BG2, fg="#ffffff",
                               font=("Courier", 9), width=6)
            val_lbl.pack(side=tk.RIGHT)
            var = (tk.IntVar(value=getattr(self.params, attr)) if is_int
                   else tk.DoubleVar(value=getattr(self.params, attr)))
            self._slider_vars[attr] = var

            def _cb(a=attr, v=var, lbl=val_lbl, i=is_int, cc=constraint_cb):
                def cb(*_):
                    val = int(v.get()) if i else float(v.get())
                    setattr(self.params, a, val)
                    lbl.config(text=str(val) if i else f"{val:.2f}")
                    if cc: cc(a)
                return cb

            cb = _cb()
            tk.Scale(
                row, variable=var, from_=from_, to=to, resolution=res,
                orient=tk.HORIZONTAL, showvalue=False,
                bg=self._BG3, troughcolor="#555555",
                highlightthickness=0, bd=0, command=cb,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            cb()

        self._slider_vars: dict[str, tk.Variable] = {}

        def _legs_constraint(attr):
            if attr == "n_legs_min" and self.params.n_legs_min > self.params.n_legs_max:
                self.params.n_legs_max = self.params.n_legs_min
                self._slider_vars["n_legs_max"].set(self.params.n_legs_min)
            elif attr == "n_legs_max" and self.params.n_legs_max < self.params.n_legs_min:
                self.params.n_legs_min = self.params.n_legs_max
                self._slider_vars["n_legs_min"].set(self.params.n_legs_max)

        # ---- Init legs ----
        section("Initial legs")
        add_slider("n_legs_min", "Min legs", 1, 8, 1, True,  _legs_constraint)
        add_slider("n_legs_max", "Max legs", 1, 8, 1, True,  _legs_constraint)

        # ---- Render ----
        separator()
        section("Render")
        self._render_size_var = tk.IntVar(value=self.params.render_size)
        sz_row = tk.Frame(p, bg=self._BG2)
        sz_row.pack()
        for sz in (192, 256, 512):
            tk.Radiobutton(
                sz_row, text=str(sz),
                variable=self._render_size_var, value=sz,
                bg=self._BG2, fg=self._FG, selectcolor="#444444",
                font=("Courier", 10), activebackground=self._BG2,
                command=self._on_size_change,
            ).pack(side=tk.LEFT, padx=6)

        def _floor_cb(_attr): self._init_renderer()
        add_slider("floor_clearance", "Floor margin (m)", 0.0, 0.30, 0.01, False, _floor_cb)

        self._photorealistic_var = tk.BooleanVar(value=self.params.photorealistic)
        tk.Checkbutton(
            p, text="Photorealistic",
            variable=self._photorealistic_var,
            bg=self._BG2, fg=self._FG, selectcolor="#444444",
            activebackground=self._BG2, activeforeground=self._FG,
            font=("Helvetica", 9),
            command=self._on_photorealistic_change,
        ).pack(anchor="w", padx=10, pady=(4, 0))

        # ---- Config summary (read-only) ----
        separator()
        section("Config (config.py)")
        summary = (
            f"length_std      {_cfg.length_std}\n"
            f"angle_std       {_cfg.angle_std}°\n"
            f"rest_angle_std  {_cfg.rest_angle_std}\n"
            f"add_remove_prob {_cfg.add_remove_prob}\n"
            f"branching       {_cfg.allow_branching} ({_cfg.branching_prob})\n"
            f"init_mutation   {_cfg.init_n_mutation}\n"
            f"body_part_prob  {_cfg.add_remove_body_part_prob}"
        )
        tk.Label(p, text=summary, bg=self._BG2, fg="#666666",
                 justify=tk.LEFT, font=("Courier", 8)).pack(padx=10, anchor="w")

        # ---- Stats ----
        separator()
        section("Session stats")
        self._stats_var = tk.StringVar()
        tk.Label(p, textvariable=self._stats_var,
                 bg=self._BG2, fg="#88ff88",
                 justify=tk.LEFT, font=("Courier", 10)).pack(padx=14, anchor="w")
        self._update_stats()

        # ---- Morph info ----
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

    def _on_size_change(self):
        new_sz = self._render_size_var.get()
        self.params.render_size = new_sz
        if new_sz != self._renderer_size:
            self._init_renderer()

    def _on_photorealistic_change(self):
        self.params.photorealistic = self._photorealistic_var.get()
        self._init_renderer()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _do_generate(self):
        """Generate and render one morphology via NewMorph()."""
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
        self._set_status(morph, tag="")
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

    def _on_positive(self):
        if self._current_image is None: return
        self._save_current(_POS_DIR)
        self._pos_count += 1
        self._update_stats()
        self._set_buttons_enabled(False)
        self.root.after(10, self._do_generate)

    def _on_negative(self):
        if self._current_image is None: return
        self._save_current(_NEG_DIR)
        self._neg_count += 1
        self._update_stats()
        self._set_buttons_enabled(False)
        self.root.after(10, self._do_generate)

    def _on_skip(self):
        if self._current_image is None: return
        self._skip_count += 1
        self._update_stats()
        self._set_buttons_enabled(False)
        self.root.after(10, self._do_generate)

    def _on_mutate(self):
        """Mutate current morphology using all mutation params from config.py."""
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
            if self._history: self._history.pop()
            return

        self._current_morph = mutated
        self._current_image = image
        self._show_image(image)
        self._update_morph_info(mutated)
        self._set_status(mutated, tag="mutated")
        self._set_buttons_enabled(True)

    def _on_previous(self):
        """Restore the most recently stored morphology from the history stack."""
        if not self._history:
            return
        morph, image = self._history.pop()
        self._current_morph = morph
        self._current_image = image
        self._show_image(image)
        self._update_morph_info(morph)
        self._set_status(morph, tag=f"restored · {len(self._history)} left")
        self._set_buttons_enabled(True)

    def _save_current(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"morph_{_next_idx(folder):04d}.png"
        self._current_image.save(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _push_history(self):
        if self._current_morph is None or self._current_image is None:
            return
        if len(self._history) >= _HISTORY_MAX:
            self._history.pop(0)
        self._history.append((self._current_morph, self._current_image))

    def _set_status(self, morph: RobotMorphology, tag: str):
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
        for btn in (self._btn_pos, self._btn_skip, self._btn_neg, self._btn_mutate):
            btn.config(state=state)
        prev_state = tk.NORMAL if (enabled and self._history) else tk.DISABLED
        self._btn_prev.config(state=prev_state)

    def _update_stats(self):
        total = self._pos_count + self._neg_count + self._skip_count
        self._stats_var.set(
            f"Positive : {self._pos_count}\n"
            f"Negative : {self._neg_count}\n"
            f"Skipped  : {self._skip_count}\n"
            f"Total    : {total}"
        )

    def _update_morph_info(self, morph: RobotMorphology):
        enc = morph.encoding()
        rx, ry, rz = morph.torso_euler
        bp_lines = ""
        for i, bp in enumerate(morph.body_parts):
            n_bp_legs = sum(1 for leg in morph.legs if leg.body_part_idx == i)
            bp_lines += f"\n  bp{i+1}: a={bp.a:.2f} b={bp.b:.2f} legs={n_bp_legs}"
        spawn_h = compute_spawn_height(morph, self.params.floor_clearance)
        self._morph_info_var.set(
            f"Name      : {morph.name}\n"
            f"Legs      : {enc['n_legs']} total\n"
            f"  root    : {enc['n_root_legs']}\n"
            f"  branch  : {enc['n_branch_legs']}\n"
            f"Body parts: {enc['n_body_parts']}"
            f"{bp_lines}\n"
            f"Symmetry  : {enc['symmetry_score']:.3f}\n"
            f"Seg len   : {enc['mean_segment_length']:.3f} m\n"
            f"Torso a   : {morph.torso_a:.3f} m\n"
            f"Torso b   : {morph.torso_b:.3f} m\n"
            f"Torso c   : {morph.torso_c:.3f} m\n"
            f"Euler     : ({rx:.1f}° {ry:.1f}° {rz:.1f}°)\n"
            f"Spawn h   : {spawn_h:.3f} m\n"
            f"History   : {len(self._history)}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not _PIL_AVAILABLE:
        print("ERROR: Pillow (PIL) is required.  Install with:  pip install Pillow")
        sys.exit(1)
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("ERROR: mujoco is required.  Install with:  pip install mujoco")
        sys.exit(1)

    _POS_DIR.mkdir(parents=True, exist_ok=True)
    _NEG_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.geometry("1020x700")
    app = MorphSorterApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
