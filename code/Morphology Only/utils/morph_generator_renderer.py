"""
utils/morph_generator_renderer.py
==================================
Interactive morphology generator and labeller.

Generates random robot morphologies, renders them with MuJoCo, and shows
each one in a tkinter window so the user can sort them into labelled folders.

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
    Q             →  quit

Live parameter sliders (right panel)
--------------------------------------
    Min / Max root legs  : range of root legs to generate
    Max branch legs      : cap on total branched legs added
    Branch depth         : maximum nesting level for branch-of-branch
    Branch prob          : probability of adding a branch at each candidate site
    Length std (m)       : Gaussian std for segment-length perturbation
    Angle std (°)        : Gaussian std for placement-angle perturbation
    Rest angle std (rad) : Gaussian std for rest-angle perturbation
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
    LegDescriptor, JointDescriptor, BodyPartDescriptor,
    MIN_LENGTH, MAX_LENGTH,
    compute_spawn_height, NewMorph
)
from rendering import MorphologyRenderer, RenderConfig, CameraView


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent
_SOURCE_ROOT = _SCRIPT_DIR.parent / "prompt_testing_source"
_POS_DIR     = _SOURCE_ROOT / "positive"
_NEG_DIR     = _SOURCE_ROOT / "negative"


# ---------------------------------------------------------------------------
# Generation parameters  (modified live by sliders)
# ---------------------------------------------------------------------------
USE_DEFAULT_MORPH_GENERATOR = True

@dataclass
class GenParams:
    """All morphology-generation hyper-parameters, editable via sliders."""
    # Legs
    n_root_min:       int   = 1
    n_root_max:       int   = 8
    max_branch_legs:  int   = 8
    max_branch_depth: int   = 2
    branch_prob:      float = 0.40   # prob of trying to branch at each parent leg
    length_std:       float = 0.05
    angle_std:        float = 15.0
    rest_angle_std:   float = 0.30
    # Torso shape
    torso_radius_min: float = 0.06
    torso_radius_max: float = 0.22
    torso_height_min: float = 0.01
    torso_height_max: float = 0.18
    torso_euler_max:  float = 0.0    # max |angle| per axis (°); 0 = no tilt
    # Body parts
    max_body_parts:       int   = 2
    body_part_prob:       float = 0.2   # prob of adding a body part to each candidate leg
    body_part_radius_min: float = 0.04
    body_part_radius_max: float = 0.12
    body_part_height_min: float = 0.01
    body_part_height_max: float = 0.08
    body_part_euler_max:  float = 0.0    # max tilt per axis (°); 0 = no tilt
    max_body_part_legs:   int   = 4      # max legs to try adding per body part
    body_part_leg_prob:   float = 0.50   # prob of adding each of those legs
    # Rendering
    render_size:      int   = 192
    floor_clearance:  float = 0.   # metres of clearance above the floor
    photorealistic:   bool  = True  # grass floor + blue-sky skybox


# ---------------------------------------------------------------------------
# Random morphology generator with full branching support
# ---------------------------------------------------------------------------

def _random_joint(rng: np.random.Generator, max_length: float = MAX_LENGTH) -> JointDescriptor:
    return JointDescriptor(
        rgba       = (float(rng.uniform(0.2, 0.9)),
                      float(rng.uniform(0.2, 0.9)),
                      float(rng.uniform(0.2, 0.9)),
                      1.0),
        length     = float(rng.uniform(MIN_LENGTH, max_length)),
        rest_angle = float(rng.uniform(-1.2, 0.8)),
    )


def generate_random_morph(params: GenParams, rng: np.random.Generator) -> RobotMorphology:
    """
    Generate a random morphology.

    1. Create N root legs  (N ~ Uniform[n_root_min, n_root_max]).
    2. Walk legs; with probability branch_prob attach a branched leg
       (up to max_branch_legs total, max_branch_depth nesting).
    3. Walk root legs; with probability body_part_prob attach a body part
       (up to max_body_parts total), then add legs to each body part.
    """
    n_root = int(rng.integers(params.n_root_min, params.n_root_max + 1))

    legs:       list[LegDescriptor]     = []
    body_parts: list[BodyPartDescriptor] = []
    depth_of:   dict[int, int]           = {}

    # --- Root legs ---
    for _ in range(n_root):
        legs.append(LegDescriptor(
            placement_angle_deg = float(rng.uniform(0, 360)),
            joints              = [_random_joint(rng)],
        ))
    for i in range(n_root):
        depth_of[i] = 0

    # --- Branched legs ---
    n_branches = 0
    leg_idx    = 0
    while leg_idx < len(legs) and n_branches < params.max_branch_legs:
        if depth_of[leg_idx] < params.max_branch_depth and rng.random() < params.branch_prob:
            parent_joint_idx = int(rng.integers(0, len(legs[leg_idx].joints)))
            new_idx          = len(legs)
            legs.append(LegDescriptor(
                placement_angle_deg = float(rng.uniform(0, 360)),
                joints              = [_random_joint(rng, max_length=0.30)],
                parent_leg_idx      = leg_idx,
                parent_joint_idx    = parent_joint_idx,
            ))
            depth_of[new_idx] = depth_of[leg_idx] + 1
            n_branches += 1
        leg_idx += 1

    # --- Body parts (only on root legs — no branched children as hosts) ---
    if params.max_body_parts > 0:
        root_leg_indices = [i for i in range(n_root)]   # indices of root legs
        n_body_parts     = 0
        for parent_leg_idx in root_leg_indices:
            if n_body_parts >= params.max_body_parts:
                break
            if rng.random() > params.body_part_prob:
                continue

            bp_idx    = len(body_parts)
            bp_radius = float(rng.uniform(params.body_part_radius_min,
                                          params.body_part_radius_max))
            bp_height = float(rng.uniform(params.body_part_height_min,
                                          params.body_part_height_max))
            if params.body_part_euler_max > 0:
                bp_euler = (
                    float(rng.uniform(-params.body_part_euler_max, params.body_part_euler_max)),
                    float(rng.uniform(-params.body_part_euler_max, params.body_part_euler_max)),
                    0.0,
                )
            else:
                bp_euler = (0.0, 0.0, 0.0)

            body_parts.append(BodyPartDescriptor(
                parent_leg_idx = parent_leg_idx,
                radius         = bp_radius,
                height         = bp_height,
                euler_deg      = bp_euler,
                rgba           = (float(rng.uniform(0.4, 0.9)),
                                  float(rng.uniform(0.4, 0.9)),
                                  float(rng.uniform(0.4, 0.9)),
                                  1.0),
            ))
            n_body_parts += 1

            # Add legs to this body part
            for _ in range(params.max_body_part_legs):
                if rng.random() < params.body_part_leg_prob:
                    legs.append(LegDescriptor(
                        placement_angle_deg = float(rng.uniform(0, 360)),
                        joints              = [_random_joint(rng, max_length=0.25)],
                        body_part_idx       = bp_idx,
                    ))

    # --- Torso shape and orientation ---
    torso_radius = float(rng.uniform(params.torso_radius_min, params.torso_radius_max))
    torso_height = float(rng.uniform(params.torso_height_min, params.torso_height_max))
    if params.torso_euler_max > 0:
        torso_euler = (
            float(rng.uniform(-params.torso_euler_max, params.torso_euler_max)),
            float(rng.uniform(-params.torso_euler_max, params.torso_euler_max)),
            float(rng.uniform(-params.torso_euler_max, params.torso_euler_max)),
        )
    else:
        torso_euler = (0.0, 0.0, 0.0)

    name = f"morph_{int(rng.integers(1000, 9999))}"
    return RobotMorphology(
        name         = name,
        legs         = legs,
        body_parts   = body_parts,
        torso_radius = torso_radius,
        torso_height = torso_height,
        torso_euler  = torso_euler,
    )


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

class MorphSorterApp:
    """
    tkinter application for interactively labelling robot morphologies.

    Rendering is intentionally done on the main thread (via root.after) to
    avoid OpenGL threading issues on macOS.  The brief freeze (<300 ms) is
    acceptable in an interactive sorting workflow.
    """

    _CANVAS_W = 700
    _CANVAS_H = 420
    _PANEL_W  = 270

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

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.rng    = np.random.default_rng()
        self.params = GenParams()

        # Track saved counts (include already-existing files)
        self._pos_count  = len(list(_POS_DIR.glob("morph_*.png"))) if _POS_DIR.exists() else 0
        self._neg_count  = len(list(_NEG_DIR.glob("morph_*.png"))) if _NEG_DIR.exists() else 0
        self._skip_count = 0

        self._current_morph: Optional[RobotMorphology] = None
        self._current_image: Optional[PILImage.Image]  = None
        self._photo:         Optional[ImageTk.PhotoImage] = None

        self._renderer:      Optional[MorphologyRenderer] = None
        self._renderer_size: int = 0

        self._build_ui()
        self._init_renderer()

        # Schedule first generation after the window draws itself
        self.root.after(100, self._do_generate)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.title("Morphology Sorter")
        self.root.configure(bg=self._BG)
        self.root.resizable(True, True)

        # ---- Status bar (top) ----
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
        self._canvas_text = self._canvas.create_text(
            self._CANVAS_W // 2, self._CANVAS_H // 2,
            text="Generating…",
            fill=self._FG_DIM, font=("Courier", 15),
        )

        # Action buttons
        btn_frame = tk.Frame(left, bg=self._BG, pady=6)
        btn_frame.pack(side=tk.TOP, fill=tk.X)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        btn_cfg = dict(
            font=("Helvetica", 13, "bold"),
            relief=tk.FLAT, cursor="hand2",
            padx=10, pady=9,
        )
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
        """
        Build the scrollable parameter + info panel.

        A Canvas + Scrollbar combo provides vertical scrolling so the panel
        can hold as many sliders as needed without clipping.
        """
        # ---- Scrollable container ----
        _canvas = tk.Canvas(parent, bg=self._BG2, highlightthickness=0)
        _scroll = tk.Scrollbar(parent, orient=tk.VERTICAL, command=_canvas.yview)
        p = tk.Frame(_canvas, bg=self._BG2)   # all content goes in p

        p.bind("<Configure>",
               lambda e: _canvas.configure(scrollregion=_canvas.bbox("all")))
        _canvas.create_window((0, 0), window=p, anchor="nw",
                              width=self._PANEL_W - 16)
        _canvas.configure(yscrollcommand=_scroll.set)

        _scroll.pack(side=tk.RIGHT, fill=tk.Y)
        _canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse-wheel scroll (macOS + Windows: delta; Linux: Button-4/5)
        def _wheel(e):
            if e.num == 4:
                _canvas.yview_scroll(-1, "units")
            elif e.num == 5:
                _canvas.yview_scroll(1, "units")
            else:
                _canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        for widget in (_canvas, p):
            widget.bind("<MouseWheel>", _wheel)
            widget.bind("<Button-4>",   _wheel)
            widget.bind("<Button-5>",   _wheel)

        # ---- Helpers ----
        def section(text: str):
            tk.Label(p, text=text, bg=self._BG2, fg="#aaaaaa",
                     font=("Helvetica", 11, "bold")).pack(pady=(10, 3))

        def subsection(text: str):
            tk.Label(p, text=text, bg=self._BG2, fg="#888888",
                     font=("Helvetica", 9, "italic")).pack(pady=(6, 1))

        def separator():
            tk.Frame(p, bg="#444444", height=1).pack(fill=tk.X, padx=10, pady=6)

        def add_slider(attr, label, from_, to, res, is_int,
                       constraint_cb=None):
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
                    raw = v.get()
                    val = int(raw) if i else float(raw)
                    setattr(self.params, a, val)
                    lbl.config(text=str(val) if i else f"{val:.2f}")
                    if cc:
                        cc(a)
                return cb

            cb = _cb()
            tk.Scale(
                row, variable=var,
                from_=from_, to=to, resolution=res,
                orient=tk.HORIZONTAL, showvalue=False,
                bg=self._BG3, troughcolor="#555555",
                highlightthickness=0, bd=0,
                command=cb,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

            cb()   # initialise label

        self._slider_vars: dict[str, tk.Variable] = {}

        # ---- Constraint callbacks ----
        def _legs_constraint(attr):
            if attr == "n_root_min" and self.params.n_root_min > self.params.n_root_max:
                self.params.n_root_max = self.params.n_root_min
                self._slider_vars["n_root_max"].set(self.params.n_root_min)
            elif attr == "n_root_max" and self.params.n_root_max < self.params.n_root_min:
                self.params.n_root_min = self.params.n_root_max
                self._slider_vars["n_root_min"].set(self.params.n_root_max)

        def _torso_r_constraint(attr):
            if attr == "torso_radius_min" and self.params.torso_radius_min > self.params.torso_radius_max:
                self.params.torso_radius_max = self.params.torso_radius_min
                self._slider_vars["torso_radius_max"].set(self.params.torso_radius_min)
            elif attr == "torso_radius_max" and self.params.torso_radius_max < self.params.torso_radius_min:
                self.params.torso_radius_min = self.params.torso_radius_max
                self._slider_vars["torso_radius_min"].set(self.params.torso_radius_max)

        def _torso_h_constraint(attr):
            if attr == "torso_height_min" and self.params.torso_height_min > self.params.torso_height_max:
                self.params.torso_height_max = self.params.torso_height_min
                self._slider_vars["torso_height_max"].set(self.params.torso_height_min)
            elif attr == "torso_height_max" and self.params.torso_height_max < self.params.torso_height_min:
                self.params.torso_height_min = self.params.torso_height_max
                self._slider_vars["torso_height_min"].set(self.params.torso_height_max)

        def _bp_r_constraint(attr):
            if attr == "body_part_radius_min" and self.params.body_part_radius_min > self.params.body_part_radius_max:
                self.params.body_part_radius_max = self.params.body_part_radius_min
                self._slider_vars["body_part_radius_max"].set(self.params.body_part_radius_min)
            elif attr == "body_part_radius_max" and self.params.body_part_radius_max < self.params.body_part_radius_min:
                self.params.body_part_radius_min = self.params.body_part_radius_max
                self._slider_vars["body_part_radius_min"].set(self.params.body_part_radius_max)

        def _bp_h_constraint(attr):
            if attr == "body_part_height_min" and self.params.body_part_height_min > self.params.body_part_height_max:
                self.params.body_part_height_max = self.params.body_part_height_min
                self._slider_vars["body_part_height_max"].set(self.params.body_part_height_min)
            elif attr == "body_part_height_max" and self.params.body_part_height_max < self.params.body_part_height_min:
                self.params.body_part_height_min = self.params.body_part_height_max
                self._slider_vars["body_part_height_min"].set(self.params.body_part_height_max)

        # ---- Legs section ----
        section("Legs")
        add_slider("n_root_min",      "Min root legs",   1,   8,    1,    True,  _legs_constraint)
        add_slider("n_root_max",      "Max root legs",   1,   8,    1,    True,  _legs_constraint)
        add_slider("max_branch_legs", "Max branches",    0,   16,   1,    True)
        add_slider("max_branch_depth","Branch depth",    1,   4,    1,    True)
        add_slider("branch_prob",     "Branch prob",     0.0, 1.0,  0.05, False)
        add_slider("length_std",      "Length std (m)",  0.0, 0.30, 0.01, False)
        add_slider("angle_std",       "Angle std (°)",   0.0, 60.0, 1.0,  False)
        add_slider("rest_angle_std",  "Rest angle std",  0.0, 1.0,  0.05, False)

        # ---- Torso section ----
        separator()
        section("Torso")
        add_slider("torso_radius_min", "Radius min (m)",  0.04, 0.30, 0.01, False, _torso_r_constraint)
        add_slider("torso_radius_max", "Radius max (m)",  0.04, 0.30, 0.01, False, _torso_r_constraint)
        add_slider("torso_height_min", "Height min (m)",  0.01, 0.25, 0.01, False, _torso_h_constraint)
        add_slider("torso_height_max", "Height max (m)",  0.01, 0.25, 0.01, False, _torso_h_constraint)
        add_slider("torso_euler_max",  "Tilt max (°)",    0.0,  90.0, 1.0,  False)

        # ---- Body parts section ----
        separator()
        section("Body parts")
        add_slider("max_body_parts",       "Max body parts",    0,    4,    1,    True)
        add_slider("body_part_prob",       "BP attach prob",    0.0,  1.0,  0.05, False)
        add_slider("body_part_radius_min", "BP radius min (m)", 0.03, 0.20, 0.01, False, _bp_r_constraint)
        add_slider("body_part_radius_max", "BP radius max (m)", 0.03, 0.20, 0.01, False, _bp_r_constraint)
        add_slider("body_part_height_min", "BP height min (m)", 0.01, 0.15, 0.01, False, _bp_h_constraint)
        add_slider("body_part_height_max", "BP height max (m)", 0.01, 0.15, 0.01, False, _bp_h_constraint)
        add_slider("body_part_euler_max",  "BP tilt max (°)",   0.0,  90.0, 1.0,  False)
        add_slider("max_body_part_legs",   "BP max legs",       0,    6,    1,    True)
        add_slider("body_part_leg_prob",   "BP leg prob",       0.0,  1.0,  0.05, False)

        # ---- Render size ----
        separator()
        section("Render size")
        self._render_size_var = tk.IntVar(value=self.params.render_size)
        sz_row = tk.Frame(p, bg=self._BG2)
        sz_row.pack()
        for sz in (256, 512, 768):
            tk.Radiobutton(
                sz_row, text=str(sz),
                variable=self._render_size_var, value=sz,
                bg=self._BG2, fg=self._FG, selectcolor="#444444",
                font=("Courier", 10), activebackground=self._BG2,
                command=self._on_size_change,
            ).pack(side=tk.LEFT, padx=6)
        def _floor_cb(_attr):
            self._init_renderer()
        add_slider("floor_clearance", "Floor margin (m)", 0.0, 0.30, 0.01, False, _floor_cb)

        self._photorealistic_var = tk.BooleanVar(value=self.params.photorealistic)
        tk.Checkbutton(
            p, text="Photorealistic (grass + sky)",
            variable=self._photorealistic_var,
            bg=self._BG2, fg=self._FG, selectcolor="#444444",
            activebackground=self._BG2, activeforeground=self._FG,
            font=("Helvetica", 9),
            command=self._on_photorealistic_change,
        ).pack(anchor="w", padx=10, pady=(4, 0))

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

        tk.Frame(p, bg=self._BG2, height=16).pack()   # bottom padding

    # ------------------------------------------------------------------
    # Renderer management
    # ------------------------------------------------------------------

    def _init_renderer(self):
        if self._renderer is not None:
            self._renderer.close()
        sz = self.params.render_size
        cfg = RenderConfig(
            width           = sz,
            height          = sz,
            camera_views    = [
                CameraView(azimuth=0,   elevation=5,   distance=2.),
                CameraView(azimuth=45,  elevation=-50, distance=2.),
            ],
            floor_clearance = self.params.floor_clearance,
            photorealistic  = self.params.photorealistic,
        )
        self._renderer = MorphologyRenderer(cfg)
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
    # Rendering  (main-thread, via root.after)
    # ------------------------------------------------------------------

    def _do_generate(self):
        """Generate and render one morphology.  Called on the main thread."""
        self._status_var.set("  Generating…")
        self.root.update_idletasks()   # flush UI before blocking render

        try:
            if USE_DEFAULT_MORPH_GENERATOR:
                morph = NewMorph()
            else:
                morph = generate_random_morph(self.params, self.rng)
            image = self._renderer.render(morph)
        except Exception as exc:
            self._status_var.set(f"  ERROR: {exc}")
            self._set_buttons_enabled(False)
            return

        self._current_morph = morph
        self._current_image = image
        self._show_image(image)
        self._update_morph_info(morph)

        enc = morph.encoding()
        self._status_var.set(
            f"  {morph.name}   "
            f"legs={enc['n_legs']} (root={enc['n_root_legs']} branch={enc['n_branch_legs']})   "
            f"bparts={enc['n_body_parts']}   "
            f"sym={enc['symmetry_score']:.2f}"
        )
        self._set_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def _show_image(self, image: PILImage.Image):
        cw = max(self._canvas.winfo_width(),  self._CANVAS_W)
        ch = max(self._canvas.winfo_height(), self._CANVAS_H)

        scale = min(cw / image.width, ch / image.height)
        nw    = max(1, int(image.width  * scale))
        nh    = max(1, int(image.height * scale))

        resized    = image.resize((nw, nh), PILImage.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)

        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2,
                                  anchor=tk.CENTER, image=self._photo)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_positive(self):
        if self._current_image is None:
            return
        self._save_current(_POS_DIR)
        self._pos_count += 1
        self._update_stats()
        self._set_buttons_enabled(False)
        self.root.after(10, self._do_generate)

    def _on_negative(self):
        if self._current_image is None:
            return
        self._save_current(_NEG_DIR)
        self._neg_count += 1
        self._update_stats()
        self._set_buttons_enabled(False)
        self.root.after(10, self._do_generate)

    def _on_skip(self):
        if self._current_image is None:
            return
        self._skip_count += 1
        self._update_stats()
        self._set_buttons_enabled(False)
        self.root.after(10, self._do_generate)

    def _save_current(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        idx  = _next_idx(folder)
        path = folder / f"morph_{idx:04d}.png"
        self._current_image.save(path)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _set_buttons_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self._btn_pos, self._btn_skip, self._btn_neg):
            btn.config(state=state)

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
            n_bp_legs = sum(1 for l in morph.legs if l.body_part_idx == i)
            bp_lines += f"\n  bp{i+1}: r={bp.radius:.2f} legs={n_bp_legs}"
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
            f"Torso r   : {morph.torso_radius:.3f} m\n"
            f"Torso h   : {morph.torso_height:.3f} m\n"
            f"Euler     : ({rx:.1f}° {ry:.1f}° {rz:.1f}°)\n"
            f"Spawn h   : {spawn_h:.3f} m"
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
    root.geometry("1020x660")
    app = MorphSorterApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
