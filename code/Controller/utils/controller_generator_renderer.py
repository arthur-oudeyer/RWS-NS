"""
utils/controller_generator_renderer.py
=======================================
Interactive controller trainer and renderer.

Pipeline per individual
-----------------------
    random_initial_weights()  or  manual weights
        mutate_weights(current_rw)
            |
            v  (background thread — UI stays live)
        train_from_scratch()  or  train_warm_start()
            + live fitness plot + progress bar
            |
            v
        rollout_to_video()
            |
            v
        Animate frames in canvas at ~20 fps

Controls
--------
    N  / button   new random reward function (train from scratch)
    M  / button   mutate current reward (warm-start)
    T  / button   train on the manually edited reward weights
    B  / <-       go back to previous individual
    S  / Space    skip / discard current
    V  / button   save current (weights + policy + video)
    Q             quit

Right panel
-----------
    Training:    Init steps, Warm steps, Envs, Episode (s)
    Mutation:    Init sigma, Mutate sigma
    Reward weights (7 editable sliders — also show current individual's weights)
    Session stats / Individual info / Config summary

Output (cleared on every launch)
---------------------------------
    utils/study_output/
        policies/   policy_NNNN.zip
        videos/     video_NNNN.mp4
        rewards/    reward_NNNN.json   (V-saved only)
        log.jsonl
"""

from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
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
from reward import RewardWeights, mutate_weights, random_initial_weights
from ppo_trainer import train_from_scratch, train_warm_start
from video_renderer import rollout_to_video
from config import ExperimentConfig

try:
    from stable_baselines3.common.callbacks import BaseCallback as _SB3Callback
except ImportError:
    _SB3Callback = object


# ---------------------------------------------------------------------------
# Module-level config
# ---------------------------------------------------------------------------

_cfg = ExperimentConfig()


def _detect_device() -> str:
    """Return 'cpu' — optimal for small PPO MLPs with SubprocVecEnv.

    MPS is theoretically available on Apple Silicon but is slower here because
    the policy network (256×256 MLP) is tiny: the CPU→MPS data-transfer
    overhead per batch exceeds the GPU compute saving, producing the 0%↔50%
    GPU usage spikes and ~5x slowdown. MPS only wins when the network is large
    enough that compute >> transfer cost.
    """
    return "cpu"


_DEFAULT_DEVICE = _detect_device()


# ---------------------------------------------------------------------------
# Output paths  (cleared on each launch — see main())
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent
_OUT_ROOT    = _SCRIPT_DIR / "study_output"
_POLICY_DIR  = _OUT_ROOT / "policies"
_VIDEO_DIR   = _OUT_ROOT / "videos"
_REWARD_DIR  = _OUT_ROOT / "rewards"
_LOG_FILE    = _OUT_ROOT / "log.jsonl"


# ---------------------------------------------------------------------------
# Slider-controlled training parameters
# ---------------------------------------------------------------------------

@dataclass
class TrainParams:
    n_init_steps:      int   = _cfg.n_init_steps
    n_warm_steps:      int   = _cfg.n_warm_steps
    n_envs:            int   = _cfg.n_envs
    episode_duration:  float = _cfg.episode_duration
    reward_init_sigma: float = _cfg.reward_init_sigma
    reward_mut_sigma:  float = _cfg.reward_mutation_sigma
    device:            str   = _DEFAULT_DEVICE


# ---------------------------------------------------------------------------
# Per-weight slider configuration  (from_, to, resolution)
# ---------------------------------------------------------------------------

_RW_SLIDER_CFG: dict[str, tuple] = {
    "forward_velocity": (0.0,  5.0,  0.05),
    "lateral_drift":    (0.0,  2.0,  0.01),
    "upright_bonus":    (0.0,  5.0,  0.05),
    "energy_penalty":   (0.0,  0.1,  0.001),
    "contact_reward":   (0.0,  2.0,  0.01),
    "alive_bonus":      (0.0,  1.0,  0.005),
    "fall_penalty":     (0.0, 50.0,  0.5),
}

# Default values shown on startup
_RW_DEFAULTS = _cfg.default_reward_weights_dict()


# ---------------------------------------------------------------------------
# Result bundle for one trained individual
# ---------------------------------------------------------------------------

@dataclass
class IndividualResult:
    reward_weights:      RewardWeights
    policy_path:         Optional[str]
    video_path:          Optional[str]
    fitness:             float
    n_steps:             int    # steps in the most recent training run
    total_steps_trained: int    # cumulative across from-scratch + all continues
    mode:                str    # "scratch" | "warm" | "manual" | "continue"
    frames:              list   # list[PIL.Image] for animated playback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_idx(folder: Path, pattern: str) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    indices = [
        int(p.stem.split("_")[-1])
        for p in folder.glob(pattern)
        if p.stem.split("_")[-1].isdigit()
    ]
    return (max(indices) + 1) if indices else 1


def _extract_frames(video_path: str, max_frames: int = 300) -> list:
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        frames = [PILImage.fromarray(f) for i, f in enumerate(reader) if i < max_frames]
        reader.close()
        return frames
    except Exception:
        return []


# ---------------------------------------------------------------------------
# SB3 training callback — posts progress + fitness points to the UI queue
# ---------------------------------------------------------------------------

_result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()


class _TrainingCallback(_SB3Callback):
    """
    Feeds the live-plot canvas and progress bar during PPO training.

    Progress is relative to the current training run (not model lifetime),
    so warm-start runs show 0→1 the same as from-scratch runs.
    """

    def __init__(self, total_steps: int):
        super().__init__(verbose=0)
        self._total    = max(1, total_steps)
        # Post at most ~80 progress ticks; at least every 500 steps.
        self._interval = max(500, total_steps // 80)
        self._start    = 0
        self._last     = 0

    def _on_training_start(self) -> None:
        self._start = self.num_timesteps
        self._last  = 0

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - self._start
        if elapsed - self._last >= self._interval:
            _result_queue.put(("progress", min(1.0, elapsed / self._total)))
            self._last = elapsed
        return True

    def _on_rollout_end(self) -> None:
        buf = list(self.model.ep_info_buffer)
        if buf:
            elapsed = self.num_timesteps - self._start
            mean_r  = float(np.mean([ep["r"] for ep in buf[-20:]]))
            _result_queue.put(("fitness_point", (elapsed, mean_r)))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

_HISTORY_MAX  = 10
_PLAYBACK_MS  = 50   # ~20 fps video playback


class ControllerTrainerApp:
    """
    tkinter app for interactively training and browsing PPO controllers.

    All PPO/MuJoCo work runs in daemon threads; the main loop polls a
    shared queue every 200 ms to stay responsive.
    """

    _CANVAS_W = 700
    _CANVAS_H = 420
    _PANEL_W  = 280

    # Palette
    _BG      = "#1e1e1e"
    _BG2     = "#252525"
    _BG3     = "#2d2d2d"
    _FG      = "#cccccc"
    _FG_DIM  = "#888888"
    _SKIP_BG = "#3a3a3a"
    _MUT_BG  = "#2d4a6a";  _MUT_FG  = "#cce0ff"
    _PREV_BG = "#4a3a2d";  _PREV_FG = "#ffe0cc"
    _NEW_BG  = "#3a2d6a";  _NEW_FG  = "#e0ccff"
    _SAVE_BG = "#2d5a3a";  _SAVE_FG = "#ccffdd"
    _MAN_BG  = "#5a3a1a";  _MAN_FG  = "#ffd8a8"
    _CONT_BG = "#3a3a1a";  _CONT_FG = "#ffff88"

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.rng    = np.random.default_rng()
        self.params = TrainParams()

        self._saved_count = 0
        self._skip_count  = 0

        self._current: Optional[IndividualResult] = None
        self._photo:   Optional[ImageTk.PhotoImage] = None

        # Video playback
        self._play_frames: list        = []
        self._play_idx:    int         = 0
        self._play_after:  Optional[str] = None

        # Live fitness plot
        self._fitness_history:    list  = []   # [(step, fitness), …]
        self._current_total_steps: int  = 1
        self._training_mode:       str  = "scratch"

        # History
        self._history: list[IndividualResult] = []
        self._training = False

        self._build_ui()
        # Enable New button immediately — no individual needed to start
        self._set_buttons_enabled(True)
        self._poll()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.title("Controller Trainer")
        self.root.configure(bg=self._BG)
        self.root.resizable(True, True)

        # Status bar
        self._status_var = tk.StringVar(value="  Ready — press [N] to train a new controller")
        tk.Label(
            self.root, textvariable=self._status_var,
            bg="#2a2a2a", fg=self._FG_DIM, anchor="w", padx=10,
            font=("Courier", 11),
        ).pack(side=tk.TOP, fill=tk.X)

        # Progress bar (thin strip)
        self._pb_canvas = tk.Canvas(
            self.root, bg="#2a2a2a", height=6, highlightthickness=0,
        )
        self._pb_canvas.pack(side=tk.TOP, fill=tk.X)
        self._pb_rect = self._pb_canvas.create_rectangle(0, 0, 0, 6,
                                                          fill="#4488cc", outline="")

        # Main content
        content = tk.Frame(self.root, bg=self._BG)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---- Left: canvas + button rows ----
        left = tk.Frame(content, bg=self._BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._canvas = tk.Canvas(
            left, bg="#2a2a2a",
            width=self._CANVAS_W, height=self._CANVAS_H,
            highlightthickness=1, highlightbackground="#444",
        )
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._show_idle_message()

        btn_cfg = dict(font=("Helvetica", 12, "bold"),
                       relief=tk.FLAT, cursor="hand2", padx=8, pady=8)

        # Nav row: Back | New | Mutate | Continue
        nav = tk.Frame(left, bg=self._BG, pady=3)
        nav.pack(side=tk.TOP, fill=tk.X)
        for i in range(4):
            nav.columnconfigure(i, weight=1)

        self._btn_prev = tk.Button(
            nav, text="<- Back  [B]",
            bg=self._PREV_BG, fg=self._PREV_FG,
            activebackground="#6a5a3a", activeforeground="#fff",
            command=self._on_previous, **btn_cfg)
        self._btn_new = tk.Button(
            nav, text="* New  [N]",
            bg=self._NEW_BG, fg=self._NEW_FG,
            activebackground="#5a3a9a", activeforeground="#fff",
            command=self._on_new, **btn_cfg)
        self._btn_mutate = tk.Button(
            nav, text="~ Mutate  [M]",
            bg=self._MUT_BG, fg=self._MUT_FG,
            activebackground="#3a6a9a", activeforeground="#fff",
            command=self._on_mutate, **btn_cfg)
        self._btn_continue = tk.Button(
            nav, text=">> Continue  [C]",
            bg=self._CONT_BG, fg=self._CONT_FG,
            activebackground="#5a5a2a", activeforeground="#fff",
            command=self._on_continue, **btn_cfg)
        self._btn_prev.grid(    row=0, column=0, sticky="ew", padx=3)
        self._btn_new.grid(     row=0, column=1, sticky="ew", padx=3)
        self._btn_mutate.grid(  row=0, column=2, sticky="ew", padx=3)
        self._btn_continue.grid(row=0, column=3, sticky="ew", padx=3)

        # Action row: Manual | Save | Skip
        act = tk.Frame(left, bg=self._BG, pady=3)
        act.pack(side=tk.TOP, fill=tk.X)
        for i in range(3):
            act.columnconfigure(i, weight=1)

        self._btn_manual = tk.Button(
            act, text="Edit Weights  [T]",
            bg=self._MAN_BG, fg=self._MAN_FG,
            activebackground="#8a5a2a", activeforeground="#fff",
            command=self._on_manual, **btn_cfg)
        self._btn_save = tk.Button(
            act, text="v Save  [V]",
            bg=self._SAVE_BG, fg=self._SAVE_FG,
            activebackground="#3a8a5a", activeforeground="#fff",
            command=self._on_save, **btn_cfg)
        self._btn_skip = tk.Button(
            act, text="> Skip  [S]",
            bg=self._SKIP_BG, fg=self._FG,
            activebackground="#555", activeforeground="#fff",
            command=self._on_skip, **btn_cfg)
        self._btn_manual.grid(row=0, column=0, sticky="ew", padx=3)
        self._btn_save.grid(  row=0, column=1, sticky="ew", padx=3)
        self._btn_skip.grid(  row=0, column=2, sticky="ew", padx=3)

        # Keyboard shortcuts
        for key, cb in (
            ("<n>", self._on_new),       ("<N>", self._on_new),
            ("<m>", self._on_mutate),    ("<M>", self._on_mutate),
            ("<c>", self._on_continue),  ("<C>", self._on_continue),
            ("<t>", self._on_manual),    ("<T>", self._on_manual),
            ("<s>", self._on_skip),      ("<S>", self._on_skip),
            ("<space>", self._on_skip),
            ("<v>", self._on_save),     ("<V>", self._on_save),
            ("<b>", self._on_previous), ("<B>", self._on_previous),
            ("<Left>", self._on_previous),
            ("<q>", lambda e: self.root.destroy()),
            ("<Q>", lambda e: self.root.destroy()),
        ):
            self.root.bind(key, lambda e, f=cb: f())

        # ---- Right: parameter panel ----
        right = tk.Frame(content, bg=self._BG2, width=self._PANEL_W)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        right.pack_propagate(False)
        self._build_param_panel(right)

    # ------------------------------------------------------------------

    def _build_param_panel(self, parent: tk.Frame):
        pc = tk.Canvas(parent, bg=self._BG2, highlightthickness=0)
        ps = tk.Scrollbar(parent, orient=tk.VERTICAL, command=pc.yview)
        p  = tk.Frame(pc, bg=self._BG2)

        p.bind("<Configure>", lambda e: pc.configure(scrollregion=pc.bbox("all")))
        pc.create_window((0, 0), window=p, anchor="nw", width=self._PANEL_W - 16)
        pc.configure(yscrollcommand=ps.set)
        ps.pack(side=tk.RIGHT, fill=tk.Y)
        pc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _wheel(e):
            if e.num == 4:   pc.yview_scroll(-1, "units")
            elif e.num == 5: pc.yview_scroll(1,  "units")
            else:            pc.yview_scroll(int(-1*(e.delta/120)), "units")
        for w in (pc, p):
            w.bind("<MouseWheel>", _wheel)
            w.bind("<Button-4>",   _wheel)
            w.bind("<Button-5>",   _wheel)

        def section(txt):
            tk.Label(p, text=txt, bg=self._BG2, fg="#aaaaaa",
                     font=("Helvetica", 11, "bold")).pack(pady=(10, 3))

        def separator():
            tk.Frame(p, bg="#444444", height=1).pack(fill=tk.X, padx=10, pady=5)

        def note(txt):
            tk.Label(p, text=txt, bg=self._BG2, fg="#666666",
                     justify=tk.LEFT, font=("Courier", 7),
                     wraplength=self._PANEL_W - 30).pack(padx=12, anchor="w")

        self._slider_vars: dict[str, tk.Variable] = {}

        def add_slider(obj, attr, label, from_, to, res, is_int):
            row = tk.Frame(p, bg=self._BG2)
            row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(row, text=label, bg=self._BG2, fg="#bbbbbb",
                     font=("Helvetica", 9), width=16, anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(row, bg=self._BG2, fg="#ffffff",
                               font=("Courier", 9), width=8)
            val_lbl.pack(side=tk.RIGHT)
            var = (tk.IntVar(value=int(getattr(obj, attr))) if is_int
                   else tk.DoubleVar(value=float(getattr(obj, attr))))
            self._slider_vars[attr] = var

            def _cb(a=attr, v=var, lbl=val_lbl, i=is_int, o=obj):
                def cb(*_):
                    val = int(v.get()) if i else float(v.get())
                    setattr(o, a, val)
                    lbl.config(text=str(val) if i else f"{val:.4f}" if val < 0.1 else f"{val:.3f}")
                return cb
            cb = _cb()
            tk.Scale(
                row, variable=var, from_=from_, to=to, resolution=res,
                orient=tk.HORIZONTAL, showvalue=False,
                bg=self._BG3, troughcolor="#555555",
                highlightthickness=0, bd=0, command=cb,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            cb()

        # ---- Training section ----
        section("Training")
        add_slider(self.params, "n_init_steps",     "Init steps",   50_000, 3_000_000, 50_000, True)
        add_slider(self.params, "n_warm_steps",     "Warm steps",   50_000,  1_000_000, 50_000, True)
        add_slider(self.params, "n_envs",           "Envs",         1,       8,     1,     True)
        note("Envs: parallel MuJoCo instances.\nMore = faster but more CPU/RAM.")
        add_slider(self.params, "episode_duration", "Episode (s)",  1.0,    10.0,   0.5,   False)

        # Device selector
        tk.Label(p, text="Device", bg=self._BG2, fg="#bbbbbb",
                 font=("Helvetica", 9), anchor="w").pack(padx=12, anchor="w", pady=(4, 0))
        self._device_var = tk.StringVar(value=self.params.device)
        btn_row = tk.Frame(p, bg=self._BG2)
        btn_row.pack(fill=tk.X, padx=12, pady=1)
        choices = [("cpu", "CPU"), ("auto", "auto")]
        if _DEFAULT_DEVICE == "mps":
            choices.append(("mps", "mps (M2 GPU)"))
        for dev, label in choices:
            tk.Radiobutton(
                btn_row, text=label, variable=self._device_var, value=dev,
                bg=self._BG2, fg=self._FG, selectcolor="#444444",
                activebackground=self._BG2, activeforeground=self._FG,
                font=("Courier", 9),
                command=lambda: setattr(self.params, "device", self._device_var.get()),
            ).pack(side=tk.LEFT, padx=(0, 8))
        note("CPU is fastest here: the 256x256 MLP\nis too small for MPS — transfer overhead\n> compute gain (see 0%%<->50%% GPU spikes).\nMain speedup: more Envs.")

        # ---- Mutation sigma ----
        separator()
        section("Mutation sigma")
        add_slider(self.params, "reward_init_sigma", "Init sigma",   0.05, 2.0, 0.05, False)
        add_slider(self.params, "reward_mut_sigma",  "Mutate sigma", 0.05, 2.0, 0.05, False)
        note("Log-normal noise on each weight.\nHigher = wider exploration.")

        # ---- Manual reward weights (editable, also display current) ----
        separator()
        section("Reward Weights")
        note("Adjust + press [T] to train.\nAuto-fill from loaded individual.")

        self._rw_manual = RewardWeights(**_RW_DEFAULTS)
        self._rw_slider_vars:  dict[str, tk.DoubleVar] = {}
        self._rw_val_labels:   dict[str, tk.Label]     = {}

        for name, (fr, to, res) in _RW_SLIDER_CFG.items():
            row = tk.Frame(p, bg=self._BG2)
            row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(row, text=name, bg=self._BG2, fg="#bbbbbb",
                     font=("Courier", 8), width=17, anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(row, bg=self._BG2, fg="#ffdd88",
                               font=("Courier", 9), width=7)
            val_lbl.pack(side=tk.RIGHT)
            self._rw_val_labels[name] = val_lbl
            default = float(_RW_DEFAULTS[name])
            var = tk.DoubleVar(value=default)
            self._rw_slider_vars[name] = var

            def _rw_cb(n=name, v=var, lbl=val_lbl):
                def cb(*_):
                    val = float(v.get())
                    setattr(self._rw_manual, n, val)
                    lbl.config(text=f"{val:.4f}" if val < 0.1 else f"{val:.3f}")
                return cb
            cb = _rw_cb()
            tk.Scale(
                row, variable=var, from_=fr, to=to, resolution=res,
                orient=tk.HORIZONTAL, showvalue=False,
                bg=self._BG3, troughcolor="#665533",
                highlightthickness=0, bd=0, command=cb,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            cb()

        # ---- Session stats ----
        separator()
        section("Session stats")
        self._stats_var = tk.StringVar()
        tk.Label(p, textvariable=self._stats_var, bg=self._BG2, fg="#88ff88",
                 justify=tk.LEFT, font=("Courier", 10)).pack(padx=14, anchor="w")
        self._refresh_stats()

        # ---- Individual info ----
        separator()
        section("Individual info")
        self._info_var = tk.StringVar(value="–")
        tk.Label(p, textvariable=self._info_var, bg=self._BG2, fg="#99ccff",
                 justify=tk.LEFT, font=("Courier", 10)).pack(padx=14, anchor="w")

        # ---- Config summary (read-only) ----
        separator()
        section("Config (config.py)")
        tk.Label(p, text=(
            f"policy_arch {_cfg.policy_arch}\n"
            f"lr          {_cfg.learning_rate}\n"
            f"gamma       {_cfg.gamma}\n"
            f"batch_size  {_cfg.batch_size}\n"
            f"fall_h      {_cfg.fall_height} m\n"
            f"render      {_cfg.render_width}x{_cfg.render_height}"
        ), bg=self._BG2, fg="#666666", justify=tk.LEFT,
           font=("Courier", 8)).pack(padx=10, anchor="w")

        tk.Frame(p, bg=self._BG2, height=16).pack()

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    def _run_new(self):
        try:
            rw = random_initial_weights(
                cfg_defaults = _RW_DEFAULTS,
                sigma        = self.params.reward_init_sigma,
                rng          = np.random.default_rng(),
            )
            self._run_training(rw, parent_policy=None, mode="scratch")
        except Exception as exc:
            _result_queue.put(("error", str(exc)))

    def _run_mutate(self, parent: IndividualResult):
        try:
            rw = mutate_weights(
                parent = parent.reward_weights,
                sigma  = self.params.reward_mut_sigma,
                rng    = np.random.default_rng(),
            )
            self._run_training(rw, parent_policy=parent.policy_path, mode="warm")
        except Exception as exc:
            _result_queue.put(("error", str(exc)))

    def _run_manual(self):
        try:
            # Copy current slider values into a fresh RewardWeights
            rw = RewardWeights(**{n: float(v.get())
                                  for n, v in self._rw_slider_vars.items()})
            self._run_training(rw, parent_policy=None, mode="manual")
        except Exception as exc:
            _result_queue.put(("error", str(exc)))

    def _run_training(
        self,
        rw:            RewardWeights,
        parent_policy: Optional[str],
        mode:          str,
    ):
        """Shared training + rollout logic for all three entry points."""
        try:
            _POLICY_DIR.mkdir(parents=True, exist_ok=True)
            _VIDEO_DIR.mkdir(parents=True,  exist_ok=True)
            idx         = _next_idx(_POLICY_DIR, "policy_*.zip")
            policy_path = str(_POLICY_DIR / f"policy_{idx:04d}.zip")
            video_path  = str(_VIDEO_DIR  / f"video_{idx:04d}.mp4")

            cb = _TrainingCallback(
                self.params.n_init_steps if (mode in ("scratch", "manual"))
                else self.params.n_warm_steps
            )

            if mode in ("scratch", "manual"):
                _result_queue.put(("status",
                    f"  Training from scratch ({self.params.n_init_steps:,} steps)…"))
                model, fitness = train_from_scratch(
                    reward_weights   = rw,
                    seed             = int(np.random.randint(0, 2**31)),
                    total_timesteps  = self.params.n_init_steps,
                    n_envs           = self.params.n_envs,
                    save_path        = policy_path,
                    episode_duration = self.params.episode_duration,
                    use_subproc      = (self.params.n_envs > 1),
                    verbose          = 0,
                    callback         = cb,
                    device           = self.params.device,
                )
            else:
                _result_queue.put(("status",
                    f"  Warm-start training ({self.params.n_warm_steps:,} steps)…"))
                model, fitness = train_warm_start(
                    reward_weights     = rw,
                    parent_policy_path = parent_policy,
                    seed               = int(np.random.randint(0, 2**31)),
                    n_warm_steps       = self.params.n_warm_steps,
                    n_envs             = self.params.n_envs,
                    save_path_warmed   = policy_path,
                    episode_duration   = self.params.episode_duration,
                    use_subproc        = (self.params.n_envs > 1),
                    verbose            = 0,
                    callback           = cb,
                    device             = self.params.device,
                )

            # Signal progress bar full before rollout starts
            _result_queue.put(("progress", 1.0))
            _result_queue.put(("status", "  Rolling out to video…"))

            from mujoco_env import RobotControllerEnv
            env = RobotControllerEnv(
                reward_weights   = rw,
                seed             = 0,
                episode_duration = self.params.episode_duration,
                render_mode      = "rgb_array",
                render_width     = _cfg.render_width,
                render_height    = _cfg.render_height,
            )
            rollout_to_video(model, env, video_path, fps=_cfg.video_fps)
            env.close()

            frames   = _extract_frames(video_path)
            n_this   = (self.params.n_init_steps if mode in ("scratch", "manual")
                        else self.params.n_warm_steps)
            _result_queue.put(("done", IndividualResult(
                reward_weights      = rw,
                policy_path         = policy_path,
                video_path          = video_path,
                fitness             = float(fitness),
                n_steps             = n_this,
                total_steps_trained = n_this,
                mode                = mode,
                frames              = frames,
            )))
        except Exception as exc:
            _result_queue.put(("error", str(exc)))

    def _run_continue(self, parent: IndividualResult):
        """Continue training the current individual (same weights, warm start)."""
        try:
            rw      = parent.reward_weights
            n_steps = self.params.n_warm_steps

            _POLICY_DIR.mkdir(parents=True, exist_ok=True)
            _VIDEO_DIR.mkdir(parents=True,  exist_ok=True)
            idx         = _next_idx(_POLICY_DIR, "policy_*.zip")
            policy_path = str(_POLICY_DIR / f"policy_{idx:04d}.zip")
            video_path  = str(_VIDEO_DIR  / f"video_{idx:04d}.mp4")

            cb = _TrainingCallback(n_steps)
            _result_queue.put(("status",
                f"  Continuing training ({n_steps:,} more steps, "
                f"total will be {parent.total_steps_trained + n_steps:,})…"))

            model, fitness = train_warm_start(
                reward_weights     = rw,
                parent_policy_path = parent.policy_path,
                seed               = int(np.random.randint(0, 2**31)),
                n_warm_steps       = n_steps,
                n_envs             = self.params.n_envs,
                save_path_warmed   = policy_path,
                episode_duration   = self.params.episode_duration,
                use_subproc        = (self.params.n_envs > 1),
                verbose            = 0,
                callback           = cb,
                device             = self.params.device,
            )

            _result_queue.put(("progress", 1.0))
            _result_queue.put(("status", "  Rolling out to video…"))

            from mujoco_env import RobotControllerEnv
            env = RobotControllerEnv(
                reward_weights   = rw,
                seed             = 0,
                episode_duration = self.params.episode_duration,
                render_mode      = "rgb_array",
                render_width     = _cfg.render_width,
                render_height    = _cfg.render_height,
            )
            rollout_to_video(model, env, video_path, fps=_cfg.video_fps)
            env.close()

            frames = _extract_frames(video_path)
            total  = parent.total_steps_trained + n_steps
            _result_queue.put(("done", IndividualResult(
                reward_weights      = rw,
                policy_path         = policy_path,
                video_path          = video_path,
                fitness             = float(fitness),
                n_steps             = n_steps,
                total_steps_trained = total,
                mode                = "continue",
                frames              = frames,
            )))
        except Exception as exc:
            _result_queue.put(("error", str(exc)))

    # ------------------------------------------------------------------
    # Queue polling — called every 200 ms via root.after
    # ------------------------------------------------------------------

    def _poll(self):
        try:
            while True:
                msg_type, payload = _result_queue.get_nowait()
                if msg_type == "status":
                    self._status_var.set(payload)

                elif msg_type == "progress":
                    self._update_progress_bar(float(payload))

                elif msg_type == "fitness_point":
                    self._fitness_history.append(payload)
                    self._draw_fitness_plot()

                elif msg_type == "done":
                    self._training = False
                    self._update_progress_bar(1.0, color="#33cc66")
                    self._apply_result(payload)

                elif msg_type == "error":
                    self._training = False
                    self._update_progress_bar(0.0)
                    self._status_var.set(f"  ERROR: {payload}")
                    self._set_buttons_enabled(True)
        except queue.Empty:
            pass
        self.root.after(200, self._poll)

    # ------------------------------------------------------------------
    # Apply result
    # ------------------------------------------------------------------

    def _apply_result(self, result: IndividualResult):
        self._push_history()
        self._current = result
        self._start_playback(result.frames)
        self._sync_rw_sliders(result.reward_weights)
        self._update_info(result)
        total = result.total_steps_trained
        last  = result.n_steps
        self._status_var.set(
            f"  [{result.mode}  +{last:,} steps / {total:,} total]   "
            f"fitness = {result.fitness:+.3f}   "
            f"frames = {len(result.frames)}"
        )
        self._set_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------

    def _update_progress_bar(self, fraction: float, color: str = "#4488cc"):
        self._pb_canvas.update_idletasks()
        w = max(1, self._pb_canvas.winfo_width())
        self._pb_canvas.coords(self._pb_rect, 0, 0, int(w * min(fraction, 1.0)), 6)
        self._pb_canvas.itemconfig(self._pb_rect, fill=color)

    # ------------------------------------------------------------------
    # Live fitness plot
    # ------------------------------------------------------------------

    def _draw_fitness_plot(self):
        if not self._fitness_history:
            return

        cw = max(self._canvas.winfo_width(),  self._CANVAS_W)
        ch = max(self._canvas.winfo_height(), self._CANVAS_H)

        PL, PR, PT, PB = 58, 18, 38, 32   # left/right/top/bottom padding
        pw = cw - PL - PR
        ph = ch - PT - PB

        steps = [p[0] for p in self._fitness_history]
        fitns = [p[1] for p in self._fitness_history]

        x_max = max(self._current_total_steps, steps[-1])
        y_min = min(fitns);  y_max = max(fitns)
        if y_min == y_max:
            y_min -= 1.0;  y_max += 1.0
        pad_y = (y_max - y_min) * 0.12
        y_min -= pad_y;  y_max += pad_y

        def px(step, fit):
            x = PL + (step / x_max) * pw
            y = PT + (1.0 - (fit - y_min) / (y_max - y_min)) * ph
            return x, y

        self._canvas.delete("all")
        self._canvas.configure(bg="#141414")

        # Horizontal grid + Y labels
        for i in range(5):
            gy  = PT + i * ph / 4
            val = y_max - i * (y_max - y_min) / 4
            self._canvas.create_line(PL, gy, PL + pw, gy, fill="#252525", dash=(3, 4))
            self._canvas.create_text(PL - 4, gy, text=f"{val:+.1f}",
                                     fill="#666", anchor="e", font=("Courier", 7))

        # Axes
        self._canvas.create_line(PL, PT, PL, PT + ph, fill="#555", width=1)
        self._canvas.create_line(PL, PT + ph, PL + pw, PT + ph, fill="#555", width=1)

        # X labels
        self._canvas.create_text(PL, PT + ph + 14, text="0",
                                 fill="#555", anchor="w", font=("Courier", 7))
        self._canvas.create_text(PL + pw, PT + ph + 14, text=f"{x_max:,}",
                                 fill="#555", anchor="e", font=("Courier", 7))
        self._canvas.create_text(PL + pw // 2, PT + ph + 24,
                                 text="env steps", fill="#555", font=("Courier", 7))

        # Title
        mode_label = {"scratch": "from scratch", "warm": "warm start",
                      "manual": "manual weights", "continue": "continue"
                      }.get(self._training_mode, self._training_mode)
        title = f"{mode_label}   fitness = {fitns[-1]:+.2f}   step {steps[-1]:,} / {x_max:,}"
        self._canvas.create_text(cw // 2, PT // 2, text=title,
                                 fill="#aaaaaa", font=("Courier", 11))

        # Curve
        if len(self._fitness_history) >= 2:
            coords = [c for s, f in self._fitness_history for c in px(s, f)]
            self._canvas.create_line(*coords, fill="#44ff88", width=2, smooth=True)

        # Last-point dot
        lx, ly = px(steps[-1], fitns[-1])
        self._canvas.create_oval(lx-4, ly-4, lx+4, ly+4, fill="#44ff88", outline="")

    # ------------------------------------------------------------------
    # Video playback
    # ------------------------------------------------------------------

    def _start_playback(self, frames: list):
        self._stop_playback()
        self._play_frames = frames
        self._play_idx    = 0
        if frames:
            self._play_next_frame()
        else:
            self._show_canvas_message("No frames rendered")

    def _stop_playback(self):
        if self._play_after is not None:
            self.root.after_cancel(self._play_after)
            self._play_after = None
        self._play_frames = []
        self._play_idx    = 0

    def _play_next_frame(self):
        if not self._play_frames:
            return
        self._show_image(self._play_frames[self._play_idx])
        self._play_idx   = (self._play_idx + 1) % len(self._play_frames)
        self._play_after = self.root.after(_PLAYBACK_MS, self._play_next_frame)

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

    def _on_new(self):
        if self._training:
            return
        self._start_training("scratch")
        threading.Thread(target=self._run_new, daemon=True).start()

    def _on_mutate(self):
        if self._training or self._current is None:
            return
        parent = self._current
        self._start_training("warm")
        threading.Thread(target=lambda: self._run_mutate(parent), daemon=True).start()

    def _on_manual(self):
        if self._training:
            return
        self._start_training("manual")
        threading.Thread(target=self._run_manual, daemon=True).start()

    def _on_continue(self):
        if self._training or self._current is None:
            return
        parent = self._current
        self._start_training("continue")
        threading.Thread(target=lambda: self._run_continue(parent), daemon=True).start()

    def _on_skip(self):
        if self._training:
            return
        self._skip_count += 1
        self._stop_playback()
        self._current = None
        self._show_idle_message()
        self._info_var.set("–")
        self._status_var.set("  Skipped")
        self._refresh_stats()
        self._set_buttons_enabled(True)

    def _on_save(self):
        if self._current is None:
            return
        _REWARD_DIR.mkdir(parents=True, exist_ok=True)
        idx  = _next_idx(_REWARD_DIR, "reward_*.json")
        path = _REWARD_DIR / f"reward_{idx:04d}.json"
        entry = {
            "idx":                 idx,
            "fitness":             self._current.fitness,
            "mode":                self._current.mode,
            "n_steps":             self._current.n_steps,
            "total_steps_trained": self._current.total_steps_trained,
            "reward_weights":      self._current.reward_weights.to_dict(),
            "policy_path":         self._current.policy_path,
            "video_path":          self._current.video_path,
        }
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._saved_count += 1
        self._refresh_stats()
        self._status_var.set(f"  Saved #{self._saved_count}  ->  {path.name}")

    def _on_previous(self):
        if not self._history:
            return
        self._stop_playback()
        result = self._history.pop()
        self._current = result
        self._start_playback(result.frames)
        self._sync_rw_sliders(result.reward_weights)
        self._update_info(result)
        self._status_var.set(
            f"  [restored]   fitness={result.fitness:+.3f}   "
            f"history={len(self._history)} left"
        )
        self._set_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _start_training(self, mode: str):
        self._training           = True
        self._fitness_history    = []
        self._training_mode      = mode
        self._current_total_steps = (
            self.params.n_init_steps if mode in ("scratch", "manual")
            else self.params.n_warm_steps
        )
        self._set_buttons_enabled(False)
        self._stop_playback()
        self._update_progress_bar(0.0)
        label = {
            "scratch":  "Sampling random weights + training…",
            "warm":     "Mutating weights + warm-start training…",
            "manual":   "Training on manual weights…",
            "continue": "Continuing training (warm start)…",
        }[mode]
        self._show_canvas_message(label)

    def _push_history(self):
        if self._current is None:
            return
        if len(self._history) >= _HISTORY_MAX:
            self._history.pop(0)
        self._history.append(self._current)

    def _show_idle_message(self):
        cw = max(self._canvas.winfo_width(),  self._CANVAS_W)
        ch = max(self._canvas.winfo_height(), self._CANVAS_H)
        self._canvas.delete("all")
        self._canvas.configure(bg="#2a2a2a")
        self._canvas.create_text(
            cw // 2, ch // 2,
            text="Press [N] to train a new controller",
            fill=self._FG_DIM, font=("Courier", 15),
        )

    def _show_canvas_message(self, msg: str):
        cw = max(self._canvas.winfo_width(),  self._CANVAS_W)
        ch = max(self._canvas.winfo_height(), self._CANVAS_H)
        self._canvas.delete("all")
        self._canvas.configure(bg="#1a1a1a")
        self._canvas.create_text(
            cw // 2, ch // 2, text=msg,
            fill=self._FG_DIM, font=("Courier", 14),
        )

    def _set_buttons_enabled(self, enabled: bool):
        has_c = self._current is not None
        has_h = bool(self._history)
        needs_c = enabled and has_c
        self._btn_new.config(     state=tk.NORMAL if enabled  else tk.DISABLED)
        self._btn_mutate.config(  state=tk.NORMAL if needs_c  else tk.DISABLED)
        self._btn_continue.config(state=tk.NORMAL if needs_c  else tk.DISABLED)
        self._btn_manual.config(  state=tk.NORMAL if enabled  else tk.DISABLED)
        self._btn_prev.config(    state=tk.NORMAL if (enabled and has_h) else tk.DISABLED)
        self._btn_save.config(    state=tk.NORMAL if needs_c  else tk.DISABLED)
        self._btn_skip.config(    state=tk.NORMAL if enabled  else tk.DISABLED)

    def _refresh_stats(self):
        self._stats_var.set(
            f"Saved    : {self._saved_count}\n"
            f"Skipped  : {self._skip_count}\n"
            f"History  : {len(self._history)}"
        )

    def _sync_rw_sliders(self, rw: RewardWeights):
        """Update the manual reward-weight sliders to reflect a loaded individual."""
        for name, var in self._rw_slider_vars.items():
            val = float(getattr(rw, name))
            var.set(val)
            setattr(self._rw_manual, name, val)
            # Also update the value label directly — var.set() doesn't fire command=
            lbl = self._rw_val_labels[name]
            lbl.config(text=f"{val:.4f}" if val < 0.1 else f"{val:.3f}")

    def _update_info(self, result: IndividualResult):
        self._info_var.set(
            f"Mode    : {result.mode}\n"
            f"Last run: {result.n_steps:,} steps\n"
            f"Total   : {result.total_steps_trained:,} steps\n"
            f"Fitness : {result.fitness:+.3f}\n"
            f"Frames  : {len(result.frames)}\n"
            f"History : {len(self._history)}"
        )
        self._refresh_stats()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not _PIL_AVAILABLE:
        print("ERROR: Pillow required.  pip install Pillow")
        sys.exit(1)
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("ERROR: mujoco required.  pip install mujoco")
        sys.exit(1)

    # Clear study_output on every launch for a clean session
    if _OUT_ROOT.exists():
        shutil.rmtree(_OUT_ROOT)
    _OUT_ROOT.mkdir(parents=True)

    root = tk.Tk()
    root.geometry("1120x730")
    ControllerTrainerApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
