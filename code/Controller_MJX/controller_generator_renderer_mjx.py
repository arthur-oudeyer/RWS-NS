"""
controller_generator_renderer_mjx.py
=====================================
Interactive MJX controller trainer and renderer.

Identical UI to Controller/utils/controller_generator_renderer.py, but the
backend is fully MJX / pure-JAX — no Stable Baselines 3 dependency.

Pipeline per individual
-----------------------
    random_initial_weights()  or  manual weights
        mutate_weights(current_rw)
            |
            v  (background thread)
        train_from_scratch_mjx()  or  train_warm_start_mjx()
            + timed progress bar (no SB3 callback — time-based estimate)
            |
            v
        rollout_to_video_mjx()
            |
            v
        Animate frames in canvas at ~20 fps

Controls
--------
    N  / button   new random reward function (train from scratch)
    M  / button   mutate current reward (warm-start)
    T  / button   train on the manually edited reward weights
    C  / button   continue training current (warm-start, same weights)
    B  / <-       go back to previous individual
    S  / Space    skip / discard current
    V  / button   save current (weights + policy + video)
    L  / button   load existing .params policy file
    Q             quit

Right panel
-----------
    Training:    Init steps, Warm steps, n_envs_mjx, Rollout len, Episode (s)
    Mutation:    Init sigma, Mutate sigma
    Reward weights (all 23 editable sliders)
    Session stats / Individual info / Config summary

Output (cleared on every launch)
---------------------------------
    mjx_study_output/
        policies/   policy_NNNN.params
        videos/     video_NNNN.mp4
        rewards/    reward_NNNN.json   (V-saved only)
        log.jsonl
"""

from __future__ import annotations

import os
# Headless rendering fallback (GUI uses GLFW by default, so EGL only matters
# if someone runs this on a remote box with X-forwarding off).
os.environ.setdefault("MUJOCO_GL", "egl")
# XLA optimizations (no-op on CPU/Metal; helps when run on a CUDA box).
os.environ.setdefault("XLA_FLAGS",
    "--xla_gpu_enable_cublaslt=true --xla_gpu_autotune_level=4"
)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import json
import queue
import shutil
import sys
import threading
import time
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
sys.path.insert(0, str(Path(__file__).parent))
from reward          import RewardWeights, mutate_weights, random_initial_weights
from ppo_trainer_mjx import (
    train_from_scratch_mjx, train_warm_start_mjx, PPOConfig, make_params,
)
from video_renderer_mjx import rollout_to_video_mjx
from evolution_mjx      import save_params, load_params
from mujoco_env_mjx     import build_env_config, _pick_mjx_device
from controller_morph   import build_model
from config             import ExperimentConfig

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Startup: force CPU (MJX doesn't support Metal)
# ---------------------------------------------------------------------------

_dev = _pick_mjx_device()
jax.config.update("jax_default_device", _dev)

# Persistent JIT cache — second run with same shapes skips XLA compile (~60s).
_jax_cache_dir = os.path.expanduser("~/.cache/jax_mjx")
os.makedirs(_jax_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


# ---------------------------------------------------------------------------
# Module-level config
# ---------------------------------------------------------------------------

_cfg = ExperimentConfig()


# ---------------------------------------------------------------------------
# Output paths  (cleared on each launch)
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent
_OUT_ROOT    = _SCRIPT_DIR / "mjx_study_output"
_POLICY_DIR  = _OUT_ROOT / "policies"
_VIDEO_DIR   = _OUT_ROOT / "videos"
_REWARD_DIR  = _OUT_ROOT / "rewards"
_LOG_FILE    = _OUT_ROOT / "log.jsonl"


# ---------------------------------------------------------------------------
# Slider-controlled training parameters
# ---------------------------------------------------------------------------

@dataclass
class TrainParams:
    # Defaults tuned for Mac M2 CPU (MJX runs on CPU, not Metal).
    # Scale up n_init_steps / n_envs_mjx when a GPU is available.
    n_init_steps:      int   = 2_000
    n_warm_steps:      int   = 1_000
    n_envs_mjx:        int   = 4
    rollout_len:       int   = 32
    episode_duration:  float = _cfg.episode_duration
    reward_init_sigma: float = _cfg.reward_init_sigma
    reward_mut_sigma:  float = _cfg.reward_mutation_sigma
    fall_height:       float = _cfg.fall_height


# ---------------------------------------------------------------------------
# Reward weight slider config (same as original)
# ---------------------------------------------------------------------------

_RW_SLIDER_CFG: dict[str, tuple] = {
    "forward_velocity":           (0.0,  5.0,  0.05),
    "lateral_drift":              (0.0,  2.0,  0.01),
    "upright_bonus":              (0.0,  5.0,  0.05),
    "energy_penalty":             (0.0,  0.1,  0.001),
    "contact_reward":             (0.0,  2.0,  0.01),
    "alive_bonus":                (0.0,  1.0,  0.005),
    "fall_penalty":               (0.0, 50.0,  0.5),
    "no_contact_reward":          (0.0,  2.0,  0.01),
    "torso_height_reward":        (0.0,  5.0,  0.05),
    "torso_rotation_reward":      (0.0,  2.0,  0.01),
    "torso_tilting_speed_reward": (0.0,  2.0,  0.01),
    "limb_coordination_reward":   (0.0,  2.0,  0.01),
    "nervosity_reward":           (0.0,  2.0,  0.01),
    "smooth_reward":              (0.0,  2.0,  0.01),
    "vertical_velocity_reward":   (0.0,  5.0,  0.05),
    "lateral_velocity_reward":    (0.0,  2.0,  0.01),
    "joint_range_reward":         (0.0,  2.0,  0.01),
    "height_target_reward":       (0.0,  5.0,  0.05),
    "tilt_penalty":               (0.0,  5.0,  0.05),
    "tilt_rate_penalty":          (0.0,  2.0,  0.01),
    "all_feet_planted_bonus":     (0.0,  1.0,  0.01),
    "vertical_velocity_penalty":  (0.0,  1.0,  0.01),
    "horizontal_velocity_penalty":(0.0, 20.0,  0.05),
}

_RW_DEFAULTS = _cfg.default_reward_weights_dict()


# ---------------------------------------------------------------------------
# Shared result queue + result bundle
# ---------------------------------------------------------------------------

_result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()


@dataclass
class IndividualResult:
    reward_weights:      RewardWeights
    policy_path:         Optional[str]
    video_path:          Optional[str]
    fitness:             float
    n_steps:             int
    total_steps_trained: int
    mode:                str    # "scratch" | "warm" | "manual" | "continue" | "loaded"
    frames:              list   # list[PIL.Image]


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
# Timed progress thread  (replaces SB3 callback)
# ---------------------------------------------------------------------------

def _timed_progress_thread(total_steps: int, stop_evt: threading.Event):
    """
    Post ("progress", fraction) messages based on elapsed wall time.

    We estimate total time from past runs (rough). The progress fraction is
    purely cosmetic — it gives the user feedback that training is running.
    """
    t0 = time.perf_counter()
    # Conservative estimate: first run pays JIT compilation (~30s on M2 CPU),
    # then ~50 effective steps/s.  We cap display at 0.92 so the bar never
    # "finishes" before training does — stop_evt.set() + ("progress", 1.0)
    # is the authoritative completion signal.
    est_seconds = max(30.0, total_steps / 50.0)
    while not stop_evt.is_set():
        elapsed = time.perf_counter() - t0
        frac = min(0.92, elapsed / est_seconds)
        _result_queue.put(("progress", frac))
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# MJX env config cache — built once, updated per-individual via dc_replace
# ---------------------------------------------------------------------------

_template_cfg: Optional[object] = None
_mj_model     = None


def _get_template_cfg(params: TrainParams):
    global _template_cfg, _mj_model
    if _template_cfg is None:
        _result_queue.put(("status", "  Building MJX model (first time, ~3s) …"))
        _template_cfg = build_env_config(
            reward_weights    = RewardWeights(),
            episode_duration  = params.episode_duration,
            control_frequency = _cfg.control_frequency,
            fall_height       = params.fall_height,
        )
        _mj_model, _ = build_model()
    return _template_cfg, _mj_model


def _env_cfg_for(rw: RewardWeights, params: TrainParams):
    from dataclasses import replace as dc_replace
    tmpl, mj_m = _get_template_cfg(params)
    return dc_replace(tmpl, reward_weights_vec=rw.to_jax_vector()), mj_m


def _ppo_cfg(params: TrainParams) -> PPOConfig:
    return PPOConfig(n_epochs=4, n_minibatches=32)


# ---------------------------------------------------------------------------
# Background training workers
# ---------------------------------------------------------------------------

def _run_training(
    rw:            RewardWeights,
    parent_params: Optional[object],
    mode:          str,
    params:        TrainParams,
):
    """Train + rollout + push result/error to _result_queue. Runs in daemon thread."""
    stop_evt = threading.Event()
    n_steps  = (params.n_init_steps if mode in ("scratch", "manual")
                else params.n_warm_steps)
    progress_thread = threading.Thread(
        target=_timed_progress_thread, args=(n_steps, stop_evt), daemon=True
    )
    progress_thread.start()

    try:
        env_cfg, mj_m = _env_cfg_for(rw, params)

        if mode in ("scratch", "manual"):
            _result_queue.put(("status",
                f"  MJX training from scratch ({n_steps:,} steps, "
                f"{params.n_envs_mjx} envs) …"))
            trained_params, fitness = train_from_scratch_mjx(
                cfg             = env_cfg,
                seed            = int(np.random.randint(0, 2**31)),
                total_steps     = n_steps,
                n_envs          = params.n_envs_mjx,
                rollout_len     = params.rollout_len,
                policy_arch     = tuple(_cfg.policy_arch),
                ppo_cfg         = _ppo_cfg(params),
                fitness_episodes = 10,
                verbose         = False,
            )
        else:
            _result_queue.put(("status",
                f"  MJX warm-start ({n_steps:,} steps, "
                f"{params.n_envs_mjx} envs) …"))
            trained_params, fitness = train_warm_start_mjx(
                parent_params   = parent_params,
                cfg             = env_cfg,
                seed            = int(np.random.randint(0, 2**31)),
                total_steps     = n_steps,
                n_envs          = params.n_envs_mjx,
                rollout_len     = params.rollout_len,
                policy_arch     = tuple(_cfg.policy_arch),
                ppo_cfg         = _ppo_cfg(params),
                fitness_episodes = 10,
                verbose         = False,
            )

        stop_evt.set()
        _result_queue.put(("progress", 1.0))
        _result_queue.put(("status", "  Rendering rollout …"))

        _POLICY_DIR.mkdir(parents=True, exist_ok=True)
        _VIDEO_DIR.mkdir(parents=True,  exist_ok=True)
        idx         = _next_idx(_POLICY_DIR, "policy_*.params")
        policy_path = str(_POLICY_DIR / f"policy_{idx:04d}.params")
        video_path  = str(_VIDEO_DIR  / f"video_{idx:04d}.mp4")

        save_params(trained_params, policy_path)

        _, rollout_info = rollout_to_video_mjx(
            params               = trained_params,
            cfg                  = env_cfg,
            mj_model             = mj_m,
            save_path            = video_path,
            fps                  = _cfg.video_fps,
            render_width         = _cfg.render_width,
            render_height        = _cfg.render_height,
            cam1_azimuth         = _cfg.cam1_azimuth,
            cam1_elevation       = _cfg.cam1_elevation,
            cam1_distance        = _cfg.cam1_distance,
            cam1_lookat_z        = _cfg.cam1_lookat_z,
            cam2_azimuth         = _cfg.cam2_azimuth,
            cam2_elevation       = _cfg.cam2_elevation,
            cam2_distance        = _cfg.cam2_distance,
            camera_track_torso   = True,
            policy_arch          = tuple(_cfg.policy_arch),
            seed                 = int(np.random.randint(0, 2**31)),
            deterministic        = True,
        )

        frames = _extract_frames(video_path)
        _result_queue.put(("done", IndividualResult(
            reward_weights      = rw,
            policy_path         = policy_path,
            video_path          = video_path,
            fitness             = float(fitness),
            n_steps             = n_steps,
            total_steps_trained = n_steps,
            mode                = mode,
            frames              = frames,
        )))

    except Exception:
        import traceback
        stop_evt.set()
        _result_queue.put(("error", traceback.format_exc()))


def _run_load(policy_params_path: str, params: TrainParams):
    """Load an existing .params file, run rollout preview, push result."""
    try:
        p = Path(policy_params_path)
        _result_queue.put(("status", f"  Loading {p.name} …"))
        loaded_params = load_params(policy_params_path)

        # Try to find companion reward JSON
        reward_data = _find_reward_json(policy_params_path)
        if reward_data and "reward_weights" in reward_data:
            rw          = RewardWeights(**reward_data["reward_weights"])
            total_steps = int(reward_data.get("total_steps_trained", 0))
        else:
            rw          = RewardWeights(**_RW_DEFAULTS)
            total_steps = 0

        env_cfg, mj_m = _env_cfg_for(rw, params)
        _result_queue.put(("progress", 0.4))
        _result_queue.put(("status", "  Rendering loaded policy …"))

        _VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        idx        = _next_idx(_VIDEO_DIR, "video_*.mp4")
        video_path = str(_VIDEO_DIR / f"video_{idx:04d}.mp4")

        _, rollout_info = rollout_to_video_mjx(
            params               = loaded_params,
            cfg                  = env_cfg,
            mj_model             = mj_m,
            save_path            = video_path,
            fps                  = _cfg.video_fps,
            render_width         = _cfg.render_width,
            render_height        = _cfg.render_height,
            cam1_azimuth         = _cfg.cam1_azimuth,
            cam1_elevation       = _cfg.cam1_elevation,
            cam1_distance        = _cfg.cam1_distance,
            cam1_lookat_z        = _cfg.cam1_lookat_z,
            cam2_azimuth         = _cfg.cam2_azimuth,
            cam2_elevation       = _cfg.cam2_elevation,
            cam2_distance        = _cfg.cam2_distance,
            camera_track_torso   = True,
            policy_arch          = tuple(_cfg.policy_arch),
            seed                 = 0,
            deterministic        = True,
        )

        frames  = _extract_frames(video_path)
        fitness = float(rollout_info.get("total_reward", 0.0))
        _result_queue.put(("progress", 1.0))
        _result_queue.put(("done", IndividualResult(
            reward_weights      = rw,
            policy_path         = policy_params_path,
            video_path          = video_path,
            fitness             = fitness,
            n_steps             = 0,
            total_steps_trained = total_steps,
            mode                = "loaded",
            frames              = frames,
        )))
    except Exception:
        import traceback
        _result_queue.put(("error", traceback.format_exc()))


def _find_reward_json(params_path: str) -> Optional[dict]:
    """Locate companion reward JSON for a .params file (same index)."""
    p       = Path(params_path)
    parts   = p.stem.split("_")
    idx_str = parts[-1] if parts and parts[-1].isdigit() else None
    if idx_str:
        for cand in [
            p.parent / f"reward_{idx_str}.json",
            _REWARD_DIR / f"reward_{idx_str}.json",
        ]:
            if cand.exists():
                try:
                    with open(cand) as f:
                        return json.load(f)
                except Exception:
                    pass
    if _LOG_FILE.exists():
        try:
            with open(_LOG_FILE) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("policy_path") == str(params_path):
                            return entry
                    except Exception:
                        pass
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

_HISTORY_MAX = 10
_PLAYBACK_MS = 50   # ~20 fps


class ControllerTrainerMJXApp:
    """
    tkinter app for interactively training MJX controllers.

    Mirrors ControllerTrainerApp from controller_generator_renderer.py
    with MJX-specific tweaks (no SB3 callback, .params files).
    """

    _CANVAS_W = 700
    _CANVAS_H = 420
    _PANEL_W  = 290

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
    _LOAD_BG = "#3a1a4a";  _LOAD_FG = "#ffccff"

    def __init__(self, root: tk.Tk, preload_path: Optional[str] = None):
        self.root   = root
        self.rng    = np.random.default_rng()
        self.params = TrainParams()

        self._saved_count = 0
        self._skip_count  = 0

        self._current: Optional[IndividualResult] = None
        self._photo:   Optional[object]           = None

        self._play_frames: list         = []
        self._play_idx:    int          = 0
        self._play_after:  Optional[str] = None

        self._fitness_history:    list = []
        self._current_total_steps: int = 1
        self._training_mode:       str = "scratch"

        self._history: list[IndividualResult] = []
        self._training = False

        self._build_ui()
        self._set_buttons_enabled(True)
        self._poll()
        if preload_path:
            self.root.after(300, lambda: self._trigger_load(preload_path))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.title("Controller Trainer  [MJX]")
        self.root.configure(bg=self._BG)
        self.root.resizable(True, True)

        # Status bar
        self._status_var = tk.StringVar(value="  Ready — press [N] to train (MJX backend)")
        tk.Label(
            self.root, textvariable=self._status_var,
            bg="#2a2a2a", fg=self._FG_DIM, anchor="w", padx=10,
            font=("Courier", 11),
        ).pack(side=tk.TOP, fill=tk.X)

        # Progress bar
        self._pb_canvas = tk.Canvas(self.root, bg="#2a2a2a", height=6, highlightthickness=0)
        self._pb_canvas.pack(side=tk.TOP, fill=tk.X)
        self._pb_rect = self._pb_canvas.create_rectangle(0, 0, 0, 6, fill="#44aacc", outline="")

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

        btn_cfg = dict(font=("Helvetica", 12, "bold"), relief=tk.FLAT, cursor="hand2",
                       padx=8, pady=8)

        # Nav row
        nav = tk.Frame(left, bg=self._BG, pady=3)
        nav.pack(side=tk.TOP, fill=tk.X)
        for i in range(4): nav.columnconfigure(i, weight=1)

        self._btn_prev     = tk.Button(nav, text="<- Back  [B]",
            bg=self._PREV_BG, fg=self._PREV_FG, activebackground="#6a5a3a",
            command=self._on_previous, **btn_cfg)
        self._btn_new      = tk.Button(nav, text="* New  [N]",
            bg=self._NEW_BG,  fg=self._NEW_FG,  activebackground="#5a3a9a",
            command=self._on_new, **btn_cfg)
        self._btn_mutate   = tk.Button(nav, text="~ Mutate  [M]",
            bg=self._MUT_BG,  fg=self._MUT_FG,  activebackground="#3a6a9a",
            command=self._on_mutate, **btn_cfg)
        self._btn_continue = tk.Button(nav, text=">> Continue  [C]",
            bg=self._CONT_BG, fg=self._CONT_FG, activebackground="#5a5a2a",
            command=self._on_continue, **btn_cfg)
        self._btn_prev.grid(    row=0, column=0, sticky="ew", padx=3)
        self._btn_new.grid(     row=0, column=1, sticky="ew", padx=3)
        self._btn_mutate.grid(  row=0, column=2, sticky="ew", padx=3)
        self._btn_continue.grid(row=0, column=3, sticky="ew", padx=3)

        # Action row
        act = tk.Frame(left, bg=self._BG, pady=3)
        act.pack(side=tk.TOP, fill=tk.X)
        for i in range(4): act.columnconfigure(i, weight=1)

        self._btn_load   = tk.Button(act, text="^ Load  [L]",
            bg=self._LOAD_BG, fg=self._LOAD_FG, activebackground="#6a2a7a",
            command=self._on_load, **btn_cfg)
        self._btn_manual = tk.Button(act, text="Edit Weights  [T]",
            bg=self._MAN_BG,  fg=self._MAN_FG,  activebackground="#8a5a2a",
            command=self._on_manual, **btn_cfg)
        self._btn_save   = tk.Button(act, text="v Save  [V]",
            bg=self._SAVE_BG, fg=self._SAVE_FG, activebackground="#3a8a5a",
            command=self._on_save, **btn_cfg)
        self._btn_skip   = tk.Button(act, text="> Skip  [S]",
            bg=self._SKIP_BG, fg=self._FG,       activebackground="#555",
            command=self._on_skip, **btn_cfg)
        self._btn_load.grid(  row=0, column=0, sticky="ew", padx=3)
        self._btn_manual.grid(row=0, column=1, sticky="ew", padx=3)
        self._btn_save.grid(  row=0, column=2, sticky="ew", padx=3)
        self._btn_skip.grid(  row=0, column=3, sticky="ew", padx=3)

        # Keyboard shortcuts
        for key, cb in (
            ("<n>", self._on_new),       ("<N>", self._on_new),
            ("<m>", self._on_mutate),    ("<M>", self._on_mutate),
            ("<c>", self._on_continue),  ("<C>", self._on_continue),
            ("<l>", self._on_load),      ("<L>", self._on_load),
            ("<t>", self._on_manual),    ("<T>", self._on_manual),
            ("<s>", self._on_skip),      ("<S>", self._on_skip),
            ("<space>", self._on_skip),
            ("<v>", self._on_save),      ("<V>", self._on_save),
            ("<b>", self._on_previous),  ("<B>", self._on_previous),
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
                               font=("Courier", 9), width=9)
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

        section("Training  [MJX]")
        add_slider(self.params, "n_init_steps",     "Init steps",     1_000, 2_000_000, 10_000, True)
        add_slider(self.params, "n_warm_steps",      "Warm steps",     1_000,   500_000,  5_000, True)
        add_slider(self.params, "n_envs_mjx",        "n_envs (vmap)",  1,      1024,      8, True)
        add_slider(self.params, "rollout_len",       "Rollout len",    8,      2048,      16, True)
        add_slider(self.params, "episode_duration",  "Episode (s)",    1.0,    10.0,    0.5, False)
        add_slider(self.params, "fall_height",       "Fall height",    0.0,     0.5,   0.01, False)
        note(f"JAX device: {_dev}\nCPU defaults: 2k steps, 4 envs (quick).\nScale up for GPU runs.")

        separator()
        section("Mutation sigma")
        add_slider(self.params, "reward_init_sigma", "Init sigma",    0.05, 2.0, 0.05, False)
        add_slider(self.params, "reward_mut_sigma",  "Mutate sigma",  0.05, 2.0, 0.05, False)

        separator()
        section("Reward Weights")
        note("Adjust + press [T] to train.\nAuto-fill from loaded individual.")

        self._rw_manual = RewardWeights(**_RW_DEFAULTS)
        self._rw_slider_vars: dict[str, tk.DoubleVar] = {}
        self._rw_val_labels:  dict[str, tk.Label]     = {}

        for name, (fr, to, res) in _RW_SLIDER_CFG.items():
            row = tk.Frame(p, bg=self._BG2)
            row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(row, text=name, bg=self._BG2, fg="#bbbbbb",
                     font=("Courier", 8), width=19, anchor="w").pack(side=tk.LEFT)
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

        separator()
        section("Session stats")
        self._stats_var = tk.StringVar()
        tk.Label(p, textvariable=self._stats_var, bg=self._BG2, fg="#88ff88",
                 justify=tk.LEFT, font=("Courier", 10)).pack(padx=14, anchor="w")
        self._refresh_stats()

        separator()
        section("Individual info")
        self._info_var = tk.StringVar(value="–")
        tk.Label(p, textvariable=self._info_var, bg=self._BG2, fg="#99ccff",
                 justify=tk.LEFT, font=("Courier", 10)).pack(padx=14, anchor="w")

        separator()
        section("Config (config.py)")
        tk.Label(p, text=(
            f"policy_arch {_cfg.policy_arch}\n"
            f"lr          {_cfg.learning_rate}\n"
            f"gamma       {_cfg.gamma}\n"
            f"batch_size  {_cfg.batch_size}\n"
            f"render      {_cfg.render_width}×{_cfg.render_height}"
        ), bg=self._BG2, fg="#666666", justify=tk.LEFT,
           font=("Courier", 8)).pack(padx=10, anchor="w")

        tk.Frame(p, bg=self._BG2, height=16).pack()

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll(self):
        try:
            while True:
                msg_type, payload = _result_queue.get_nowait()
                if msg_type == "status":
                    self._status_var.set(payload)
                elif msg_type == "progress":
                    self._update_progress_bar(float(payload))
                elif msg_type == "done":
                    self._training = False
                    self._update_progress_bar(1.0, color="#33cc66")
                    self._apply_result(payload)
                elif msg_type == "error":
                    self._training = False
                    self._update_progress_bar(0.0)
                    self._status_var.set(f"  ERROR — see terminal")
                    print(f"\n[ERROR]\n{payload}")
                    self._set_buttons_enabled(True)
        except queue.Empty:
            pass
        self.root.after(200, self._poll)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_new(self):
        if self._training: return
        self._start_training("scratch")
        rw = random_initial_weights(_RW_DEFAULTS, sigma=self.params.reward_init_sigma,
                                     rng=np.random.default_rng())
        p  = self.params
        threading.Thread(target=lambda: _run_training(rw, None, "scratch", p), daemon=True).start()

    def _on_mutate(self):
        if self._training or self._current is None: return
        parent = self._current
        rw = mutate_weights(parent.reward_weights, sigma=self.params.reward_mut_sigma,
                            rng=np.random.default_rng())
        parent_params = load_params(parent.policy_path) if parent.policy_path else None
        p = self.params
        self._start_training("warm")
        threading.Thread(target=lambda: _run_training(rw, parent_params, "warm", p), daemon=True).start()

    def _on_manual(self):
        if self._training: return
        rw = RewardWeights(**{n: float(v.get()) for n, v in self._rw_slider_vars.items()})
        p  = self.params
        self._start_training("manual")
        threading.Thread(target=lambda: _run_training(rw, None, "manual", p), daemon=True).start()

    def _on_continue(self):
        if self._training or self._current is None: return
        parent = self._current
        rw = RewardWeights(**{n: float(v.get()) for n, v in self._rw_slider_vars.items()})
        parent_params = load_params(parent.policy_path) if parent.policy_path else None
        p = self.params
        self._start_training("continue")
        threading.Thread(target=lambda: _run_training(rw, parent_params, "continue", p), daemon=True).start()

    def _on_load(self):
        if self._training: return
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Load MJX policy (.params)",
            filetypes=[("MJX Policy", "*.params"), ("All files", "*.*")],
            initialdir=str(_SCRIPT_DIR),
        )
        if path:
            self._trigger_load(path)

    def _trigger_load(self, path: str):
        self._start_training("loaded")
        p = self.params
        threading.Thread(target=lambda: _run_load(path, p), daemon=True).start()

    def _on_skip(self):
        if self._training: return
        self._skip_count += 1
        self._stop_playback()
        self._current = None
        self._show_idle_message()
        self._info_var.set("–")
        self._status_var.set("  Skipped")
        self._refresh_stats()
        self._set_buttons_enabled(True)

    def _on_save(self):
        if self._current is None: return
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
        self._status_var.set(f"  Saved #{self._saved_count}  →  {path.name}")

    def _on_previous(self):
        if not self._history: return
        self._stop_playback()
        result = self._history.pop()
        self._current = result
        self._start_playback(result.frames)
        self._sync_rw_sliders(result.reward_weights)
        self._update_info(result)
        self._status_var.set(
            f"  [restored]  fitness={result.fitness:+.3f}  "
            f"history={len(self._history)} left"
        )
        self._set_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Apply result
    # ------------------------------------------------------------------

    def _apply_result(self, result: IndividualResult):
        self._push_history()
        self._current = result
        self._start_playback(result.frames)
        self._sync_rw_sliders(result.reward_weights)
        self._update_info(result)
        self._status_var.set(
            f"  [{result.mode}  +{result.n_steps:,} steps / "
            f"{result.total_steps_trained:,} total]   "
            f"fitness = {result.fitness:+.3f}   "
            f"frames = {len(result.frames)}"
        )
        self._set_buttons_enabled(True)

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
        if not self._play_frames: return
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
    # Progress bar
    # ------------------------------------------------------------------

    def _update_progress_bar(self, fraction: float, color: str = "#44aacc"):
        self._pb_canvas.update_idletasks()
        w = max(1, self._pb_canvas.winfo_width())
        self._pb_canvas.coords(self._pb_rect, 0, 0, int(w * min(fraction, 1.0)), 6)
        self._pb_canvas.itemconfig(self._pb_rect, fill=color)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _start_training(self, mode: str):
        self._training      = True
        self._training_mode = mode
        self._set_buttons_enabled(False)
        self._stop_playback()
        self._update_progress_bar(0.0)
        labels = {
            "scratch":  "Sampling random weights + MJX training…",
            "warm":     "Mutating weights + MJX warm-start…",
            "manual":   "Training manual weights with MJX…",
            "continue": "Continuing MJX training (warm start)…",
            "loaded":   "Loading .params + rendering rollout…",
        }
        self._show_canvas_message(labels.get(mode, "Training…"))

    def _push_history(self):
        if self._current is None: return
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
            text="MJX Backend — Press [N] to train",
            fill=self._FG_DIM, font=("Courier", 15),
        )

    def _show_canvas_message(self, msg: str):
        cw = max(self._canvas.winfo_width(),  self._CANVAS_W)
        ch = max(self._canvas.winfo_height(), self._CANVAS_H)
        self._canvas.delete("all")
        self._canvas.configure(bg="#1a1a1a")
        self._canvas.create_text(cw // 2, ch // 2, text=msg,
                                  fill=self._FG_DIM, font=("Courier", 14))

    def _set_buttons_enabled(self, enabled: bool):
        has_c = self._current is not None
        has_h = bool(self._history)
        needs_c = enabled and has_c
        self._btn_new.config(     state=tk.NORMAL if enabled  else tk.DISABLED)
        self._btn_mutate.config(  state=tk.NORMAL if needs_c  else tk.DISABLED)
        self._btn_continue.config(state=tk.NORMAL if needs_c  else tk.DISABLED)
        self._btn_load.config(    state=tk.NORMAL if enabled  else tk.DISABLED)
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
        for name, var in self._rw_slider_vars.items():
            val = float(getattr(rw, name))
            var.set(val)
            setattr(self._rw_manual, name, val)
            self._rw_val_labels[name].config(
                text=f"{val:.4f}" if val < 0.1 else f"{val:.3f}"
            )

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
        import mujoco  # noqa
    except ImportError:
        print("ERROR: mujoco required.  pip install mujoco")
        sys.exit(1)

    preload_path: Optional[str] = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if not Path(arg).exists():
            print(f"ERROR: policy file not found: {arg}")
            sys.exit(1)
        preload_path = arg
        print(f"  Pre-loading policy: {arg}")

    # Clear output dir on each launch
    if _OUT_ROOT.exists():
        shutil.rmtree(_OUT_ROOT)
    _OUT_ROOT.mkdir(parents=True)

    print(f"  JAX device : {_dev}")
    print(f"  Output dir : {_OUT_ROOT}")

    root = tk.Tk()
    root.geometry("1160x760")
    ControllerTrainerMJXApp(root, preload_path=preload_path)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
