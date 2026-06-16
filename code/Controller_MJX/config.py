"""
config.py
=========
Single source of truth for the controller-study experiment parameters.

Mirrors `Morphology/config.py` so the directory layout and run conventions
match. The fields specific to the controller study are:

  - reward weight defaults and per-dimension mutation σ
  - PPO budgets (`n_init_steps`, `n_warm_steps`, `n_envs_mjx`, …)
  - episode duration and physics rollout length
  - video render settings for the rollout MP4

Usage
-----
    from config import ExperimentConfig

    cfg = ExperimentConfig(
        run_id        = "ctrl_001",
        strategy      = "mu_lambda",
        mu            = 4,
        lambda_       = 8,
        n_generations = 10,
        prompt_name   = "walk_forward",
    )
    cfg.save()    # writes to {output_dir}/{run_id}/config.json

Debug
-----
Run this file directly to print the default config and round-trip a JSON
copy. No MuJoCo / Gemini calls are made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    All parameters that define one controller-study run.

    Sections
    --------
    Identity        : run_id, seed, description, strategy
    Population      : mu, lambda_, sigma (random injections), n_generations
    Reward weights  : default values + per-mutation σ
    PPO inner loop  : n_init_steps, n_warm_steps, n_envs_mjx, policy_arch
    Env / episode   : episode_duration, fall_height, control_frequency
    Video render    : video_fps, render_w/h, episode_seconds_recorded
    Grader          : gemini_model, batching, prompt_name, descriptor_config
    Output          : output_dir, save_every_n_gen, save_best_every_n_gen
    """

    # ---- Identity -----------------------------------------------------------
    run_id:        str = ""
    seed:          int = 14
    description:   str = ""
    strategy:      str = "map_elite"   # "mu_lambda" | "map_elite"

    # ---- Population ---------------------------------------------------------
    mu:            int = 1
    lambda_:       int = 1
    sigma:         int = 0          # fresh random individuals injected per gen
    n_generations: int = 1

    # init_population_size : number of random individuals trained from scratch
    # at gen 0. mu_lambda → defaults to mu*3; map_elite → max(mu, lambda_)*2.
    init_population_size: int = 0   # 0 = strategy default

    # ---- Morphology / Env ----------------------------------------------------
    morphology = "robot" # Morphology, default None -> QUADRIPOD
    photorealistic = True

    # ---- Reward weights (defaults; mutation σ controls per-gen jitter) -------
    # Default vector — opinionated starting prior. See instruction.md §5.
    # Original 7 terms
    rw_forward_velocity: float = 0.05
    rw_lateral_drift:    float = 0.1
    rw_upright_bonus:    float = 0.1
    rw_energy_penalty:   float = 0.02
    rw_contact_reward:   float = 0.02
    rw_alive_bonus:      float = 0.01
    rw_fall_penalty:     float = 10.0
    # Extended 10 terms (small positive so log-normal mutation can activate them)
    rw_no_contact_reward:           float = 0.02
    rw_torso_height_reward:         float = 0.05
    rw_torso_rotation_reward:       float = 0.05
    rw_torso_tilting_speed_reward:  float = 0.02
    rw_limb_coordination_reward:    float = 0.05
    rw_nervosity_reward:            float = 0.02
    rw_smooth_reward:               float = 0.05
    rw_vertical_velocity_reward:    float = 0.02
    rw_lateral_velocity_reward:     float = 0.05
    rw_joint_range_reward:          float = 0.03
    rw_height_target_reward:        float = 2.0
    rw_tilt_penalty:                float = 0.02
    rw_tilt_rate_penalty:           float = 0.005
    rw_all_feet_planted_bonus:      float = 0.01
    rw_vertical_velocity_penalty:     float = 0.01
    rw_horizontal_velocity_penalty:    float = 0.01

    # Mutation σ for the per-generation log-normal noise on each weight.
    # σ_init is used to widen the *initial* population around the default
    # vector so gen-0 individuals do not all collapse to the same prior.
    reward_mutation_sigma:     float = 0.5
    reward_init_sigma:         float = 1.5

    # ---- MJX / JAX backend ---------------------------------------------------
    # Number of parallel environments vectorised via jax.vmap.
    # On Mac M2 (Metal) start low (64–128); bump to 512–2048 on a CUDA GPU.
    n_envs_mjx:   int = 2048
    # JAX backend: "cpu" | "gpu" | "metal"  (set before importing jax)
    jax_backend:  str = "metal"

    # ---- PPO inner loop -----------------------------------------------------
    n_init_steps: int = 4_000_000      # from-scratch training budget (gen 0)
    n_warm_steps: int = 2_000_000      # warm-start budget for mutated children
    policy_arch:  list = field(default_factory=lambda: [128, 128])
    learning_rate:    float = 3e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    ent_coef:         float = 0.0
    vf_coef:          float = 0.5
    n_steps_per_env:  int   = 32   # PPO rollout length before each update
    batch_size:       int   = 256

    # Print throttled per-update PPO progress for each individual during
    # training (update #, steps/s, mean reward, π/V losses, entropy) — same
    # monitoring output as jump_experiment_mjx.py. Throttled to ~every 5 s.
    verbose_training: bool = True

    # ---- Env / episode ------------------------------------------------------
    # Episode length used both for PPO rollouts and for the recorded MP4.
    episode_duration:  float = 4.0   # seconds of simulation per episode
    control_frequency: int   = 20    # Hz — how often the policy outputs an action
    # MuJoCo timestep is set by the morphology XML (0.019 s); the env applies
    # the same action for `physics_steps_per_action` mj_steps.
    fall_height:       float = 0.1  # torso z below this terminates the episode
    # Action → joint-angle delta scale = prediction_factor / control_frequency.
    # -60 mirrors the CPU mujoco_env.py (≈3 rad/tick at 20 Hz — very fast/nervous).
    # Use a smaller magnitude (e.g. -15) for slower, smoother, more stable motion.
    prediction_factor: float = -30.0

    # ---- Video / VLM render -------------------------------------------------
    video_fps:           int  = 20
    render_width:        int  = 360   # per-camera; total video width = 2 × this
    render_height:       int  = 360
    camera_track_torso:  bool = False

    # Origin tile — colored marker at (0,0) so the VLM can gauge displacement.
    origin_tile_rgba:    tuple = (0.4, 0.55, 0.5, 0.5)  # grey-green
    origin_tile_size:    float = 0.3                    # half-extent in metres

    # Camera 1 — ground-level side view (azimuth 90 = right side of robot)
    cam1_azimuth:    float = 90.0
    cam1_elevation:  float = -5.0     # 0 = perfectly horizontal; camera is at lookat height
    cam1_distance:   float = 4.0
    cam1_lookat_z:   float = 0.1    # look at leg height so camera sits near ground level

    # Camera 2 — diagonal front view; gives VLM a second spatial reference
    cam2_azimuth:    float = 60.0
    cam2_elevation:  float = -30.0
    cam2_distance:   float = 4.0

    # ---- Grader (VLM only) --------------------------------------------------
    # The fitness scorer is always the Gemini VLM grader (vlm_grader.py).
    grader_type:    str = "gemini"
    gemini_model:   str = "gemini-3-flash-preview"
    batching:       int = 8           # videos per Gemini request (batch size)
    # The VLM score has high variance between identical requests. Scoring each
    # batch n_score_request times and averaging the per-dimension scores gives
    # a more consistent fitness. 1 = single request (no averaging).
    n_score_request: int = 3

    # The target behaviour is NOT stored here — it lives in a plain-text file so
    # it can be edited with `nano`. `target_file` is resolved relative to the
    # Controller_MJX package directory when not absolute. One line of natural
    # language, e.g. "a forced and awkward gait" or
    # "a dance where the arms are lifted to the sky".
    target_file:    str = "target.txt"
    # Short label recorded in results for this target (GraderOutput.prompt_set).
    prompt_name:    str = "target"

    # Fitness = weighted mean of the three VLM dimensions (each on 0..1).
    vlm_weight_coherence:   float = 1.0
    vlm_weight_originality: float = 0.5
    vlm_weight_potential:    float = 1.5

    # When True, the current best individual's video is uploaded as a labelled
    # "reference" alongside every batch. The reference only exists from
    # generation 1 onward (gen 0 has no best yet), so this gives exactly the
    # "build the prompt with the reference video if it's not the first
    # generation" behaviour.
    reference_best_in_batch: bool = False

    # Use synthetic VLM responses (no network / no API cost) for wiring tests.
    use_fake_grader: bool = False

    # MAP-Elites feature space (see descriptor.py). Used only by strategy="map_elite";
    # ignored by mu_lambda. Selects which 2-D behavioural axes the VLM scores and the
    # grid diversifies over. "" disables descriptors (collapses to a single cell).
    descriptor_config_name: str = "similitude_feeling"

    # ---- Output -------------------------------------------------------------
    output_dir:            str  = "results"
    save_every_n_gen:      int  = 1
    save_best_every_n_gen: int  = 1     # 0 = disable
    save_final_best:       bool = True
    save_all_render_tmp:   bool = True  # always keep last rollout videos

    # -------------------------------------------------------------------------

    def __post_init__(self):
        if not self.run_id:
            # Readable, filesystem-safe: run_2026-06-12_11h44m03s
            self.run_id = datetime.now().strftime("run_%Y-%m-%d_%Hh%Mm%Ss")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id

    # ---- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Optional[str] = None) -> Path:
        target = Path(path) if path is not None else self.run_dir / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ---- Reward-weight helpers ---------------------------------------------

    def default_reward_weights_dict(self) -> dict:
        """The starting reward-weight vector as a plain dict."""
        return {
            "forward_velocity":          self.rw_forward_velocity,
            "lateral_drift":             self.rw_lateral_drift,
            "upright_bonus":             self.rw_upright_bonus,
            "energy_penalty":            self.rw_energy_penalty,
            "contact_reward":            self.rw_contact_reward,
            "alive_bonus":               self.rw_alive_bonus,
            "fall_penalty":              self.rw_fall_penalty,
            "no_contact_reward":         self.rw_no_contact_reward,
            "torso_height_reward":       self.rw_torso_height_reward,
            "torso_rotation_reward":     self.rw_torso_rotation_reward,
            "torso_tilting_speed_reward":self.rw_torso_tilting_speed_reward,
            "limb_coordination_reward":  self.rw_limb_coordination_reward,
            "nervosity_reward":          self.rw_nervosity_reward,
            "smooth_reward":             self.rw_smooth_reward,
            "vertical_velocity_reward":  self.rw_vertical_velocity_reward,
            "lateral_velocity_reward":   self.rw_lateral_velocity_reward,
            "joint_range_reward":        self.rw_joint_range_reward,
            "height_target_reward":      self.rw_height_target_reward,
            "tilt_penalty":              self.rw_tilt_penalty,
            "tilt_rate_penalty":         self.rw_tilt_rate_penalty,
            "all_feet_planted_bonus":    self.rw_all_feet_planted_bonus,
            "vertical_velocity_penalty":   self.rw_vertical_velocity_penalty,
            "horizontal_velocity_penalty": self.rw_horizontal_velocity_penalty,
        }

    # ---- Population helpers --------------------------------------------------

    def resolved_init_population_size(self) -> int:
        """Actual number of gen-0 from-scratch individuals.

        Set it explicitly via `init_population_size` (config) or `--init_ind`.
        When 0, a strategy default applies:
          mu_lambda → mu * 3
          map_elite → lambda_ * 2   (μ is unused by MAP-Elites)
        """
        if self.init_population_size:
            return self.init_population_size
        if self.strategy == "map_elite":
            return self.lambda_ * 2
        return self.mu * 3

    # ---- Display ------------------------------------------------------------

    def describe(self) -> None:
        print(f"\nExperimentConfig: {self.run_id}")
        print(f"  strategy     : {self.strategy}")
        if self.strategy == "mu_lambda":
            print(f"  population   : μ={self.mu}  λ={self.lambda_}  σ={self.sigma}  generations={self.n_generations}")
        else:
            print(f"  population   : λ={self.lambda_}  σ={self.sigma}  generations={self.n_generations}  descriptors={self.descriptor_config_name}")
        print(f"  init pop     : {self.resolved_init_population_size()} individuals trained from scratch (gen 0)")
        print(f"  PPO          : init={self.n_init_steps:,}  warm={self.n_warm_steps:,}  envs={self.n_envs_mjx}  arch={self.policy_arch}")
        print(f"  episode      : {self.episode_duration}s  ctrl_freq={self.control_frequency} Hz  fall_h={self.fall_height} m")
        print(f"  reward σ     : init={self.reward_init_sigma}  mut={self.reward_mutation_sigma}")
        print(f"  reward defaults : {self.default_reward_weights_dict()}")
        print(f"  video        : {self.render_width}×{self.render_height}  {self.video_fps} fps  track_torso={self.camera_track_torso}")
        print(f"  grader       : {self.gemini_model}  batch={self.batching}  "
              f"n_score_request={self.n_score_request}  "
              f"fake={self.use_fake_grader}  reference={self.reference_best_in_batch}")
        print(f"  target file  : {self.target_file}")
        print(f"  output       : {self.run_dir}  (archive every {self.save_every_n_gen} gen)")


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    print("=" * 60)
    print("  config.py — debug mode")
    print("=" * 60)

    # Default config
    cfg = ExperimentConfig(description="debug test")
    cfg.describe()

    # JSON round-trip
    print("\n[2] JSON round-trip\n")
    with tempfile.TemporaryDirectory() as tmp:
        path  = os.path.join(tmp, "config.json")
        saved = cfg.save(path)
        cfg2  = ExperimentConfig.load(path)
        assert cfg2.run_id              == cfg.run_id
        assert cfg2.n_init_steps        == cfg.n_init_steps
        assert cfg2.policy_arch         == cfg.policy_arch
        assert cfg2.default_reward_weights_dict() == cfg.default_reward_weights_dict()
        print(f"  Saved to   : {saved}")
        print(f"  Round-trip : OK")

    print("\nAll config checks passed.")
