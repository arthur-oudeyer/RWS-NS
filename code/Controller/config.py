"""
config.py
=========
Single source of truth for the controller-study experiment parameters.

Mirrors `Morphology/config.py` so the directory layout and run conventions
match. The fields specific to the controller study are:

  - reward weight defaults and per-dimension mutation σ
  - PPO budgets (`n_init_steps`, `n_warm_steps`, `n_envs`, …)
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
    PPO inner loop  : n_init_steps, n_warm_steps, n_envs, policy_arch
    Env / episode   : episode_duration, fall_height, control_frequency
    Video render    : video_fps, render_w/h, episode_seconds_recorded
    Grader          : gemini_model, batching, prompt_name, descriptor_config
    Output          : output_dir, save_every_n_gen, save_best_every_n_gen
    """

    # ---- Identity -----------------------------------------------------------
    run_id:        str = ""
    seed:          int = 11
    description:   str = ""
    strategy:      str = "mu_lambda"   # "mu_lambda" | "map_elite"

    # ---- Population ---------------------------------------------------------
    mu:            int = 3
    lambda_:       int = 12
    sigma:         int = 0          # fresh random individuals injected per gen
    n_generations: int = 5

    # init_population_size : number of random individuals trained from scratch
    # at gen 0. mu_lambda → defaults to mu*2; map_elite → max(mu, lambda_)*2.
    init_population_size: int = mu + lambda_   # 0 = strategy default

    # ---- Morphology / Env ----------------------------------------------------
    morphology = "tripod" # Morphology, default None -> QUADRIPOD
    photorealistic = True

    # ---- Reward weights (defaults; mutation σ controls per-gen jitter) -------
    # Default vector — opinionated starting prior. See instruction.md §5.
    rw_forward_velocity: float = 1.0
    rw_lateral_drift:    float = 0.1
    rw_upright_bonus:    float = 0.5
    rw_energy_penalty:   float = 0.001
    rw_contact_reward:   float = 0.1
    rw_alive_bonus:      float = 0.05
    rw_fall_penalty:     float = 10.0

    # Mutation σ for the per-generation log-normal noise on each weight.
    # σ_init is used to widen the *initial* population around the default
    # vector so gen-0 individuals do not all collapse to the same prior.
    reward_mutation_sigma:     float = 0.2
    reward_init_sigma:         float = 0.4

    # ---- PPO inner loop -----------------------------------------------------
    n_init_steps: int = 80_000      # from-scratch training budget (gen 0)
    n_warm_steps: int = 20_000      # warm-start budget for mutated children
    n_envs:       int = 4
    policy_arch:  list = field(default_factory=lambda: [256, 256])
    learning_rate:    float = 3e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    ent_coef:         float = 0.0
    vf_coef:          float = 0.5
    n_steps_per_env:  int   = 2048   # PPO rollout length before each update
    batch_size:       int   = 256

    # ---- Env / episode ------------------------------------------------------
    # Episode length used both for PPO rollouts and for the recorded MP4.
    episode_duration:  float = 5.0   # seconds of simulation per episode
    control_frequency: int   = 20    # Hz — how often the policy outputs an action
    # MuJoCo timestep is set by the morphology XML (0.005 s); the env applies
    # the same action for `physics_steps_per_action` mj_steps.
    fall_height:       float = 0.05  # torso z below this terminates the episode

    # ---- Video / VLM render -------------------------------------------------
    video_fps:           int  = 20
    render_width:        int  = 192
    render_height:       int  = 192
    camera_track_torso:  bool = True

    # ---- Grader -------------------------------------------------------------
    use_fake_grader = True

    grader_type:    str = "gemini"
    gemini_model:   str = "gemini-3-flash-preview"
    batching:       int = 12           # videos per Gemini request
    prompt_name:    str = "walk_forward"
    descriptor_config_name: str = ""  # "" = no MAP-Elites descriptors

    # When True (batch mode only): the current best individual's video is
    # uploaded as a labelled "reference" alongside every batch.
    reference_best_in_batch: bool = False

    # ---- Output -------------------------------------------------------------
    output_dir:            str  = "results"
    save_every_n_gen:      int  = 1
    save_best_every_n_gen: int  = 1     # 0 = disable
    save_final_best:       bool = True
    save_all_render_tmp:   bool = True  # always keep last rollout videos

    # -------------------------------------------------------------------------

    def __post_init__(self):
        if not self.run_id:
            self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

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
            "forward_velocity": self.rw_forward_velocity,
            "lateral_drift":    self.rw_lateral_drift,
            "upright_bonus":    self.rw_upright_bonus,
            "energy_penalty":   self.rw_energy_penalty,
            "contact_reward":   self.rw_contact_reward,
            "alive_bonus":      self.rw_alive_bonus,
            "fall_penalty":     self.rw_fall_penalty,
        }

    # ---- Display ------------------------------------------------------------

    def describe(self) -> None:
        print(f"\nExperimentConfig: {self.run_id}")
        print(f"  strategy     : {self.strategy}")
        if self.strategy == "mu_lambda":
            print(f"  population   : μ={self.mu}  λ={self.lambda_}  σ={self.sigma}  generations={self.n_generations}")
        else:
            print(f"  population   : λ={self.lambda_}  σ={self.sigma}  generations={self.n_generations}  descriptors={self.descriptor_config_name}")
        print(f"  PPO          : init={self.n_init_steps:,}  warm={self.n_warm_steps:,}  envs={self.n_envs}  arch={self.policy_arch}")
        print(f"  episode      : {self.episode_duration}s  ctrl_freq={self.control_frequency} Hz  fall_h={self.fall_height} m")
        print(f"  reward σ     : init={self.reward_init_sigma}  mut={self.reward_mutation_sigma}")
        print(f"  reward defaults : {self.default_reward_weights_dict()}")
        print(f"  video        : {self.render_width}×{self.render_height}  {self.video_fps} fps  track_torso={self.camera_track_torso}")
        print(f"  grader       : {self.gemini_model}  batch={self.batching}  prompt={self.prompt_name}")
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
