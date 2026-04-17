"""
config.py
=========
Single source of truth for all experiment parameters.

Creating a config and saving it to JSON at the start of each run makes
every experiment fully reproducible: re-running with the same config.json
should produce equivalent results (given the same seed).

Usage
-----
    from config import ExperimentConfig

    cfg = ExperimentConfig(
        run_id          = "run_001",
        strategy        = "mu_lambda",
        mu              = 10,
        lambda_         = 20,
        n_generations   = 50,
        prompt_set_name = "spider_body",
    )
    cfg.save("results/run_001/config.json")

    # Load back
    cfg2 = ExperimentConfig.load("results/run_001/config.json")

Debug
-----
    Run this file to print the default config and test JSON round-trip.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Camera view (mirrors rendering.CameraView without the import)
# Stored as a plain dict so config.py has no dependency on rendering.py.
# ExperimentConfig.camera_views returns list[dict]; rendering.py converts.
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_VIEWS = [
    {"azimuth": 0,   "elevation": 5, "distance": 2., "lookat": [0.0, 0.0, 0.25]},
    {"azimuth": 45, "elevation": -50, "distance": 2., "lookat": [0.0, 0.0, 0.25]},
]

# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    All parameters that define one experiment run.

    Sections
    --------
    Identity      : run_id, seed, description, strategy
    Population    : mu, lambda_, n_generations
    Init          : init_n_legs_min/max  (random start population)
    Mutation      : length_std, angle_std, rest_angle_std,
                    add_remove_prob, allow_branching, branching_prob
    Rendering     : render_width, render_height, camera_views
    Grader        : clip_model, clip_pretrained, clip_cache_dir,
                    scoring_method, prompt_set_name
    Output        : output_dir, save_every_n_gen
    MapElite      : symmetry_bins  (bin edges for symmetry feature)
    """

    # ---- Identity -----------------------------------------------------------
    run_id:          str = ""            # filled automatically if empty
    seed:            int = 42
    description:     str = ""
    strategy:        str = "mu_lambda"   # "mu_lambda" | "map_elite"

    # ---- Population ---------------------------------------------------------
    mu:              int = 5            # number of parents kept each generation
    lambda_:         int = 5            # number of offspring produced each generation
    n_generations:   int = 3

    # ---- Initial population -------------------------------------------------
    init_n_legs_min: int = 2
    init_n_legs_max: int = 6

    # ---- Mutation -----------------------------------------------------------
    length_std:       float = 0.05        # Gaussian std for segment length (m) (base length ~0.25)
    angle_std:        float = 12.0        # Gaussian std for placement angle (deg) (angle pos torso / relative angle for swing axis for branched)
    rest_angle_std:   float = 0.15        # Gaussian std for rest angle (rad)
    add_remove_prob:  float = 0.5         # probability of adding or removing a leg
    allow_branching:  bool  = True        # whether mutation can create branched legs
    branching_prob:   float = 0.5        # conditional prob of adding a branched leg
    # Torso mutation (0.0 = keep fixed, i.e. torso shape/orientation unchanged)
    torso_radius_std: float = 0.05         # Gaussian std for torso radius (m)
    torso_height_std: float = 0.05         # Gaussian std for torso half-height (m)
    torso_euler_std:  float = 5.0         # Gaussian std per euler axis (degrees)
    # Body part mutation
    add_remove_body_part_prob: float = 0.1  # prob of adding/removing a body part (0 = disabled)
    body_part_radius_std:      float = 0.02
    body_part_height_std:      float = 0.01
    body_part_euler_std:       float = 5.0
    body_part_leg_prob:        float = 0.5  # when adding a leg, prob of attaching to a body part

    # ---- Rendering ----------------------------------------------------------
    render_width:    int   = 192
    render_height:   int   = 192
    camera_views:    list  = field(default_factory=lambda: list(DEFAULT_CAMERA_VIEWS))
    floor_clearance: float = 0   # metres of clearance above z=0 (auto spawn-height)

    # ---- Grader -------------------------------------------------------------
    grader_type = "gemini" # clip | gemini

    """
    CLIP ViT-B-32   350 MB   RAM ~1 GB   ~50ms/image
    CLIP ViT-B-16   350 MB   RAM ~1 GB   ~80ms/image
    CLIP ViT-L-14   890 MB   RAM ~2 GB   ~150ms/image
    """
    clip_model:      str = "ViT-L-14"
    clip_pretrained: str = "openai"
    clip_cache_dir:  str = "/Volumes/T7_AO/clip-models"
    scoring_method:  str = "cosine"      # "cosine" | "softmax"

    """
    Gemini 3.1 Flash-Lite -> gemini-3.1-flash-lite-preview
    Gemini 3 Flash        -> gemini-3-flash-preview
    Gemini 3.1 Pro        -> gemini-3.1-pro-preview
    """
    gemini_model = "gemini-3-flash-preview"

    # ---- Prompt -------------------------------------------------------------
    prompt_name = "crab_morph"

    # ---- Output -------------------------------------------------------------
    output_dir:            str  = "results"
    save_every_n_gen:      int  = 5     # save archive snapshot every N generations
    # Render saving — which individuals to render and keep as PNGs.
    # save_best_every_n_gen : render the best individual every N generations
    #                         (0 = disabled).  Saved to renders/best/gen{N:04d}.png
    # save_final_best       : always render and save the overall best at end of run.
    #                         Saved to renders/best_final.png
    save_best_every_n_gen: int  = 1     # 0 to disable
    save_final_best:       bool = True

    # ---- MapElite -----------------------------------------------------------
    # Bin edges for the symmetry_score feature dimension.
    # Produces len(symmetry_bins)+1 buckets:
    #   [0, 0.5) → asymmetric | [0.5, 0.8) → semi-sym | [0.8, 1] → symmetric
    symmetry_bins: list = field(default_factory=lambda: [0.5, 0.8])

    # -------------------------------------------------------------------------

    def __post_init__(self):
        if not self.run_id:
            self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    @property
    def run_dir(self) -> Path:
        """Directory where this run's outputs are stored."""
        return Path(self.output_dir) / self.run_id

    # ---- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Optional[str] = None) -> Path:
        """
        Save config to JSON.  If path is None, saves to run_dir/config.json.
        Creates parent directories automatically.
        """
        if path is None:
            target = self.run_dir / "config.json"
        else:
            target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

    @classmethod
    def load(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def describe(self) -> None:
        """Print a human-readable summary."""
        print(f"\nExperimentConfig: {self.run_id}")
        print(f"  strategy     : {self.strategy}")
        if self.strategy == "mu_lambda":
            print(f"  population   : μ={self.mu}  λ={self.lambda_}  generations={self.n_generations}")
        elif self.strategy == "map_elite":
            print(f"  population   : μ={self.mu}  λ={self.lambda_}  generations={self.n_generations}")
        print(f"  init legs    : [{self.init_n_legs_min}, {self.init_n_legs_max}]")
        print(f"  mutation     : length_std={self.length_std}  angle_std={self.angle_std}  "
              f"rest_angle_std={self.rest_angle_std}")
        print(f"  add_remove   : {self.add_remove_prob:.0%}  branching={self.allow_branching} ({self.branching_prob:.0%})")
        print(f"  torso mut    : radius_std={self.torso_radius_std}  height_std={self.torso_height_std}  "
              f"euler_std={self.torso_euler_std}°")
        print(f"  seed         : {self.seed}")
        if self.grader_type == "clip":
            print(f"  grader       : {self.clip_model}  method={self.scoring_method}  "
                  f"prompts={self.prompt_name}")
        else:
            print(f"  grader       : {self.gemini_model}  prompts={self.prompt_name}")
        print(f"  render       : {self.render_width}×{self.render_height}  "
              f"views={len(self.camera_views)}")
        best_render = (f"every {self.save_best_every_n_gen} gen"
                       if self.save_best_every_n_gen > 0 else "disabled")
        print(f"  output       : {self.run_dir}  (archive every {self.save_every_n_gen} gen)")
        print(f"  renders      : best {best_render}  final={self.save_final_best}")


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    print("=" * 60)
    print("  config.py — debug mode")
    print("=" * 60)

    # --- 1. Default config ---
    print("\n[1] Default config\n")
    cfg = ExperimentConfig(description="debug test")
    cfg.describe()

    # --- 2. Custom config ---
    print("\n[2] Custom config\n")
    cfg2 = ExperimentConfig(
        run_id          = "exp_spider_001",
        strategy        = "mu_lambda",
        mu              = 15,
        lambda_         = 30,
        n_generations   = 100,
        seed            = 7,
        prompt_set_name = "spider_body",
        allow_branching = True,
        description     = "Spider body optimisation with branching",
    )
    cfg2.describe()

    # --- 3. JSON round-trip ---
    print("\n[3] JSON round-trip\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        saved = cfg2.save(path)
        cfg3  = ExperimentConfig.load(path)

        assert cfg3.run_id          == cfg2.run_id
        assert cfg3.mu              == cfg2.mu
        assert cfg3.prompt_set_name == cfg2.prompt_set_name
        assert cfg3.allow_branching == cfg2.allow_branching
        print(f"  Saved to   : {saved}")
        print(f"  Loaded run : {cfg3.run_id}")
        print(f"  Round-trip : OK")

    # --- 4. run_dir ---
    print(f"\n[4] run_dir = {cfg2.run_dir}")

    print("\nAll config checks passed.")
