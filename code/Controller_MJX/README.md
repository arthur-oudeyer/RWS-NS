# Controller_MJX — GPU controller evolution (MJX / JAX)

Part 4 of [RWS-NS-Proto](../../README.md), and the **current, full-scale version**
of the controller study (part 3, [`Controller/`](../Controller/README.md), is the
older CPU version). Same idea — evolve locomotion behaviours whose fitness is a
**VLM score of the motion video** — but the physics and the PPO training loop are
compiled through **JAX/MJX**, so thousands of environments run in parallel.

> ### 🖥️ This part is meant to run on a GPU
> A **CUDA GPU** is strongly recommended (`n_envs_mjx` defaults to 2048 parallel
> environments). It also runs on **CPU** (slow) and **Apple Metal** (experimental)
> for small tests. Install the right **JAX** build for your machine first — see
> *Setup* below.

```
ExperimentConfig (reward-weight prior)  +  target.txt (natural-language goal)
      │
      ▼
 Mutate reward weights ──▶ Train PPO policy (JAX/MJX, N envs in parallel) ──▶ Film rollout (MP4) ──▶ VLM grader ──▶ fitness
      ▲                                                                                                                │
      └──────────────────────────────── Archive  (μ+λ  or  MAP-Elites) ◀──────────────────────────────────────────────┘
```

**What differs from the CPU `Controller/`:**
- MuJoCo physics runs through `mujoco.mjx` (JAX), vectorised over `n_envs_mjx`
  environments with `jax.vmap`; the PPO inner loop is JIT-compiled once and reused.
- Trained policies are saved as Flax params (`policies/id*.params`) instead of
  Stable-Baselines3 `.zip`.
- The target behaviour is a **plain-text file** (`target.txt`) you can edit,
  not a fixed entry in a prompt registry.

---

## Files

| File | Role |
|------|------|
| **`config.py`** | **All experiment parameters** (`ExperimentConfig`). Edit this. |
| **`target.txt`** | **The target behaviour**, one line of natural language. Edit this. |
| `experiment_mjx.py` | Main entry point — `run_mjx()`, `resume_mjx()`, and the CLI. |
| `controller_cli_mjx.py` | SSH-friendly CLI (train / mutate / render / benchmark), no GUI. |
| `jump_experiment_mjx.py` | EA validation run scored by a **deterministic metric** (no VLM). |
| `benchmark_mjx.py` | Wall-clock benchmark of the MJX pipeline. |
| `experiment_analyser.py` | Interactive Tk explorer for a finished run (plays rollout MP4s). |
| `mujoco_env_mjx.py` | MJX environment (vectorised physics). |
| `ppo_trainer_mjx.py` | JAX/Flax/Optax PPO trainer (from-scratch + warm-start). |
| `reward.py` | Reward terms and `RewardWeights`. |
| `vlm_grader.py` | Gemini locomotion grader (video → fitness). |
| `performance_grader.py` | Deterministic metric graders (used by `jump_experiment_mjx.py`). |
| `descriptor.py` | VLM feature descriptors used as MAP-Elites axes. |
| `gemini_prompts.py` | Prompt scaffolding for the VLM grader. |
| `video_renderer_mjx.py` | Renders a trained policy's rollout to MP4. |
| `evolution_mjx.py` / `archive.py` | `(μ+λ)` / MAP-Elites + population archive. |
| `morphology.py`, `controller_morph.py`, `data_handler.py` | Body, policy builder, result records. |
| `tests/` | `pytest` suite (`test_phase1..4.py`). |
| `results/` | Run outputs (git-ignored). |

---

## Setup

**1. Install JAX for your platform first** (this selects the accelerator build):

```bash
# CUDA 12 GPU (recommended):
pip install -U "jax[cuda12]"
# CPU only:
pip install -U "jax"
# Apple Metal (experimental, small tests only):
pip install -U jax-metal
```
See <https://docs.jax.dev/en/latest/installation.html> for details.

**2. Install the rest of the dependencies:**

```bash
cd code/Controller_MJX
pip install -r requirements.txt
# and make sure code/api_keys.py exists with your APIKEY_GEMINI  (see main README)
```

**Choosing a GPU (shared machines).** Pin the process to one GPU with the
`CUDA_VISIBLE_DEVICES` environment variable *before* the script starts (JAX
otherwise grabs memory on every visible GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python experiment_mjx.py ...
# experiment_mjx.py and jump_experiment_mjx.py also accept --gpu 0 as a shortcut.
```

---

## Configuration — edit `config.py` (+ `target.txt`)

### The target behaviour — `target.txt`
The behaviour you want to evolve toward is **one line of natural language** in
`target.txt`, e.g. `an upright unique rotating dance` or
`a dance where the arms are lifted to the sky`. Edit that file (or pass
`--target-file path/to/other.txt`). `prompt_name` in `config.py` is only a short
label recorded in the results.

### Strategy & population
| Parameter | Meaning |
|-----------|---------|
| `strategy` | `"mu_lambda"` or `"map_elite"`. |
| `mu`, `lambda_`, `sigma`, `n_generations` | Parents / offspring / random injections / generations. |
| `init_population_size` | Individuals trained from scratch at gen 0 (`0` = strategy default). |
| `seed` | Random seed. |

### MJX / JAX backend  ← the GPU knobs
| Parameter | Meaning |
|-----------|---------|
| `n_envs_mjx` | **Parallel environments** (`jax.vmap`). Start at 64–128 on CPU/Metal; 512–2048 on a CUDA GPU. |
| `jax_backend` | `"gpu"`, `"cpu"`, or `"metal"`. |

### PPO inner loop
| Parameter | Meaning |
|-----------|---------|
| `n_init_steps` / `n_warm_steps` | Training budget for from-scratch / warm-started individuals. |
| `policy_arch` | Hidden-layer sizes, e.g. `[128, 128]`. |
| `learning_rate`, `gamma`, `gae_lambda`, `ent_coef`, `vf_coef`, `n_steps_per_env`, `batch_size` | PPO hyper-parameters. |
| `verbose_training` | Print throttled per-update PPO progress. |

### Reward weights (what the search explores)
Same design as the CPU version: a ~23-term reward whose `rw_*` defaults form the
starting prior, mutated per generation with multiplicative log-normal noise
(terms at `0.0` stay disabled). Tune with `reward_init_sigma` (gen-0 spread) and
`reward_mutation_sigma` (per-generation strength). Full list in `config.py` /
`reward.py`.

### Episode & control
| Parameter | Meaning |
|-----------|---------|
| `episode_duration` | Seconds per episode (rollout + video). |
| `control_frequency` | Policy actions per second (Hz). |
| `fall_height` | Torso-z fall threshold (episode termination). |
| `prediction_factor` | Action → joint-angle scale. Larger magnitude = faster/nervier motion; smaller = smoother/more stable. |

### Video / cameras
`video_fps`, `render_width`, `render_height`, `camera_track_torso`, `cam1_*`,
`cam2_*`, `origin_tile_*` — the two camera angles and the origin marker the VLM
uses to judge displacement.

### Grader (VLM)
| Parameter | Meaning |
|-----------|---------|
| `gemini_model` | Gemini model id (e.g. `gemini-3-flash-preview`). |
| `batching` | Videos per Gemini request. |
| `n_score_request` | Score each batch this many times and average (reduces VLM variance). |
| `vlm_weight_coherence` / `vlm_weight_originality` / `vlm_weight_potential` | Weights combining the three VLM dimensions into the fitness. |
| `reference_best_in_batch` | Upload the current best as a labelled reference each batch. |
| `use_fake_grader` | Synthetic VLM responses — no network / no API cost. |
| `target_file`, `prompt_name` | Path to the target text (default `target.txt`) and its results label. |

### Descriptors (MAP-Elites axes — only for `strategy="map_elite"`)
Set `descriptor_config_name` to a config in `descriptor.py`:
`coordination_amplitude` · `similitude_feeling` · `energy_abstraction`
(or `""` to disable). Each names the two VLM-rated features that become the grid
axes.

### Output
`output_dir`, `save_every_n_gen`, `save_best_every_n_gen`, `save_final_best` —
where results go and how often snapshots / best videos are saved.

> Tip: `python config.py` prints the current config and checks the JSON round-trip.

---

## How to run

### Full experiment (the main entry point)
```bash
cd code/Controller_MJX

# Defaults from config.py + target.txt, pinned to GPU 0
CUDA_VISIBLE_DEVICES=0 python experiment_mjx.py

# Override common parameters
python experiment_mjx.py --strategy map_elite --descriptor energy_abstraction \
       --lambda_ 16 --n_gen 20 --n_envs_mjx 2048 --gpu 0

# Use a different target file
python experiment_mjx.py --target-file my_target.txt

# Resume an interrupted run
python experiment_mjx.py --resume results/run_2026-06-25_14h30m00s

# Tiny smoke run with a fake grader (no GPU-scale training, no Gemini)
python experiment_mjx.py --debug --fake-grader
```

**CLI flags:** `--strategy` · `--descriptor` · `--mu` · `--lambda_` · `--n_gen` ·
`--init_ind` · `--n_init_steps` · `--n_warm_steps` · `--n_envs_mjx` · `--prompt` ·
`--target-file` · `--fake-grader` · `--seed` · `--output_dir` · `--resume` ·
`--gpu` · `--debug`.

### Other entry points
```bash
# SSH-friendly CLI — train / warm-start / render / benchmark, outputs saved to disk
python controller_cli_mjx.py new    --steps 200000 --envs 512
python controller_cli_mjx.py mutate run_0001/policy.params
python controller_cli_mjx.py render run_0001/policy.params
python controller_cli_mjx.py bench

# Validate the EA machinery with a DETERMINISTIC metric (no VLM / no API):
python jump_experiment_mjx.py --target jump      # or: walk | rotate | crawl

# Benchmark the pipeline
python benchmark_mjx.py            # tiny;  --full for realistic scale

# Explore a finished run interactively (plays rollout videos)
python experiment_analyser.py                    # newest run
python experiment_analyser.py results/run_XXXX   # a specific run
```

### Tests
```bash
pytest tests/
```

---

## Outputs (`results/<run_id>/`)

```
config.json                       frozen ExperimentConfig
log.txt                           full captured terminal output
archive_gen{NNNN}.json / archive_final.json   archive snapshots
log.jsonl                         one line per generation
individuals_log.jsonl             one line per evaluated individual
policies/id{NNNNNN}.params        trained Flax policy params per individual
videos/gen{GGGG}_id{NNNNNN}.mp4   rollout video per individual
```
