# Controller — evolving locomotion behaviours scored by a VLM (CPU version)

Part 3 of [RWS-NS-Proto](../../README.md). This experiment **evolves locomotion
behaviours**: instead of hand-designing a reward function, each individual *is* a
reward-weight vector. A PPO policy is trained to maximise that reward on the
robot, the resulting motion is filmed, and a **VLM scores the video** against a
natural-language target behaviour. That VLM score is the fitness of the
Quality-Diversity search.

> ### ⚠️ Status: superseded by `Controller_MJX/`
> This is the **original CPU implementation** (MuJoCo + Stable-Baselines3). It has
> been **replaced by [`Controller_MJX/`](../Controller_MJX/README.md)**, which runs
> the same study on the GPU (MJX/JAX) and is what full-scale runs use.
>
> This CPU version is still useful: it **runs on a laptop**, has **fewer/simpler
> dependencies**, and is **easier to read and to get started with** — it was used
> for testing and for small-scale experiments. Use it to understand the pipeline
> or for quick local tests; use `Controller_MJX/` for real runs.

```
ExperimentConfig (reward-weight prior)
      │
      ▼
 Mutate reward weights ──▶ Train PPO policy (SB3) ──▶ Film rollout (2-cam MP4) ──▶ VLM grader ──▶ fitness
      ▲                                                                                              │
      └───────────────────────────────── Archive  (μ+λ) ◀───────────────────────────────────────────┘
```

---

## Files

| File | Role |
|------|------|
| **`config.py`** | **All experiment parameters** (`ExperimentConfig`). Edit this. |
| `experiment.py` | Entry point — `run()`, `resume()`, and the CLI. |
| `reward.py` | The reward terms and `RewardWeights` (what PPO optimises). |
| `ppo_trainer.py` | PPO training of one individual (Stable-Baselines3). |
| `mujoco_env.py` | The Gymnasium MuJoCo environment (one robot). |
| `morphology.py` | Robot body used for the controller study. |
| `controller_morph.py` | Builds the policy network for a morphology. |
| `video_renderer.py` | Renders a trained policy's rollout to MP4. |
| `grader.py` | Gemini locomotion grader (video → fitness). |
| `gemini_prompts.py` | Named target-behaviour prompt sets. |
| `evolution.py` / `archive.py` | `(μ+λ)` strategy + population archive. |
| `data_handler.py` | Per-individual result records. |
| `behavior_correlation_experiment.py` | Human vs. VLM score correlation (validation). |
| `utils/controller_generator_renderer.py` | Interactive tool to train/render controllers. |
| `utils/behavior_human_evaluation.py` | UI to build the human-evaluation dataset. |
| `utils/grader_test_prompt.py` | Try a grader prompt on an existing video. |
| `results/` | Run outputs (git-ignored). |

---

## Setup

```bash
cd code/Controller
pip install -r requirements.txt
# and make sure code/api_keys.py exists with your APIKEY_GEMINI  (see main README)
```

This installs MuJoCo (CPU), Gymnasium, Stable-Baselines3 (which pulls in PyTorch),
and the Gemini client. No GPU is required.

---

## Configuration — edit `config.py`

Change the defaults of the `ExperimentConfig` dataclass. The key groups:

### Strategy & population
| Parameter | Meaning |
|-----------|---------|
| `strategy` | `"mu_lambda"` (default). *(MAP-Elites with locomotion descriptors is provided by `Controller_MJX/`; leave `descriptor_config_name = ""` here.)* |
| `mu`, `lambda_`, `sigma` | Parents kept / offspring per gen / fresh random injections. |
| `n_generations` | Number of generations. |
| `init_population_size` | Individuals trained from scratch at generation 0 (`0` = strategy default). |
| `seed` | Random seed. |

### PPO inner loop (how each individual is trained)
| Parameter | Meaning |
|-----------|---------|
| `n_init_steps` | Training budget (env steps) for a from-scratch individual (gen 0). |
| `n_warm_steps` | Training budget for a mutated child (warm-started from its parent). |
| `n_envs` | Parallel environments for PPO. |
| `policy_arch` | Hidden-layer sizes, e.g. `[256, 256]`. |
| `learning_rate`, `gamma`, `gae_lambda`, `ent_coef`, `vf_coef`, `n_steps_per_env`, `batch_size` | Standard PPO hyper-parameters. |

### Reward weights (what the search explores)
The reward is a weighted sum of ~23 terms (`rw_forward_velocity`,
`rw_upright_bonus`, `rw_energy_penalty`, `rw_fall_penalty`,
`rw_height_target_reward`, … — full list in `config.py` / `reward.py`). The
`rw_*` values are the **starting prior**; each generation mutates them with
multiplicative log-normal noise.
| Parameter | Meaning |
|-----------|---------|
| `rw_*` (23 terms) | Default reward-weight vector the search starts from. Terms left at `0.0` stay disabled (mutation is multiplicative), which keeps the search in a small, relevant subspace. |
| `reward_init_sigma` | How widely generation-0 individuals spread around the prior. |
| `reward_mutation_sigma` | Per-generation mutation strength on each weight. |

### Episode & environment
| Parameter | Meaning |
|-----------|---------|
| `episode_duration` | Seconds of simulation per episode (PPO rollout **and** the recorded video). |
| `control_frequency` | Policy actions per second (Hz). |
| `fall_height` | Torso-z below which the episode terminates as a fall. |

### Video / cameras (what the VLM sees)
| Parameter | Meaning |
|-----------|---------|
| `video_fps`, `render_width`, `render_height` | Rollout video settings (final width = 2× per-camera). |
| `camera_track_torso` | Camera follows the torso vs. stays fixed. |
| `cam1_*`, `cam2_*` | The two camera angles (side + diagonal). |
| `origin_tile_*` | Coloured marker at (0,0) so the VLM can gauge displacement. |

### Grader
| Parameter | Meaning |
|-----------|---------|
| `gemini_model` | Gemini model id (e.g. `gemini-3-flash-preview`). |
| `batching` | Videos per Gemini request. |
| `prompt_name` | Target behaviour: `walk_forward`, `jump_high`, or `crawl` (see `gemini_prompts.py`). |
| `reference_best_in_batch` | Upload the current best as a labelled reference each batch. |
| `use_fake_grader` | Synthetic VLM responses — no network / no API cost (wiring tests). |

### Output
| Parameter | Meaning |
|-----------|---------|
| `output_dir`, `save_every_n_gen`, `save_best_every_n_gen`, `save_final_best` | Where results go and how often snapshots / best videos are saved. |

> **Add a target behaviour:** copy a block in `gemini_prompts.py`, give it a new
> `name` and `target` sentence, then set `prompt_name` to it.
>
> Tip: `python config.py` prints the current config and checks the JSON
> round-trip — a fast validity check after editing.

---

## How to run

```bash
cd code/Controller

# Run with the defaults in config.py
python experiment.py

# Override common parameters
python experiment.py --strategy mu_lambda --mu 3 --lambda_ 7 --n_gen 10 \
                     --n_init_steps 2000000 --n_warm_steps 250000 --n_envs 8 \
                     --prompt walk_forward --seed 14

# Resume an interrupted run
python experiment.py --resume results/run_20260626_151400

# Tiny end-to-end smoke run with a fake grader (no Gemini)
python experiment.py --debug
```

**CLI flags:** `--strategy` · `--mu` · `--lambda_` · `--n_gen` ·
`--n_init_steps` · `--n_warm_steps` · `--n_envs` · `--prompt` · `--seed` ·
`--output_dir` · `--resume` · `--debug`. Everything else is set in `config.py`.

---

## Outputs (`results/<run_id>/`)

```
config.json                       frozen ExperimentConfig
archive_gen{NNNN}.json / archive_final.json   archive snapshots
log.jsonl                         one line per generation (best / mean / elapsed)
individuals_log.jsonl             one line per evaluated individual
vlm_responses.jsonl               raw Gemini responses (audit trail)
policies/id{NNNNNN}.zip           trained SB3 policy per individual
videos/gen{GGGG}_id{NNNNNN}.mp4   rollout video per individual
tb/                               TensorBoard logs from the PPO inner loops
```

---

## Extra tools

| Command | Purpose |
|---------|---------|
| `python utils/controller_generator_renderer.py` | Interactively train and render a controller (visual tuning). |
| `python behavior_correlation_experiment.py` | Score a human-annotated video dataset with Gemini and plot human-vs-VLM correlation. |
| `python utils/behavior_human_evaluation.py` | UI to collect the human evaluations for the correlation study. |
| `python utils/grader_test_prompt.py` | Score a single existing video to test a grader prompt. |
