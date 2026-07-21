# Morphology — evolving robot bodies scored by a VLM

Part 2 of [RWS-NS-Proto](../../README.md). This experiment **evolves robot
morphologies** (bodies) and uses a Vision-Language Model to score how much each
one *looks like* a chosen target — an insect, a spider, a tree, a lamp, … — and
uses that score as the fitness signal of a Quality-Diversity search.

No physics simulation of motion is involved here: each body is only **rendered**
from a few camera angles into images, and a **VLM grades the images**.

```
ExperimentConfig
      │
      ▼
 MutateMorphology ──▶ Renderer (MuJoCo, N camera views → PNG) ──▶ VLM grader ──▶ fitness
      ▲                                                                              │
      └──────────────────────── Archive  (μ+λ  or  MAP-Elites) ◀─────────────────────┘
```

- **Grader** — Gemini (default) or CLIP. Gemini returns three scored dimensions
  (coherence, originality, interest/potential) plus a written observation.
- **Strategies** — `mu_lambda` keeps the best μ each generation; `map_elite`
  fills a 2-D behaviour grid whose axes are VLM-rated features (see *Descriptors*).

---

## Files

| File | Role |
|------|------|
| **`config.py`** | **All experiment parameters** (`ExperimentConfig`). Edit this. |
| `experiment.py` | Entry point — `run()`, `resume()`, and the CLI. |
| `morphology.py` | `RobotMorphology` (procedural body) + `MutateMorphology`. |
| `rendering.py` | MuJoCo image renderer (multi-camera → PNG). |
| `grader.py` | CLIP and Gemini graders (image → fitness). |
| `gemini_prompts.py` | Named Gemini prompt sets (the target creatures). |
| `CLIP_prompts.py` | Named CLIP prompt sets. |
| `descriptor.py` | VLM feature descriptors used as MAP-Elites axes. |
| `evolution.py` | `(μ+λ)` and MAP-Elites strategies. |
| `archive.py` | Population archives (best-μ list / MAP-Elites grid). |
| `data_handler.py` | `MorphologyResult` + evaluation glue. |
| `report.py` | Auto-generated human-readable run report. |
| `prompt_tester.py` | Tool to test prompt sensitivity on hand-sorted images. |
| `correlation_experiment.py` | Compares human vs. VLM scores (validation). |
| `utils/morph_generator_renderer.py` | Interactive morphology viewer. |
| `utils/morph_human_evaluation.py` | UI to build the human-evaluation dataset. |
| `results/` | Run outputs (git-ignored). |

---

## Setup

```bash
cd code/Morphology
pip install -r requirements.txt
# and make sure code/api_keys.py exists with your APIKEY_GEMINI  (see main README)
```

CLIP is only needed if you set `grader_type = "clip"`. Gemini is the default.

---

## Configuration — edit `config.py`

Open `config.py` and change the defaults of the `ExperimentConfig` dataclass.
The most important parameters:

### Strategy & population
| Parameter | Meaning |
|-----------|---------|
| `strategy` | `"mu_lambda"` or `"map_elite"`. |
| `mu` | Parents kept each generation (μ+λ only). |
| `lambda_` | Offspring produced each generation. |
| `sigma` | Fresh random morphologies injected each generation. |
| `n_generations` | Number of generations to run. |
| `init_population_size` | Random individuals evaluated at generation 0. |
| `init_n_legs_min` / `init_n_legs_max` | Leg-count range of the random start bodies. |
| `init_n_mutation` | Random mutations applied to each initial body. |
| `seed` | Random seed (reproducibility). |

### Mutation (how bodies change between generations)
| Parameter | Meaning |
|-----------|---------|
| `length_std`, `angle_std`, `rest_angle_std` | Gaussian noise on segment length / placement angle / joint rest angle. |
| `add_remove_prob` | Probability of adding or removing a leg. |
| `allow_branching`, `branching_prob` | Allow / probability of branched (multi-segment) legs. |
| `torso_a_std … torso_euler_std` | Torso shape / orientation mutation (`0` = keep torso fixed). |
| `add_remove_body_part_prob`, `body_part_*` | Add/remove and mutate extra body parts. |

### Rendering
| Parameter | Meaning |
|-----------|---------|
| `render_width`, `render_height` | Pixel size of each rendered view. |
| `camera_views` | List of camera angles (azimuth/elevation/distance/lookat). |
| `photorealistic` | Grass + sky background (VLMs respond better to realistic scenes). |
| `reference_best_in_batch` | Send the current best as a labelled reference so the VLM rewards genuine novelty (batch mode). |

### Grader
| Parameter | Meaning |
|-----------|---------|
| `grader_type` | `"gemini"` (default) or `"clip"`. |
| `gemini_model` | Gemini model id, e.g. `gemini-3-flash-preview`. |
| `batching` | Images per Gemini request (batch scoring). |
| `gemini_max_retries`, `gemini_retry_base_delay` | Auto-retry on transient API errors. |
| `clip_model`, `clip_pretrained`, `scoring_method` | CLIP model + `"cosine"`/`"softmax"` (CLIP only). |
| ⚠️ `clip_cache_dir` | Where CLIP weights are cached. **The default points to an external drive (`/Volumes/T7_AO/...`) — change it to a local folder** if you use CLIP. |

### Prompt (the target the VLM scores against)
Set `prompt_name` to one of the named sets. For **Gemini** (`gemini_prompts.py`):

`insect_morph` · `spider_morph` · `crab_morph` · `centipede_morph` ·
`kangaroo_morph` · `elephant_morph` · `goal_keeper_morph` · `lamp_morph` ·
`tree_morph`

For **CLIP** (`CLIP_prompts.py`): `spider_body` · `compact_stable` · `many_legs`.

> **Add your own target:** copy an existing block in `gemini_prompts.py`
> (or `CLIP_prompts.py`), give it a new `name`, edit the `target`/prompt text,
> and set `prompt_name` to that new name.

### Descriptors (MAP-Elites feature axes — only used by `strategy="map_elite"`)
Set `descriptor_config_name` to one of the configs in `descriptor.py`:
`generic_descriptors` · `lamp_descriptors` · `elephant_descriptors` ·
`tree_descriptors` (or `""` to disable and use structural features only).
Each config names the two VLM-rated features that become the grid's X/Y axes.

### Output
| Parameter | Meaning |
|-----------|---------|
| `output_dir` | Root results directory (default `results`). |
| `save_every_n_gen` | Save an archive snapshot every N generations. |
| `save_best_every_n_gen` | Render + keep the best individual every N generations (`0` = off). |
| `save_final_best` | Always render the overall best at the end. |

> Tip: run `python config.py` to print the current default config and confirm it
> round-trips to/from JSON — a quick check that your edits are valid.

---

## How to run

```bash
cd code/Morphology

# Run with the defaults in config.py
python experiment.py

# Override a few common parameters from the command line
python experiment.py --strategy map_elite --mu 10 --lambda_ 20 --n_gen 100
python experiment.py --prompt_set tree_morph --seed 7 --save_renders

# Resume an interrupted run (continues from the last saved snapshot)
python experiment.py --resume results/run_20260417_124808

# Wiring smoke-test with a fake renderer + grader (no MuJoCo / no API calls)
python experiment.py --debug
```

**CLI flags** (each overrides the matching `config.py` field):
`--strategy` · `--mu` · `--lambda_` · `--n_gen` · `--prompt_set` · `--seed` ·
`--output_dir` · `--save_renders` · `--resume`. Anything without a flag is set by
editing `config.py`.

---

## Outputs (`results/<run_id>/`)

```
config.json                 frozen copy of the ExperimentConfig
archive_gen{NNNN}.json       archive snapshot every save_every_n_gen
archive_final.json           final archive
log.jsonl                    one JSON line per generation (best / mean / elapsed)
individuals_log.jsonl        one line per evaluated individual (genealogy)
renders/                     rendered PNGs (best per gen, final best, …)
report.txt / report.md       auto-generated human-readable summary
```

---

## Extra tools

| Command | Purpose |
|---------|---------|
| `python utils/morph_generator_renderer.py` | Interactively generate & view morphologies (tune mutation visually). |
| `python prompt_tester.py --pos good_folder --neg bad_folder` | Check whether a prompt separates images you consider good vs. bad, before a full run. |
| `python correlation_experiment.py` | Score a human-annotated dataset with Gemini and plot human-vs-VLM correlation (validation). |
| `python utils/morph_human_evaluation.py` | UI to collect the human evaluations used by `correlation_experiment.py`. |
