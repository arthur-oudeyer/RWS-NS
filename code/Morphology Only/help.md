# Morphology-Only Evolution — Guide

## File overview

| File | Role |
|------|------|
| `experiment.py` | **Entry point.** Ties everything together: `run()` and `resume()` |
| `config.py` | `ExperimentConfig` — all parameters in one place |
| `morphology.py` | `RobotMorphology` — robot body descriptor + mutation + `compute_spawn_height()` |
| `rendering.py` | `MorphologyRenderer` — renders morphologies via MuJoCo |
| `grader.py` | `CLIPGrader` / `GeminiGrader` — scores rendered images |
| `CLIP_prompts.py` | Prompt sets for CLIPGrader |
| `gemini_prompts.py` | Prompt configs for GeminiGrader |
| `evolution.py` | `MuLambdaEvolution` / `MapEliteEvolution` — selection + mutation |
| `archive.py` | `MuLambdaArchive` / `MapEliteArchive` — population storage |
| `data_handler.py` | `MorphologyResult` + `evaluate()` — evaluation pipeline |
| `report.py` | Human-readable run report from any archive snapshot |
| `prompt_tester.py` | Interactive tool to compare prompts on rendered images |

---

## Configuring an experiment

All parameters live in `config.py → ExperimentConfig`.  Edit them directly or override via the CLI.

### Key parameters

**Identity**
```python
run_id       = ""           # auto-generated timestamp if left empty
seed         = 42
description  = "my run"
strategy     = "mu_lambda"  # "mu_lambda" | "map_elite"
```

**Population**
```python
mu           = 5    # parents kept per generation
lambda_      = 5    # offspring produced per generation
n_generations = 50
```

**Mutation** (all Gaussian std values)
```python
length_std      = 0.05    # segment length (m)
angle_std       = 12.0    # placement angle (deg)
rest_angle_std  = 0.15    # rest angle (rad)
add_remove_prob = 0.5     # probability of adding/removing a leg
allow_branching = True
branching_prob  = 0.5
torso_radius_std = 0.05
torso_height_std = 0.05
torso_euler_std  = 5.0    # deg
# Body part mutation
add_remove_body_part_prob = 0.1   # 0 = body parts disabled
body_part_radius_std      = 0.02
body_part_height_std      = 0.01
body_part_euler_std       = 5.0
body_part_leg_prob        = 0.5   # prob of attaching new leg to a body part
```

**Grader**
```python
grader_type  = "gemini"   # "clip" | "gemini"
prompt_name  = "crab_morph"

# CLIP only
clip_model      = "ViT-L-14"
clip_pretrained = "openai"
scoring_method  = "cosine"   # "cosine" | "softmax"

# Gemini only
gemini_model = "gemini-3-flash-preview"
# Model IDs:
#   gemini-3.1-flash-lite-preview
#   gemini-3-flash-preview
#   gemini-3.1-pro-preview
```

**Rendering**
```python
render_width     = 192
render_height    = 192
floor_clearance  = 0.05   # metres; auto-lifts torso so no part clips z=0
camera_views     = [
    {"azimuth": 0,  "elevation": 5,   "distance": 2., "lookat": [0, 0, 0.25]},
    {"azimuth": 45, "elevation": -50, "distance": 2., "lookat": [0, 0, 0.25]},
]
```

**Output**
```python
output_dir           = "results"
save_every_n_gen     = 5      # archive snapshot frequency
save_best_every_n_gen = 5     # render best individual every N gen (0 = off)
save_final_best      = True
```

---

## Available prompt sets

### CLIP prompts (`CLIP_prompts.py`)

| Name | Description |
|------|-------------|
| `spider_body` | Reward spider-like morphologies: many legs spread radially |
| `compact_stable` | Reward low-profile, wide-stance morphologies |
| `many_legs` | Reward high limb count (6+) regardless of arrangement |

### Gemini prompts (`gemini_prompts.py`)

| Name | Target | Description |
|------|--------|-------------|
| `insect_morph` | insect | 6-legged insect-like body plan |
| `spider_morph` | spider | 8-legged spider-like body plan |
| `crab_morph` | crab | Wide-stance crab-like body plan |

To add a new Gemini prompt config:
```python
from gemini_prompts import build_morphology_prompt, GeminiPromptConfig, GeminiScoringWeights

MY_MORPH = GeminiPromptConfig(
    name    = "my_morph",
    target  = "scorpion",
    prompt  = build_morphology_prompt("scorpion"),
    weights = GeminiScoringWeights(coherence=1.0, originality=0.5, interest=1.5),
)
```

---

## Running an experiment

### From Python

```python
from experiment import run
from config import ExperimentConfig

cfg = ExperimentConfig(
    run_id       = "exp_001",
    strategy     = "mu_lambda",
    mu           = 10,
    lambda_      = 20,
    n_generations = 50,
    grader_type  = "gemini",
    prompt_name  = "crab_morph",
    seed         = 42,
    output_dir   = "results",
)

archive = run(cfg, save_renders=True)
print(archive.best().fitness)
```

### From the CLI

```bash
# Default config
python experiment.py

# Override strategy
python experiment.py --strategy map_elite

# Override population size and generations
python experiment.py --mu 10 --lambda_ 20 --n_gen 100

# Save rendered PNGs each generation
python experiment.py --save_renders

# Custom output directory
python experiment.py --output_dir /path/to/results

# Resume an interrupted run
python experiment.py --resume results/run_20241015_143022
```

---

## Resuming an interrupted run

Snapshots are saved every `save_every_n_gen` generations as `archive_gen{N:04d}.json`.  
To resume from the latest snapshot:

```python
from experiment import resume

archive = resume("results/run_20241015_143022", save_renders=True)
```

Or via CLI:
```bash
python experiment.py --resume results/run_20241015_143022
```

`resume()` will:
1. Load `config.json` from the run directory
2. Find the latest `archive_gen*.json` snapshot
3. Continue the evolution loop from the next generation

---

## Output directory layout

```
results/
└── run_20241015_143022/
    ├── config.json              ← frozen copy of ExperimentConfig
    ├── log.jsonl                ← one JSON line per generation
    ├── archive_gen0000.json     ← population snapshot at gen 0
    ├── archive_gen0005.json     ← snapshot every save_every_n_gen
    ├── archive_final.json       ← population at end of run
    ├── report.txt               ← human-readable report (auto-generated at end of run)
    └── renders/
        ├── best/
        │   ├── gen0005_id000012.png
        │   └── gen0010_id000031.png
        └── best_final_id000087.png
```

---

## Save formats

### `config.json`
Frozen copy of `ExperimentConfig` as JSON.  Used by `resume()` to restore all parameters.

### `log.jsonl`
One JSON object per line, one per generation:
```json
{"generation": 3, "phase": "step", "n_evaluated": 5, "best_fitness": 0.712, "best_id": 23, "elapsed_s": 4.21, "population_size": 10}
```

### `archive_gen{N}.json` / `archive_final.json`
Contains the full population.  Each individual entry looks like:

**CLIP grader:**
```json
{
  "generation": 2,
  "individual_id": 17,
  "fitness": 0.31415,
  "grader_method": "cosine",
  "prompt_set": "spider_body",
  "raw_scores": {
    "a 3D simulation render of a spider-like robot...": 0.28,
    "a robot with only 2 or 3 legs": 0.15
  },
  "grader_extra": {},
  "descriptors": {"n_legs": 6, "symmetry_score": 0.87, "mean_leg_length": 0.24},
  "render_path": "renders/best/gen0005_id000017.png",
  "morphology": { ... }
}
```

**Gemini grader** — same structure, with `grader_extra` populated:
```json
{
  "fitness": 0.68,
  "grader_method": "gemini",
  "prompt_set": "crab_morph",
  "raw_scores": {
    "coherence": 7.0,
    "originality": 5.0,
    "interest": 8.0
  },
  "grader_extra": {
    "observation": "The robot has a white cylindrical torso elevated ~0.3m above the floor...",
    "interpretation": "The wide lateral limb spread and low centre of mass closely resemble a crab...",
    "coherence_reason": "Strong match: 4 lateral pairs of legs, wide stance, low torso",
    "originality_reason": "Standard quadruped arrangement, no unusual features",
    "interest_reason": "Good ground contact geometry, stable base suggests viable locomotion"
  },
  ...
}
```

**Gemini fitness formula:**
```
fitness = (1.0 * coherence + 0.5 * originality + 1.5 * interest) / (10 * 3.0)
```
→ always in [0, 1]. Weights are set per `GeminiPromptConfig`.

---

## Evolution strategies

### `mu_lambda`
Classic (μ, λ) selection. Each generation:
1. Select μ best individuals from the archive as parents
2. Produce λ mutated offspring
3. Evaluate all offspring
4. Keep the best μ as the new population

### `map_elite`
Grid-based quality-diversity. Each cell stores the best individual
for a combination of `(n_legs, symmetry_bin)`.  
Bin edges default to `[0.5, 0.8]` → 3 symmetry bins.

---

## Progress output

```
[gen   0 / 50]  init     n=10   best=+0.31415  mean=+0.11234   2.1s
[gen   1 / 50]  step     n=5    best=+0.41234  mean=+0.15678   1.8s
```

---

## Generating a report

`report.py` reads any archive snapshot and writes a human-readable `report.txt`
sorted by fitness (worst → best).  Each entry shows scores, structural
descriptors, and the full Gemini analysis (observation, interpretation,
per-dimension reasoning).

A report is **automatically saved** at the end of every `run()` and `resume()` call.

To generate or regenerate a report manually:

```python
from report import generate_report

# From the latest snapshot (archive_final.json or highest gen snapshot)
generate_report("results/run_20241015_143022")

# From a specific snapshot
generate_report("results/run_20241015_143022", archive_name="archive_gen0010.json")

# Print to stdout without saving
generate_report("results/run_20241015_143022", save=False, print_report=True)
```

Or via CLI:

```bash
python report.py results/run_20241015_143022
python report.py results/run_20241015_143022 --archive archive_gen0010.json
python report.py results/run_20241015_143022 --no-save   # print only
```

---

## Floor clearance — automatic spawn height

Robots with long or downward-angled legs could clip through the floor (z = 0).
`compute_spawn_height(morph, floor_clearance)` in `morphology.py` computes the
minimum torso Z that keeps every part above the floor:

```
drop = max over all chains of: Σ L·cos(α)  (per segment, at rest pose)
spawn_height = drop + floor_clearance
```

This is applied **automatically** on every `renderer.render()` call — the
`spawn_height` stored in the morphology object is ignored.  The clearance is
controlled by `floor_clearance` (default 0.05 m) in both `ExperimentConfig`
and `RenderConfig`.

The `morph_generator_renderer.py` tool exposes a **Floor margin** slider and
shows the computed spawn height in the morphology info panel.

---

## Testing individual components

Each module has a `__main__` block for standalone testing:

```bash
python morphology.py      # test mutation + serialisation
python rendering.py       # test renderer (requires MuJoCo)
python grader.py          # test CLIP grader (requires open_clip)
python data_handler.py    # test full evaluation pipeline
python archive.py         # test both archive types
python config.py          # test config serialisation
python CLIP_prompts.py    # list all CLIP prompt sets
python gemini_prompts.py  # list all Gemini prompt configs
python report.py results/my_run   # generate report for a run
python experiment.py --debug   # smoke-test with fake renderer/grader
```
