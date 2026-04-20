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
| `utils/morph_generator_renderer.py` | Interactive morphology viewer with floor/camera controls |
| `utils/data_analyser.py` | **Interactive results explorer** — graphs, individual browser, genealogy |

---

## Configuring an experiment

All parameters live in `config.py → ExperimentConfig`. Edit them directly or pass as kwargs.

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
lambda_      = 10   # offspring produced per generation
sigma        = 0    # fresh random morphologies injected per generation (0 = disabled)
n_generations = 50
```

**Mutation** (all Gaussian std values)
```python
length_std      = 0.1     # segment length (m)
angle_std       = 12.0    # placement angle (deg)
rest_angle_std  = 0.2     # rest angle (rad)
add_remove_prob = 0.5     # probability of adding/removing a leg
allow_branching = True
branching_prob  = 0.6
torso_radius_std = 0.05
torso_height_std = 0.05
torso_euler_std  = 5.0    # deg
# Body part mutation
add_remove_body_part_prob = 0.25   # 0 = body parts disabled
body_part_radius_std      = 0.02
body_part_height_std      = 0.01
body_part_euler_std       = 5.0
body_part_leg_prob        = 0.5   # prob of attaching a new leg to a body part
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
# Available models:
#   gemini-3.1-flash-lite-preview
#   gemini-3-flash-preview
#   gemini-3.1-pro-preview
```

**Rendering**
```python
render_width     = 192
render_height    = 192
floor_clearance  = 0.0    # metres above z=0; auto-lifts torso
camera_views     = [
    {"azimuth": 0,  "elevation": 5,   "distance": 2., "lookat": [0, 0, 0.25]},
    {"azimuth": 45, "elevation": -50, "distance": 2., "lookat": [0, 0, 0.25]},
]
```

**Output**
```python
output_dir            = "results"
save_every_n_gen      = 1     # archive snapshot frequency
save_best_every_n_gen = 1     # render best individual every N gen (0 = off)
save_final_best       = True
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
    run_id        = "exp_001",
    strategy      = "mu_lambda",
    mu            = 10,
    lambda_       = 20,
    sigma         = 5,    # inject 5 fresh randoms per generation
    n_generations = 50,
    grader_type   = "gemini",
    prompt_name   = "crab_morph",
    seed          = 42,
    output_dir    = "results",
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
python experiment.py --resume results/run_20260417_124808
```

---

## Evolution strategies

### `mu_lambda` — (μ+λ+σ)

Each generation:
1. Sample λ parents randomly from the current population
2. Mutate each parent → λ offspring
3. Optionally generate σ fresh random morphologies (no parent)
4. Evaluate offspring + randoms
5. Pool = μ incumbents + λ offspring + σ randoms → keep best μ

Set `sigma = 0` (default) for classic (μ+λ) behaviour.

### `map_elite`

Grid-based quality-diversity. Each cell stores the best individual for a `(n_legs, symmetry_bin)` combination.  
Bin edges default to `[0.5, 0.8]` → 3 symmetry bins (asymmetric / semi-symmetric / symmetric).

Each generation: sample λ parents from filled cells → mutate → evaluate → insert into grid if fitness improves.

---

## Resuming an interrupted run

Snapshots are saved every `save_every_n_gen` generations as `archive_gen{N:04d}.json`.

```python
from experiment import resume
archive = resume("results/run_20260417_124808", save_renders=True)
```

Or via CLI:
```bash
python experiment.py --resume results/run_20260417_124808
```

`resume()` will:
1. Load `config.json` from the run directory
2. Find the latest `archive_gen*.json` snapshot
3. Continue the evolution loop from the next generation

---

## Output directory layout

```
results/
└── run_20260417_124808/
    ├── config.json              ← frozen copy of ExperimentConfig
    ├── log.jsonl                ← one JSON line per generation (stats)
    ├── individuals_log.jsonl    ← one JSON line per individual (full data + parent_id)
    ├── archive_gen0000.json     ← population snapshot at gen 0
    ├── archive_gen0005.json     ← snapshot every save_every_n_gen gens
    ├── archive_final.json       ← population at end of run
    ├── report.txt               ← human-readable report (auto-generated)
    └── renders/
        ├── best/
        │   ├── gen0001_id000012.png
        │   └── gen0005_id000031.png
        └── best_final_id000087.png
```

---

## Save formats

### `log.jsonl`
One JSON object per line, one per generation:
```json
{"generation": 3, "phase": "step", "n_evaluated": 10, "best_fitness": 0.712, "best_id": 23, "elapsed_s": 4.21, "population_size": 5}
```

### `individuals_log.jsonl`
One JSON object per individual evaluated (streaming):
```json
{"generation": 1, "individual_id": 12, "parent_id": 4, "fitness": 0.68, "raw_scores": {...}, "descriptors": {...}, "grader_extra": {...}, ...}
```
Used by the data analyser for genealogy tracking.

### `archive_gen{N}.json` / `archive_final.json`

**CLIP grader:**
```json
{
  "generation": 2, "individual_id": 17, "fitness": 0.314,
  "grader_method": "cosine", "prompt_set": "spider_body",
  "raw_scores": {"a 3D render of a spider-like robot": 0.28, ...},
  "descriptors": {"n_legs": 6, "symmetry_score": 0.87, "mean_leg_length": 0.24},
  "parent_id": 9
}
```

**Gemini grader** — same structure, with `grader_extra` populated:
```json
{
  "fitness": 0.68, "grader_method": "gemini", "prompt_set": "crab_morph",
  "raw_scores": {"coherence": 7.0, "originality": 5.0, "interest": 8.0},
  "grader_extra": {
    "observation": "...", "interpretation": "...",
    "coherence_reason": "...", "originality_reason": "...", "interest_reason": "..."
  },
  "parent_id": 9
}
```

**Gemini fitness formula:**
```
fitness = (coherence_w * coherence + originality_w * originality + interest_w * interest) / (10 * sum_of_weights)
```
→ always in [0, 1]. Weights are set per `GeminiPromptConfig`.

---

## Data analyser

`utils/data_analyser.py` is an interactive tkinter app for exploring run results.

```bash
python utils/data_analyser.py
```

**Features:**
- Select any run from the `results/` directory (or browse manually)
- Two configurable graph panels:
  - Best fitness over generations
  - Mean fitness over generations
  - All individual fitnesses (scatter)
  - Per-criterion score breakdown
  - Morphology render (best or selected individual)
  - Genealogy tree (ancestor chain to root)
  - Morphology descriptors (n_legs, symmetry, leg length)
- General info panel: strategy, μ/λ/σ, grader, prompt text, run timings, best fitness
- Individual info panel: ID, generation, fitness, parent, per-criterion scores, Gemini reasoning
- Navigate individuals with Prev/Next or dropdown; defaults to the best individual

---

## Genealogy tracking

Every individual records its `parent_id` — the `individual_id` of the morphology it was mutated from. Fresh random morphologies (init population or σ injections) have `parent_id = None`.

The full lineage is streamed to `individuals_log.jsonl` and can be traced root → individual in the data analyser's genealogy graph.

---

## Generating a report

A `report.txt` is saved automatically at the end of every `run()` and `resume()` call.

To generate or regenerate manually:

```python
from report import generate_report

generate_report("results/run_20260417_124808")
generate_report("results/run_20260417_124808", archive_name="archive_gen0010.json")
generate_report("results/run_20260417_124808", save=False, print_report=True)
```

Or via CLI:
```bash
python report.py results/run_20260417_124808
python report.py results/run_20260417_124808 --archive archive_gen0010.json
python report.py results/run_20260417_124808 --no-save
```

---

## Testing individual components

Each module has a `__main__` block for standalone testing:

```bash
python morphology.py          # mutation + serialisation
python rendering.py           # renderer (requires MuJoCo)
python grader.py              # CLIP grader (requires open_clip)
python data_handler.py        # full evaluation pipeline
python archive.py             # both archive types
python config.py              # config serialisation + round-trip
python CLIP_prompts.py        # list all CLIP prompt sets
python gemini_prompts.py      # list all Gemini prompt configs
python evolution.py           # both evolution strategies (fake grader)
python report.py results/run  # generate report for a run
python experiment.py --debug  # smoke-test with fake renderer/grader
```
