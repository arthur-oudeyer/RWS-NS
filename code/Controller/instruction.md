# Controller Study — Instructions

This document is the single entry point for anyone starting work on the
controller-optimisation stage of the project. It assumes **no prior knowledge**
of the rest of the repository.

Read it top-to-bottom before writing any code under `code/Controller/`.

---

## Baseline (the loop in one paragraph)

> This is the single source of truth for the study's logic. Every module
> under `code/Controller/` must be consistent with this paragraph. If
> implementation forces a deviation, update this paragraph first, then
> the relevant section below, then the code — never the other way round.

The body is fixed (QUADRIPOD by default), so the only thing the evolutionary algorithm
controls is a small **reward weight vector** — coefficients on hand-designed
shaping terms (forward velocity, energy penalty, upright bonus, contact
reward, alive bonus, fall penalty, …). Each generation, the EA picks random
parents from the archive and mutates them 𝝺 times by applying log-normal noise to each parent's
reward weights to produce λ child weight vectors. **For every child**, an
inner-loop PPO training run **warm-starts from that parent's saved policy**
— instead of random init, the child inherits the parent's already-learned
locomotion competence and only needs to adapt for `n_warm_steps` against
its own mutated reward (initial individuals, having no parent, train from
scratch for the longer `n_init_steps` budget). Each trained policy is
rolled out once in a deterministic episode and recorded as an MP4 with a
torso-tracking camera. **Only after all λ children of the generation have
been trained and rendered** are the videos handed to **Gemini in
batched call** (`cfg.batching` videos per request, mirroring
`Morphology/grader.py:GeminiGrader.score_batch`). Those batched call
returns, for each video, a scalar **fitness in [0, 1]** (weighted
combination of behavioural scores like coherence / originality / interest)
and a **descriptor vector** that locates the individual on MAP-Elites'
diversity axes. Each `(reward_weights, trained_policy)` pair is then
either inserted into the archive (if it beats the cell's occupant in
MAP-Elites or makes the top μ in a μ+λ run) or discarded. Over many
generations the archive fills with diverse `(reward function, behaviour)`
pairs and lineages of inherited locomotion drift across reward-weight
space — the VLM is the sole arbiter of what counts as interesting (called
once per generation, not per individual), the reward weights are the
search variable, and PPO is an amortised solver that takes each candidate
reward and the parent's policy and produces the controller that reward
implies.

---

## 1. High-level framework

Across all three stages of the project the same evaluation loop is used —
only **what is being varied** changes. This is the unifying flow:

```
              ┌────────────────────────────────────────┐
              │ Evolutionary algorithm (μ+λ / MAP-E)   │
              │  - keeps an archive of individuals     │
              │  - selects parents, mutates them       │
              └────────────────────┬───────────────────┘
                                   │  generates / mutates
                                   ▼
              ┌────────────────────────────────────────┐
              │ Robot specification                    │
              │  Stage 1/2: morphology (body)          │
              │  Stage 3 :  reward weights + policy NN │
              │             (PPO bridges EA → MuJoCo)  │
              └────────────────────┬───────────────────┘
                                   │  instantiated in MuJoCo
                                   ▼
              ┌────────────────────────────────────────┐
              │ MuJoCo simulation                      │
              │  Stage 1/2: static rest pose           │
              │  Stage 3 :  dynamic rollout            │
              └────────────────────┬───────────────────┘
                                   │  rendered to
                                   ▼
              ┌────────────────────────────────────────┐
              │ Visual artefact                        │
              │  Stage 1/2: PNG (multiple views)       │
              │  Stage 3 :  MP4 video                  │
              └────────────────────┬───────────────────┘
                                   │  scored by
                                   ▼
              ┌────────────────────────────────────────┐
              │ VLM (Gemini Files API)                 │
              │  - batched calls at end of generation  │
              │  - prompt = target-aware questions     │
              │  - returns JSON: scores + descriptors  │
              └────────────────────┬───────────────────┘
                                   │  feeds back
                                   ▼
                          fitness, descriptors
                           → archive update
                           → next generation
```

Two artefacts come out of every VLM call:

- a scalar **fitness** in `[0, 1]` (weighted combination of scored
  dimensions such as coherence / originality / interest) — used for
  selection;
- a **descriptor vector** (named integer ratings, e.g. bilateral symmetry,
  limb count, gait style, stability, ...) — used to place the individual in a MAP-Elites
  grid for diversity-aware search.

The VLM call is **batched, not per-individual**: every generation, all
inner-loop training is performed first, every individual is rolled out and
recorded as an MP4, and then a single (chunked) Gemini request scores the
whole batch at once. This mirrors `Morphology/grader.py:GeminiGrader.score_batch`
and is essential — Gemini latency dominates wall-clock time, so amortising
it across the generation is what makes the loop tractable.

Stage 3 (this study) is the same loop with three differences:

1. The "individual" is a **(reward weight vector for PPO, PPO-trained policy)**
   pair. The body is fixed; the EA evolves the reward weights, and an
   inner PPO training loop converts each candidate reward into the
   controller it implies.
2. There is therefore an **inner optimisation loop** (PPO) nested inside
   the outer one (EA). PPO is the bridge between an evolved reward and a
   policy that MuJoCo can step.
3. The visual artefact is no longer a still render but a **video of the
   robot acting in the world**, so the VLM is asked about *behaviour*
   instead of *form*.

---

## 2. Research context

The overall research question is:

> Can a Visual Language Model (VLM) be used as a reward / fitness signal
> for an evolutionary algorithm that designs robots?

The study is split into three stages:

| Stage | Folder              | What evolves        | Fitness signal                       |
|-------|---------------------|---------------------|--------------------------------------|
| 1     | `code/proto/`       | Morphology + brain  | Hand-crafted metrics (proof of concept) |
| 2     | `code/Morphology/`  | Morphology only     | VLM (Gemini) on rendered images      |
| 3     | `code/Controller/`  | **Controller only** | **VLM (Gemini) on rollout videos**   |

Stages 1 and 2 are complete. **This document covers stage 3.**

The goal of stage 3 is to verify that the VLM-as-reward approach works for
**dynamic behaviour** (locomotion / motion patterns), not just static
appearance — the morphology is frozen so any improvement must come from the
controller learning to move better.

---

## 3. Repository tour (what to read in the existing codebase)

You **do not** need to write anything that already exists. Reuse aggressively.

### 3.1  `code/proto/` — early prototype

A working but un-evolutionary loop where N robots run side-by-side in MuJoCo
with hand-coded fully-connected controllers. Read these files; they show how
the simulation is wired today:

| File                          | What you should learn from it                                  |
|-------------------------------|----------------------------------------------------------------|
| `Mujoco/main_sim.py`          | Per-step control loop: read sensors → compute control → `mj_step` |
| `Mujoco/control.py`           | `RobotSensorData` dataclass + `read_robot_sensors()` (reusable as the env's observation builder) |
| `Mujoco/video_render.py`      | Async MP4 recorder using `mujoco.Renderer` + `imageio` (port directly to the controller env) |
| `Robot/simple_brain.py`       | The hand-rolled NN we are **replacing** with PPO. Note its `PREDICTION_FACTOR` Δ-angle output convention — we keep the same convention so morphology actuator ranges still make sense |
| `VLM/gemini_flash.py`         | Reference for uploading a video file to Gemini Files API and parsing the JSON response |

### 3.2  `code/Morphology/` — stage 2, the architectural template

This is the codebase whose structure stage 3 must mirror. Read these files
carefully; the controller study reuses 80% of the patterns:

| File                | Why it matters for stage 3                                   |
|---------------------|--------------------------------------------------------------|
| `config.py`         | `ExperimentConfig` dataclass + JSON round-trip + auto `run_id`. Copy and extend |
| `experiment.py`     | `run()` / `resume()` driver with archive snapshots, `log.jsonl`, `individuals_log.jsonl`, auto-`report.txt`. Copy structure verbatim |
| `evolution.py`      | `BaseEvolution` + `MuLambdaEvolution` + `MapEliteEvolution` — same interface, new "individual" type |
| `archive.py`        | `MuLambdaArchive` + `MapEliteArchive` (grid storage, feature bins) |
| `grader.py`         | `GeminiGrader.score_batch()` — uploads a chunk of images, parses JSON, supports a "reference best" image and an optional `descriptor_config`. **This is the exact pattern we mirror for the locomotion grader, swapping images for videos.** |
| `descriptor.py`     | `DescriptorItem` / `DescriptorConfig` — VLM-driven MAP-Elites feature axes. Reuse as-is |
| `gemini_prompts.py` | `GeminiPromptConfig` + `build_morphology_prompt()` — prompt-engineering pattern for fitness scoring |
| `morphology.py`     | `MorphologyManager.get_model(morph)` returns a ready-to-step `mujoco.MjModel` with position-servo actuators. **This is how the static body is built in stage 3** |

### 3.3  Conventions inherited from `Morphology/`

- Single dataclass `ExperimentConfig` is the only place hyper-parameters live.
- `run_id` defaults to `run_YYYYMMDD_HHMMSS`; everything for a run lives under
  `output_dir/run_id/`.
- Every run produces, inside `run_dir`:
  - `config.json`             — frozen copy of the config
  - `log.jsonl`               — one line per generation (best/mean/elapsed)
  - `individuals_log.jsonl`   — one line per evaluated individual
  - `archive_gen{N:04d}.json` — periodic snapshots
  - `archive_final.json`      — last snapshot
  - `report.txt`              — auto-generated human-readable summary
  - `vlm_responses.jsonl`     — raw Gemini responses (debugging / audit)
- `__main__` blocks in every module run a smoke test with fakes (no
  MuJoCo / no Gemini). Replicate this convention.

---

## 4. Objective for stage 3

Implement an evolutionary loop where:

1. The robot **morphology is frozen** to the `QUADRIPOD` preset
   (`from morphology import QUADRIPOD`) — 4 legs at 0/90/180/270°, 1 joint each,
   so 4 actuators, 4 hip angles + 4 hip velocities + 6-DoF torso pose.
2. Each "individual" is a **(reward weight vector, PPO-trained controller)**
   pair. The reward weight vector parametrises a hand-designed shaped
   reward (forward velocity, energy penalty, upright bonus, contact reward,
   alive bonus, fall penalty, …). The controller is the policy network PPO
   produces when trained against that reward for a fixed timestep budget.
3. After training, the policy is rolled out, the rollout is recorded as an MP4,
   and a **Gemini VLM grades the video** to produce the evolutionary fitness
   *and* the MAP-Elites descriptors.
4. Evolution iterates **per generation**, in this order:
   (a) select λ parents from the archive and mutate each parent's reward
       weight vector (small log-normal noise);
   (b) for every child, **warm-start PPO from the parent's saved policy**
       against the child's reward for `n_warm_steps` and record an MP4
       rollout (initial individuals, with no parent, train from scratch
       for the longer `n_init_steps` budget);
   (c) **score the whole batch of MP4s in a single Gemini call**
       (chunked by `cfg.batching` if the batch exceeds the per-request
       cap) — this is the *last* step of the generation and the only place
       the VLM is used;
   (d) attempt insertion of each scored individual into the archive.

The output of one full run is the same kind of archive that `Morphology/`
produces, so `utils/data_analyser.py` can be adapted to browse it.

---

## 5. Decisions already made (do not re-litigate without explicit ask)

These were settled with the user before writing this document:

| Question | Decision                                                                                                                                                                                             |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Which morphology to freeze? | `QUADRIPOD` preset by default.                                                                                                                                                                       |
| Reward structure | VLM is the **only fitness signal for evolution**. PPO still needs a dense per-step shaped reward to actually learn; that shaped reward is an internal training detail, not the evolutionary fitness. |
| Mutation strategy (v1) | **Reward-weight perturbation + warm-start PPO**. A child is created by adding small log-normal noise to each component of the parent's reward weight vector. PPO is then **warm-started from the parent's saved policy weights** and trained for `n_warm_steps` against the child's (mutated) reward. Initial individuals (no parent yet) train from scratch for `n_init_steps` instead. Rationale: the parent has already learned competent locomotion; the child inherits that competence and only needs to adapt to the new reward shape, saving substantial inner-loop compute. Trade-off accepted: the child's behaviour partially reflects the parent's policy bias on top of the new reward — confounding we accept in exchange for the speed-up. |
| Map-Elites descriptors | Returned by Gemini in the same call as fitness, using the existing `DescriptorConfig` mechanism (no separate per-rollout structural descriptors required for v1).                                    |
| VLM model | Gemini, same `google-genai` client and Files API as `Morphology/`.                                                                                                                                   |

---

## 6. Library stack

Already installed (verified): `mujoco 3.2.5`, `torch 2.5.1`, `numpy`,
`google-genai`, `imageio`, `Pillow`, `colorama`.

Still to install:

```bash
pip install gymnasium stable-baselines3
```

Decisions:

| Need              | Pick                                | Why |
|-------------------|-------------------------------------|-----|
| PPO               | **Stable-Baselines3**               | Mature PyTorch impl, sensible `MlpPolicy`, callbacks, save/load, TensorBoard. We do **not** hand-roll the NN. Crucially, SB3's `PPO.load(path, env=...)` followed by `model.learn(n)` is exactly the **warm-start** mechanism v1 uses: every mutated child loads its parent's saved policy and trains briefly against the child's (mutated) reward. |
| Env API           | **Gymnasium**                       | SB3-native; lets us swap RL libs later. |
| NN backend        | PyTorch (via SB3 default)           | `policy_kwargs=dict(net_arch=[256, 256])`; no custom NN code. |
| Vec envs          | `SubprocVecEnv` with `start_method="spawn"` | macOS-safe; MuJoCo offscreen rendering is fork-unsafe. |
| Video             | `mujoco.Renderer` + `imageio` (libx264) | Direct port of `proto/Mujoco/video_render.py` retargeted to a torso-tracking camera. |
| VLM eval          | `google-genai` (already used)       | Mirror `Morphology/grader.py:GeminiGrader.score_batch`, swap PIL images for MP4 file uploads (Files API). |
| Logging           | TensorBoard (SB3 built-in) + `log.jsonl` | TB for the inner PPO training, JSONL for the outer evolution loop (matches `Morphology/`). |

Skipped on purpose:
- `dm_control` — overkill, MJCF is already generated.
- `Brax` / `MJX` — would force a JAX rewrite of `morphology.py`.
- `CleanRL` — single-file PPO is nice, but more glue work for callbacks.

---

## 7. Proposed module layout for `code/Controller/`

Mirror `code/Morphology/` so existing tooling and conventions transfer.

```
code/Controller/
├── instruction.md               ← this file
├── config.py                    ← ExperimentConfig (extends the Morphology fields)
├── controller_morph.py          ← STATIC_MORPH = QUADRIPOD; build_model()
├── reward.py                    ← RewardWeights dataclass, mutate_weights(), compute_step_reward()
├── mujoco_env.py                ← RobotControllerEnv(gym.Env), parameterised by RewardWeights
├── ppo_trainer.py               ← train_from_scratch(reward_weights, ...) and train_warm_start(reward_weights, parent_policy_path, ...)
├── video_renderer.py            ← rollout_to_video(policy, env, save_path)
├── grader.py                    ← LocomotionGrader (Gemini, score_batch over MP4s)
├── gemini_prompts.py            ← LocomotionPromptConfig + build_locomotion_prompt()
├── descriptor.py                ← (optional) reuse Morphology/descriptor.py via import
├── evolution.py                 ← MuLambdaEvolution / MapEliteEvolution (controller version)
├── archive.py                   ← (optional) reuse Morphology/archive.py via import
├── data_handler.py              ← ControllerResult dataclass + evaluate_batch()
├── experiment.py                ← run() / resume() driver
├── report.py                    ← (optional) reuse Morphology/report.py
└── results/                     ← run_YYYYMMDD_HHMMSS/ output directories
```

Where the comment says "reuse via import": prefer importing from `Morphology/`
to avoid duplication. Only fork the file when the controller version genuinely
diverges from the morphology version.

Note: mutation operates on the **reward weight vector** (a small, structured,
human-interpretable object), not on the policy network. The mutation operator
therefore lives in `reward.py:mutate_weights`, not in `ppo_trainer.py`. PPO
is the inner-loop *solver* that maps each candidate reward to a controller.
For mutated children PPO warm-starts from the parent's policy weights — the
weights themselves are not directly perturbed, they are inherited and then
refined by gradient descent against the child's reward.

---

## 8. Component contracts (what each module must expose)

### 8.1  `controller_morph.py`

```python
from morphology import QUADRIPOD, MorphologyManager   # from code/Morphology/

STATIC_MORPH = QUADRIPOD     # 4 legs, 4 actuators

def build_model() -> mujoco.MjModel:
    """Return a ready-to-step MjModel for STATIC_MORPH (with floor, lighting,
    actuators). Wraps MorphologyManager.get_model() and also sets
    spawn_height via compute_spawn_height()."""
```

### 8.2  `reward.py`

The reward weight vector is the **thing that evolves**. It parametrises the
shaped per-step reward PPO uses to train one controller. Mutation lives here.

```python
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class RewardWeights:
    """
    Coefficients of the shaped reward fed to PPO.
    These are the variables the EA mutates and stores in the archive.
    Each term is signed so that a *positive* weight always means "more of
    this is better" — log-normal mutation can never flip a term's role.
    """
    forward_velocity: float = 1.0    # +v_x torso (forward progress)
    lateral_drift:    float = 0.1    # |v_y| torso (penalises sideways drift)
    upright_bonus:    float = 0.5    # cos(roll) * cos(pitch) (encourages upright)
    energy_penalty:   float = 0.001  # sum(action**2) (control cost)
    contact_reward:   float = 0.1    # number of feet in ground contact
    alive_bonus:      float = 0.05   # +1 per step the robot has not fallen
    fall_penalty:     float = 10.0   # one-off penalty when torso drops below threshold

    def to_vector(self) -> np.ndarray: ...
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "RewardWeights": ...
    def to_dict(self) -> dict: return asdict(self)


def mutate_weights(parent: RewardWeights,
                   sigma:  float = 0.2,
                   rng:    np.random.Generator | None = None) -> RewardWeights:
    """
    Multiplicative log-normal noise on each weight:
        w_child[i] = w_parent[i] * exp(N(0, sigma))
    Rationale: keeps signs sane and gives roughly proportional scaling
    regardless of a weight's magnitude (so a 100× difference between
    `fall_penalty` and `energy_penalty` does not require dimension-specific σ).
    """


def compute_step_reward(weights: RewardWeights,
                        sensors: "RobotSensorData",
                        action:  np.ndarray,
                        fell:    bool) -> float:
    """
    Compute the shaped reward for one env step from weights and sensor readings.
    Used by RobotControllerEnv.step(); never by the evolutionary loop itself.
    """
```

The seven-coefficient default is an opinionated starting set — extending the
list (e.g. adding `feet_air_time`, `joint_smoothness`) is a v1.x knob.
Whatever the dimensionality, no term may flip sign through mutation; the
human-chosen *direction* of each term is part of the prior.

### 8.3  `mujoco_env.py`

`class RobotControllerEnv(gym.Env)`:

- `__init__(reward_weights: RewardWeights, seed, episode_duration, render_mode=None)`
  — the env is **parameterised by the weight vector**. A new individual = a
  new env instance with a new `RewardWeights`.
- `observation_space`: `Box`. Composition (matches `RobotSensorData`):
  - 3 clock signals `sin(ω·t)` for ω in {1, 5, 15} (mirrors `simple_brain.get_input_clocks`)
  - hip angles (4)
  - hip velocities (4)
  - torso quaternion (4) and linear+angular velocity (6)
- `action_space`: `Box(low=-1, high=1, shape=(4,))` — interpreted as
  Δ-angle scaled by `PREDICTION_FACTOR` (same convention as
  `simple_brain.get_simplebrain_controller`), then added to current hip angles
  and clipped to actuator `ctrl_range`.
- `reset(seed)` → re-creates `MjData`, sets `qpos[2] = morph.spawn_height`.
- `step(action)`:
  - apply Δ-angle, `mj_step`
  - `r = compute_step_reward(self.reward_weights, sensors, action, fell)`
  - terminate early if torso falls below a height threshold (e.g. 0.05 m
    after low-pass filtering), or on time-out.
- `render() -> np.ndarray` (rgb_array) for video capture.

The shaped reward is **never** the evolutionary fitness. Its role is purely
to give PPO a dense gradient inside one training run. Different individuals
have different weight vectors → different shaped rewards → PPO finds different
controllers → the VLM judges which behaviour is interesting.

### 8.4  `ppo_trainer.py`

Two entry points: one for the initial population (random init, longer
budget), one for mutated children (warm-start from the parent's policy,
shorter budget).

```python
def train_from_scratch(
    reward_weights:  RewardWeights,
    seed:            int,
    total_timesteps: int  = 200_000,   # = cfg.n_init_steps
    n_envs:          int  = 4,
    save_path:       str  = None,
    tensorboard_log: str  = None,
) -> "PPO":
    """
    Train one PPO policy from random initial weights on RobotControllerEnv
    parameterised by reward_weights. Used only for the initial population
    (no parent yet to inherit from).
    Returns the trained model. Saves to save_path if given.
    """


def train_warm_start(
    reward_weights:     RewardWeights,
    parent_policy_path: str,
    seed:               int,
    n_warm_steps:       int  = 50_000,  # = cfg.n_warm_steps
    n_envs:             int  = 4,
    save_path:          str  = None,
    tensorboard_log:    str  = None,
) -> "PPO":
    """
    Load the parent's PPO policy from parent_policy_path, then continue
    training on RobotControllerEnv parameterised by reward_weights (the
    *mutated* child weights) for n_warm_steps additional timesteps with a
    fresh seed. The child therefore inherits the parent's learned
    locomotion competence and only needs to adapt to the new reward shape.
    Returns the trained model. Saves to save_path if given.
    """
```

Use `SubprocVecEnv` with `start_method="spawn"` on macOS. Default
`policy_kwargs=dict(net_arch=[256, 256])`. Log to TensorBoard under
`run_dir/tb/individual_<id>` so every individual's training curve is
inspectable post-hoc.

`n_warm_steps` should default to ~25–50 % of `n_init_steps` —
warm-starting from a competent parent buys most of the value of a full
training run for a fraction of the cost. Both numbers live in
`ExperimentConfig`. If `n_warm_steps` is too small the child stays a clone
of the parent; if it is too large warm-start loses its compute advantage
over from-scratch.

### 8.5  `video_renderer.py`

```python
def rollout_to_video(policy, env, save_path: str,
                     fps: int = 20, camera_track_torso: bool = True) -> str:
    """Roll out one deterministic episode, write MP4 to save_path,
    return the path."""
```

Adapt `proto/Mujoco/video_render.py` (libx264 ultrafast, async encoder
thread). Camera follows the torso so the VLM sees the gait clearly.

### 8.6  `gemini_prompts.py`

`LocomotionPromptConfig` mirroring `GeminiPromptConfig`. Replace
"resemble a {static_target}" with behavioural criteria like:

- `coherence` — does the gait look relevant toward the target?
- `originality` — does the robot make specific unconventional movement?
- `interest` — is the gait pattern interesting or biologically plausible and gives a real potential for optimization toward the target?

Same JSON output schema, same fitness formula
(`fitness = weighted_sum / (10 * sum_of_weights)`).

Provide at least one preset, e.g. `WALK_FORWARD`. Keep weights as
`coherence=1.0, progress=1.5, interest=0.5`.

### 8.7  `grader.py`

`LocomotionGrader(MorphologyGrader)`:

- `score_batch(videos: list[tuple[str, str]], ...) -> dict[str, GraderOutput]`
  - input: `(individual_id, mp4_path)` pairs
  - upload all videos via Files API in one `generate_content` call
    (chunked by `batch_size`)
  - parse the same JSON schema as the morphology grader
  - support the `descriptor_config` arg so the same call returns
    MAP-Elites descriptors
- Reuse `_log_response`, `_extract_vlm_descriptors`, the chunking pattern,
  and the optional `reference_image`/`reference_video` argument.

### 8.8  `data_handler.py`

```python
@dataclass
class ControllerResult:
    generation:     int
    individual_id:  int
    parent_id:      int | None
    reward_weights: dict[str, float]   # the evolved variable, serialised for JSONL
    policy_path:    str                # .zip from SB3 (controller PPO found for these weights)
    video_path:     str                # .mp4 from rollout
    n_train_steps:  int                # PPO timesteps spent on this individual
    fitness:        float              # from the VLM
    raw_scores:     dict[str, float]   # per-dimension VLM scores (coherence, progress, ...)
    descriptors:    dict[str, float | int]
    grader_method:  str
    prompt_set:     str
    grader_extra:   dict
```

`reward_weights` is stored as a plain dict (not a `RewardWeights` instance)
so the result is JSON-serialisable for `individuals_log.jsonl`. Reconstruct
via `RewardWeights(**result.reward_weights)`.

Plus `evaluate_batch(individuals, ...) -> (list[ControllerResult], new_id_counter)`
mirroring `Morphology/data_handler.py:evaluate_batch`. An "individual" passed
in is a `(reward_weights, policy_path)` pair (the policy is produced by the
trainer just before evaluation).

### 8.9  `evolution.py`

Same `BaseEvolution` interface as `Morphology/evolution.py`. Two strategies:

- `MuLambdaEvolution` — sample λ parents from the archive, mutate each
  parent's **reward weights**, **warm-start PPO from each parent's policy**
  against the child's reward, roll out each child to MP4, **batch-score
  all λ MP4s with one VLM call**, keep best μ.
- `MapEliteEvolution` — sample λ parents from filled grid cells, mutate
  reward weights, warm-start PPO from each parent's policy, roll out,
  **batch-score all λ MP4s with one VLM call**, insert if cell improves.

Each generation has the same coarse shape — the VLM call comes once at the
end, after every child has been trained and rendered:

```
for g in range(n_generations):
    children = [mutate_weights(p.reward_weights, ...) for p in select_parents(archive, lambda_)]
    for w, parent in zip(children, parents):
        train_warm_start(reward_weights=w, parent_policy_path=parent.policy_path, ...)
        rollout_to_video(...)                             # produces an MP4 per child
    results = grader.score_batch(videos, ...)             # ← single batched VLM call
    archive.insert_many(results)
```

Initialisation: `init_population_size` random `RewardWeights` (sampled around
the default vector with a wider σ than the per-generation mutation), each
trained **from scratch** with a different seed; the initial population is
also batch-scored once after all from-scratch trainings finish.

Per-child mutation is two function calls (`reward.mutate_weights` then
`ppo_trainer.train_warm_start`):
```python
child_weights = mutate_weights(parent.reward_weights,
                               sigma = cfg.reward_mutation_sigma,
                               rng   = rng)
child_policy_path = run_dir / "policies" / f"id{child_id:06d}.zip"
train_warm_start(
    reward_weights     = child_weights,
    parent_policy_path = parent.policy_path,     # inherit parent's locomotion
    seed               = generation_seed + i,
    n_warm_steps       = cfg.n_warm_steps,
    save_path          = child_policy_path,
)
```

Lineage is tracked through `parent_id`. Cumulative inner-loop compute along
a lineage is `n_init_steps + depth * n_warm_steps`, which the analyser can
plot against fitness to see how much extra training each individual
inherited from its ancestors.

### 8.10  `experiment.py`

`run(cfg)` and `resume(run_dir)` — same shape and side effects as
`Morphology/experiment.py`. The only difference is that "evaluation" now
means: for each child of the generation, mutate reward weights and
warm-start PPO from the parent's policy (or train from scratch for the
initial population), then record an MP4 rollout; **once all children of
the generation are rendered, run one batched VLM call to score the whole
batch and update the archive**. In `Morphology/` the batched call already
existed; here every child first costs an inner-loop PPO budget on top.

`__main__` runs a debug smoke test with a fake env and fake grader.

---

## 9. Build order (recommended path)

Do these in order. Each step is independently runnable; do not move on until
its `__main__` debug block prints success.

1. **Install missing libs**
   `pip install gymnasium stable-baselines3`

2. **`controller_morph.py`** — print model.nq / nu, save one offscreen
   PNG render of the static body.

3. **`reward.py`** — implement `RewardWeights`, `to_vector` / `from_vector`
   round-trip, and `mutate_weights`. `__main__` should sample 100 mutations
   from the default vector and print summary statistics (mean, std, min/max
   per dimension) so the σ behaviour can be eyeballed.

4. **`mujoco_env.py`** — implement env taking a `RewardWeights` argument,
   run `gymnasium.utils.env_checker.check_env`, then run a 1000-step
   random-action episode and print returns + termination reason.

5. **`ppo_trainer.py:train_from_scratch`** — train one policy against the
   *default* `RewardWeights` for ~100 k steps, log to TensorBoard, confirm
   forward velocity climbs. Save the resulting `.zip` — it becomes the
   "parent" for steps 6–8.

6. **`video_renderer.py`** — roll out the trained policy, write an MP4,
   manually open it and confirm the camera tracks the torso.

7. **`gemini_prompts.py` + `grader.py`** — score one MP4 with the VLM
   (single-video smoke test, not batch). Confirm the JSON parses and the
   fitness is in `[0, 1]`.

8. **`ppo_trainer.py:train_warm_start` + reward mutation smoke test** —
   sample one mutated `RewardWeights` from the default with `sigma=0.3`,
   load step 5's policy as the parent, **warm-start PPO** for ~25 k steps
   against the mutated reward, re-render, re-score with the VLM. Confirm
   (a) the gait visibly differs from step 5's, (b) warm-start wall-clock
   time is well under from-scratch, and (c) the fitness moves. This
   validates the *full* (mutate → warm-start → render → score) pipeline
   end-to-end on one individual before stitching it into evolution.

9. **`data_handler.py` + `evolution.py` + `archive.py`** — run a tiny
   `MuLambdaEvolution` (μ=2, λ=2, n_gen=2) end-to-end, no VLM (use a fake
   grader from the `__main__` of `Morphology/experiment.py` as a template).

10. **Real run** — small `MuLambdaEvolution` with VLM (μ=3, λ=4, n_gen=5,
    `n_init_steps=50_000`, `n_warm_steps=15_000` per individual). Verify
    the archive populates, `report.txt` generates, and analyser-compatible
    files are written.

11. **Scale up** — bump generations, training budgets, switch to
    `MapEliteEvolution` with a `LOCOMOTION_DESCRIPTORS` config.

---

## 10. Open / future work

These are explicitly **out of scope for v1**. Note them in
`individuals_log.jsonl`'s `grader_extra` if they come up early so they are
not lost.

- Richer mutation operators on `RewardWeights`: per-dimension σ, dimension
  dropout (set a weight to zero — reward-component ablation as evolution),
  uniform reset of one dimension, two-parent recombination (averaging or
  per-dimension crossover).
- Richer warm-start strategies: LR / entropy-coefficient reset on warm-start,
  partial weight perturbation on top of warm-start (to break out of the
  parent's local optimum), separate value-function reset (the parent's
  critic is calibrated for the *parent's* reward, not the child's).
- Periodic "from-scratch" injection during evolution (re-seed the population
  with new random `RewardWeights` trained from scratch every K generations)
  to prevent the whole archive from converging onto one ancestor's policy
  bias.
- Extending the `RewardWeights` schema (`feet_air_time`, `joint_smoothness`,
  `head_height`, …) — every new term widens the search space and the
  PPO budget needed to converge against it.
- VLM-shaped per-step reward (currently far too expensive — Gemini takes
  ~5 s per call).
- Per-rollout structural descriptors (gait frequency from FFT of joint
  angles, average stride length, energy spent) as a complement to
  VLM-returned descriptors.
- Switching MuJoCo → MJX for batched, GPU-parallel rollouts (would require
  a JAX rewrite of `morphology.py`).
- Loading a top morphology from a `Morphology/` run instead of `QUADRIPOD`
  (couples stages 2 and 3; do once stage 3 alone is validated).

---

## 11. Questions that should pause coding and surface to the user

If any of the following arises during implementation, **stop and ask** —
do not silently decide:

- The default `RewardWeights` vector is fundamentally broken (e.g. PPO with
  the default vector converges to "stand still" because the alive bonus
  dominates). The starting prior is a real research question and should be
  discussed before letting evolution proceed — bad seeds poison the archive
  *and* every warm-started descendant inherits that bias.
- Reward-weight mutation σ is **too small** (children behave identically to
  parents; archive stagnates) or **too large** (warm-started PPO can't bend
  the parent's policy to the new reward within `n_warm_steps`). Calls for
  per-dimension σ or capped log-normal magnitudes.
- Warm-started children are **indistinguishable from their parent** even
  though the reward weights have moved (PPO snaps back to the parent's
  local optimum on the new reward). Calls for increasing `n_warm_steps`,
  adding LR / entropy-coefficient reset on warm-start, or layering small
  weight noise on top of warm-start (§10).
- A single from-scratch PPO run takes more than ~10 min wall-clock on the
  target hardware; the total initial-population budget needs renegotiation
  or `init_population_size` must shrink.
- Gemini's JSON parsing fails on > 5 % of videos despite a clean prompt.
  May need to switch from MP4 upload to a keyframe strip.
- The VLM consistently scores "no progress" gaits highly because the
  static frame looks plausible. Would mean the prompt needs richer
  temporal cues or video resolution must increase.

---

## 12. Glossary

| Term | Meaning here |
|------|--------------|
| Individual | A **(reward weight vector, PPO-trained policy)** pair for the static QUADRIPOD body. Stored in the archive together. |
| Reward weights | The small, structured, human-interpretable vector of coefficients (forward velocity, energy penalty, upright bonus, …) that parametrises the per-step shaped reward PPO sees. **The variable that actually evolves.** |
| Generation | One full evolutionary step: select parents → mutate reward weights → warm-start PPO from each parent's policy against the child reward → roll out each child to MP4 → **score the whole batch of MP4s in one VLM call** → archive update. The VLM is called once per generation, never per individual. |
| Inner loop | PPO training for one individual, optimising the policy NN against that individual's shaped reward. From scratch for the initial population, warm-started thereafter. |
| Outer loop | The evolution loop across generations. Mutates reward weights, uses the VLM fitness to drive selection. |
| From-scratch training | Initial PPO training from random weights against the individual's `RewardWeights`. Used only for the initial population (no parent yet). Costs `n_init_steps` timesteps. |
| Warm-start training | PPO training that resumes from the parent's saved policy with a fresh seed and the *child's* (mutated) reward weights. Costs `n_warm_steps` (typically 25–50 % of `n_init_steps`). The compute-saving inner-loop optimisation; the trade-off is that the child's behaviour partly inherits the parent's policy bias on top of the new reward. |
| Fitness | The scalar from the VLM in `[0, 1]`; **only this drives selection** in the outer loop |
| Descriptor | A categorical or scalar feature returned by the VLM that places an individual in a MAP-Elites cell |
| Static morph | The frozen QUADRIPOD body, identical for every individual |
| Lineage | The chain of parent → child relationships traced through `parent_id`. Cumulative inner-loop compute for an individual at depth k is `n_init_steps + k * n_warm_steps`. Useful for analysing reward-weight drift, inherited policy bias, and the compute–fitness curve over generations. |

---

## 13. Interactive development tools

Two tkinter desktop tools are provided for studying the PPO inner loop outside
of a full evolutionary run. They share no shared state with `experiment.py` and
write to their own `results/study_output/` directory.

### 13.1  `utils/controller_generator_renderer.py` — Controller Trainer & Generator

An interactive GUI for manually exploring the PPO training loop: generate or
mutate reward weight vectors, watch a live fitness plot as PPO trains, then
play back the rendered MP4 rollout — all without running a full evolutionary
experiment.

**Launch:**
```bash
cd code/Controller
python utils/controller_generator_renderer.py
```

**Session model.** The app maintains a session history (a list of
`IndividualResult` objects). Navigation buttons operate on this history:

| Button | Key | Action |
|--------|-----|--------|
| Back   | B   | Re-display previous individual (restores its weights/video) |
| New    | N   | Train a fresh individual from scratch with random-init weights |
| Mutate | M   | Mutate the current individual's reward weights; warm-start PPO from its policy |
| Continue | C | Warm-start the current individual's policy against its **same** reward weights for another `n_warm_steps` |
| Edit Weights | T | Train from the manually set reward weight sliders (ignoring any parent) |
| Save video | V | Open file dialog to save the current MP4 |
| Skip   | S   | Skip playback; mark individual as done without rendering |

**Panels (left sidebar):**

- **Training params** — `n_init_steps` / `n_warm_steps` / `n_envs` / `episode_duration` sliders and a device radio (`cpu` / `mps`). `n_envs` controls how many parallel MuJoCo environments PPO collects data from simultaneously — more envs = more diverse rollouts per update = faster wall-clock, but higher RAM usage.
- **Mutation sigma** — Controls the per-dimension log-normal noise applied by `mutate_weights()`. Higher σ → more aggressive jumps in reward space.
- **Reward Weights** — Seven labelled sliders, one per `RewardWeights` field. These sliders always reflect the **current individual's** weights after every Mutate / New / Load operation. When you click [T] (Edit Weights), PPO is trained against whatever values are currently set in these sliders.
- **Session stats** — Live counter of individuals trained this session, plus the current individual's `total_steps_trained` (cumulative PPO steps across from-scratch + all [C] continue runs).
- **Individual info** — Shows the last individual's mode (`scratch` / `warm` / `manual` / `continue`), fitness (mean reward of last 10 PPO episodes), and step count.
- **Config summary** — Prints the effective `ExperimentConfig` fields for reference.

**Live training feedback:**

- A **progress bar** tracks the current PPO run from 0 → 100%.
- A **fitness plot** (live, no matplotlib) updates every rollout epoch. X-axis = steps elapsed in the current run; Y-axis = mean episode reward of the last 20 PPO episodes. The plot resets at the start of each new training run. It gives immediate feedback on whether a reward weight configuration leads to a climbing, stagnant, or collapsing fitness curve — the key diagnostic for reward shaping.

**Output directory** is cleared on every launch. All policies and videos are
written to `results/study_output/`.

**Interaction notes:**
- `var.set()` on a tkinter Scale does **not** fire the `command=` callback; the
  app maintains a separate `_rw_val_labels` dict and updates labels directly in
  `_sync_rw_sliders()`.
- All PPO / MuJoCo work runs in a daemon thread. The UI polls a
  `queue.Queue` every 200 ms via `root.after(200, self._poll)` — this keeps the
  tkinter main loop responsive during training.
- Video playback uses `imageio` frame extraction into a list of `PIL.Image`
  objects; animation is driven by `root.after(50, ...)` at ~20 fps.

### 13.2  `results/data_analyser.py` — Run Explorer

Post-hoc interactive viewer for completed evolutionary runs. Reads the
`log.jsonl` and `individuals_log.jsonl` files that `experiment.py` writes.

**Launch:**
```bash
cd code/Controller
python results/data_analyser.py                      # opens file dialog to pick a run dir
python results/data_analyser.py results/run_XXXXXX   # open run dir directly
```

**Features:**
- Fitness-over-generations chart (matplotlib embedded via TkAgg backend)
- Individual browser: click any point to inspect that individual's reward
  weights, fitness, descriptor, and lineage
- Embedded video player (`SimpleVideoPlayer`) backed by `av` + Pillow for
  looping MP4 playback in-app — no external player needed
- Archive grid heatmap for MAP-Elites runs (descriptor axes as X/Y, fitness as
  colour)

**Dependencies:** `matplotlib`, `Pillow`, `av` (PyAV for video decode).

---

## 14. PPO internals — callbacks and hyperparameter tuning

### 14.1  The `_TrainingCallback` (live feedback during training)

`utils/controller_generator_renderer.py` injects a `stable_baselines3`
callback into every PPO training call to stream progress and fitness data back
to the UI without polling the process or blocking the main thread.

```python
class _TrainingCallback(BaseCallback):
    def __init__(self, total_steps, result_queue, interval=500):
        super().__init__(verbose=0)
        self._total    = total_steps
        self._queue    = result_queue
        self._interval = interval   # post a progress update every N steps
        self._start    = 0
        self._last     = 0

    def _on_training_start(self) -> None:
        # Capture the model's timestep counter at the *start* of this run.
        # For warm-start (reset_num_timesteps=False) the counter is already
        # > 0 (the parent's step count), so we compute progress relative to
        # this snapshot, not to the model's lifetime counter.
        self._start = self.num_timesteps
        self._last  = 0

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - self._start
        if elapsed - self._last >= self._interval:
            self._queue.put(("progress", min(1.0, elapsed / self._total)))
            self._last = elapsed
        return True

    def _on_rollout_end(self) -> None:
        # Called once per rollout collection (every n_steps_per_env steps).
        # Post the mean reward of the last 20 finished episodes as a fitness
        # point so the UI can update its live plot.
        buf = list(self.model.ep_info_buffer)
        if buf:
            elapsed  = self.num_timesteps - self._start
            mean_r   = float(np.mean([ep["r"] for ep in buf[-20:]]))
            self._queue.put(("fitness_point", (elapsed, mean_r)))
```

The callback is passed to `ppo_trainer.train_from_scratch` and
`ppo_trainer.train_warm_start` via the `callback=` argument (added to both
functions as an optional parameter, defaulting to `None` for backward
compatibility with `experiment.py`).

**Why capture `self._start` in `_on_training_start`?** — In a warm-start run
`PPO.learn(reset_num_timesteps=False)` continues counting from where the parent
left off. Without the snapshot, `elapsed / total` would immediately exceed 1.0
and the progress bar would appear full. The snapshot makes the fraction relative
to the current run, regardless of the model's accumulated lifetime step count.

**Queue message types:**

| Type | Payload | When posted |
|------|---------|-------------|
| `"status"` | `str` | Human-readable status change (starting, done, error) |
| `"progress"` | `float` in [0, 1] | Every `interval` steps during training |
| `"fitness_point"` | `(int, float)` — (step, mean_r) | End of each rollout collection |
| `"done"` | `IndividualResult` | Training + rollout complete |
| `"error"` | `str` | Exception in the worker thread |

### 14.2  PPO hyperparameter reference

All values live in `config.py:ExperimentConfig`. This table covers influence
and tuning guidance specific to this locomotion task (5 s episodes, 4-actuator
quadruped, `n_envs` parallel environments).

#### `policy_arch` — default `[256, 256]`

Defines the MLP hidden layer sizes shared by the policy head and value head.
`[256, 256]` = two hidden layers of 256 neurons each; this is the standard for
continuous-control locomotion and is large enough for the ~20-dim observation
space and 4-dim action space.

- **Too small** (`[32, 32]`): Policy lacks capacity to represent a coherent
  gait; training may converge to a degenerate stand-still strategy.
- **Too large** (`[1024, 1024]`): Slower per-update; diminishing returns for
  this task; also makes warm-start less stable since more parameters need
  to "re-orient" to the new reward.
- **Tuning signal**: If from-scratch training plateaus early with a clean
  fitness curve, the architecture is not the bottleneck. If the curve is noisy
  and the gait looks chaotic even after `n_init_steps`, try adding a third
  layer or switching to `[512, 512]`.

#### `learning_rate` — default `3e-4`

Adam optimiser step size for both policy and value network updates.

- **Too high** (`> 1e-3`): Policy updates destroy previously learned locomotion;
  particularly dangerous for warm-starts where the parent's competence must
  be preserved and gradually redirected.
- **Too low** (`< 1e-5`): Children barely diverge from their parent within
  `n_warm_steps`; the EA cannot explore the reward-weight space effectively.
- **Tuning signal**: For warm-starts where children look identical to the
  parent, try raising to `5e-4`. For unstable from-scratch runs (fitness
  spikes up then collapses), lower to `1e-4`.

#### `gamma` — default `0.99`

Discount factor on future rewards. At γ=0.99, a reward 100 steps away
(= 5 s at 20 Hz, one full episode) is worth `0.99^100 ≈ 0.37` of an immediate
reward.

- **Critical for `fall_penalty`**: This is a one-off terminal penalty. If γ
  is too low the agent cannot "see" the fall cost far enough in advance to
  avoid it — resulting in policies that walk boldly then fall.
- **Tuning signal**: Do not lower below 0.97 for this task. If the agent
  never learns to avoid falls, this is the first parameter to raise (try
  0.995).

#### `gae_lambda` — default `0.95`

λ parameter for Generalized Advantage Estimation. Controls bias-variance
trade-off in advantage estimates used by the policy gradient.

- λ = 1 → Monte Carlo advantage (low bias, high variance — noisy training).
- λ = 0 → 1-step TD advantage (high bias, low variance — smooth but slow).
- `0.95` is robust for dense-reward locomotion (alive_bonus + contact_reward
  + forward_velocity fire every step).
- **Tuning signal**: Only lower (to 0.9) if the training loss oscillates wildly
  and `learning_rate` tuning did not help. Almost never the root cause.

#### `ent_coef` — default `0.0`

Weight on the entropy bonus added to the PPO loss, encouraging the policy
to stay stochastic and explore.

- `0.0` = no exploration bonus; the policy converges greedily.
- In this EA, diversity comes from the **outer loop** (reward weight mutation)
  — not from within a single PPO run. The inner loop should converge, not
  explore. `0.0` is correct.
- **Exception**: If from-scratch training gets trapped in a degenerate local
  optimum early (e.g. robot falls on the first step and the policy never
  recovers), try `0.005`–`0.01` for `n_init_steps` only. Immediately reset to
  `0.0` for subsequent warm-starts.

#### `vf_coef` — default `0.5`

Weight of the value function (critic) loss in the combined PPO loss:
`loss = policy_loss + vf_coef × value_loss − ent_coef × entropy`.

The critic produces baseline estimates for GAE — a well-fitted critic reduces
variance in the advantage estimates and improves policy gradient quality.
`0.5` is the SB3 default and is almost never the bottleneck.

- **Tuning signal**: Only lower (to `0.25`) if the critic appears to be
  overfitting the rollout data (value loss drops to near-zero while policy
  loss stagnates). In practice, leave at `0.5`.

#### `n_steps_per_env` — default `2048`

Number of env steps each vectorised environment collects before one PPO
update round. Total rollout buffer = `n_steps_per_env × n_envs`.

With the defaults (`n_steps_per_env=2048, n_envs=8`), each PPO update
processes `16,384` steps — roughly **163 full 5 s episodes**. This gives
each update diverse, representative experience.

- **Too large**: Updates are well-calibrated but infrequent; the policy adapts
  slowly to the new reward — a problem for short warm-starts.
- **Too small**: More frequent updates but high variance; can cause erratic
  gait oscillations.
- **Tuning signal**: For warm-starts where `n_warm_steps` is short (< 50 k),
  consider reducing to `512`–`1024` so the policy gets more gradient steps per
  timestep budget. Keep at `2048` for long from-scratch runs.
- **Constraint**: `(n_steps_per_env × n_envs)` must be divisible by
  `batch_size`.

#### `batch_size` — default `256`

Mini-batch size within each PPO update round. The full rollout buffer is
shuffled and split into mini-batches of this size; each mini-batch produces
one gradient update. SB3 default `n_epochs=10` means the buffer is swept 10
times per update round.

- Must evenly divide `n_steps_per_env × n_envs` (= 16,384 with defaults).
  Valid choices: 128, 256, 512, 1024, 2048, 4096, 8192.
- **Too large** (→ 8192): Smoother gradients but risk overfitting the rollout
  buffer; each update round produces fewer gradient steps.
- **Too small** (→ 64): Noisy gradients; can help escape local optima but
  may destabilise warm-starts.
- **Tuning signal**: Change only when you change `n_steps_per_env` or
  `n_envs`. Scale proportionally: `batch_size ≈ total_rollout / 64` is a
  reasonable heuristic. For quick debug runs (`n_steps_per_env=256, n_envs=1`)
  use `batch_size=64`.

### 14.3  Device selection on Apple Silicon

MuJoCo simulation and data collection always run on CPU. For tiny MLPs
(`[256, 256]` with 4-dim actions), the cost of transferring each mini-batch
between CPU and MPS exceeds the GPU compute saving — resulting in **~5×
slower training** than CPU-only (observable as 0% ↔ 50% GPU usage spikes:
the GPU is idle during rollout collection on CPU, then saturated briefly
during the mini-batch update, then idle again).

The default device is therefore `"cpu"`. To experiment with MPS anyway, set
`device = "mps"` in the Training Params panel of the interactive tool or in
`ExperimentConfig`. Only likely to benefit if you increase `policy_arch`
substantially (e.g. `[1024, 1024, 1024]`) or use very large batch sizes.