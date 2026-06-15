"""
evolution_mjx.py
================
MJX drop-in replacement for Controller/evolution.py.

Same `BaseEvolution` interface (initialise / step), same archive / data_handler
contracts — only the PPO inner loop and renderer are swapped for their JAX
equivalents:

  train_from_scratch_mjx  instead of  train_from_scratch   (SB3)
  train_warm_start_mjx    instead of  train_warm_start      (SB3)
  rollout_to_video_mjx    instead of  rollout_to_video      (mujoco_env Gym)

Policy persistence
------------------
SB3 saves policies as .zip files.  Flax params are plain pytrees (nested dicts
of jax arrays) — we persist them as msgpack bytes via
`flax.serialization.to_bytes / from_bytes` in a .params file.

`ControllerResult.policy_path` still points to a file on disk; callers just
need to use `load_params()` instead of `PPO.load()`.

MJX model caching
-----------------
`build_env_config()` compiles the MuJoCo model into a JAX-traceable MJX model,
which is relatively slow.  Since the *morphology* is fixed for an entire run,
we build a single template config once in `__init__` and swap only
`reward_weights_vec` per individual using `dataclasses.replace()`.

Debug
-----
Run this file with a fake grader to exercise the full loop without Gemini.
"""

from __future__ import annotations

import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from config       import ExperimentConfig
from reward       import RewardWeights, mutate_weights, random_initial_weights
from data_handler import ControllerResult, evaluate_batch, _IndividualSpec
from archive      import MuLambdaArchive, MapEliteArchive
from controller_morph import build_model
import optax
from flax.training.train_state import TrainState

from mujoco_env_mjx   import (
    MJXEnvConfig, build_env_config, _pick_mjx_device,
    make_reward_agnostic_batch_fns,
)
from ppo_trainer_mjx  import (
    ActorCritic,
    train_from_scratch_mjx,
    train_warm_start_mjx,
    PPOConfig,
    make_train_step_fn,
)
from video_renderer_mjx import rollout_to_video_mjx, make_reusable_render_fns


# ---------------------------------------------------------------------------
# Policy persistence helpers
# ---------------------------------------------------------------------------

def save_params(params: Any, path: str) -> None:
    """Pickle Flax params (nested dict of jax arrays → numpy) to `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cpu_params = jax.tree.map(np.asarray, params)
    with open(path, "wb") as f:
        pickle.dump(cpu_params, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_params(path: str) -> Any:
    """Load Flax params from a .params pickle file, converting to jax arrays."""
    with open(path, "rb") as f:
        cpu_params = pickle.load(f)
    return jax.tree.map(jnp.asarray, cpu_params)


# ---------------------------------------------------------------------------
# Path helpers (mirror evolution.py)
# ---------------------------------------------------------------------------

def _policies_dir(run_dir: Path) -> Path:
    return run_dir / "policies"


def _videos_dir(run_dir: Path) -> Path:
    return run_dir / "videos"


def _policy_path(run_dir: Path, individual_id: int) -> str:
    p = _policies_dir(run_dir) / f"id{individual_id:06d}.params"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _video_path(run_dir: Path, individual_id: int, generation: int) -> str:
    p = _videos_dir(run_dir) / f"gen{generation:04d}_id{individual_id:06d}.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# ---------------------------------------------------------------------------
# BaseEvolutionMJX
# ---------------------------------------------------------------------------

class BaseEvolutionMJX(ABC):
    """
    Shared interface for MJX-backed evolution strategies.

    One major difference from BaseEvolution: `policy_path` in a
    `ControllerResult` points to a `.params` pickle file (Flax pytree)
    rather than an SB3 .zip.  The warm-start step loads this file and
    passes the params directly to `train_warm_start_mjx`.
    """

    def __init__(
        self,
        cfg:     ExperimentConfig,
        run_dir: Path,
        rng:     Optional[np.random.Generator] = None,
    ):
        self.cfg     = cfg
        self.run_dir = Path(run_dir)
        self.rng     = rng if rng is not None else np.random.default_rng(cfg.seed)
        # When True, _run_individual prints throttled per-update PPO progress
        # (update, steps/s, reward, losses, entropy) — useful for spotting a
        # divergent inner loop without waiting for the whole individual to finish.
        self.verbose_training = False

        # Force CPU device for MJX (Metal not supported)
        _dev = _pick_mjx_device()
        jax.config.update("jax_default_device", _dev)

        # Build a template MJXEnvConfig once (reused by swapping reward_weights_vec)
        print("[evolution_mjx] Building MJX template env config (model compile) …", flush=True)
        self._template_cfg: MJXEnvConfig = build_env_config(
            reward_weights    = RewardWeights(),
            episode_duration  = cfg.episode_duration,
            control_frequency = cfg.control_frequency,
            fall_height       = cfg.fall_height,
            prediction_factor = cfg.prediction_factor,
        )
        # Keep the CPU-side mj_model for rendering (mjx.get_data needs it)
        self._mj_model, _ = build_model()

        # Build shared JIT-compiled training infrastructure ONCE.
        # Because rw_vec is a runtime argument (not baked into the kernel),
        # the same compiled function handles all individuals without recompilation.
        self._init_shared_training()
        print("[evolution_mjx] Template config ready.", flush=True)

    def _init_shared_training(self) -> None:
        """
        Build JIT-compiled training infrastructure shared across all individuals.

        Key insight: make_reward_agnostic_batch_fns creates a batch_step_rw function
        where rw_vec is a RUNTIME argument (not a JIT closure constant).  Combined
        with a single make_train_step_fn call, this means:
          - JIT compiles ONCE (~30-60 s) during __init__
          - Every subsequent individual reuses the same XLA kernel
          - Different reward weights → different rw_vec arg, no recompile

        Without this, each individual causes a fresh 30-60 s JIT recompilation
        because the old design baked rw_vec into the closure.  For 50 individuals
        that wasted >40 minutes per generation.
        """
        n_envs      = self.cfg.n_envs_mjx
        rollout_len = self.cfg.n_steps_per_env
        arch        = tuple(self.cfg.policy_arch)
        ppo_cfg     = PPOConfig(n_epochs=4, n_minibatches=32)

        # Env functions: batch_reset with mjx.forward (used for initial reset),
        # batch_step_rw (rw_vec at runtime), fast_batch_reset (no mjx.forward).
        self._shared_batch_reset, self._shared_batch_step_rw, self._shared_fast_reset = \
            make_reward_agnostic_batch_fns(self._template_cfg, n_envs)

        # Single ActorCritic module — architecture fixed for the entire run.
        self._shared_net = ActorCritic(
            obs_dim = self._template_cfg.obs_dim,
            act_dim = self._template_cfg.n_joints,
            hidden  = arch,
        )

        # ONE compiled training step function for all individuals.
        self._shared_train_step = make_train_step_fn(
            cfg              = self._template_cfg,
            net              = self._shared_net,
            n_envs           = n_envs,
            rollout_len      = rollout_len,
            ppo_cfg          = ppo_cfg,
            gamma            = self.cfg.gamma,
            gae_lambda       = self.cfg.gae_lambda,
            batch_step_rw    = self._shared_batch_step_rw,
            fast_batch_reset = self._shared_fast_reset,
        )

    def _run_individual(
        self,
        params:           Any,
        rw_vec:           Any,    # (reward_dim,) JAX float32
        total_steps:      int,
        seed:             int,
        learning_rate:    float = 3e-4,
        fitness_episodes: int   = 20,
        verbose:          bool  = False,
    ) -> tuple[Any, float]:
        """
        Train one individual using the shared compiled training step.
        No recompilation — the kernel was compiled once in _init_shared_training.
        """
        n_envs      = self.cfg.n_envs_mjx
        rollout_len = self.cfg.n_steps_per_env
        ppo_cfg     = PPOConfig(n_epochs=4, n_minibatches=32)

        tx = optax.chain(
            optax.clip_by_global_norm(ppo_cfg.max_grad_norm),
            optax.adam(learning_rate),
        )
        train_state = TrainState.create(
            apply_fn = self._shared_net.apply,
            params   = params,
            tx       = tx,
        )

        # Initial reset (full, with mjx.forward — needed for valid kinematics)
        rng = jax.random.PRNGKey(seed)
        rng, key_rst = jax.random.split(rng)
        rst_keys = jax.random.split(key_rst, n_envs)
        env_states, obs = self._shared_batch_reset(rst_keys)

        runner_state    = (env_states, obs, rng)
        n_updates       = max(1, total_steps // (n_envs * rollout_len))
        tail_rewards    = []
        keep_tail       = max(2, 1 + fitness_episodes // max(1, n_envs))

        show     = verbose or self.verbose_training
        t_start  = time.perf_counter()
        last_log = t_start

        for update_idx in range(n_updates):
            train_state, runner_state, metrics, raw_rewards = self._shared_train_step(
                train_state, runner_state, rw_vec
            )
            tail_rewards.append(raw_rewards)
            if len(tail_rewards) > keep_tail:
                tail_rewards.pop(0)

            # Throttled progress: at most every ~5 s plus the final update.
            # The block_until_ready sync only fires when we actually print, so
            # logging does not stall the pipelined GPU rollout/update loop.
            now = time.perf_counter()
            if show and (now - last_log >= 5.0 or update_idx == n_updates - 1):
                jax.block_until_ready(metrics["value_loss"])
                steps_done = (update_idx + 1) * n_envs * rollout_len
                fps = steps_done / (now - t_start)
                rw_mean = float(jnp.mean(raw_rewards))
                print(f"        update {update_idx+1}/{n_updates}  "
                      f"steps={steps_done:,}  fps={fps:,.0f}  "
                      f"rw={rw_mean:+.3f}  π={float(metrics['actor_loss']):+.3f}  "
                      f"V={float(metrics['value_loss']):.1f}  "
                      f"ent={float(metrics['entropy']):.3f}", flush=True)
                last_log = now

        if tail_rewards:
            last = jnp.concatenate([r.ravel() for r in tail_rewards])
            fitness = float(jnp.mean(last))
        else:
            fitness = 0.0

        return train_state.params, fitness

    def _env_cfg_for(self, rw: RewardWeights) -> MJXEnvConfig:
        """Return a config with the given reward weights, reusing the compiled model."""
        return dc_replace(
            self._template_cfg,
            reward_weights_vec = rw.to_jax_vector(),
        )

    def _ppo_cfg(self) -> PPOConfig:
        return PPOConfig(n_epochs=4, n_minibatches=32)

    # ------------------------------------------------------------------
    # Per-child operations
    # ------------------------------------------------------------------

    def _train_from_scratch(
        self,
        rw:           RewardWeights,
        seed:         int,
        individual_id: int,
    ) -> tuple[str, Any, float]:
        """Train from random params using the shared compiled training step."""
        from ppo_trainer_mjx import init_actor_critic
        key    = jax.random.PRNGKey(seed)
        _, init_params = init_actor_critic(
            key, self._template_cfg.obs_dim, self._template_cfg.n_joints,
            tuple(self.cfg.policy_arch),
        )
        rw_vec = rw.to_jax_vector()
        params, fitness = self._run_individual(
            params        = init_params,
            rw_vec        = rw_vec,
            total_steps   = self.cfg.n_init_steps,
            seed          = seed,
            learning_rate = self.cfg.learning_rate,
        )
        path = _policy_path(self.run_dir, individual_id)
        save_params(params, path)
        return path, params, fitness

    def _train_warm_start(
        self,
        rw:            RewardWeights,
        parent_params: Any,
        seed:          int,
        individual_id: int,
    ) -> tuple[str, Any, float]:
        """Warm-start from parent params using the shared compiled training step."""
        rw_vec = rw.to_jax_vector()
        params, fitness = self._run_individual(
            params        = parent_params,
            rw_vec        = rw_vec,
            total_steps   = self.cfg.n_warm_steps,
            seed          = seed,
            learning_rate = self.cfg.learning_rate,
        )
        path = _policy_path(self.run_dir, individual_id)
        save_params(params, path)
        return path, params, fitness

    def _render_fns(self):
        """Build the compile-once render functions lazily and cache them.

        params/rw_vec are runtime arguments, so the same XLA executables serve
        every individual — avoiding the per-individual recompile that leaked
        VRAM and OOM'd the render after a few individuals.
        """
        if getattr(self, "_render_fns_cache", None) is None:
            self._render_fns_cache = make_reusable_render_fns(
                self._template_cfg,
                policy_arch   = tuple(self.cfg.policy_arch),
                deterministic = True,
            )
        return self._render_fns_cache

    def _render(
        self,
        params:       Any,
        rw:           RewardWeights,
        individual_id: int,
        generation:   int,
        seed:         int,
    ) -> str:
        env_cfg = self._env_cfg_for(rw)
        mp4     = _video_path(self.run_dir, individual_id, generation)
        policy_apply, reset_fn, step_rw_fn = self._render_fns()
        rollout_to_video_mjx(
            params        = params,
            cfg           = env_cfg,
            mj_model      = self._mj_model,
            save_path     = mp4,
            fps           = self.cfg.video_fps,
            render_width  = self.cfg.render_width,
            render_height = self.cfg.render_height,
            cam1_azimuth  = self.cfg.cam1_azimuth,
            cam1_elevation= self.cfg.cam1_elevation,
            cam1_distance = self.cfg.cam1_distance,
            cam1_lookat_z = self.cfg.cam1_lookat_z,
            cam2_azimuth  = self.cfg.cam2_azimuth,
            cam2_elevation= self.cfg.cam2_elevation,
            cam2_distance = self.cfg.cam2_distance,
            camera_track_torso = self.cfg.camera_track_torso,
            seed          = seed,
            policy_arch   = tuple(self.cfg.policy_arch),
            deterministic = True,
            policy_apply  = policy_apply,
            reset_fn      = reset_fn,
            step_rw_fn    = step_rw_fn,
        )
        return mp4

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialise(
        self, grader, id_counter: int = 0,
    ) -> tuple[list[ControllerResult], int]:
        ...

    @abstractmethod
    def step(
        self, archive, grader, generation: int, id_counter: int,
    ) -> tuple[list[ControllerResult], int]:
        ...


# ---------------------------------------------------------------------------
# MuLambdaEvolutionMJX
# ---------------------------------------------------------------------------

class MuLambdaEvolutionMJX(BaseEvolutionMJX):
    """(μ+λ) evolution using MJX training and rendering."""

    def initialise(self, grader, id_counter: int = 0):
        size     = self.cfg.resolved_init_population_size()
        defaults = self.cfg.default_reward_weights_dict()

        specs: list[_IndividualSpec] = []
        ids   = list(range(id_counter, id_counter + size))

        for k, ind_id in enumerate(ids):
            rw   = random_initial_weights(defaults, sigma=self.cfg.reward_init_sigma, rng=self.rng)
            seed = int(self.cfg.seed) + ind_id
            print(f"\r  [init {k+1}/{size}] training id={ind_id} from scratch "
                  f"({self.cfg.n_init_steps:,} steps) …", end="", flush=True)
            t0 = time.perf_counter()
            policy_path, params, fitness = self._train_from_scratch(rw, seed, ind_id)
            print(f"\r  [init {k+1}/{size}] rendering id={ind_id} …", end="", flush=True)
            video_path = self._render(params, rw, ind_id, generation=0, seed=seed)
            specs.append(_IndividualSpec(
                reward_weights = rw.to_dict(),
                policy_path    = policy_path,
                video_path     = video_path,
                parent_id      = None,
                n_train_steps  = self.cfg.n_init_steps,
            ))
            print(f"\r  [init {k+1}/{size}] id={ind_id} done in "
                  f"{time.perf_counter()-t0:.1f}s  fitness={fitness:.3f}")

        results, new_id = evaluate_batch(
            specs, grader, generation=0, id_counter=id_counter,
            reference_video=None, debug=False,
        )
        return results, new_id

    def step(
        self,
        archive:    MuLambdaArchive,
        grader,
        generation: int,
        id_counter: int,
    ):
        sampled_parents = archive.get_parent_results(self.cfg.lambda_)
        ids = list(range(id_counter, id_counter + self.cfg.lambda_))

        reference_video = None
        if self.cfg.reference_best_in_batch:
            best = archive.best()
            if best is not None and best.video_path:
                reference_video = best.video_path

        specs: list[_IndividualSpec] = []
        for k, parent in enumerate(sampled_parents):
            ind_id        = ids[k]
            parent_rw     = RewardWeights(**parent.reward_weights)
            child_rw      = mutate_weights(parent_rw, sigma=self.cfg.reward_mutation_sigma, rng=self.rng)
            seed          = int(self.cfg.seed) + 1000 * generation + ind_id
            parent_params = load_params(parent.policy_path)
            print(f"\r  [step] {k+1}/{len(sampled_parents)} gen={generation} "
                  f"id={ind_id} warm-start {self.cfg.n_warm_steps:,} steps …",
                  end="", flush=True)
            t0 = time.perf_counter()
            policy_path, params, fitness = self._train_warm_start(
                child_rw, parent_params, seed, ind_id
            )
            video_path = self._render(params, child_rw, ind_id, generation, seed)
            specs.append(_IndividualSpec(
                reward_weights = child_rw.to_dict(),
                policy_path    = policy_path,
                video_path     = video_path,
                parent_id      = parent.individual_id,
                n_train_steps  = self.cfg.n_warm_steps,
            ))
            print(f"\r  [step] {k+1}/{len(sampled_parents)} id={ind_id} done in "
                  f"{time.perf_counter()-t0:.1f}s  fitness={fitness:.3f}")

        children_results, new_id = evaluate_batch(
            specs, grader, generation=generation, id_counter=id_counter,
            reference_video=reference_video, debug=False,
        )

        # Pool = re-tagged parents + λ children (mirrors evolution.py)
        parent_pool = [
            ControllerResult(
                generation     = generation,
                individual_id  = r.individual_id,
                parent_id      = r.parent_id,
                reward_weights = r.reward_weights,
                policy_path    = r.policy_path,
                video_path     = r.video_path,
                n_train_steps  = r.n_train_steps,
                fitness        = r.fitness,
                raw_scores     = r.raw_scores,
                descriptors    = r.descriptors,
                grader_method  = r.grader_method,
                prompt_set     = r.prompt_set,
                grader_extra   = r.grader_extra,
            )
            for r in archive.population
        ]
        return parent_pool + children_results, new_id


# ---------------------------------------------------------------------------
# MapEliteEvolutionMJX
# ---------------------------------------------------------------------------

class MapEliteEvolutionMJX(BaseEvolutionMJX):
    """MAP-Elites variant using MJX training and rendering."""

    def initialise(self, grader, id_counter: int = 0):
        size     = self.cfg.resolved_init_population_size()
        defaults = self.cfg.default_reward_weights_dict()

        specs: list[_IndividualSpec] = []
        ids = list(range(id_counter, id_counter + size))
        for k, ind_id in enumerate(ids):
            rw         = random_initial_weights(defaults, sigma=self.cfg.reward_init_sigma, rng=self.rng)
            seed       = int(self.cfg.seed) + ind_id
            print(f"\r  [init {k+1}/{size}] training id={ind_id} from scratch "
                  f"({self.cfg.n_init_steps:,} steps) …", end="", flush=True)
            t0 = time.perf_counter()
            policy_path, params, fitness = self._train_from_scratch(rw, seed, ind_id)
            print(f"\r  [init {k+1}/{size}] rendering id={ind_id} …", end="", flush=True)
            video_path = self._render(params, rw, ind_id, generation=0, seed=seed)
            specs.append(_IndividualSpec(
                reward_weights = rw.to_dict(),
                policy_path    = policy_path,
                video_path     = video_path,
                parent_id      = None,
                n_train_steps  = self.cfg.n_init_steps,
            ))
            print(f"\r  [init {k+1}/{size}] id={ind_id} done in "
                  f"{time.perf_counter()-t0:.1f}s  fitness={fitness:.3f}")

        return evaluate_batch(
            specs, grader, generation=0, id_counter=id_counter,
            reference_video=None, debug=False,
        )

    def step(self, archive: MapEliteArchive, grader, generation: int, id_counter: int):
        sampled_parents = archive.get_parent_results(self.cfg.lambda_)
        ids = list(range(id_counter, id_counter + self.cfg.lambda_))

        reference_video = None
        if self.cfg.reference_best_in_batch:
            best = archive.best()
            if best is not None and best.video_path:
                reference_video = best.video_path

        specs: list[_IndividualSpec] = []
        for k, parent in enumerate(sampled_parents):
            ind_id        = ids[k]
            parent_rw     = RewardWeights(**parent.reward_weights)
            child_rw      = mutate_weights(parent_rw, sigma=self.cfg.reward_mutation_sigma, rng=self.rng)
            seed          = int(self.cfg.seed) + 1000 * generation + ind_id
            parent_params = load_params(parent.policy_path)
            print(f"\r  [step] {k+1}/{len(sampled_parents)} gen={generation} "
                  f"id={ind_id} warm-start {self.cfg.n_warm_steps:,} steps …",
                  end="", flush=True)
            t0 = time.perf_counter()
            policy_path, params, fitness = self._train_warm_start(child_rw, parent_params, seed, ind_id)
            video_path = self._render(params, child_rw, ind_id, generation, seed)
            specs.append(_IndividualSpec(
                reward_weights = child_rw.to_dict(),
                policy_path    = policy_path,
                video_path     = video_path,
                parent_id      = parent.individual_id,
                n_train_steps  = self.cfg.n_warm_steps,
            ))
            print(f"\r  [step] {k+1}/{len(sampled_parents)} id={ind_id} done in "
                  f"{time.perf_counter()-t0:.1f}s  fitness={fitness:.3f}")

        return evaluate_batch(
            specs, grader, generation=generation, id_counter=id_counter,
            reference_video=reference_video, debug=False,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_evolution_mjx(
    cfg:     ExperimentConfig,
    run_dir: Path,
    rng:     Optional[np.random.Generator] = None,
) -> BaseEvolutionMJX:
    if cfg.strategy == "mu_lambda":
        return MuLambdaEvolutionMJX(cfg, run_dir, rng)
    if cfg.strategy == "map_elite":
        return MapEliteEvolutionMJX(cfg, run_dir, rng)
    raise ValueError(f"Unknown strategy '{cfg.strategy}'.")


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  evolution_mjx.py — debug mode (fake grader)")
    print("=" * 60)

    class _FakeOut:
        def __init__(self, fit):
            self.fitness = fit
            self.raw_scores = {"coherence": fit}
            self.method = "fake"
            self.prompt_set = "fake"
            self.extra = {"vlm_descriptors": {}}

    class _FakeGrader:
        def __init__(self):
            self.rng = np.random.default_rng(99)
        def score_batch(self, videos, debug=False, reference_video=None):
            return {vid: _FakeOut(float(self.rng.uniform(0.3, 0.9))) for vid, _ in videos}

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_debug"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = ExperimentConfig(
            run_id               = "evo_mjx_debug",
            strategy             = "mu_lambda",
            mu                   = 2,
            lambda_              = 2,
            n_generations        = 1,
            init_population_size = 2,
            n_init_steps         = 2 * 8,    # tiny: 2 envs × 8 steps
            n_warm_steps         = 2 * 8,
            n_envs_mjx           = 2,
            n_steps_per_env      = 8,
            batch_size           = 8,
            episode_duration     = 1.5,
            output_dir           = tmp,
        )

        rng = np.random.default_rng(0)
        evo = MuLambdaEvolutionMJX(cfg, run_dir, rng)
        grader = _FakeGrader()

        print("\n[1] initialise")
        t0 = time.perf_counter()
        init_results, id_counter = evo.initialise(grader, id_counter=0)
        print(f"  {len(init_results)} results in {time.perf_counter()-t0:.1f}s")
        for r in init_results:
            print(f"    {r}")

        archive = MuLambdaArchive(mu=cfg.mu)
        archive.update(init_results)

        print("\n[2] step (gen 1)")
        t0 = time.perf_counter()
        step_results, id_counter = evo.step(archive, grader, generation=1, id_counter=id_counter)
        print(f"  {len(step_results)} results in {time.perf_counter()-t0:.1f}s")
        for r in step_results:
            print(f"    {r}")
        archive.update(step_results)
        archive.summary()

        # Verify .params files exist and are loadable
        policy_files = list((run_dir / "policies").glob("*.params"))
        print(f"\n  Policy files on disk : {len(policy_files)}")
        for pf in policy_files:
            loaded = load_params(str(pf))
            leaves = jax.tree.leaves(loaded)
            print(f"    {pf.name}  → {len(leaves)} param tensors")

    print("\nAll evolution_mjx.py checks passed.")
