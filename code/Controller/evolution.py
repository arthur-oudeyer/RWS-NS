"""
evolution.py
============
Evolution-strategy operators for the controller study.

Two strategies, same shared interface (`BaseEvolution`):

  MuLambdaEvolution  — pick λ parents, mutate reward weights, warm-start
                       PPO from each parent's policy, roll out, BATCH-SCORE
                       all λ MP4s in one VLM call, archive keeps best μ.

  MapEliteEvolution  — pick λ parents from filled grid cells, same per-child
                       train/render/score loop, archive inserts/replaces by
                       cell.

The crucial sequencing constraint (see instruction.md §1, §4 step c):
**all per-child PPO training and MP4 rendering happens first**; only after
every child of the generation has its video on disk does the grader fire
*one* batched call. Gemini latency dominates wall-clock time, so this is
the cheap way to make a generation tractable.

Per-child cost
--------------
Initial population (no parent): each child runs `train_from_scratch` for
`cfg.n_init_steps`. Subsequent generations: each child runs
`train_warm_start(parent_policy_path)` for `cfg.n_warm_steps`.

Cumulative inner-loop compute along a lineage at depth k is
`n_init_steps + k * n_warm_steps` — a useful axis for analysis.

Debug
-----
Run this file with the bundled fake grader to exercise the full loop in
~15 s without hitting Gemini.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from config        import ExperimentConfig
from reward        import RewardWeights, mutate_weights, random_initial_weights
from data_handler  import (
    ControllerResult, evaluate_batch, _IndividualSpec
)
from archive       import MuLambdaArchive, MapEliteArchive
from ppo_trainer   import train_from_scratch, train_warm_start
from video_renderer import rollout_to_video


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _policies_dir(run_dir: Path) -> Path:
    return run_dir / "policies"


def _videos_dir(run_dir: Path) -> Path:
    return run_dir / "videos"


def _policy_path(run_dir: Path, individual_id: int) -> str:
    p = _policies_dir(run_dir) / f"id{individual_id:06d}.zip"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _video_path(run_dir: Path, individual_id: int, generation: int) -> str:
    p = _videos_dir(run_dir) / f"gen{generation:04d}_id{individual_id:06d}.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# ---------------------------------------------------------------------------
# BaseEvolution
# ---------------------------------------------------------------------------

class BaseEvolution(ABC):
    """
    Shared interface for evolution strategies.

    Parameters
    ----------
    cfg     : ExperimentConfig.
    run_dir : where policies, videos, and snapshots are written.
    rng     : numpy Generator. Created from cfg.seed if None.
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
        self._tb_dir = str(self.run_dir / "tb")

    # ------------------------------------------------------------------
    # Per-child operations (shared)
    # ------------------------------------------------------------------

    def _train_from_scratch(self, rw: RewardWeights, seed: int, individual_id: int) -> (str, float):
        save_path = _policy_path(self.run_dir, individual_id)
        ppo, fitness = train_from_scratch(
            reward_weights    = rw,
            seed              = seed,
            total_timesteps   = self.cfg.n_init_steps,
            n_envs            = self.cfg.n_envs,
            save_path         = save_path,
            tensorboard_log   = self._tb_dir,
            tb_run_name       = f"id{individual_id:06d}_init",
            episode_duration  = self.cfg.episode_duration,
            control_frequency = self.cfg.control_frequency,
            fall_height       = self.cfg.fall_height,
            policy_arch       = self.cfg.policy_arch,
            learning_rate     = self.cfg.learning_rate,
            gamma             = self.cfg.gamma,
            gae_lambda        = self.cfg.gae_lambda,
            ent_coef          = self.cfg.ent_coef,
            vf_coef           = self.cfg.vf_coef,
            n_steps_per_env   = self.cfg.n_steps_per_env,
            batch_size        = self.cfg.batch_size,
            use_subproc       = self.cfg.n_envs > 1,
            verbose           = 0,
        )
        return save_path, fitness

    def _train_warm_start(
        self,
        rw:                  RewardWeights,
        parent_policy_path:  str,
        seed:                int,
        individual_id:       int,
    ) -> (str, float):
        save_path = _policy_path(self.run_dir, individual_id)
        _, fitness = train_warm_start(
            reward_weights     = rw,
            parent_policy_path = parent_policy_path,
            seed               = seed,
            n_warm_steps       = self.cfg.n_warm_steps,
            n_envs             = self.cfg.n_envs,
            save_path_warmed   = save_path,
            tensorboard_log    = self._tb_dir,
            tb_run_name        = f"id{individual_id:06d}_warm",
            episode_duration   = self.cfg.episode_duration,
            control_frequency  = self.cfg.control_frequency,
            fall_height        = self.cfg.fall_height,
            use_subproc        = self.cfg.n_envs > 1,
            verbose            = 0,
        )
        return save_path, fitness

    def _render(self, policy_path: str, rw: RewardWeights, individual_id: int, generation: int, seed: int) -> str:
        # Lazy-import SB3 + env so multi-process workers don't pay the cost.
        from stable_baselines3 import PPO
        from mujoco_env import RobotControllerEnv

        env = RobotControllerEnv(
            reward_weights    = rw,
            seed              = seed,
            episode_duration  = self.cfg.episode_duration,
            control_frequency = self.cfg.control_frequency,
            fall_height       = self.cfg.fall_height,
            render_mode       = "rgb_array",
            render_width      = self.cfg.render_width,
            render_height     = self.cfg.render_height,
        )

        try:
            policy = PPO.load(policy_path)
            mp4 = _video_path(self.run_dir, individual_id, generation)
            rollout_to_video(
                policy, env, mp4,
                fps                = self.cfg.video_fps,
                camera_track_torso = self.cfg.camera_track_torso,
                deterministic      = True,
            )
            return mp4
        finally:
            env.close()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialise(
        self,
        grader,
        id_counter: int = 0,
    ) -> tuple[list[ControllerResult], int]:
        ...

    @abstractmethod
    def step(
        self,
        archive,
        grader,
        generation: int,
        id_counter: int,
    ) -> tuple[list[ControllerResult], int]:
        ...


# ---------------------------------------------------------------------------
# MuLambdaEvolution
# ---------------------------------------------------------------------------

class MuLambdaEvolution(BaseEvolution):
    """
    Classical (μ+λ): random init at gen 0, then per gen pick λ parents,
    mutate reward weights, warm-start from each parent's policy, roll out,
    batch-score, archive keeps best μ.
    """

    # ------------------------------------------------------------------
    # Initial population — train λ random individuals from scratch
    # ------------------------------------------------------------------

    def initialise(self, grader, id_counter: int = 0):
        size = self.cfg.init_population_size or self.cfg.mu * 2
        defaults = self.cfg.default_reward_weights_dict()

        specs: list[_IndividualSpec] = []
        ids   = list(range(id_counter, id_counter + size))
        for k, ind_id in enumerate(ids):
            rw   = random_initial_weights(defaults, sigma=self.cfg.reward_init_sigma, rng=self.rng)
            seed = int(self.cfg.seed) + ind_id
            print(f"\r  [init {k}/{size} ] training individual {ind_id} from scratch ({self.cfg.n_init_steps:,} steps) ...", end='')
            t0 = time.perf_counter()
            policy_path, fitness = self._train_from_scratch(rw, seed, ind_id)
            print(f"\r  [init {k}/{size} ] rendering rollout for id={ind_id} ...", end='')
            video_path = self._render(policy_path, rw, ind_id, generation=0, seed=seed)
            specs.append(_IndividualSpec(
                reward_weights = rw.to_dict(),
                policy_path    = policy_path,
                video_path     = video_path,
                parent_id      = None,
                n_train_steps  = self.cfg.n_init_steps,
            ))
            print(f"\r [init {k}/{size}] id={ind_id} trained and render in {time.perf_counter()-t0:.1f}s (fitness {fitness:.3f})")
        print(f"\r  [init {size}/{size} ] Training and Rendering succeeded. evaluating..",  end='')

        results, new_id = evaluate_batch(
            specs, grader, generation=0, id_counter=id_counter,
            reference_video=None, debug=False,
        )
        print(f"\r  [init] Initialization done.")
        return results, new_id

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------

    def step(self, archive: MuLambdaArchive, grader, generation: int, id_counter: int):
        # 1. Sample λ parents from the archive
        sampled_parents = archive.get_parent_results(self.cfg.lambda_)
        ids = list(range(id_counter, id_counter + self.cfg.lambda_))

        # 2. Optional reference video (current best)
        reference_video = None
        if self.cfg.reference_best_in_batch:
            best = archive.best()
            if best is not None and best.video_path:
                reference_video = best.video_path

        # 3. For each child: mutate → warm-start → render
        specs: list[_IndividualSpec] = []
        for k, parent in enumerate(sampled_parents):
            ind_id = ids[k]
            parent_rw = RewardWeights(**parent.reward_weights)
            child_rw  = mutate_weights(parent_rw, sigma=self.cfg.reward_mutation_sigma, rng=self.rng)
            seed      = int(self.cfg.seed) + 1000 * generation + ind_id
            print(f"\r  [step] {k}/{len(sampled_parents) - 1} gen={generation} id={ind_id} parent={parent.individual_id} "
                  f"warm-start {self.cfg.n_warm_steps:,} steps ...", end='', flush=True)
            t0 = time.perf_counter()
            policy_path, fitness = self._train_warm_start(child_rw, parent.policy_path, seed, ind_id)
            video_path  = self._render(policy_path, child_rw, ind_id, generation, seed)
            specs.append(_IndividualSpec(
                reward_weights = child_rw.to_dict(),
                policy_path    = policy_path,
                video_path     = video_path,
                parent_id      = parent.individual_id,
                n_train_steps  = self.cfg.n_warm_steps,
            ))
            print(f"\r [step] {k}/{len(sampled_parents) - 1} trained and render in {time.perf_counter() - t0:.1f}s (fitness {fitness:.3f})")

        print(f"\r  [step] Training and Rendering succeeded. evaluating..",  end='')

        # 4. Single batched grading call for all λ children
        children_results, new_id = evaluate_batch(
            specs, grader, generation=generation, id_counter=id_counter,
            reference_video=reference_video, debug=False,
        )

        # 5. Pool = current μ parents (re-tagged for this generation so the
        #    history stats include them) + λ children
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
        print(f"\r  [step] Step done.")
        return parent_pool + children_results, new_id


# ---------------------------------------------------------------------------
# MapEliteEvolution
# ---------------------------------------------------------------------------

class MapEliteEvolution(BaseEvolution):
    """
    MAP-Elites variant. Identical per-child plumbing to MuLambdaEvolution,
    but parents come from filled grid cells and the archive only ever sees
    the offspring (no re-evaluation of incumbents).
    """

    def initialise(self, grader, id_counter: int = 0):
        size = self.cfg.init_population_size or max(self.cfg.mu, self.cfg.lambda_) * 2
        defaults = self.cfg.default_reward_weights_dict()

        specs: list[_IndividualSpec] = []
        ids = list(range(id_counter, id_counter + size))
        for ind_id in ids:
            rw   = random_initial_weights(defaults, sigma=self.cfg.reward_init_sigma, rng=self.rng)
            seed = int(self.cfg.seed) + ind_id
            policy_path, fitness = self._train_from_scratch(rw, seed, ind_id)
            video_path  = self._render(policy_path, rw, ind_id, generation=0, seed=seed)
            specs.append(_IndividualSpec(
                reward_weights = rw.to_dict(),
                policy_path    = policy_path,
                video_path     = video_path,
                parent_id      = None,
                n_train_steps  = self.cfg.n_init_steps,
            ))

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
            ind_id = ids[k]
            parent_rw = RewardWeights(**parent.reward_weights)
            child_rw  = mutate_weights(parent_rw, sigma=self.cfg.reward_mutation_sigma, rng=self.rng)
            seed      = int(self.cfg.seed) + 1000 * generation + ind_id
            policy_path, _ = self._train_warm_start(child_rw, parent.policy_path, seed, ind_id)
            video_path  = self._render(policy_path, child_rw, ind_id, generation, seed)
            specs.append(_IndividualSpec(
                reward_weights = child_rw.to_dict(),
                policy_path    = policy_path,
                video_path     = video_path,
                parent_id      = parent.individual_id,
                n_train_steps  = self.cfg.n_warm_steps,
            ))

        return evaluate_batch(
            specs, grader, generation=generation, id_counter=id_counter,
            reference_video=reference_video, debug=False,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_evolution(cfg: ExperimentConfig, run_dir: Path,
                   rng: Optional[np.random.Generator] = None) -> BaseEvolution:
    if cfg.strategy == "mu_lambda":
        return MuLambdaEvolution(cfg, run_dir, rng)
    if cfg.strategy == "map_elite":
        return MapEliteEvolution(cfg, run_dir, rng)
    raise ValueError(f"Unknown strategy '{cfg.strategy}'. "
                     f"Expected 'mu_lambda' or 'map_elite'.")


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  evolution.py — debug mode (no Gemini, real PPO)")
    print("=" * 60)

    class _FakeOut:
        def __init__(self, fit):
            self.fitness = fit
            self.raw_scores = {"coherence": fit, "progress": fit, "interest": fit}
            self.method = "fake"
            self.prompt_set = "fake"
            self.extra = {"vlm_descriptors": {}}

    class _FakeGrader:
        def __init__(self, rng):
            self.rng = rng
        def score_batch(self, videos, debug=False, reference_video=None):
            return {vid: _FakeOut(float(self.rng.uniform(0.0, 1.0))) for vid, _ in videos}

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_debug"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = ExperimentConfig(
            run_id           = "evo_debug",
            strategy         = "mu_lambda",
            mu               = 2,
            lambda_          = 2,
            n_generations    = 1,
            n_init_steps     = 2_000,
            n_warm_steps     = 1_000,
            n_envs           = 1,
            n_steps_per_env  = 256,
            batch_size       = 64,
            episode_duration = 1.5,
            output_dir       = tmp,
        )
        rng = np.random.default_rng(0)
        evo = MuLambdaEvolution(cfg, run_dir, rng)
        grader = _FakeGrader(np.random.default_rng(1))

        print("\n[1] initialise (train + render + grade)")
        t0 = time.perf_counter()
        init_results, id_counter = evo.initialise(grader, id_counter=0)
        print(f"   {len(init_results)} init results in {time.perf_counter()-t0:.1f}s")
        for r in init_results:
            print(f"     {r}")

        from archive import MuLambdaArchive
        archive = MuLambdaArchive(mu=cfg.mu)
        archive.update(init_results)

        print("\n[2] step (gen 1)")
        t0 = time.perf_counter()
        step_results, id_counter = evo.step(archive, grader, generation=1, id_counter=id_counter)
        print(f"   {len(step_results)} step results in {time.perf_counter()-t0:.1f}s")
        for r in step_results:
            print(f"     {r}")
        archive.update(step_results)
        archive.summary()

    print("\nAll evolution.py checks passed.")
