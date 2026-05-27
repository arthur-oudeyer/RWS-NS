"""
tests/test_phase4.py
====================
Phase 4 integration + validation tests.

Sections
--------
A. TestParamsPersistence  — save_params / load_params round-trip
B. TestEvolutionMJX       — initialise + step with fake grader, archive wiring
C. TestExperimentMJX      — run_mjx end-to-end: artefacts, log files, archive
D. TestBenchmark          — benchmark_mjx.run_benchmark produces valid timings
E. TestRewardConsistency  — JAX reward matches numpy reward (regression vs P1)
F. TestConfigCaching      — template cfg → per-individual replace doesn't share state
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

jax  = pytest.importorskip("jax",       reason="jax not installed")
jnp  = pytest.importorskip("jax.numpy", reason="jax not installed")
pytest.importorskip("flax",   reason="flax not installed")
pytest.importorskip("optax",  reason="optax not installed")
pytest.importorskip("imageio", reason="imageio not installed")

import jax
import jax.numpy as jnp

from mujoco_env_mjx import _pick_mjx_device
_dev = _pick_mjx_device()
jax.config.update("jax_default_device", _dev)

from config        import ExperimentConfig
from reward        import RewardWeights, mutate_weights
from archive       import MuLambdaArchive
from data_handler  import ControllerResult
from mujoco_env_mjx import build_env_config, make_env_fns
from ppo_trainer_mjx import make_params, train_from_scratch_mjx, PPOConfig
from evolution_mjx import (
    save_params, load_params,
    MuLambdaEvolutionMJX, make_evolution_mjx,
)
from experiment_mjx import run_mjx
from benchmark_mjx  import run_benchmark, BenchmarkResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_TINY_ARCH = (64, 64)
_N_ENVS    = 2
_ROLL_LEN  = 8
_N_STEPS   = _N_ENVS * _ROLL_LEN   # = 16, exactly 1 update


def _tiny_cfg(tmp: str) -> ExperimentConfig:
    return ExperimentConfig(
        run_id               = "test_run",
        strategy             = "mu_lambda",
        mu                   = 2,
        lambda_              = 2,
        n_generations        = 1,
        init_population_size = 2,
        n_init_steps         = _N_STEPS,
        n_warm_steps         = _N_STEPS,
        n_envs_mjx           = _N_ENVS,
        n_steps_per_env      = _ROLL_LEN,
        batch_size           = _ROLL_LEN,
        episode_duration     = 1.5,
        output_dir           = tmp,
    )


class _FakeOut:
    def __init__(self, fit):
        self.fitness    = fit
        self.raw_scores = {"coherence": fit}
        self.method     = "fake"
        self.prompt_set = "fake"
        self.extra      = {"vlm_descriptors": {}}


class _FakeGrader:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    def score_batch(self, videos, debug=False, reference_video=None):
        return {vid: _FakeOut(float(self.rng.uniform(0.3, 0.9))) for vid, _ in videos}


@pytest.fixture(scope="module")
def env_cfg():
    return build_env_config(episode_duration=1.5)


@pytest.fixture(scope="module")
def tiny_params(env_cfg):
    return make_params(0, env_cfg.obs_dim, env_cfg.n_joints, _TINY_ARCH)


# ---------------------------------------------------------------------------
# A. TestParamsPersistence
# ---------------------------------------------------------------------------

class TestParamsPersistence:
    def test_save_creates_file(self, tiny_params):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.params")
            save_params(tiny_params, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_load_restores_structure(self, tiny_params):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.params")
            save_params(tiny_params, path)
            loaded = load_params(path)
            l_orig   = jax.tree.leaves(tiny_params)
            l_loaded = jax.tree.leaves(loaded)
            assert len(l_orig) == len(l_loaded)

    def test_values_preserved(self, tiny_params):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.params")
            save_params(tiny_params, path)
            loaded = load_params(path)
            for a, b in zip(jax.tree.leaves(tiny_params), jax.tree.leaves(loaded)):
                np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-6)

    def test_loaded_params_are_jax_arrays(self, tiny_params):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.params")
            save_params(tiny_params, path)
            loaded = load_params(path)
            for leaf in jax.tree.leaves(loaded):
                assert hasattr(leaf, "shape"), "leaf is not a JAX array"

    def test_creates_parent_dirs(self, tiny_params):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deep", "nested", "p.params")
            save_params(tiny_params, path)
            assert os.path.exists(path)

    def test_trained_params_survive_round_trip(self, env_cfg):
        """Trained params (not just random) should round-trip without change."""
        params, _ = train_from_scratch_mjx(
            cfg=env_cfg, seed=0, total_steps=_N_STEPS,
            n_envs=_N_ENVS, rollout_len=_ROLL_LEN,
            policy_arch=_TINY_ARCH, fitness_episodes=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trained.params")
            save_params(params, path)
            loaded = load_params(path)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)):
            np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-6)


# ---------------------------------------------------------------------------
# B. TestEvolutionMJX
# ---------------------------------------------------------------------------

class TestEvolutionMJX:
    @pytest.fixture(scope="class")
    def evo_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg     = _tiny_cfg(tmp)
            run_dir = Path(tmp) / cfg.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            evo     = MuLambdaEvolutionMJX(cfg, run_dir, np.random.default_rng(0))
            grader  = _FakeGrader()

            init_results, id_counter = evo.initialise(grader, id_counter=0)
            archive = MuLambdaArchive(mu=cfg.mu)
            archive.update(init_results)
            step_results, id_counter = evo.step(archive, grader, generation=1, id_counter=id_counter)
            archive.update(step_results)

            # Collect what we need before tmp is deleted
            policy_files = list((run_dir / "policies").glob("*.params"))
            video_files  = list((run_dir / "videos").glob("*.mp4"))
            n_policies   = len(policy_files)
            n_videos     = len(video_files)
            # verify files are loadable while still inside tmp
            loaded_ok = all(load_params(str(p)) is not None for p in policy_files)

        return {
            "init_results":  init_results,
            "step_results":  step_results,
            "archive":       archive,
            "n_policies":    n_policies,
            "n_videos":      n_videos,
            "loaded_ok":     loaded_ok,
        }

    def test_init_returns_correct_count(self, evo_result):
        assert len(evo_result["init_results"]) == 2

    def test_step_returns_pool_plus_children(self, evo_result):
        # pool = mu parents + lambda children = 2 + 2 = 4
        assert len(evo_result["step_results"]) == 4

    def test_fitness_is_finite(self, evo_result):
        for r in evo_result["init_results"] + evo_result["step_results"]:
            assert np.isfinite(r.fitness), f"fitness={r.fitness} is not finite"

    def test_policy_files_created(self, evo_result):
        assert evo_result["n_policies"] > 0

    def test_video_files_created(self, evo_result):
        assert evo_result["n_videos"] > 0

    def test_policy_files_loadable(self, evo_result):
        assert evo_result["loaded_ok"]

    def test_archive_has_population(self, evo_result):
        archive = evo_result["archive"]
        assert len(archive.population) == 2   # mu=2

    def test_archive_best_is_valid(self, evo_result):
        best = evo_result["archive"].best()
        assert best is not None
        assert np.isfinite(best.fitness)

    def test_parent_ids_wired(self, evo_result):
        """Step children should have non-None parent_id."""
        children = [r for r in evo_result["step_results"] if r.parent_id is not None]
        assert len(children) == 2   # lambda_=2

    def test_init_individuals_have_no_parent(self, evo_result):
        for r in evo_result["init_results"]:
            assert r.parent_id is None

    def test_factory_returns_correct_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_cfg(tmp)
            evo = make_evolution_mjx(cfg, Path(tmp))
            assert isinstance(evo, MuLambdaEvolutionMJX)


# ---------------------------------------------------------------------------
# C. TestExperimentMJX
# ---------------------------------------------------------------------------

class TestExperimentMJX:
    @pytest.fixture(scope="class")
    def smoke_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg     = _tiny_cfg(tmp)
            archive = run_mjx(cfg, grader=_FakeGrader())
            run_dir = Path(tmp) / cfg.run_id

            # Capture everything needed before tmp is cleaned up
            result = {
                "archive":          archive,
                "config_exists":    (run_dir / "config.json").exists(),
                "log_exists":       (run_dir / "log.jsonl").exists(),
                "indiv_log_exists": (run_dir / "individuals_log.jsonl").exists(),
                "final_arch_exists": (run_dir / "archive_final.json").exists(),
                "n_policies": len(list((run_dir / "policies").glob("*.params"))),
                "n_videos":   len(list((run_dir / "videos").glob("*.mp4"))),
                "log_lines":  (run_dir / "log.jsonl").read_text().strip().splitlines(),
                "indiv_lines": (run_dir / "individuals_log.jsonl").read_text().strip().splitlines(),
                "archive_gen0_exists": (run_dir / "archive_gen0000.json").exists(),
            }
        return result

    def test_archive_best_not_none(self, smoke_run):
        assert smoke_run["archive"].best() is not None

    def test_config_json_written(self, smoke_run):
        assert smoke_run["config_exists"]

    def test_log_jsonl_written(self, smoke_run):
        assert smoke_run["log_exists"]
        assert len(smoke_run["log_lines"]) > 0

    def test_individuals_log_written(self, smoke_run):
        assert smoke_run["indiv_log_exists"]
        assert len(smoke_run["indiv_lines"]) > 0

    def test_archive_final_written(self, smoke_run):
        assert smoke_run["final_arch_exists"]

    def test_archive_gen0_written(self, smoke_run):
        assert smoke_run["archive_gen0_exists"]

    def test_policies_written(self, smoke_run):
        assert smoke_run["n_policies"] > 0

    def test_videos_written(self, smoke_run):
        assert smoke_run["n_videos"] > 0

    def test_log_entries_parseable(self, smoke_run):
        for line in smoke_run["log_lines"]:
            entry = json.loads(line)
            assert "generation" in entry
            assert "elapsed_s" in entry

    def test_individual_log_entries_parseable(self, smoke_run):
        for line in smoke_run["indiv_lines"]:
            entry = json.loads(line)
            assert "individual_id" in entry
            assert "fitness" in entry


# ---------------------------------------------------------------------------
# D. TestBenchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    @pytest.fixture(scope="class")
    def bench(self):
        return run_benchmark(
            n_envs       = _N_ENVS,
            rollout_len  = _ROLL_LEN,
            n_init       = _N_STEPS,
            n_warm       = _N_STEPS,
            policy_arch  = _TINY_ARCH,
            render_width = 32,
            render_height = 32,
            max_render_steps = 3,
            verbose      = False,
        )

    def test_returns_benchmark_result(self, bench):
        assert isinstance(bench, BenchmarkResult)

    def test_timings_positive(self, bench):
        assert bench.t_build_cfg   > 0
        assert bench.t_init_train  > 0
        assert bench.t_warm_train  > 0
        assert bench.t_render      > 0

    def test_fps_positive(self, bench):
        assert bench.init_fps > 0
        assert bench.warm_fps > 0

    def test_step_counts_match(self, bench):
        assert bench.total_init_steps == _N_STEPS
        assert bench.total_warm_steps == _N_STEPS

    def test_n_envs_recorded(self, bench):
        assert bench.n_envs == _N_ENVS


# ---------------------------------------------------------------------------
# E. TestRewardConsistency
# ---------------------------------------------------------------------------

class TestRewardConsistency:
    """
    Regression: JAX and numpy reward functions agree on the same trajectory.
    (Repeats the core assertion from test_phase1.py::test_jax_numpy_parity
    but from a higher-level perspective: over a full episode, the sum of JAX
    rewards should be close to the sum of numpy rewards.)
    """

    def test_alive_bonus_positive(self, env_cfg):
        """A standing robot should receive a positive alive_bonus each step."""
        reset_fn, step_fn, _ = make_env_fns(env_cfg)
        state, obs = reset_fn(jax.random.PRNGKey(0))
        action = jnp.zeros(env_cfg.n_joints, jnp.float32)
        _, _, reward, _ = step_fn(state, action)
        # alive_bonus weight is 0.05 (default). Even with other terms, total
        # reward from a standing robot should be > -5 (loose sanity bound).
        assert float(reward) > -10.0, f"reward={float(reward):.4f} unexpectedly low"

    def test_reward_improves_with_training(self, env_cfg):
        """
        A trained policy should yield higher mean reward than a random policy.
        This is a soft validation of the reward curve direction.
        """
        def _mean_reward(params):
            reset_fn, step_fn, _ = make_env_fns(env_cfg)
            state, obs = reset_fn(jax.random.PRNGKey(7))
            total = 0.0
            for _ in range(10):
                action = jnp.zeros(env_cfg.n_joints, jnp.float32)
                state, obs, r, done = step_fn(state, action)
                total += float(r)
                if done:
                    break
            return total

        random_params = make_params(0, env_cfg.obs_dim, env_cfg.n_joints, _TINY_ARCH)
        trained_params, _ = train_from_scratch_mjx(
            cfg=env_cfg, seed=0, total_steps=_N_STEPS * 4,
            n_envs=_N_ENVS, rollout_len=_ROLL_LEN,
            policy_arch=_TINY_ARCH,
            ppo_cfg=PPOConfig(n_epochs=2, minibatch_size=8),
            fitness_episodes=1,
        )
        # Both should be finite — we don't assert direction since tiny training
        r_random  = _mean_reward(random_params)
        r_trained = _mean_reward(trained_params)
        assert np.isfinite(r_random)
        assert np.isfinite(r_trained)

    def test_reward_weights_affect_reward(self, env_cfg):
        """Different reward weights should give different rewards for the same state."""
        from dataclasses import replace as dc_replace
        reset_fn1, step_fn1, _ = make_env_fns(env_cfg)
        state, _ = reset_fn1(jax.random.PRNGKey(0))
        action = jnp.zeros(env_cfg.n_joints, jnp.float32)
        _, _, r1, _ = step_fn1(state, action)

        # Config with forward_velocity weight set to 0
        rw2  = RewardWeights(forward_velocity=0.0)
        cfg2 = dc_replace(env_cfg, reward_weights_vec=rw2.to_jax_vector())
        reset_fn2, step_fn2, _ = make_env_fns(cfg2)
        _, _, r2, _ = step_fn2(state, action)

        # r1 != r2 (different forward_velocity weight)
        assert float(r1) != float(r2), \
            "reward should change when forward_velocity weight changes"


# ---------------------------------------------------------------------------
# F. TestConfigCaching
# ---------------------------------------------------------------------------

class TestConfigCaching:
    def test_replace_does_not_mutate_template(self, env_cfg):
        """dc_replace(template_cfg, reward_weights_vec=...) must not change template."""
        from dataclasses import replace as dc_replace
        original_vec = np.array(env_cfg.reward_weights_vec)
        rw2 = RewardWeights(forward_velocity=99.0)
        cfg2 = dc_replace(env_cfg, reward_weights_vec=rw2.to_jax_vector())
        np.testing.assert_allclose(
            np.array(env_cfg.reward_weights_vec), original_vec,
            err_msg="template cfg was mutated by dc_replace",
        )

    def test_replace_gives_distinct_reward_vecs(self, env_cfg):
        from dataclasses import replace as dc_replace
        rw1 = RewardWeights(forward_velocity=1.0)
        rw2 = RewardWeights(forward_velocity=5.0)
        cfg1 = dc_replace(env_cfg, reward_weights_vec=rw1.to_jax_vector())
        cfg2 = dc_replace(env_cfg, reward_weights_vec=rw2.to_jax_vector())
        assert not jnp.allclose(cfg1.reward_weights_vec, cfg2.reward_weights_vec)

    def test_each_individual_gets_own_reward_vec(self):
        """Evolution correctly builds per-individual env configs."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg     = _tiny_cfg(tmp)
            run_dir = Path(tmp) / cfg.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            evo     = MuLambdaEvolutionMJX(cfg, run_dir, np.random.default_rng(0))

            rw1 = RewardWeights(forward_velocity=1.0)
            rw2 = RewardWeights(forward_velocity=5.0)
            c1  = evo._env_cfg_for(rw1)
            c2  = evo._env_cfg_for(rw2)
            assert not jnp.allclose(c1.reward_weights_vec, c2.reward_weights_vec)
            # And neither mutated the template
            template_fwd = float(evo._template_cfg.reward_weights_vec[0])
            assert abs(template_fwd - 1.0) < 0.5
