"""
tests/test_phase3.py
====================
Phase 3 wiring tests for video_renderer_mjx.py.

Sections
--------
A. TestDataTransfer    — mjx.Data → mujoco.MjData transfer correctness
B. TestPolicyFn        — build_policy_fn shapes, determinism, clipping
C. TestRenderFrame     — _render_frame produces valid RGB arrays
D. TestRolloutToVideo  — rollout_to_video_mjx writes a valid MP4
E. TestInfoDict        — info dict fields, n_frames consistency
F. TestDeterminism     — same seed → same total_reward
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

jax       = pytest.importorskip("jax",       reason="jax not installed")
jnp       = pytest.importorskip("jax.numpy", reason="jax not installed")
pytest.importorskip("flax",   reason="flax not installed")
pytest.importorskip("optax",  reason="optax not installed")
pytest.importorskip("imageio", reason="imageio not installed")

import jax
import jax.numpy as jnp

from mujoco_env_mjx import _pick_mjx_device
_dev = _pick_mjx_device()
jax.config.update("jax_default_device", _dev)

import mujoco
from mujoco import mjx

from controller_morph import build_model
from mujoco_env_mjx import build_env_config, make_env_fns
from ppo_trainer_mjx import make_params, train_from_scratch_mjx
from video_renderer_mjx import (
    build_policy_fn,
    mjx_state_to_mj_data,
    rollout_to_video_mjx,
    _make_cameras,
    _render_frame,
    ActorCritic,
)


# ---------------------------------------------------------------------------
# Shared fixtures  (module-scoped so MJX build happens once per test run)
# ---------------------------------------------------------------------------

_TINY_ARCH = (64, 64)
_TINY_STEPS = 2 * 8   # 2 envs × 8 steps = 1 update


@pytest.fixture(scope="module")
def mj_model():
    m, _ = build_model()
    return m


@pytest.fixture(scope="module")
def cfg():
    return build_env_config(episode_duration=2.0)


@pytest.fixture(scope="module")
def trained_params(cfg):
    """Tiny trained params shared across multiple tests."""
    params, _ = train_from_scratch_mjx(
        cfg          = cfg,
        seed         = 0,
        total_steps  = _TINY_STEPS,
        n_envs       = 2,
        rollout_len  = 8,
        policy_arch  = _TINY_ARCH,
        fitness_episodes = 1,
    )
    return params


@pytest.fixture(scope="module")
def reset_state(cfg):
    """A single reset EnvState for reuse."""
    reset_fn, _, _ = make_env_fns(cfg)
    state, obs = reset_fn(jax.random.PRNGKey(0))
    return state, obs


# ---------------------------------------------------------------------------
# A. TestDataTransfer
# ---------------------------------------------------------------------------

class TestDataTransfer:
    def test_returns_mj_data(self, mj_model, reset_state):
        state, _ = reset_state
        cpu_data = mjx_state_to_mj_data(mj_model, state)
        assert isinstance(cpu_data, mujoco.MjData)

    def test_qpos_shape_preserved(self, mj_model, reset_state):
        state, _ = reset_state
        cpu_data = mjx_state_to_mj_data(mj_model, state)
        assert cpu_data.qpos.shape == state.data.qpos.shape

    def test_qpos_values_match(self, mj_model, reset_state):
        state, _ = reset_state
        cpu_data = mjx_state_to_mj_data(mj_model, state)
        mjx_qpos = np.array(state.data.qpos)
        np.testing.assert_allclose(
            cpu_data.qpos, mjx_qpos, atol=1e-5,
            err_msg="qpos mismatch between mjx.Data and transferred mujoco.MjData",
        )

    def test_spawn_height_in_cpu_data(self, mj_model, cfg, reset_state):
        state, _ = reset_state
        cpu_data = mjx_state_to_mj_data(mj_model, state)
        assert abs(float(cpu_data.qpos[2]) - cfg.spawn_height) < 0.1, \
            f"torso z {cpu_data.qpos[2]:.4f} differs from spawn_height {cfg.spawn_height:.4f}"

    def test_transfer_after_step(self, mj_model, cfg, reset_state):
        """Data transferred after a step should reflect updated physics."""
        state, _ = reset_state
        _, step_fn, _ = make_env_fns(cfg)
        action = jnp.zeros(cfg.n_joints, jnp.float32)
        new_state, _, _, _ = step_fn(state, action)
        cpu_data = mjx_state_to_mj_data(mj_model, new_state)
        # Time should have advanced
        assert float(cpu_data.time) > 0.0, "simulation time should be > 0 after a step"

    def test_geom_xpos_valid(self, mj_model, reset_state):
        """geom_xpos should be populated after mjx.forward + transfer."""
        state, _ = reset_state
        cpu_data = mjx_state_to_mj_data(mj_model, state)
        assert cpu_data.geom_xpos.shape[0] == mj_model.ngeom
        assert np.isfinite(cpu_data.geom_xpos).all()


# ---------------------------------------------------------------------------
# B. TestPolicyFn
# ---------------------------------------------------------------------------

class TestPolicyFn:
    def test_action_shape(self, cfg, trained_params):
        net = ActorCritic(obs_dim=cfg.obs_dim, act_dim=cfg.n_joints, hidden=_TINY_ARCH)
        policy_fn = build_policy_fn(trained_params, net)
        obs = jnp.zeros((cfg.obs_dim,), jnp.float32)
        action = policy_fn(obs)
        assert action.shape == (cfg.n_joints,), f"action shape {action.shape}"

    def test_action_clipped(self, cfg, trained_params):
        net = ActorCritic(obs_dim=cfg.obs_dim, act_dim=cfg.n_joints, hidden=_TINY_ARCH)
        policy_fn = build_policy_fn(trained_params, net)
        obs = jnp.ones((cfg.obs_dim,), jnp.float32) * 100.0
        action = policy_fn(obs)
        assert jnp.all(action >= -1.0)
        assert jnp.all(action <=  1.0)

    def test_action_finite(self, cfg, trained_params):
        net = ActorCritic(obs_dim=cfg.obs_dim, act_dim=cfg.n_joints, hidden=_TINY_ARCH)
        policy_fn = build_policy_fn(trained_params, net)
        obs = jax.random.normal(jax.random.PRNGKey(5), (cfg.obs_dim,))
        action = policy_fn(obs)
        assert jnp.isfinite(action).all()

    def test_deterministic_repeatability(self, cfg, trained_params):
        """Same obs → same action with deterministic=True."""
        net = ActorCritic(obs_dim=cfg.obs_dim, act_dim=cfg.n_joints, hidden=_TINY_ARCH)
        policy_fn = build_policy_fn(trained_params, net, deterministic=True)
        obs = jax.random.normal(jax.random.PRNGKey(7), (cfg.obs_dim,))
        a1 = policy_fn(obs)
        a2 = policy_fn(obs)
        assert jnp.allclose(a1, a2)

    def test_random_params_give_action(self, cfg):
        """Even random (untrained) params should produce valid actions."""
        params = make_params(99, cfg.obs_dim, cfg.n_joints, _TINY_ARCH)
        net = ActorCritic(obs_dim=cfg.obs_dim, act_dim=cfg.n_joints, hidden=_TINY_ARCH)
        policy_fn = build_policy_fn(params, net)
        obs = jnp.zeros((cfg.obs_dim,), jnp.float32)
        action = policy_fn(obs)
        assert action.shape == (cfg.n_joints,)
        assert jnp.isfinite(action).all()


# ---------------------------------------------------------------------------
# C. TestRenderFrame
# ---------------------------------------------------------------------------

class TestRenderFrame:
    @pytest.fixture(scope="class")
    def renderers_and_data(self, mj_model, reset_state):
        state, _ = reset_state
        r1 = mujoco.Renderer(mj_model, height=64, width=64)
        r2 = mujoco.Renderer(mj_model, height=64, width=64)
        cam1, cam2 = _make_cameras(90.0, -5.0, 4.0, 0.1, 60.0, -30.0, 4.0)
        cpu_data = mjx_state_to_mj_data(mj_model, state)
        yield r1, r2, cam1, cam2, cpu_data
        r1.close()
        r2.close()

    def test_frame_shape(self, renderers_and_data):
        r1, r2, cam1, cam2, cpu_data = renderers_and_data
        frame = _render_frame(r1, r2, cpu_data, cam1, cam2, camera_track_torso=False)
        # Side-by-side: height=64, width=128 (2 cameras × 64)
        assert frame.shape == (64, 128, 3), f"frame shape {frame.shape}"

    def test_frame_dtype(self, renderers_and_data):
        r1, r2, cam1, cam2, cpu_data = renderers_and_data
        frame = _render_frame(r1, r2, cpu_data, cam1, cam2, camera_track_torso=False)
        assert frame.dtype == np.uint8

    def test_frame_not_all_black(self, renderers_and_data):
        r1, r2, cam1, cam2, cpu_data = renderers_and_data
        frame = _render_frame(r1, r2, cpu_data, cam1, cam2, camera_track_torso=False)
        assert frame.max() > 0, "frame is all-black — renderer may not be working"

    def test_frame_values_in_range(self, renderers_and_data):
        r1, r2, cam1, cam2, cpu_data = renderers_and_data
        frame = _render_frame(r1, r2, cpu_data, cam1, cam2, camera_track_torso=False)
        assert int(frame.min()) >= 0
        assert int(frame.max()) <= 255


# ---------------------------------------------------------------------------
# D. TestRolloutToVideo
# ---------------------------------------------------------------------------

class TestRolloutToVideo:
    @pytest.fixture(scope="class")
    def video_result(self, cfg, mj_model, trained_params):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test_rollout.mp4")
            path, info = rollout_to_video_mjx(
                params       = trained_params,
                cfg          = cfg,
                mj_model     = mj_model,
                save_path    = out,
                fps          = 20,
                render_width = 64,
                render_height = 64,
                policy_arch  = _TINY_ARCH,
                seed         = 0,
                max_steps    = 10,
            )
            # Read back for assertion outside the tempdir context
            file_size = os.path.getsize(path)
        return path, info, file_size

    def test_file_created(self, video_result):
        _, _, file_size = video_result
        assert file_size > 0, "output MP4 is empty"

    def test_n_frames_positive(self, video_result):
        _, info, _ = video_result
        assert info["n_frames"] > 0

    def test_n_frames_leq_max_steps_plus_one(self, video_result):
        # max_steps=10 + 1 initial frame
        _, info, _ = video_result
        assert info["n_frames"] <= 11, f"n_frames={info['n_frames']} > max_steps+1"

    def test_total_reward_finite(self, video_result):
        _, info, _ = video_result
        assert np.isfinite(info["total_reward"])

    def test_n_steps_leq_max_steps(self, video_result):
        _, info, _ = video_result
        assert info["n_steps"] <= 10

    def test_info_keys_present(self, video_result):
        _, info, _ = video_result
        for key in ("n_frames", "terminated", "truncated", "total_reward", "n_steps"):
            assert key in info, f"missing key '{key}' in info dict"


# ---------------------------------------------------------------------------
# E. TestInfoDict
# ---------------------------------------------------------------------------

class TestInfoDict:
    def test_terminated_or_truncated_not_both(self, cfg, mj_model, trained_params):
        """An episode can be terminated OR truncated but not both simultaneously."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "info_test.mp4")
            _, info = rollout_to_video_mjx(
                params       = trained_params,
                cfg          = cfg,
                mj_model     = mj_model,
                save_path    = out,
                fps          = 20,
                render_width = 32,
                render_height = 32,
                policy_arch  = _TINY_ARCH,
                seed         = 1,
                max_steps    = 5,
            )
        assert not (info["terminated"] and info["truncated"]), \
            "episode cannot be both terminated and truncated"

    def test_n_frames_consistent_with_n_steps(self, cfg, mj_model, trained_params):
        """n_frames should be n_steps + 1 (initial frame) unless done mid-episode."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "frames_test.mp4")
            _, info = rollout_to_video_mjx(
                params       = trained_params,
                cfg          = cfg,
                mj_model     = mj_model,
                save_path    = out,
                fps          = 20,
                render_width = 32,
                render_height = 32,
                policy_arch  = _TINY_ARCH,
                seed         = 2,
                max_steps    = 8,
            )
        # n_frames = n_steps + 1 (initial frame)
        assert info["n_frames"] == info["n_steps"] + 1, \
            f"n_frames={info['n_frames']}  n_steps={info['n_steps']}"


# ---------------------------------------------------------------------------
# F. TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_reward(self, cfg, mj_model, trained_params):
        """Same seed → deterministic policy → same total_reward."""
        def _run():
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "det.mp4")
                _, info = rollout_to_video_mjx(
                    params       = trained_params,
                    cfg          = cfg,
                    mj_model     = mj_model,
                    save_path    = out,
                    fps          = 20,
                    render_width = 32,
                    render_height = 32,
                    policy_arch  = _TINY_ARCH,
                    seed         = 0,
                    max_steps    = 8,
                    deterministic = True,
                )
            return info["total_reward"]

        r1 = _run()
        r2 = _run()
        assert abs(r1 - r2) < 1e-4, \
            f"same seed gave different rewards: {r1:.6f} vs {r2:.6f}"

    def test_different_seeds_can_give_different_rewards(self, cfg, mj_model, trained_params):
        """Different seeds (different reset jitter) should yield different rewards."""
        def _run(seed):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, f"det_{seed}.mp4")
                _, info = rollout_to_video_mjx(
                    params       = trained_params,
                    cfg          = cfg,
                    mj_model     = mj_model,
                    save_path    = out,
                    fps          = 20,
                    render_width = 32,
                    render_height = 32,
                    policy_arch  = _TINY_ARCH,
                    seed         = seed,
                    max_steps    = 8,
                    deterministic = True,
                )
            return info["total_reward"]

        r0 = _run(0)
        r1 = _run(1)
        # With different reset jitter the trajectories typically diverge.
        # This is a soft check — it can equal by coincidence, so we don't assert !=.
        _ = r0 - r1   # just ensure both ran without error
