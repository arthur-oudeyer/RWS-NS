"""
ppo_trainer.py
==============
Inner-loop PPO training driver. Two entry points:

  train_from_scratch(reward_weights, ...)
      Random-init PPO. Used **only** for the initial population (no parent
      to inherit from). Costs `n_init_steps` timesteps.

  train_warm_start(reward_weights, parent_policy_path, ...)
      Loads the parent's saved PPO policy and continues training against
      the *child's* (mutated) reward for `n_warm_steps` extra timesteps.
      The child inherits the parent's locomotion competence and only has
      to adapt to the new reward shape — the compute-saving trick of v1.

Both functions return a `stable_baselines3.PPO` instance (already saved
to disk if `save_path` is given) and log to TensorBoard at
`<tensorboard_log>/individual_<id>` so each individual's training curve
is inspectable post-hoc.

We use `SubprocVecEnv` with `start_method="spawn"` on macOS — MuJoCo's
offscreen renderer is fork-unsafe.

Debug
-----
Run this file directly to do a tiny ~5 k-step from-scratch training run
on the default `RewardWeights` and confirm the model can be loaded back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from reward import RewardWeights


# ---------------------------------------------------------------------------
# Vec-env factory (top-level so it pickles for SubprocVecEnv on macOS)
# ---------------------------------------------------------------------------

def _make_env_fn(
    rw_dict:           dict,
    seed:              int,
    episode_duration:  float,
    control_frequency: int,
    fall_height:       float,
):
    """
    Return a thunk that constructs a fresh `RobotControllerEnv` with the
    given reward weights and seed.

    The thunk imports inside the closure so SubprocVecEnv child processes
    do not inherit a half-initialised mujoco context from the parent.
    """
    def _thunk():
        from mujoco_env import RobotControllerEnv
        from reward     import RewardWeights as _RW
        from stable_baselines3.common.monitor import Monitor
        return Monitor(RobotControllerEnv(
            reward_weights    = _RW(**rw_dict),
            seed              = seed,
            episode_duration  = episode_duration,
            control_frequency = control_frequency,
            fall_height       = fall_height,
        ))
    return _thunk


def _build_vec_env(
    reward_weights:    RewardWeights,
    n_envs:            int,
    seed:              int,
    episode_duration:  float,
    control_frequency: int,
    fall_height:       float,
    use_subproc:       bool = True,
):
    """Build a SubprocVecEnv (default) or DummyVecEnv (for debug)."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    rw_dict = reward_weights.to_dict()
    env_fns = [
        _make_env_fn(rw_dict, seed + i, episode_duration, control_frequency, fall_height)
        for i in range(n_envs)
    ]
    if not use_subproc or n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_from_scratch(
    reward_weights:    RewardWeights,
    seed:              int,
    total_timesteps:   int   = 200_000,
    n_envs:            int   = 4,
    save_path:         Optional[str] = None,
    tensorboard_log:   Optional[str] = None,
    tb_run_name:       Optional[str] = None,
    episode_duration:  float = 5.0,
    control_frequency: int   = 20,
    fall_height:       float = 0.05,
    policy_arch:       Optional[list] = None,
    learning_rate:     float = 3e-4,
    gamma:             float = 0.99,
    gae_lambda:        float = 0.95,
    ent_coef:          float = 0.0,
    vf_coef:           float = 0.5,
    n_steps_per_env:   int   = 2048,
    batch_size:        int   = 256,
    use_subproc:       bool  = True,
    verbose:           int   = 0,
):
    """
    Train one PPO policy from random initial weights against
    `RobotControllerEnv(reward_weights=reward_weights)`.

    Returns the trained `PPO` instance. Saves to `save_path` if given.
    """
    from stable_baselines3 import PPO

    vec = _build_vec_env(
        reward_weights, n_envs, seed,
        episode_duration, control_frequency, fall_height,
        use_subproc=use_subproc,
    )
    try:
        policy_kwargs = dict(net_arch=list(policy_arch or [256, 256]))
        model = PPO(
            "MlpPolicy",
            vec,
            seed             = seed,
            learning_rate    = learning_rate,
            gamma            = gamma,
            gae_lambda       = gae_lambda,
            ent_coef         = ent_coef,
            vf_coef          = vf_coef,
            n_steps          = n_steps_per_env,
            batch_size       = batch_size,
            policy_kwargs    = policy_kwargs,
            tensorboard_log  = tensorboard_log,
            verbose          = verbose,
        )
        model.learn(
            total_timesteps  = int(total_timesteps),
            tb_log_name      = tb_run_name or "from_scratch",
            reset_num_timesteps = True,
            progress_bar     = False,
        )

        # --- Get end fitness from ep_info_buffer ---
        if len(model.ep_info_buffer) > 10:
            # Get the average reward of the last 10 episodes
            last_episodes = list(model.ep_info_buffer)[-10:]
            end_fitness = np.mean([ep_info["r"] for ep_info in last_episodes])
        else:
            # Fallback: If no episodes were logged, use the last reward from the monitor
            end_fitness = model.ep_info_buffer[-1]["r"] if len(model.ep_info_buffer) > 0 else 0.0

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)

        return model, end_fitness
    finally:
        vec.close()

def train_warm_start(
    reward_weights:     RewardWeights,
    parent_policy_path: str,
    seed:               int,
    n_warm_steps:       int   = 50_000,
    n_envs:             int   = 4,
    save_path_warmed:          Optional[str] = None,
    tensorboard_log:    Optional[str] = None,
    tb_run_name:        Optional[str] = None,
    episode_duration:   float = 5.0,
    control_frequency:  int   = 20,
    fall_height:        float = 0.05,
    use_subproc:        bool  = True,
    verbose:            int   = 0,
):
    """
    Continue training the parent's PPO policy against the child's mutated
    reward.

    Mechanism: `PPO.load(parent_policy_path, env=...)` swaps in the child's
    env (which has the new reward_weights) while keeping the parent's
    policy + value-function weights. `model.learn(n_warm_steps)` then runs
    PPO updates with the new reward — the policy starts where the parent
    left off, drifting toward whatever optimum the new reward shape implies.

    Returns the trained `PPO` instance. Saves to `save_path` if given.
    """
    from stable_baselines3 import PPO

    vec = _build_vec_env(
        reward_weights, n_envs, seed,
        episode_duration, control_frequency, fall_height,
        use_subproc=use_subproc,
    )
    try:
        model = PPO.load(
            parent_policy_path,
            env             = vec,
            tensorboard_log = tensorboard_log,
            verbose         = verbose,
        )
        model.set_random_seed(seed)
        model.learn(
            total_timesteps  = int(n_warm_steps),
            tb_log_name      = tb_run_name or "warm_start",
            reset_num_timesteps = False,
            progress_bar     = False,
        )

        # --- Get end fitness from ep_info_buffer ---
        if len(model.ep_info_buffer) > 10:
            # Get the average reward of the last 10 episodes
            last_episodes = list(model.ep_info_buffer)[-10:]
            end_fitness = np.mean([ep_info["r"] for ep_info in last_episodes])
        else:
            # Fallback: If no episodes were logged, use the last reward from the monitor
            end_fitness = model.ep_info_buffer[-1]["r"] if len(model.ep_info_buffer) > 0 else 0.0

        if save_path_warmed:
            Path(save_path_warmed).parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path_warmed)

        return model, end_fitness
    finally:
        vec.close()


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Tiny smoke test — trains for a few thousand steps on a single env
    (DummyVecEnv) so the full SubprocVecEnv subprocess machinery isn't
    invoked. Verifies `learn() → save() → load() → predict()` works.
    """
    import tempfile

    print("=" * 60)
    print("  ppo_trainer.py — debug mode")
    print("=" * 60)

    rw = RewardWeights()
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "smoke_policy.zip")
        tb_dir    = os.path.join(tmp, "tb")

        print("\n[1] train_from_scratch (5k steps, 1 env)\n")
        model, fitness = train_from_scratch(
            reward_weights   = rw,
            seed             = 0,
            total_timesteps  = 5_000,
            n_envs           = 1,
            save_path        = save_path,
            tensorboard_log  = tb_dir,
            tb_run_name      = "smoke",
            episode_duration = 2.0,
            n_steps_per_env  = 256,
            batch_size       = 64,
            use_subproc      = False,
            verbose          = 0,
        )
        print(f"  trained ({fitness}).   saved → {save_path}  size={os.path.getsize(save_path)} B")

        # Reload + single-step predict
        from stable_baselines3 import PPO
        loaded = PPO.load(save_path)
        from mujoco_env import RobotControllerEnv
        env  = RobotControllerEnv(reward_weights=rw, seed=1, episode_duration=2.0)
        obs, _ = env.reset(seed=1)
        action, _ = loaded.predict(obs, deterministic=True)
        print(f"  predict action shape={action.shape}  finite={np.isfinite(action).all()}")
        env.close()

        # Tiny warm-start
        print("\n[2] train_warm_start (2k steps, 1 env, mutated weights)\n")
        from reward import mutate_weights
        rw_child = mutate_weights(rw, sigma=0.2, rng=np.random.default_rng(0))
        save_path_warm = os.path.join(tmp, "smoke_warm.zip")
        train_warm_start(
            reward_weights     = rw_child,
            parent_policy_path = save_path,
            seed               = 1,
            n_warm_steps       = 2_000,
            n_envs             = 1,
            save_path          = save_path_warm,
            tensorboard_log    = tb_dir,
            tb_run_name        = "smoke_warm",
            episode_duration   = 2.0,
            use_subproc        = False,
            verbose            = 0,
        )
        print(f"  warm-start saved → {save_path_warm}  size={os.path.getsize(save_path_warm)} B")

    print("\nAll ppo_trainer.py checks passed.")
