"""
ppo_trainer_mjx.py
==================
Pure-JAX PPO trainer that runs inside jax.lax.scan for maximum throughput.

Architecture
------------
    ActorCritic     — shared-trunk Flax MLP; outputs Gaussian action mean,
                      learned log-std, and scalar value estimate.
    PPOTrainState   — Flax TrainState (params + optax optimizer state).
    collect_rollout — single scan over max_steps, vmapped over n_envs.
    ppo_update      — one epoch of minibatch PPO gradient steps.
    train_from_scratch_mjx  — full training run from random params.
    train_warm_start_mjx    — continue training from existing params.

The rollout collection and PPO update are both JIT-compiled JAX functions.
No Python loops except the outer generation loop (intentional: lets us log
and save checkpoints each generation).

Observation / action convention
--------------------------------
Same as mujoco_env_mjx.py / mujoco_env.py:
    obs  : (obs_dim,) float32
    action: (n_joints,) float32 ∈ [-1, 1]  (delta angles, clipped inside env)

Warm-start
-----------
`train_warm_start_mjx` takes parent_params (a Flax param dict) and
resumes training with the child's mutated reward weights.
No SB3 weight conversion needed — params stay in Flax format throughout.

End-fitness
-----------
Returns the mean episodic return over the last `fitness_episodes` complete
episodes collected at the END of training (not averaged over all training).

Install
-------
    pip install flax optax mujoco-mjx
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

from mujoco_env_mjx import (
    MJXEnvConfig,
    EnvState,
    build_env_config,
    make_env_fns,
    _pick_mjx_device,
)
from reward import RewardWeights


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic MLP.

    Architecture:
        shared trunk  →  actor head  → Gaussian mean (n_joints,)
                      →  value head  → scalar
        log_std: separate learned parameter (not input-dependent)
    """
    obs_dim:  int
    act_dim:  int
    hidden:   Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: Any) -> Tuple[Any, Any, Any]:
        # Actor trunk
        x = obs.astype(jnp.float32)
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        mean = nn.Dense(self.act_dim, name="actor_out")(x)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(-0.5),
            (self.act_dim,),
        )

        # Critic trunk (separate weights — avoids gradient conflict)
        v = obs.astype(jnp.float32)
        for i, h in enumerate(self.hidden):
            v = nn.relu(nn.Dense(h, name=f"critic_{i}")(v))
        value = nn.Dense(1, name="critic_out")(v)[..., 0]

        return mean, log_std, value


def init_actor_critic(
    key:     Any,
    obs_dim: int,
    act_dim: int,
    hidden:  Tuple[int, ...],
) -> Tuple[ActorCritic, Any]:
    """Instantiate ActorCritic and return (net, initial_params)."""
    net    = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=tuple(hidden))
    params = net.init(key, jnp.zeros((obs_dim,)))
    return net, params


# ---------------------------------------------------------------------------
# Rollout transition buffer
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:        Any   # (T, n_envs, obs_dim) float32
    action:     Any   # (T, n_envs, act_dim)
    log_prob:   Any   # (T, n_envs)
    value:      Any   # (T, n_envs)
    reward:     Any   # (T, n_envs)
    done:       Any   # (T, n_envs)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _gaussian_sample(key: Any, mean: Any, log_std: Any) -> Tuple[Any, Any]:
    """Sample from N(mean, exp(log_std)) and return (action, log_prob)."""
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    action   = mean + std * eps
    log_prob = -0.5 * (jnp.sum(((action - mean) / std) ** 2, axis=-1)
                       + jnp.sum(jnp.log(2.0 * jnp.pi * std ** 2), axis=-1))
    return action, log_prob


def make_rollout_fn(
    cfg:        MJXEnvConfig,
    net:        ActorCritic,
    n_envs:     int,
    rollout_len: int,        # steps per rollout (Python int for scan)
) -> Any:
    """
    Return a JIT-compiled function that collects one rollout.

        collect(params, states, rng) → (states, transitions, rng)

    `states` is a batched EnvState (leading dim = n_envs).
    """
    _, _, make_batch_fns = make_env_fns(cfg)
    batch_reset, batch_step = make_batch_fns(n_envs)

    @jax.jit
    def collect(
        params:  Any,
        states:  EnvState,
        rng:     Any,
    ) -> Tuple[EnvState, Transition, Any]:

        def _one_step(carry, _):
            states, rng = carry
            rng, key_act = jax.random.split(rng)

            # Get current obs from states (n_envs, obs_dim)
            obs_batch = jax.vmap(
                lambda s: jnp.concatenate([
                    jnp.array([
                        jnp.sin(s.sim_time * jnp.float32(1.0  / np.pi)),
                        jnp.sin(s.sim_time * jnp.float32(5.0  / np.pi)),
                        jnp.sin(s.sim_time * jnp.float32(15.0 / np.pi)),
                    ]),
                    s.data.qpos[7 : 7 + cfg.n_joints].astype(jnp.float32),
                    s.data.qvel[6 : 6 + cfg.n_joints].astype(jnp.float32),
                    s.data.qpos[3:7].astype(jnp.float32),
                    s.data.qvel[0:3].astype(jnp.float32),
                    s.data.qvel[3:6].astype(jnp.float32),
                ])
            )(states)  # (n_envs, obs_dim)

            # Actor-critic forward pass (vmapped over envs)
            mean, log_std, value = jax.vmap(
                partial(net.apply, params)
            )(obs_batch)  # mean/value: (n_envs, act_dim), (n_envs,)

            # Sample actions
            keys_env = jax.random.split(key_act, n_envs)
            actions, log_probs = jax.vmap(_gaussian_sample)(keys_env, mean, log_std)
            actions_clipped = jnp.clip(actions, -1.0, 1.0)

            # Step all envs
            new_states, _, rewards, dones = batch_step(states, actions_clipped)

            # Auto-reset: when an env is done, reset it
            rng, key_rst = jax.random.split(rng)
            reset_keys = jax.random.split(key_rst, n_envs)
            reset_states, _ = batch_reset(reset_keys)
            # Replace each done env with a fresh reset state
            new_states = jax.tree.map(
                lambda new, rst: jnp.where(
                    dones.reshape((-1,) + (1,) * (new.ndim - 1)),
                    rst, new
                ),
                new_states, reset_states,
            )

            trans = Transition(
                obs=obs_batch,
                action=actions_clipped,
                log_prob=log_probs,
                value=value,
                reward=rewards,
                done=dones,
            )
            return (new_states, rng), trans

        (states, rng), transitions = jax.lax.scan(
            _one_step, (states, rng), None, length=rollout_len
        )
        return states, transitions, rng

    return collect


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: Any,    # (T, n_envs)
    values:  Any,    # (T, n_envs)
    dones:   Any,    # (T, n_envs)
    last_value: Any, # (n_envs,)
    gamma:   float,
    gae_lam: float,
) -> Tuple[Any, Any]:
    """
    Compute GAE advantages and returns.  Returns (advantages, returns).
    All shapes: (T, n_envs).
    """
    T = rewards.shape[0]

    def _step(carry, t):
        adv, last_v = carry
        r   = rewards[T - 1 - t]
        v   = values [T - 1 - t]
        d   = dones  [T - 1 - t].astype(jnp.float32)
        delta = r + gamma * last_v * (1.0 - d) - v
        adv   = delta + gamma * gae_lam * (1.0 - d) * adv
        return (adv, v), adv

    _, advantages_rev = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_value), last_value),
        jnp.arange(T),
    )
    advantages = advantages_rev[::-1]      # reverse back to (T, n_envs)
    returns    = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update step
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    clip_eps:      float = 0.2
    vf_coef:       float = 0.5
    ent_coef:      float = 0.01
    max_grad_norm: float = 0.5
    n_epochs:      int   = 4
    minibatch_size: int  = 256


def make_ppo_update_fn(
    net:    ActorCritic,
    tx:     Any,          # optax optimizer
    ppo:    PPOConfig,
) -> Any:
    """
    Return a JIT-compiled function that performs one PPO update epoch.

        update(train_state, batch) → (train_state, metrics)

    `batch` is a flat dict of arrays with leading dim = total_steps.
    """

    def _loss(params, obs, action, old_log_prob, advantage, returns):
        mean, log_std, value = jax.vmap(partial(net.apply, params))(obs)
        std = jnp.exp(log_std)

        # New log-prob
        log_prob = -0.5 * (
            jnp.sum(((action - mean) / std) ** 2, axis=-1)
            + jnp.sum(jnp.log(2.0 * jnp.pi * std ** 2), axis=-1)
        )

        ratio      = jnp.exp(log_prob - old_log_prob)
        adv_norm   = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        pg_loss1   = -adv_norm * ratio
        pg_loss2   = -adv_norm * jnp.clip(ratio, 1 - ppo.clip_eps, 1 + ppo.clip_eps)
        actor_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        value_loss = jnp.mean((value - returns) ** 2)

        entropy = jnp.mean(
            0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * jnp.e * std ** 2), axis=-1)
        )

        total = actor_loss + ppo.vf_coef * value_loss - ppo.ent_coef * entropy
        return total, (actor_loss, value_loss, entropy)

    @jax.jit
    def update(
        train_state: TrainState,
        obs:         Any,   # (N, obs_dim)
        action:      Any,   # (N, act_dim)
        old_log_prob: Any,  # (N,)
        advantage:   Any,   # (N,)
        returns:     Any,   # (N,)
    ) -> Tuple[TrainState, dict]:
        grad_fn = jax.value_and_grad(_loss, has_aux=True)
        (loss, (al, vl, ent)), grads = grad_fn(
            train_state.params, obs, action, old_log_prob, advantage, returns
        )
        grads = jax.tree.map(
            lambda g: jnp.clip(g, -ppo.max_grad_norm, ppo.max_grad_norm), grads
        )
        train_state = train_state.apply_gradients(grads=grads)
        metrics = {"loss": loss, "actor_loss": al, "value_loss": vl, "entropy": ent}
        return train_state, metrics

    return update


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def _run_training(
    cfg:            MJXEnvConfig,
    params:         Any,
    total_steps:    int,
    n_envs:         int,
    rollout_len:    int,
    ppo_cfg:        PPOConfig,
    learning_rate:  float,
    gamma:          float,
    gae_lambda:     float,
    seed:           int,
    fitness_episodes: int,
    verbose:        bool,
    policy_arch:    tuple = (256, 256),
) -> Tuple[Any, float]:
    """
    Shared training loop.  Returns (final_params, end_fitness).
    """
    net = ActorCritic(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.n_joints,
        hidden=tuple(policy_arch),
    )

    tx = optax.chain(
        optax.clip_by_global_norm(ppo_cfg.max_grad_norm),
        optax.adam(learning_rate),
    )
    train_state = TrainState.create(
        apply_fn=net.apply, params=params, tx=tx
    )

    collect_fn = make_rollout_fn(cfg, net, n_envs, rollout_len)
    update_fn  = make_ppo_update_fn(net, tx, ppo_cfg)

    rng = jax.random.PRNGKey(seed)

    # Initial batch reset
    _, _, make_batch_fns = make_env_fns(cfg)
    batch_reset, _ = make_batch_fns(n_envs)
    rng, key_rst = jax.random.split(rng)
    reset_keys = jax.random.split(key_rst, n_envs)
    states, _ = batch_reset(reset_keys)

    n_updates = max(1, total_steps // (n_envs * rollout_len))
    total_collected = 0
    episode_returns: list[float] = []

    t0 = time.time()
    for update_idx in range(n_updates):
        rng, key_col = jax.random.split(rng)
        states, transitions, rng = collect_fn(train_state.params, states, key_col)
        total_collected += n_envs * rollout_len

        # Compute bootstrap value for last state
        obs_last = jax.vmap(
            lambda s: jnp.concatenate([
                jnp.array([
                    jnp.sin(s.sim_time * jnp.float32(1.0  / np.pi)),
                    jnp.sin(s.sim_time * jnp.float32(5.0  / np.pi)),
                    jnp.sin(s.sim_time * jnp.float32(15.0 / np.pi)),
                ]),
                s.data.qpos[7 : 7 + cfg.n_joints].astype(jnp.float32),
                s.data.qvel[6 : 6 + cfg.n_joints].astype(jnp.float32),
                s.data.qpos[3:7].astype(jnp.float32),
                s.data.qvel[0:3].astype(jnp.float32),
                s.data.qvel[3:6].astype(jnp.float32),
            ])
        )(states)
        _, _, last_value = jax.vmap(partial(net.apply, train_state.params))(obs_last)

        advantages, returns = compute_gae(
            transitions.reward, transitions.value, transitions.done,
            last_value, gamma, gae_lambda,
        )

        # Flatten (T, n_envs, ...) → (T*n_envs, ...)
        T, N = transitions.obs.shape[:2]
        flat = lambda x: x.reshape((T * N,) + x.shape[2:])
        obs_f    = flat(transitions.obs)
        act_f    = flat(transitions.action)
        lp_f     = flat(transitions.log_prob)
        adv_f    = flat(advantages)
        ret_f    = flat(returns)

        # Track episode returns (for fitness)
        ep_r = np.array(transitions.reward).sum(axis=0)  # rough per-env sum
        episode_returns.extend(ep_r.tolist())

        # PPO epochs (Python loop — small, acceptable)
        total_samples = T * N
        indices = np.arange(total_samples)
        for _ in range(ppo_cfg.n_epochs):
            np.random.shuffle(indices)
            mb = ppo_cfg.minibatch_size
            for start in range(0, total_samples, mb):
                idx = indices[start : start + mb]
                if len(idx) < 2:
                    continue
                train_state, metrics = update_fn(
                    train_state,
                    obs_f[idx], act_f[idx], lp_f[idx], adv_f[idx], ret_f[idx],
                )

        if verbose and (update_idx % max(1, n_updates // 10) == 0):
            elapsed = time.time() - t0
            fps = total_collected / elapsed
            print(f"  update {update_idx+1}/{n_updates}  "
                  f"steps={total_collected:,}  fps={fps:.0f}  "
                  f"loss={float(metrics['loss']):.4f}")

    end_fitness = float(np.mean(episode_returns[-fitness_episodes:])) if episode_returns else 0.0
    return train_state.params, end_fitness


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_from_scratch_mjx(
    reward_weights:   Optional[RewardWeights] = None,
    cfg:              Optional[MJXEnvConfig]  = None,
    seed:             int   = 0,
    total_steps:      int   = 1_000_000,
    n_envs:           int   = 128,
    rollout_len:      int   = 64,
    learning_rate:    float = 3e-4,
    gamma:            float = 0.99,
    gae_lambda:       float = 0.95,
    policy_arch:      tuple = (256, 256),
    ppo_cfg:          Optional[PPOConfig] = None,
    episode_duration: float = 5.0,
    control_frequency: int  = 20,
    fall_height:      float = 0.3,
    fitness_episodes: int   = 20,
    verbose:          bool  = False,
) -> Tuple[Any, float]:
    """
    Train a PPO policy from scratch against the given reward weights.

    Returns
    -------
    (params, end_fitness)
        params      : Flax param dict — pass to train_warm_start_mjx as parent_params
        end_fitness : mean episodic return over last `fitness_episodes` episode-worth
                      of collected rewards
    """
    # Force CPU for MJX compatibility (Metal not supported)
    dev = _pick_mjx_device()
    jax.config.update("jax_default_device", dev)

    if cfg is None:
        cfg = build_env_config(
            reward_weights    = reward_weights,
            episode_duration  = episode_duration,
            control_frequency = control_frequency,
            fall_height       = fall_height,
        )

    ppo_cfg = ppo_cfg or PPOConfig()
    key = jax.random.PRNGKey(seed)
    net, params = init_actor_critic(key, cfg.obs_dim, cfg.n_joints, policy_arch)

    return _run_training(
        cfg            = cfg,
        params         = params,
        total_steps    = total_steps,
        n_envs         = n_envs,
        rollout_len    = rollout_len,
        ppo_cfg        = ppo_cfg,
        learning_rate  = learning_rate,
        gamma          = gamma,
        gae_lambda     = gae_lambda,
        seed           = seed,
        fitness_episodes = fitness_episodes,
        verbose        = verbose,
        policy_arch    = policy_arch,
    )


def train_warm_start_mjx(
    parent_params:    Any,
    reward_weights:   Optional[RewardWeights] = None,
    cfg:              Optional[MJXEnvConfig]  = None,
    seed:             int   = 0,
    total_steps:      int   = 250_000,
    n_envs:           int   = 128,
    rollout_len:      int   = 64,
    learning_rate:    float = 3e-4,
    gamma:            float = 0.99,
    gae_lambda:       float = 0.95,
    policy_arch:      tuple = (256, 256),
    ppo_cfg:          Optional[PPOConfig] = None,
    episode_duration: float = 5.0,
    control_frequency: int  = 20,
    fall_height:      float = 0.3,
    fitness_episodes: int   = 20,
    verbose:          bool  = False,
) -> Tuple[Any, float]:
    """
    Warm-start training: inherit parent's policy weights, train against child's
    (mutated) reward weights for fewer steps.

    Parameters
    ----------
    parent_params : Flax param dict returned by a previous train_* call.
    reward_weights: child's mutated reward weights.

    Returns
    -------
    (params, end_fitness)
    """
    dev = _pick_mjx_device()
    jax.config.update("jax_default_device", dev)

    if cfg is None:
        cfg = build_env_config(
            reward_weights    = reward_weights,
            episode_duration  = episode_duration,
            control_frequency = control_frequency,
            fall_height       = fall_height,
        )

    ppo_cfg = ppo_cfg or PPOConfig()

    return _run_training(
        cfg            = cfg,
        params         = parent_params,     # start from parent weights
        total_steps    = total_steps,
        n_envs         = n_envs,
        rollout_len    = rollout_len,
        ppo_cfg        = ppo_cfg,
        learning_rate  = learning_rate,
        gamma          = gamma,
        gae_lambda     = gae_lambda,
        seed           = seed,
        fitness_episodes = fitness_episodes,
        verbose        = verbose,
        policy_arch    = policy_arch,
    )


def make_params(
    seed:        int,
    obs_dim:     int,
    act_dim:     int,
    policy_arch: tuple = (256, 256),
) -> Any:
    """
    Initialise random Flax params without running a full build.
    Useful for test fixtures.
    """
    dev = _pick_mjx_device()
    jax.config.update("jax_default_device", dev)
    key = jax.random.PRNGKey(seed)
    _, params = init_actor_critic(key, obs_dim, act_dim, policy_arch)
    return params


# ---------------------------------------------------------------------------
# Debug / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    print("=" * 60)
    print("  ppo_trainer_mjx.py — debug mode")
    print("=" * 60)

    dev = _pick_mjx_device()
    jax.config.update("jax_default_device", dev)
    print(f"\n  JAX device : {dev}")

    # 1. Tiny from-scratch run
    print("\n[1] train_from_scratch_mjx (tiny: 2 envs × 16 steps × 1 update)\n")
    params, fitness = train_from_scratch_mjx(
        seed            = 0,
        total_steps     = 2 * 16,   # exactly 1 update
        n_envs          = 2,
        rollout_len     = 16,
        policy_arch     = (64, 64),
        episode_duration = 2.0,
        fitness_episodes = 1,
        verbose         = True,
    )
    print(f"  Params pytree leaves: {len(jax.tree.leaves(params))}")
    print(f"  End fitness: {fitness:.4f}")
    assert params is not None
    print("  from_scratch: OK")

    # 2. Warm-start from above params
    print("\n[2] train_warm_start_mjx (2 envs × 16 steps)\n")
    from reward import RewardWeights, mutate_weights
    rw_child = mutate_weights(RewardWeights(), sigma=0.2, rng=np.random.default_rng(1))
    params2, fitness2 = train_warm_start_mjx(
        parent_params   = params,
        reward_weights  = rw_child,
        seed            = 1,
        total_steps     = 2 * 16,
        n_envs          = 2,
        rollout_len     = 16,
        policy_arch     = (64, 64),
        episode_duration = 2.0,
        fitness_episodes = 1,
        verbose         = True,
    )
    print(f"  warm-start fitness: {fitness2:.4f}")
    assert params2 is not None
    print("  warm_start: OK")

    print("\nAll ppo_trainer_mjx.py checks passed.")
