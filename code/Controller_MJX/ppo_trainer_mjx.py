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
    make_fast_reset_fn,
    make_reward_agnostic_batch_fns,
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

# Numerical safety rails for the Gaussian policy. The evolutionary search drives
# a wide range of reward scales; without these, log_std can collapse toward -inf
# (std→0 ⇒ log_prob→±inf) or the importance ratio can overflow, NaN-ing the whole
# update. Both clamps only bite at extremes (init log_std=-0.5 ⇒ std≈0.61).
_LOG_STD_MIN   = -5.0    # std ≈ 0.0067
_LOG_STD_MAX   = 2.0     # std ≈ 7.39
_RATIO_LOG_CLIP = 10.0   # exp(±10) before the PPO clip ratio


def _gaussian_sample(key: Any, mean: Any, log_std: Any) -> Tuple[Any, Any]:
    """Sample from N(mean, exp(log_std)) and return (action, log_prob)."""
    log_std = jnp.clip(log_std, _LOG_STD_MIN, _LOG_STD_MAX)
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
    ent_coef:      float = 0.0
    max_grad_norm: float = 0.5
    n_epochs:      int   = 4
    n_minibatches: int   = 32   # minibatches per epoch; total grad steps = n_epochs × n_minibatches


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
        log_std = jnp.clip(log_std, _LOG_STD_MIN, _LOG_STD_MAX)
        std = jnp.exp(log_std)

        # New log-prob
        log_prob = -0.5 * (
            jnp.sum(((action - mean) / std) ** 2, axis=-1)
            + jnp.sum(jnp.log(2.0 * jnp.pi * std ** 2), axis=-1)
        )

        ratio      = jnp.exp(jnp.clip(log_prob - old_log_prob, -_RATIO_LOG_CLIP, _RATIO_LOG_CLIP))
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
# Optimised training step (single XLA kernel per PPO update)
# ---------------------------------------------------------------------------

def make_train_step_fn(
    cfg:              MJXEnvConfig,
    net:              ActorCritic,
    n_envs:           int,
    rollout_len:      int,
    ppo_cfg:          PPOConfig,
    gamma:            float,
    gae_lambda:       float,
    batch_step_rw:    Any,   # (states, actions, rw_vec) → (states, obs, rewards, dones)
    fast_batch_reset: Any,   # (keys) → (states, obs)
) -> Any:
    """
    Return a single @jax.jit function that performs one full PPO update:
        rollout → GAE → n_epochs × n_minibatches of gradient steps.

    Everything compiles into one XLA kernel.  No Python between minibatches.

    Signature of the returned function
    -----------------------------------
        train_step(train_state, runner_state, rw_vec)
            → (train_state, runner_state, metrics, rewards)

        runner_state = (env_states, obs, rng)
        rw_vec       = (reward_dim,) float32 — runtime, NOT baked into kernel
        metrics      = {'loss', 'actor_loss', 'value_loss', 'entropy'}  (means)
        rewards      = (rollout_len, n_envs) float32

    rw_vec as runtime argument
    --------------------------
    By accepting rw_vec at call time rather than closing over it, the JIT-compiled
    kernel can be REUSED across all individuals in an evolutionary run.
    A single 30-60 s compile serves the entire experiment instead of one per individual.
    """
    total_samples = n_envs * rollout_len

    # Clamp so each minibatch has at least one sample (matters for tiny test configs).
    n_minibatches = max(1, min(ppo_cfg.n_minibatches, total_samples))
    effective_mb  = total_samples // n_minibatches

    @jax.jit
    def train_step(train_state: Any, runner_state: Any, rw_vec: Any) -> Any:
        env_states, obs, rng = runner_state

        # ------------------------------------------------------------------
        # Phase 1 — Rollout (lax.scan over rollout_len steps)
        #   obs carried through scan — no duplicate computation per step.
        #   fast_batch_reset skips mjx.forward (~10× cheaper).
        # ------------------------------------------------------------------
        def env_step(carry, _):
            states, obs, rng = carry
            rng, key_act, key_rst = jax.random.split(rng, 3)

            mean, log_std, value = jax.vmap(
                partial(net.apply, train_state.params)
            )(obs)

            act_keys = jax.random.split(key_act, n_envs)
            actions_raw, log_probs = jax.vmap(_gaussian_sample)(act_keys, mean, log_std)
            actions = jnp.clip(actions_raw, jnp.float32(-1.0), jnp.float32(1.0))

            # rw_vec is broadcast over envs (in_axes=None in vmap)
            new_states, new_obs, rewards, dones = batch_step_rw(states, actions, rw_vec)

            rst_keys             = jax.random.split(key_rst, n_envs)
            rst_states, rst_obs  = fast_batch_reset(rst_keys)

            final_states = jax.tree.map(
                lambda n, r: jnp.where(
                    dones.reshape((-1,) + (1,) * (n.ndim - 1)), r, n
                ),
                new_states, rst_states,
            )
            final_obs = jnp.where(dones[:, None], rst_obs, new_obs)

            # Store unclipped action so PPO log_prob recomputation is consistent
            # with the sampled log_prob.  Clipped actions sent to env are correct
            # for physics; storing them here would corrupt the ratio computation
            # for ~26% of samples and systematically push mean toward zero.
            trans = Transition(obs, actions_raw, log_probs, value, rewards, dones)
            return (final_states, final_obs, rng), trans

        (env_states, obs, rng), transitions = jax.lax.scan(
            env_step, (env_states, obs, rng), None, length=rollout_len
        )

        # ------------------------------------------------------------------
        # Phase 2 — Bootstrap value
        # ------------------------------------------------------------------
        _, _, last_value = jax.vmap(
            partial(net.apply, train_state.params)
        )(obs)

        # ------------------------------------------------------------------
        # Phase 3 — GAE
        # ------------------------------------------------------------------
        advantages, returns = compute_gae(
            transitions.reward, transitions.value, transitions.done,
            last_value, gamma, gae_lambda,
        )

        # ------------------------------------------------------------------
        # Phase 4 — Flatten (T, N, ...) → (T*N, ...)
        # ------------------------------------------------------------------
        flat  = lambda x: x.reshape((total_samples,) + x.shape[2:])
        obs_f = flat(transitions.obs)
        act_f = flat(transitions.action)
        lp_f  = flat(transitions.log_prob)
        adv_f = flat(advantages)
        ret_f = flat(returns)

        # ------------------------------------------------------------------
        # Phase 5 — PPO: lax.scan over epochs × minibatches
        # ------------------------------------------------------------------
        def _ppo_loss(params, mb_obs, mb_act, mb_lp, mb_adv, mb_ret):
            mean, log_std, value = jax.vmap(partial(net.apply, params))(mb_obs)
            log_std  = jnp.clip(log_std, _LOG_STD_MIN, _LOG_STD_MAX)
            std      = jnp.exp(log_std)
            log_prob = -0.5 * (
                jnp.sum(((mb_act - mean) / std) ** 2, axis=-1)
                + jnp.sum(jnp.log(2.0 * jnp.pi * std ** 2), axis=-1)
            )
            ratio    = jnp.exp(jnp.clip(log_prob - mb_lp, -_RATIO_LOG_CLIP, _RATIO_LOG_CLIP))
            adv_norm = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg1      = -adv_norm * ratio
            pg2      = -adv_norm * jnp.clip(ratio, 1 - ppo_cfg.clip_eps,
                                                    1 + ppo_cfg.clip_eps)
            al  = jnp.mean(jnp.maximum(pg1, pg2))
            vl  = jnp.mean((value - mb_ret) ** 2)
            ent = jnp.mean(
                0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * jnp.e * std ** 2), axis=-1)
            )
            return al + ppo_cfg.vf_coef * vl - ppo_cfg.ent_coef * ent, (al, vl, ent)

        def _mb_update(ts, mb):
            mb_obs, mb_act, mb_lp, mb_adv, mb_ret = mb
            (loss, (al, vl, ent)), grads = jax.value_and_grad(
                _ppo_loss, has_aux=True
            )(ts.params, mb_obs, mb_act, mb_lp, mb_adv, mb_ret)
            return ts.apply_gradients(grads=grads), (loss, al, vl, ent)

        def _epoch(ts, epoch_key):
            perm = jax.random.permutation(epoch_key, total_samples)
            perm = perm[: n_minibatches * effective_mb]
            mk   = lambda x: x[perm].reshape(
                (n_minibatches, effective_mb) + x.shape[1:]
            )
            mbs = (mk(obs_f), mk(act_f), mk(lp_f), mk(adv_f), mk(ret_f))
            ts, ep_metrics = jax.lax.scan(_mb_update, ts, mbs)
            return ts, ep_metrics

        # Safe key splitting inside JIT: use index slicing, not Python unpacking.
        _all_keys  = jax.random.split(rng, ppo_cfg.n_epochs + 1)
        rng        = _all_keys[0]
        epoch_keys = _all_keys[1:]          # (n_epochs, 2)
        train_state, all_ep_metrics = jax.lax.scan(_epoch, train_state, epoch_keys)

        loss_all, al_all, vl_all, ent_all = all_ep_metrics
        metrics = {
            "loss":       jnp.mean(loss_all),
            "actor_loss": jnp.mean(al_all),
            "value_loss": jnp.mean(vl_all),
            "entropy":    jnp.mean(ent_all),
        }

        return train_state, (env_states, obs, rng), metrics, transitions.reward

    return train_step


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

    Uses make_train_step_fn so that each PPO update (rollout + GAE + all
    minibatch gradient steps) compiles into a single XLA kernel.
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
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # Build env functions — reward-agnostic so the same compiled kernel works
    # for all individuals (no recompile when rw_vec changes between individuals).
    batch_reset, batch_step_rw, fast_reset = make_reward_agnostic_batch_fns(cfg, n_envs)

    train_step = make_train_step_fn(
        cfg, net, n_envs, rollout_len, ppo_cfg, gamma, gae_lambda,
        batch_step_rw, fast_reset,
    )

    # Initial reset (full reset with mjx.forward for valid kinematics at step 0)
    rng = jax.random.PRNGKey(seed)
    rng, key_rst = jax.random.split(rng)
    rst_keys = jax.random.split(key_rst, n_envs)
    env_states, obs = batch_reset(rst_keys)

    runner_state    = (env_states, obs, rng)
    n_updates       = max(1, total_steps // (n_envs * rollout_len))
    total_collected = 0
    rw_vec          = cfg.reward_weights_vec   # (reward_dim,) JAX array

    # Keep last few rollout reward arrays for fitness (lazy — no sync per step).
    tail_rewards: list = []
    keep_tail = max(2, 1 + fitness_episodes // max(1, n_envs))

    t0 = time.time()
    for update_idx in range(n_updates):
        train_state, runner_state, metrics, raw_rewards = train_step(
            train_state, runner_state, rw_vec
        )
        total_collected += n_envs * rollout_len
        tail_rewards.append(raw_rewards)
        if len(tail_rewards) > keep_tail:
            tail_rewards.pop(0)

        if verbose and (
            update_idx % max(1, n_updates // 10) == 0
            or update_idx == n_updates - 1
        ):
            jax.block_until_ready(metrics["value_loss"])   # sync only for logging
            elapsed = time.time() - t0
            fps = total_collected / elapsed
            rw_mean = float(jnp.mean(jnp.concatenate([r.ravel() for r in tail_rewards]))) if tail_rewards else 0.0
            print(f"  update {update_idx+1}/{n_updates}  steps={total_collected:,}  fps={fps:.0f}"
                  f"  rw={rw_mean:+.3f}"
                  f"  π={float(metrics['actor_loss']):+.3f}"
                  f"  V={float(metrics['value_loss']):.1f}"
                  f"  ent={float(metrics['entropy']):.3f}")

    # Fitness: mean step reward over the last few rollouts (proxy for episodic return).
    if tail_rewards:
        last = jnp.concatenate([r.ravel() for r in tail_rewards])
        end_fitness = float(jnp.mean(last))
    else:
        end_fitness = 0.0

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
