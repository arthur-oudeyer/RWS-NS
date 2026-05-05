"""
reward.py
=========
Reward weight vector — the **variable that evolves** in stage 3.

The body is fixed (QUADRIPOD), so the only thing the EA mutates is a small,
structured, human-interpretable set of coefficients on hand-designed shaping
terms. Each child's vector is fed to PPO as the per-step reward function;
PPO is the inner-loop solver that turns a candidate reward into a
controller, and the VLM judges the resulting behaviour.

Sign convention
---------------
Each term is signed so that a *positive weight always means "more of this
is better"*. Log-normal mutation can never flip a term's role — the
human-chosen direction of each term is part of the prior.

Components (17-dim vector)
--------------------------
  forward_velocity          : +v_x of the torso (forward progress)
  lateral_drift             : −|v_y| of the torso (penalises sideways drift)
  upright_bonus             : cos(roll)*cos(pitch) (rewards staying upright)
  energy_penalty            : −sum(action²) (control cost)
  contact_reward            : +number of feet in ground contact
  alive_bonus               : +1 per step the robot has not fallen
  fall_penalty              : one-off −1 when torso drops below threshold
  no_contact_reward         : airborne fraction of feet [0,1]
  torso_height_reward       : raw torso z (rewards tall/jumping posture)
  torso_rotation_reward     : |ω_z| torso spin around vertical axis
  torso_tilting_speed_reward: ‖ω_xy‖ torso roll/pitch angular speed
  limb_coordination_reward  : exp(−std(hip_velocities)) — uniform leg rhythm
  nervosity_reward          : mean|Δaction| — rewards jerky/fast changes
  smooth_reward             : exp(−mean|Δaction|) — rewards smooth control
  vertical_velocity_reward  : v_z of torso (rewards going upward / jumping)
  lateral_velocity_reward   : |v_y| of torso (rewards sideways crab-walk)
  joint_range_reward        : std(hip_angles) — rewards wide-range postures

The signs are baked into compute_step_reward() — weights always multiply a
non-negative term, then the whole term is negated when the *intent* is a
penalty (energy_penalty, lateral_drift, fall_penalty). Mutation acts on the
weight magnitude only.

Mutation
--------
Multiplicative log-normal noise per dimension:
    w_child[i] = w_parent[i] * exp(N(0, σ))
This keeps signs sane (already enforced by the convention above) and gives
roughly proportional scaling regardless of a weight's magnitude.

Debug
-----
Run this file to round-trip vector ↔ dataclass and to sample 100 mutations
from the default vector — per-dimension mean / std / min / max are printed
so the σ behaviour can be eyeballed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid forcing an import cycle in the type hints
    from mujoco_env import RobotSensorReading


# ---------------------------------------------------------------------------
# RewardWeights — the evolved variable
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """
    Coefficients of the shaped per-step reward fed to PPO.

    These are stored verbatim in the archive. Default values come from
    `ExperimentConfig.default_reward_weights_dict()`.
    """
    # ---- Original 7 terms -------------------------------------------------------
    forward_velocity: float = 1.0    # rewards +v_x torso
    lateral_drift:    float = 0.1    # penalises |v_y| torso
    upright_bonus:    float = 0.5    # rewards upright posture
    energy_penalty:   float = 0.01   # penalises sum(action**2)
    contact_reward:   float = 0.1    # rewards feet in contact
    alive_bonus:      float = 0.05   # rewards every alive step
    fall_penalty:     float = 10.0   # one-off penalty on fall

    # ---- Extended 10 terms (small positive defaults so log-normal mutation ----
    # ---- can activate them; set to 0 in config.py to fully disable)  ---------

    # Airborne fraction of feet [0,1]: promotes hopping / jumping
    no_contact_reward: float = 0.05

    # Raw torso z height: promotes tall posture or jumping
    torso_height_reward: float = 0.05

    # |ω_z|: rewards spinning around the vertical axis
    torso_rotation_reward: float = 0.01

    # ‖ω_xy‖: rewards rolling / tilting (angular speed in roll+pitch plane)
    torso_tilting_speed_reward: float = 0.01

    # exp(−std(hip_vels)) ∈ [0,1]: rewards limbs swinging in unison
    limb_coordination_reward: float = 0.05

    # mean|Δaction| per step: rewards twitchy / high-acceleration control
    nervosity_reward: float = 0.01

    # exp(−mean|Δaction|) ∈ [0,1]: rewards smooth, low-jerk control
    smooth_reward: float = 0.05

    # v_z of torso: rewards upward movement (jumping impulse)
    vertical_velocity_reward: float = 0.05

    # |v_y| of torso: rewards lateral movement (crab-walk, side-step)
    lateral_velocity_reward: float = 0.01

    # std(hip_angles): rewards diverse joint configurations (acrobatics)
    joint_range_reward: float = 0.01

    # ---- vector form ------------------------------------------------------

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.field_names()], dtype=np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "RewardWeights":
        names = cls.field_names()
        if len(v) != len(names):
            raise ValueError(f"vector length {len(v)} ≠ #fields {len(names)}")
        return cls(**{n: float(v[i]) for i, n in enumerate(names)})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardWeights":
        return cls(**{k: float(d[k]) for k in cls.field_names() if k in d})


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate_weights(
    parent: RewardWeights,
    sigma:  float = 0.2,
    rng:    Optional[np.random.Generator] = None,
) -> RewardWeights:
    """
    Multiplicative log-normal noise on each weight:
        w_child[i] = w_parent[i] * exp(N(0, σ))

    The output preserves sign and stays positive whenever the parent is
    positive — i.e. no term ever flips role through mutation.
    """
    if rng is None:
        rng = np.random.default_rng()
    v = parent.to_vector()
    noise = rng.normal(0.0, sigma, size=v.shape)
    return RewardWeights.from_vector(v * np.exp(1.38 * noise))


def random_initial_weights(
    cfg_defaults: dict,
    sigma:        float = 0.4,
    rng:          Optional[np.random.Generator] = None,
) -> RewardWeights:
    """
    Sample one initial-population weight vector by widening each default
    component with the larger `sigma` (usually `reward_init_sigma` from
    config). Lets gen-0 individuals explore the prior instead of all
    converging to the literal defaults.
    """
    if rng is None:
        rng = np.random.default_rng()
    base = RewardWeights(**cfg_defaults)
    return mutate_weights(base, sigma=sigma, rng=rng)


# ---------------------------------------------------------------------------
# Per-step reward
# ---------------------------------------------------------------------------

def _quat_upright_factor(quat_wxyz: np.ndarray) -> float:
    """
    cos(roll) * cos(pitch) computed from a [w, x, y, z] torso quaternion.

    Equivalent to dot(R · ẑ_world, ẑ_world) — the world-Z component of the
    body's local +Z axis after rotation. 1.0 = perfectly upright, 0.0 = on
    its side, −1.0 = upside down. We clamp to 0 so the bonus never goes
    negative (the fall_penalty handles "upside down" separately).
    """
    w, x, y, z = quat_wxyz
    # zhat_body_in_world_z = 1 - 2*(x^2 + y^2)
    return float(max(0.0, 1.0 - 2.0 * (x * x + y * y)))


def compute_step_reward(
    weights:     RewardWeights,
    sensors:     "RobotSensorReading",
    action:      np.ndarray,
    prev_action: np.ndarray,
    fell:        bool,
) -> float:
    """
    Shaped per-step reward used by PPO inside one training run.

    `prev_action` is the clipped action from the previous step (zeros on the
    first step after reset). It is used to compute action jerk for the
    nervosity / smooth terms.
    """
    vx = float(sensors.torso_velocity[0])
    vy = float(sensors.torso_velocity[1])
    vz = float(sensors.torso_velocity[2])

    upright = _quat_upright_factor(sensors.torso_orientation)
    energy  = float(np.sum(np.square(action)))
    contact = float(sensors.n_contacts)

    # ---- Original 7 terms -------------------------------------------------------
    r = (
          weights.forward_velocity * vx
        - weights.lateral_drift    * abs(vy)
        + weights.upright_bonus    * upright
        - weights.energy_penalty   * energy
        + weights.contact_reward   * contact
        + weights.alive_bonus
    )

    # ---- Extended terms ---------------------------------------------------------

    # Airborne fraction: how many feet are NOT on the ground, in [0, 1]
    n_feet = max(1, sensors.n_feet_total)
    airborne = max(0.0, float(n_feet - contact) / n_feet)
    r += weights.no_contact_reward * airborne

    # Torso height (raw z value)
    r += weights.torso_height_reward * float(sensors.torso_height)

    # Torso spin around vertical axis
    r += weights.torso_rotation_reward * abs(float(sensors.torso_angular_velocity[2]))

    # Torso roll/pitch angular speed
    tilt_speed = float(np.linalg.norm(sensors.torso_angular_velocity[:2]))
    r += weights.torso_tilting_speed_reward * tilt_speed

    # Limb coordination: exp(−std) ∈ (0, 1], 1.0 = all joints in perfect unison
    if len(sensors.hip_velocities) > 1:
        coordination = float(np.exp(-np.std(sensors.hip_velocities)))
    else:
        coordination = 1.0
    r += weights.limb_coordination_reward * coordination

    # Action jerk (mean absolute change from previous action)
    jerk = float(np.mean(np.abs(action - prev_action)))
    r += weights.nervosity_reward * jerk
    r += weights.smooth_reward    * float(np.exp(-jerk))   # exp(−jerk) ∈ (0, 1]

    # Vertical velocity (rewards moving upward)
    r += weights.vertical_velocity_reward * vz

    # Lateral velocity (rewards sideways movement)
    r += weights.lateral_velocity_reward * abs(vy)

    # Joint angle diversity (rewards wide range of postures)
    if len(sensors.hip_angles) > 1:
        r += weights.joint_range_reward * float(np.std(sensors.hip_angles))

    if fell:
        r -= weights.fall_penalty
    return r


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  reward.py — debug mode")
    print("=" * 60)

    # 1. Round-trip
    print("\n[1] Round-trip RewardWeights ↔ vector ↔ dict\n")
    rw = RewardWeights()
    v  = rw.to_vector()
    rw2 = RewardWeights.from_vector(v)
    rw3 = RewardWeights.from_dict(rw.to_dict())
    assert rw == rw2 == rw3
    print(f"  defaults: {rw.to_dict()}")
    print(f"  vector  : {v.round(4).tolist()}")
    print(f"  Round-trip : OK")

    # 2. Mutation statistics
    print("\n[2] Mutation statistics (1000 samples, σ=0.2)\n")
    rng = np.random.default_rng(0)
    samples = np.stack([
        mutate_weights(rw, sigma=0.2, rng=rng).to_vector()
        for _ in range(1000)
    ])
    names = RewardWeights.field_names()
    print(f"  {'name':<18} {'default':>10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'-'*70}")
    for i, n in enumerate(names):
        col  = samples[:, i]
        dflt = v[i]
        print(f"  {n:<18} {dflt:>10.4f} {col.mean():>10.4f} {col.std():>10.4f} "
              f"{col.min():>10.4f} {col.max():>10.4f}")

    # 3. Sign preservation
    print("\n[3] Sign preservation (1000 samples)\n")
    assert np.all(samples > 0), "log-normal noise must keep weights positive"
    print("  All 1000 samples remained positive: OK")

    # 4. Wider initial-population sigma
    print("\n[4] Initial-population sampling (σ=0.4)\n")
    rng2 = np.random.default_rng(1)
    inits = np.stack([
        random_initial_weights(rw.to_dict(), sigma=0.4, rng=rng2).to_vector()
        for _ in range(500)
    ])
    print(f"  std (init σ=0.4)  → {inits.std(axis=0).round(4).tolist()}")
    print(f"  std (mut  σ=0.2)  → {samples.std(axis=0).round(4).tolist()}")
    print("  init std is wider per dimension (expected): "
          f"{(inits.std(axis=0) > samples.std(axis=0)).all()}")

    # 5. Reward sanity (no MuJoCo): build a fake sensor reading
    print("\n[5] compute_step_reward sanity (no MuJoCo)\n")
    class _FakeSensors:
        torso_velocity         = np.array([1.0, -0.2, 0.1])
        torso_orientation      = np.array([1.0, 0.0, 0.0, 0.0])  # upright
        torso_angular_velocity = np.array([0.1, 0.0, 0.5])
        torso_height           = 0.3
        n_contacts             = 4
        n_feet_total           = 4
        hip_velocities         = np.array([0.5, -0.3, 0.5, -0.3])
        hip_angles             = np.array([0.1, -0.1, 0.2, -0.2])
    prev = np.zeros(4)
    r_alive = compute_step_reward(rw, _FakeSensors(), np.zeros(4), prev, fell=False)
    r_fell  = compute_step_reward(rw, _FakeSensors(), np.zeros(4), prev, fell=True)
    print(f"  upright + moving fwd, fell=False : r = {r_alive:+.5f}")
    print(f"  same step  but   fell=True  : r = {r_fell:+.5f}")
    assert r_fell < r_alive, "fall penalty must lower reward"
    print("  fall penalty lowers reward: OK")

    print("\nAll reward.py checks passed.")
