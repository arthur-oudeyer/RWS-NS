"""
reward.py  (MJX edition)
========================
Identical to Controller/reward.py for the numpy/CPU side (RewardWeights,
mutate_weights, random_initial_weights, compute_step_reward).

Additions
---------
JaxSensorReading  : NamedTuple mirror of RobotSensorReading using JAX arrays.
compute_step_reward_jax : same formula as compute_step_reward but fully in
    jnp so it is jit/vmap-able alongside MJX physics.

The weights are passed as a (22,) float32 jnp.ndarray whose slots follow the
same order as RewardWeights.field_names(). Use RewardWeights.to_jax_vector()
to convert.

Index constants at the bottom of the file document the slot layout so
compute_step_reward_jax can index into the weights vector without a
named-field lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Optional, TYPE_CHECKING, NamedTuple, Any

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# RewardWeights — the evolved variable  (unchanged from Controller)
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """
    Coefficients of the shaped per-step reward fed to PPO.
    Positive weight = more of this term is better.
    """
    # ---- Original 7 terms ---------------------------------------------------
    forward_velocity: float = 1.0
    lateral_drift:    float = 0.1
    upright_bonus:    float = 0.5
    energy_penalty:   float = 0.01
    contact_reward:   float = 0.1
    alive_bonus:      float = 0.05
    fall_penalty:     float = 10.0

    # ---- Extended 15 terms --------------------------------------------------
    no_contact_reward:           float = 0.05
    torso_height_reward:         float = 0.05
    torso_rotation_reward:       float = 0.01
    torso_tilting_speed_reward:  float = 0.01
    limb_coordination_reward:    float = 0.05
    nervosity_reward:            float = 0.01
    smooth_reward:               float = 0.05
    vertical_velocity_reward:    float = 0.05
    lateral_velocity_reward:     float = 0.01
    joint_range_reward:          float = 0.01
    height_target_reward:        float = 0.3
    tilt_penalty:                float = 0.2
    tilt_rate_penalty:           float = 0.1
    all_feet_planted_bonus:      float = 0.2
    vertical_velocity_penalty:   float = 0.05
    horizontal_velocity_penalty: float = 0.05

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.field_names()], dtype=np.float64)

    def to_jax_vector(self):
        """Return a float32 jnp.ndarray for use in compute_step_reward_jax."""
        import jax.numpy as jnp
        return jnp.array(self.to_vector(), dtype=jnp.float32)

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
# Mutation  (unchanged from Controller)
# ---------------------------------------------------------------------------

def mutate_weights(
    parent: RewardWeights,
    sigma:  float = 0.2,
    rng:    Optional[np.random.Generator] = None,
) -> RewardWeights:
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
    if rng is None:
        rng = np.random.default_rng()
    base = RewardWeights(**cfg_defaults)
    return mutate_weights(base, sigma=sigma, rng=rng)


# ---------------------------------------------------------------------------
# numpy per-step reward  (unchanged from Controller)
# ---------------------------------------------------------------------------

def _quat_upright_factor(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return float(max(0.0, 1.0 - 2.0 * (x * x + y * y)))


def compute_step_reward(
    weights:              RewardWeights,
    sensors:              Any,   # RobotSensorReading
    action:               np.ndarray,
    prev_action:          np.ndarray,
    fell:                 bool,
    initial_torso_position: Optional[np.ndarray] = None,
) -> float:
    _init_z = float(initial_torso_position[2]) if initial_torso_position is not None else 0.0
    vx = float(sensors.torso_velocity[0])
    vy = float(sensors.torso_velocity[1])
    vz = float(sensors.torso_velocity[2])

    upright = _quat_upright_factor(sensors.torso_orientation)
    energy  = float(np.sum(np.square(action)))
    contact = float(sensors.n_contacts)

    r = (
          weights.forward_velocity * vx
        - weights.lateral_drift    * abs(vy)
        + weights.upright_bonus    * upright
        - weights.energy_penalty   * energy
        + weights.contact_reward   * contact
        + weights.alive_bonus
    )

    n_feet = max(1, sensors.n_feet_total)
    airborne = max(0.0, float(n_feet - contact) / n_feet)
    r += weights.no_contact_reward * airborne

    r += weights.torso_height_reward * (float(sensors.torso_height) - 0.9 * _init_z)
    r += weights.torso_rotation_reward * abs(float(sensors.torso_angular_velocity[2]))

    tilt_speed = float(np.linalg.norm(sensors.torso_angular_velocity[:2]))
    r += weights.torso_tilting_speed_reward * tilt_speed

    if len(sensors.hip_velocities) > 1:
        coordination = float(np.exp(-np.std(sensors.hip_velocities)))
    else:
        coordination = 1.0
    r += weights.limb_coordination_reward * coordination

    jerk = float(np.mean(np.abs(action - prev_action)))
    r += weights.nervosity_reward * jerk
    r += weights.smooth_reward    * float(np.exp(-jerk))

    r += weights.vertical_velocity_reward  * vz
    r += weights.lateral_velocity_reward   * abs(vy)

    if len(sensors.hip_angles) > 1:
        r += weights.joint_range_reward * float(np.std(sensors.hip_angles))

    h_target = _init_z if _init_z > 0.0 else float(sensors.torso_height)
    r += weights.height_target_reward * float(np.exp(-25.0 * (float(sensors.torso_height) - h_target) ** 2))
    r -= weights.tilt_penalty      * float((1.0 - upright) ** 2)
    r -= weights.tilt_rate_penalty * float(tilt_speed ** 2)

    if sensors.n_feet_total > 0:
        r += weights.all_feet_planted_bonus * float(sensors.n_contacts >= sensors.n_feet_total)

    r -= weights.vertical_velocity_penalty   * float(vz ** 2)
    r -= weights.horizontal_velocity_penalty * float(vx ** 2 + vy ** 2)

    if fell:
        r -= weights.fall_penalty
    return r


# ---------------------------------------------------------------------------
# JAX sensor reading
# ---------------------------------------------------------------------------

class JaxSensorReading(NamedTuple):
    """Mirror of RobotSensorReading using JAX arrays (all float32 / int32)."""
    torso_pos:              Any   # (3,)   float32
    torso_height:           Any   # ()     float32 scalar
    torso_orientation:      Any   # (4,)   float32  [w, x, y, z]
    torso_velocity:         Any   # (3,)   float32
    torso_angular_velocity: Any   # (3,)   float32
    hip_angles:             Any   # (n_joints,) float32
    hip_velocities:         Any   # (n_joints,) float32
    n_contacts:             Any   # ()     int32 scalar
    n_feet_total:           int   # Python int (static at compile time)


# ---------------------------------------------------------------------------
# Weight vector slot indices  (must match RewardWeights.field_names() order)
# ---------------------------------------------------------------------------

_W_FORWARD_VELOCITY          = 0
_W_LATERAL_DRIFT             = 1
_W_UPRIGHT_BONUS             = 2
_W_ENERGY_PENALTY            = 3
_W_CONTACT_REWARD            = 4
_W_ALIVE_BONUS               = 5
_W_FALL_PENALTY              = 6
_W_NO_CONTACT_REWARD         = 7
_W_TORSO_HEIGHT_REWARD       = 8
_W_TORSO_ROTATION_REWARD     = 9
_W_TORSO_TILTING_SPEED       = 10
_W_LIMB_COORDINATION         = 11
_W_NERVOSITY                 = 12
_W_SMOOTH                    = 13
_W_VERTICAL_VELOCITY         = 14
_W_LATERAL_VELOCITY          = 15
_W_JOINT_RANGE               = 16
_W_HEIGHT_TARGET             = 17
_W_TILT_PENALTY              = 18
_W_TILT_RATE_PENALTY         = 19
_W_ALL_FEET_PLANTED          = 20
_W_VERTICAL_VEL_PENALTY      = 21
_W_HORIZONTAL_VEL_PENALTY    = 22   # index 22 = last field


def compute_step_reward_jax(
    weights_vec:         Any,   # (23,) float32 jnp.ndarray
    sensors:             JaxSensorReading,
    action:              Any,   # (n_joints,) float32
    prev_action:         Any,   # (n_joints,) float32
    fell:                Any,   # () bool JAX scalar
    initial_torso_pos:   Any,   # (3,) float32
) -> Any:                       # () float32 scalar
    """
    Fully JAX implementation of the per-step shaped reward.
    jit- and vmap-compatible. All arguments must be JAX arrays or Python
    scalars. `sensors.n_feet_total` stays a Python int (static).
    """
    import jax.numpy as jnp

    w = weights_vec  # short alias

    vx = sensors.torso_velocity[0]
    vy = sensors.torso_velocity[1]
    vz = sensors.torso_velocity[2]
    init_z = initial_torso_pos[2]

    # Upright factor: cos(roll)*cos(pitch) from quaternion [w,x,y,z]
    qx = sensors.torso_orientation[1]
    qy = sensors.torso_orientation[2]
    upright = jnp.maximum(jnp.float32(0.0), jnp.float32(1.0) - jnp.float32(2.0) * (qx * qx + qy * qy))

    energy  = jnp.sum(jnp.square(action))
    contact = sensors.n_contacts.astype(jnp.float32)

    # ---- Original 7 terms --------------------------------------------------
    r = (
          w[_W_FORWARD_VELOCITY] * vx
        - w[_W_LATERAL_DRIFT]    * jnp.abs(vy)
        + w[_W_UPRIGHT_BONUS]    * upright
        - w[_W_ENERGY_PENALTY]   * energy
        + w[_W_CONTACT_REWARD]   * contact
        + w[_W_ALIVE_BONUS]
    )

    # ---- Extended terms ----------------------------------------------------
    # n_feet_total can be a Python int (eager) OR a traced JAX scalar (JIT).
    # jnp.asarray handles both; jnp.maximum avoids Python max() on traced values.
    n_feet = jnp.maximum(jnp.float32(1.0),
                         jnp.asarray(sensors.n_feet_total, dtype=jnp.float32))
    airborne = jnp.maximum(jnp.float32(0.0), (n_feet - contact) / n_feet)
    r += w[_W_NO_CONTACT_REWARD] * airborne

    r += w[_W_TORSO_HEIGHT_REWARD] * (sensors.torso_height - jnp.float32(0.9) * init_z)

    r += w[_W_TORSO_ROTATION_REWARD] * jnp.abs(sensors.torso_angular_velocity[2])

    tilt_speed = jnp.linalg.norm(sensors.torso_angular_velocity[:2])
    r += w[_W_TORSO_TILTING_SPEED] * tilt_speed

    # Limb coordination: exp(-std(hip_vels)); gracefully handle single joint
    hip_vel_std = jnp.where(
        action.shape[0] > 1,
        jnp.std(sensors.hip_velocities),
        jnp.float32(0.0),
    )
    r += w[_W_LIMB_COORDINATION] * jnp.exp(-hip_vel_std)

    jerk = jnp.mean(jnp.abs(action - prev_action))
    r += w[_W_NERVOSITY] * jerk
    r += w[_W_SMOOTH]    * jnp.exp(-jerk)

    r += w[_W_VERTICAL_VELOCITY]  * vz
    r += w[_W_LATERAL_VELOCITY]   * jnp.abs(vy)

    hip_angle_std = jnp.where(
        action.shape[0] > 1,
        jnp.std(sensors.hip_angles),
        jnp.float32(0.0),
    )
    r += w[_W_JOINT_RANGE] * hip_angle_std

    # ---- Standing-specific terms -------------------------------------------
    h_target = jnp.where(init_z > jnp.float32(0.0), init_z, sensors.torso_height)
    r += w[_W_HEIGHT_TARGET] * jnp.exp(jnp.float32(-25.0) * (sensors.torso_height - h_target) ** 2)

    r -= w[_W_TILT_PENALTY]      * (jnp.float32(1.0) - upright) ** 2
    r -= w[_W_TILT_RATE_PENALTY] * tilt_speed ** 2

    all_planted = (sensors.n_contacts >= sensors.n_feet_total).astype(jnp.float32)
    r += w[_W_ALL_FEET_PLANTED] * all_planted

    r -= w[_W_VERTICAL_VEL_PENALTY]   * vz ** 2
    r -= w[_W_HORIZONTAL_VEL_PENALTY] * (vx ** 2 + vy ** 2)

    # Fall penalty fires exactly once on the transition step
    r -= w[_W_FALL_PENALTY] * fell.astype(jnp.float32)

    return r.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  reward.py (MJX) — debug mode")
    print("=" * 60)

    # 1. numpy round-trip (unchanged)
    print("\n[1] RewardWeights round-trip\n")
    rw  = RewardWeights()
    v   = rw.to_vector()
    rw2 = RewardWeights.from_vector(v)
    assert rw == rw2
    print(f"  fields : {len(v)}   OK")
    assert len(v) == 23, f"expected 23 fields, got {len(v)}"

    # 2. Index constants stay in sync with field order
    print("\n[2] Index constants vs field order\n")
    names = RewardWeights.field_names()
    idx_map = {
        _W_FORWARD_VELOCITY: "forward_velocity",
        _W_LATERAL_DRIFT:    "lateral_drift",
        _W_FALL_PENALTY:     "fall_penalty",
        _W_ALL_FEET_PLANTED: "all_feet_planted_bonus",
        _W_HORIZONTAL_VEL_PENALTY: "horizontal_velocity_penalty",
    }
    for idx, name in idx_map.items():
        assert names[idx] == name, f"slot {idx}: expected {name!r}, got {names[idx]!r}"
    print("  All spot-checked index constants match field order: OK")

    # 3. JAX reward sanity
    print("\n[3] compute_step_reward_jax sanity (no MuJoCo)\n")
    try:
        import jax.numpy as jnp
        w_vec = rw.to_jax_vector()
        fake = JaxSensorReading(
            torso_pos              = jnp.array([0.0, 0.0, 0.3]),
            torso_height           = jnp.float32(0.3),
            torso_orientation      = jnp.array([1.0, 0.0, 0.0, 0.0]),
            torso_velocity         = jnp.array([1.0, -0.1, 0.05]),
            torso_angular_velocity = jnp.array([0.0, 0.0, 0.3]),
            hip_angles             = jnp.array([0.1, -0.1, 0.1, -0.1]),
            hip_velocities         = jnp.array([0.5, -0.4, 0.5, -0.4]),
            n_contacts             = jnp.int32(4),
            n_feet_total           = 4,
        )
        action      = jnp.zeros(4, jnp.float32)
        prev_action = jnp.zeros(4, jnp.float32)
        init_pos    = jnp.array([0.0, 0.0, 0.3])

        r_alive = compute_step_reward_jax(w_vec, fake, action, prev_action, jnp.bool_(False), init_pos)
        r_fell  = compute_step_reward_jax(w_vec, fake, action, prev_action, jnp.bool_(True),  init_pos)
        print(f"  fell=False : r = {float(r_alive):+.5f}")
        print(f"  fell=True  : r = {float(r_fell):+.5f}")
        assert float(r_fell) < float(r_alive), "fall penalty must lower reward"
        print("  Fall penalty lowers reward: OK")

        # Verify JAX and numpy agree (same inputs)
        class _Np:
            torso_velocity         = np.array([1.0, -0.1, 0.05])
            torso_orientation      = np.array([1.0, 0.0, 0.0, 0.0])
            torso_angular_velocity = np.array([0.0, 0.0, 0.3])
            torso_height           = 0.3
            n_contacts             = 4
            n_feet_total           = 4
            hip_velocities         = np.array([0.5, -0.4, 0.5, -0.4])
            hip_angles             = np.array([0.1, -0.1, 0.1, -0.1])
        r_np = compute_step_reward(rw, _Np(), np.zeros(4), np.zeros(4), False, np.array([0.0, 0.0, 0.3]))
        diff = abs(float(r_alive) - r_np)
        print(f"  numpy={r_np:+.5f}  jax={float(r_alive):+.5f}  diff={diff:.2e}")
        assert diff < 1e-4, f"numpy/jax mismatch too large: {diff}"
        print("  numpy ≈ JAX: OK")
    except ImportError:
        print("  JAX not available — skipping JAX reward test")

    print("\nAll reward.py (MJX) checks passed.")
