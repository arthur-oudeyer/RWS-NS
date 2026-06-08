"""
mujoco_env_mjx.py
=================
Functional, vmappable JAX environment backed by MJX physics.

Unlike the Gymnasium-based RobotControllerEnv, every function here is pure:
all mutable state lives in EnvState (a JAX NamedTuple / pytree) and is
returned from each call. This makes the env jit- and vmap-able, enabling
massively parallel rollouts on GPU / Apple Metal.

Architecture
------------
    MJXEnvConfig  — Python dataclass holding the MJX model, pre-computed
                    JAX arrays, and Python-scalar constants.  Built ONCE per
                    individual (CPU side) via build_env_config().

    EnvState      — JAX NamedTuple (pytree) with the per-step mutable state.
                    Vmapped over the batch dimension during training.

    make_env_fns()  — Given an MJXEnvConfig, returns JIT-compiled callables:
                      reset_fn(rng_key)  → (EnvState, obs)
                      step_fn(state, action) → (EnvState, obs, reward, done)
                      make_batch_fns(n_envs) → (batch_reset, batch_step)

Observation / Action spaces
----------------------------
Same layout as mujoco_env.py:
    obs = [ sin(ω·t) × 3 | hip_angles × n_j | hip_vels × n_j |
            torso_quat × 4 | torso_lin_vel × 3 | torso_ang_vel × 3 ]
    act = (n_joints,) Δ-angle in [-1, 1], float32

Contact detection
-----------------
Foot geom IDs are discovered ONCE at build time using the regular mujoco
CPU API (mj_id2name). Inside JAX steps, a vectorised check over the fixed-
size MJX contact buffer identifies which foot geoms are in contact.

Spawn-height correction
-----------------------
Identical to controller_morph.build_model(): the regular mujoco CPU API
runs forward kinematics at rest pose, finds the lowest foot-sphere bottom,
and shifts the torso up if needed. This happens at build time; the corrected
spawn_height is then used inside the JAX reset function.

Install (before first use)
--------------------------
    # MJX  (ships separately from the main mujoco pip package)
    pip install mujoco-mjx

    # JAX — choose ONE backend:
    pip install "jax[cpu]"                  # CPU only (always works)
    pip install jax-metal                   # Apple Silicon (Mac M2/M3)
    pip install "jax[cuda12]"               # CUDA 12 GPU

    On Mac M2 / Metal, JAX defaults to the CPU backend unless jax-metal
    is installed. With jax-metal, jax.default_backend() returns "METAL".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Tuple

import mujoco
import numpy as np

# ---- Lazy MJX import with helpful error -----------------------------------------------
try:
    from mujoco import mjx
    import jax
    import jax.numpy as jnp
    _MJX_AVAILABLE = True
except ImportError as _e:
    _MJX_AVAILABLE = False
    _MJX_IMPORT_ERROR = str(_e)


def _require_mjx() -> None:
    if not _MJX_AVAILABLE:
        raise ImportError(
            "MJX / JAX not available.\n"
            f"  Original error: {_MJX_IMPORT_ERROR}\n\n"
            "Install with:\n"
            "    pip install mujoco-mjx\n"
            "    pip install jax-metal      # Mac M2 / Metal\n"
            "    pip install 'jax[cpu]'     # CPU fallback\n"
        )


def _pick_mjx_device():
    """
    Return the best JAX device that MJX supports.

    MJX supports: CUDA GPU, CPU.
    MJX does NOT support: Metal (Apple M-series) as of mujoco-mjx 3.x.

    On Mac M2 with jax-metal installed, jax.default_backend() == 'METAL'.
    We fall through to CPU in that case so all MJX ops run on CPU.
    On a CUDA machine, GPU is preferred.
    """
    try:
        gpu = jax.devices("gpu")
        if gpu:
            return gpu[0]
    except (RuntimeError, Exception):
        pass
    return jax.devices("cpu")[0]


import config as _cfg
from controller_morph import build_model
from reward import (
    RewardWeights,
    JaxSensorReading,
    compute_step_reward_jax,
)

_PREDICTION_FACTOR = -60.0   # same as mujoco_env.py


# ---------------------------------------------------------------------------
# EnvState — dynamic per-env state (a JAX pytree via NamedTuple)
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """
    All mutable per-environment state.  Every field is a JAX array so the
    whole NamedTuple is a pytree and can be vmapped / JIT-compiled.
    """
    data:               Any   # mjx.Data  — full MJX physics state
    step_idx:           Any   # ()  int32
    sim_time:           Any   # ()  float32  (seconds)
    prev_action:        Any   # (n_joints,) float32
    initial_torso_pos:  Any   # (3,)  float32  — snapshot at reset
    fell:               Any   # ()  bool


# ---------------------------------------------------------------------------
# MJXEnvConfig — static build-time config (NOT a JAX pytree)
# ---------------------------------------------------------------------------

@dataclass
class MJXEnvConfig:
    """
    Immutable environment configuration built once per individual.

    JAX arrays (mjx_model, base_data, …) are closed over by the functions
    returned from make_env_fns() — JAX traces through them at JIT time.

    Python scalar fields (n_joints, physics_steps_per_action, …) are used
    directly in Python expressions inside the closures (e.g., as the
    `length` argument of jax.lax.scan, or as slice bounds), so they must
    stay as plain Python ints/floats.
    """
    # ---- MJX objects -------------------------------------------------------
    mjx_model:   Any   # mjx.Model  — compiled JAX physics model
    base_data:   Any   # mjx.Data   — zero-initialised state for reset

    # ---- Pre-computed JAX arrays (same for all envs in a generation) -------
    foot_geom_ids:     Any   # (n_feet,)   int32  — geom IDs of foot spheres
    rest_angles:       Any   # (n_joints,) float32
    ctrl_low_init:     Any   # (n_joints,) float32  — for reset jitter clip
    ctrl_high_init:    Any   # (n_joints,) float32
    ctrl_low:          Any   # (n_joints,) float32  — step ctrl clip
    ctrl_high:         Any   # (n_joints,) float32
    ctrl_to_qpos:      Any   # (n_joints,) int32
    reward_weights_vec: Any  # (23,)       float32

    # ---- Python scalars (static at JIT compile time) -----------------------
    n_joints:                 int
    n_feet:                   int
    nconmax:                  int   # mj_model.nconmax
    spawn_height:             float
    timestep:                 float
    physics_steps_per_action: int   # Python int! used as lax.scan length
    max_steps:                int
    fall_height:              float
    delta_scale:              float
    obs_dim:                  int


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_env_config(
    reward_weights:    Optional[RewardWeights] = None,
    episode_duration:  float = 5.0,
    control_frequency: int   = 20,
    fall_height:       float = 0.3,
    prediction_factor: Optional[float] = None,
) -> MJXEnvConfig:
    """
    Build an MJXEnvConfig for the static morphology.

    Performs all CPU-side setup (model compilation, spawn-height correction,
    geom-ID discovery) and returns a config whose closures can be JIT-
    compiled and vmapped.

    Parameters
    ----------
    reward_weights    : evolved reward weights for this individual.
    episode_duration  : episode length in seconds.
    control_frequency : policy firing rate (Hz).
    fall_height       : torso z below which the episode terminates.
    """
    _require_mjx()

    # ---- Select the MJX-compatible JAX device and set it as default --------
    # MJX does NOT support the Metal backend (Apple M-series). If jax-metal is
    # installed, the JAX default device becomes METAL, which causes MJX to fail
    # with "Unsupported device". We detect this and redirect to CPU (or CUDA GPU
    # if available). Setting jax_default_device globally ensures that
    # jax.random.PRNGKey, jnp.zeros, etc. all create arrays on the right device,
    # which in turn makes the JIT-compiled env functions run on that device.
    _mjx_dev = _pick_mjx_device()
    jax.config.update("jax_default_device", _mjx_dev)

    rw = reward_weights or RewardWeights()

    # ---- Build CPU-side model (includes spawn-height correction) -----------
    mj_model, morph = build_model()

    n_joints = morph.n_joints
    _body_part_parents = {bp.parent_leg_idx for bp in morph.body_parts}
    n_feet = sum(1 for i in range(len(morph.legs)) if i not in _body_part_parents)

    timestep = float(mj_model.opt.timestep)
    physics_steps_per_action = max(
        1, int(round(1.0 / (control_frequency * timestep)))
    )
    # Action → joint-angle delta per control tick. Larger |factor| = bigger,
    # faster joint moves (and more physics instability). Default mirrors
    # mujoco_env.py (-60); pass a smaller magnitude to slow the robot down.
    pf = _PREDICTION_FACTOR if prediction_factor is None else float(prediction_factor)
    delta_scale = pf / float(control_frequency)
    max_steps   = int(episode_duration * control_frequency)

    # ---- Discover foot geom IDs at compile time ----------------------------
    # Geom names "footN_geom" come from Morphology/morphology.py.
    # We record their integer IDs here so the JAX contact check never needs
    # a name lookup at step time.
    foot_geom_ids_np = []
    for gi in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if name and name.startswith("foot") and name.endswith("_geom"):
            foot_geom_ids_np.append(gi)

    if not foot_geom_ids_np:
        # Fall back: any geom whose name contains "foot"
        for gi in range(mj_model.ngeom):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if name and "foot" in name:
                foot_geom_ids_np.append(gi)

    # ---- Build rest_angles and ctrl range arrays (qpos order) --------------
    _jnt_info: dict[str, tuple] = {}
    for _li, _leg in enumerate(morph.legs):
        for _ji, _jd in enumerate(_leg.joints):
            _jname = (f"hip{_li + 1}" if len(_leg.joints) == 1
                      else f"leg{_li + 1}_j{_ji + 1}")
            _jnt_info[_jname] = (_jd.rest_angle, _jd.ctrl_range[0], _jd.ctrl_range[1])

    rest_angles_np   = np.zeros(n_joints, dtype=np.float32)
    ctrl_low_init_np = np.zeros(n_joints, dtype=np.float32)
    ctrl_high_init_np= np.zeros(n_joints, dtype=np.float32)
    _act_name_to_qi: dict[str, int] = {}

    for _qi in range(n_joints):
        _jid   = _qi + 1  # joint 0 is the freejoint "root"
        _jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, _jid)
        if _jname and _jname in _jnt_info:
            _ra, _cl, _ch = _jnt_info[_jname]
            rest_angles_np[_qi]    = float(_ra)
            ctrl_low_init_np[_qi]  = float(_cl)
            ctrl_high_init_np[_qi] = float(_ch)
            _act_name_to_qi[f"servo_{_jname}"] = _qi

    # ctrl_to_qpos: for each actuator slot, which qpos index does it drive?
    ctrl_to_qpos_np = np.arange(n_joints, dtype=np.int32)
    for _ai in range(mj_model.nu):
        _aname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, _ai)
        if _aname and _aname in _act_name_to_qi:
            ctrl_to_qpos_np[_ai] = _act_name_to_qi[_aname]

    ctrl_low_np  = mj_model.actuator_ctrlrange[:, 0].astype(np.float32)
    ctrl_high_np = mj_model.actuator_ctrlrange[:, 1].astype(np.float32)

    # ---- Create MJX model and zero-state base_data -------------------------
    # ---- Limit collision pairs before MJX conversion -------------------------
    # Default: all 25 geoms have contype=1/conaffinity=1 → 238 potential pairs.
    # MJX pre-allocates a contact buffer for ALL potential pairs (static shapes)
    # and processes them EVERY physics step, even when empty.  For locomotion
    # training, only foot↔floor contacts are needed:
    #   - Fall detection: torso height check (no contact needed)
    #   - Reward: foot-contact count (foot↔floor only)
    # Disabling all non-foot geom collisions reduces potential pairs from 238 to
    # ~11 (one per foot), giving ~5-20× speedup in the contact-resolution step.
    for _gi in range(mj_model.ngeom):
        _name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, _gi)
        if _name in ("floor",):
            pass   # keep floor as-is (contype=1, conaffinity=1)
        elif _name and _name.startswith("foot") and _name.endswith("_geom"):
            # Feet: only collide with floor
            mj_model.geom_contype[_gi]     = 1
            mj_model.geom_conaffinity[_gi] = 0
        elif _name and _name.endswith("torso_geom"):
            # Torso: collide with floor only (prevents passing through ground)
            mj_model.geom_contype[_gi]     = 1
            mj_model.geom_conaffinity[_gi] = 0
        else:
            # All other geoms (legs, body parts, origin_tile): no collisions
            mj_model.geom_contype[_gi]     = 0
            mj_model.geom_conaffinity[_gi] = 0

    mx = mjx.put_model(mj_model, device=_mjx_dev)

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    base_dx = mjx.put_data(mj_model, mj_data, device=_mjx_dev)

    # Read the actual allocated size (should now be ~11 instead of 238).
    nconmax = int(base_dx._impl.contact.geom1.shape[0])

    # ---- Convert everything to JAX arrays ----------------------------------
    obs_dim = 3 + 2 * n_joints + 4 + 6

    return MJXEnvConfig(
        mjx_model            = mx,
        base_data            = base_dx,
        foot_geom_ids        = jnp.array(foot_geom_ids_np, dtype=jnp.int32),
        rest_angles          = jnp.array(rest_angles_np,   dtype=jnp.float32),
        ctrl_low_init        = jnp.array(ctrl_low_init_np, dtype=jnp.float32),
        ctrl_high_init       = jnp.array(ctrl_high_init_np,dtype=jnp.float32),
        ctrl_low             = jnp.array(ctrl_low_np,      dtype=jnp.float32),
        ctrl_high            = jnp.array(ctrl_high_np,     dtype=jnp.float32),
        ctrl_to_qpos         = jnp.array(ctrl_to_qpos_np,  dtype=jnp.int32),
        reward_weights_vec   = rw.to_jax_vector(),
        n_joints             = n_joints,
        n_feet               = n_feet,
        nconmax              = nconmax,
        spawn_height         = float(morph.spawn_height),
        timestep             = timestep,
        physics_steps_per_action = physics_steps_per_action,
        max_steps            = max_steps,
        fall_height          = float(fall_height),
        delta_scale          = float(delta_scale),
        obs_dim              = obs_dim,
    )


# ---------------------------------------------------------------------------
# Internal helpers (pure functions, closed over by make_env_fns)
# ---------------------------------------------------------------------------

def _count_foot_contacts(
    data:      Any,   # mjx.Data
    foot_gids: Any,   # (n_feet,) int32 JAX array
) -> Any:             # () int32 scalar
    """
    Count how many distinct foot geoms are currently in active contact.

    MJX pre-allocates a fixed contact buffer. In MuJoCo 3.x, data.ncon equals
    the buffer capacity (not active count). We use contact.dist <= 0 to identify
    genuinely penetrating contacts, matching the original mujoco_env.py logic.

    Access via data._impl (non-deprecated MJX API in mujoco-mjx 3.x).
    """
    contact = data._impl.contact

    # Active slot: penetration distance is negative (geoms overlap)
    active = contact.dist <= jnp.float32(0.0)   # (nconmax,)

    g1 = contact.geom1   # (nconmax,)
    g2 = contact.geom2   # (nconmax,)

    # (nconmax, n_feet): does each contact slot involve each foot geom?
    g1_match = g1[:, None] == foot_gids[None, :]   # (nconmax, n_feet)
    g2_match = g2[:, None] == foot_gids[None, :]   # (nconmax, n_feet)

    # Slot involves a foot AND is actively penetrating
    foot_active = (g1_match | g2_match) & active[:, None]   # (nconmax, n_feet)

    # For each foot, is it in ANY active contact?
    per_foot = jnp.any(foot_active, axis=0)                  # (n_feet,)
    return jnp.sum(per_foot.astype(jnp.int32))


# Numerical safety rails for physics divergence (see _step / _step_rw).
# MJX integration can blow up to inf/NaN for an unlucky reward/action combo;
# these bands only bite on pathological values (normal rewards ~±10, obs ~±10)
# and keep a single bad env from poisoning GAE / the optimiser with NaNs.
_REWARD_CLIP = 100.0
_OBS_CLIP    = 100.0


def _read_sensors(
    data:      Any,   # mjx.Data
    n_joints:  int,   # Python int
    n_feet:    int,   # Python int
    foot_gids: Any,   # (n_feet,) int32 JAX array
) -> JaxSensorReading:
    """Extract JaxSensorReading from mjx.Data. Pure function."""
    qpos = data.qpos.astype(jnp.float32)
    qvel = data.qvel.astype(jnp.float32)

    n_contacts = _count_foot_contacts(data, foot_gids)

    return JaxSensorReading(
        torso_pos              = qpos[0:3],
        torso_height           = qpos[2],
        torso_orientation      = qpos[3:7],      # [w, x, y, z]
        torso_velocity         = qvel[0:3],
        torso_angular_velocity = qvel[3:6],
        hip_angles             = qpos[7 : 7 + n_joints],
        hip_velocities         = qvel[6 : 6 + n_joints],
        n_contacts             = n_contacts,
        n_feet_total           = n_feet,         # Python int, stays static
    )


def _build_obs(state: EnvState, n_joints: int) -> Any:
    """Build the flat observation vector from EnvState. Pure function."""
    t = state.sim_time
    clocks = jnp.array([
        jnp.sin(t * jnp.float32(1.0  / np.pi)),
        jnp.sin(t * jnp.float32(5.0  / np.pi)),
        jnp.sin(t * jnp.float32(15.0 / np.pi)),
    ], dtype=jnp.float32)

    qpos = state.data.qpos.astype(jnp.float32)
    qvel = state.data.qvel.astype(jnp.float32)

    return jnp.concatenate([
        clocks,                       # (3,)
        qpos[7 : 7 + n_joints],       # (n_joints,) hip angles
        qvel[6 : 6 + n_joints],       # (n_joints,) hip velocities
        qpos[3:7],                    # (4,)  torso quat
        qvel[0:3],                    # (3,)  torso lin vel
        qvel[3:6],                    # (3,)  torso ang vel
    ], axis=0).astype(jnp.float32)


# ---------------------------------------------------------------------------
# make_env_fns — entry point
# ---------------------------------------------------------------------------

def make_env_fns(
    cfg: MJXEnvConfig,
) -> Tuple[Callable, Callable, Callable]:
    """
    Return JIT-compiled single-env functions plus a factory for batch fns.

    Returns
    -------
    reset_fn(rng_key) → (EnvState, obs)
        rng_key : (2,) uint32 PRNGKey

    step_fn(state, action) → (EnvState, obs, reward, done)
        state  : EnvState
        action : (n_joints,) float32

    make_batch_fns(n_envs) → (batch_reset, batch_step)
        batch_reset(keys)           : keys shape (n_envs, 2)
        batch_step(states, actions) : states batched, actions (n_envs, n_joints)
    """
    _require_mjx()

    # ---- Capture all config values as local names -------------------------
    # JAX arrays: closed over and traced at JIT compile time.
    # Python scalars: used directly (e.g. as lax.scan length), never traced.
    mx           = cfg.mjx_model
    base_dx      = cfg.base_data
    foot_gids    = cfg.foot_geom_ids
    rest_angles  = cfg.rest_angles
    cl_init      = cfg.ctrl_low_init
    ch_init      = cfg.ctrl_high_init
    ctrl_low     = cfg.ctrl_low
    ctrl_high    = cfg.ctrl_high
    c2q          = cfg.ctrl_to_qpos
    rw_vec       = cfg.reward_weights_vec

    n_joints     = cfg.n_joints               # Python int
    n_feet       = cfg.n_feet                 # Python int
    spawn_h      = jnp.float32(cfg.spawn_height)
    n_substeps   = cfg.physics_steps_per_action  # Python int → lax.scan length
    max_steps    = cfg.max_steps              # Python int
    fall_h       = jnp.float32(cfg.fall_height)
    dscale       = jnp.float32(cfg.delta_scale)
    timestep     = jnp.float32(cfg.timestep)

    # ---- reset  ------------------------------------------------------------

    def _reset(rng_key: Any) -> Tuple[EnvState, Any]:
        key, subkey = jax.random.split(rng_key)
        jitter = jax.random.uniform(
            subkey, (n_joints,), minval=-0.05, maxval=0.05, dtype=jnp.float32
        )
        init_joints = jnp.clip(rest_angles + jitter, cl_init, ch_init)

        # Set spawn height and rest-angle joints on the zeroed base state.
        qpos = base_dx.qpos.at[2].set(spawn_h)
        qpos = qpos.at[7 : 7 + n_joints].set(init_joints)
        data = base_dx.replace(qpos=qpos)

        # Forward kinematics so geom_xpos / sensor outputs are valid.
        data = mjx.forward(mx, data)

        state = EnvState(
            data              = data,
            step_idx          = jnp.int32(0),
            sim_time          = jnp.float32(0.0),
            prev_action       = jnp.zeros(n_joints, jnp.float32),
            initial_torso_pos = data.qpos[0:3].astype(jnp.float32),
            fell              = jnp.bool_(False),
        )
        obs = _build_obs(state, n_joints)
        return state, obs

    # ---- step  -------------------------------------------------------------

    def _step(
        state:  EnvState,
        action: Any,          # (n_joints,) float32
    ) -> Tuple[EnvState, Any, Any, Any]:
        action = jnp.clip(action.astype(jnp.float32), jnp.float32(-1.0), jnp.float32(1.0))

        # Read pre-step hip angles to compute absolute target position.
        sensors_pre = _read_sensors(state.data, n_joints, n_feet, foot_gids)
        target_qpos = sensors_pre.hip_angles + dscale * action

        # Reindex from qpos order → ctrl order (identity for QUADRIPOD).
        target_ctrl = target_qpos[c2q]
        target_ctrl = jnp.clip(target_ctrl, ctrl_low, ctrl_high)

        data = state.data.replace(ctrl=target_ctrl)

        # Step physics n_substeps times.
        # n_substeps is a Python int so lax.scan unrolls statically.
        def _substep(d: Any, _: Any) -> Tuple[Any, None]:
            return mjx.step(mx, d), None

        data, _ = jax.lax.scan(_substep, data, None, length=n_substeps)

        new_sim_time  = state.sim_time  + timestep * jnp.float32(n_substeps)
        new_step_idx  = state.step_idx  + jnp.int32(1)

        sensors = _read_sensors(data, n_joints, n_feet, foot_gids)

        # Check fall both before and after physics: torso collision can push
        # the body back above fall_height in one step, so pre-physics position
        # must also be checked to avoid "healing" a fallen state.
        # Divergence guard (see _step_rw): non-finite physics → terminate + reset,
        # and keep reward/obs finite.
        blew_up         = ~(jnp.all(jnp.isfinite(data.qpos))
                            & jnp.all(jnp.isfinite(data.qvel)))
        fell_now        = ((sensors.torso_height < fall_h)
                           | (sensors_pre.torso_height < fall_h) | blew_up)
        fell_transition = fell_now & ~state.fell     # fires exactly once
        fell            = state.fell | fell_now

        reward = compute_step_reward_jax(
            rw_vec, sensors, action, state.prev_action,
            fell_transition, state.initial_torso_pos,
        )
        reward = jnp.clip(
            jnp.nan_to_num(reward, nan=0.0, posinf=_REWARD_CLIP, neginf=-_REWARD_CLIP),
            -_REWARD_CLIP, _REWARD_CLIP,
        )

        terminated = fell
        truncated  = new_step_idx >= jnp.int32(max_steps)
        done       = terminated | truncated

        new_state = EnvState(
            data              = data,
            step_idx          = new_step_idx,
            sim_time          = new_sim_time,
            prev_action       = action,
            initial_torso_pos = state.initial_torso_pos,
            fell              = fell,
        )
        obs = jnp.nan_to_num(_build_obs(new_state, n_joints), nan=0.0,
                             posinf=_OBS_CLIP, neginf=-_OBS_CLIP)
        return new_state, obs, reward, done

    # ---- JIT-compile single-env versions -----------------------------------
    reset_fn = jax.jit(_reset)
    step_fn  = jax.jit(_step)

    # ---- Batch factory (Phase 2) -------------------------------------------
    def make_batch_fns(n_envs: int) -> Tuple[Callable, Callable]:
        """
        Return vmapped + JIT-compiled batch functions.

        batch_reset(keys)           : keys (n_envs, 2) → (batched_state, obs batch)
        batch_step(states, actions) : batched state + (n_envs, n_joints) actions
                                      → (batched_state, obs, rewards, dones)
        """
        batch_reset = jax.jit(jax.vmap(_reset, in_axes=0))
        batch_step  = jax.jit(jax.vmap(_step,  in_axes=(0, 0)))
        return batch_reset, batch_step

    return reset_fn, step_fn, make_batch_fns


# ---------------------------------------------------------------------------
# Fast-reset factory (no mjx.forward — for use inside training lax.scan)
# ---------------------------------------------------------------------------

def make_fast_reset_fn(cfg: MJXEnvConfig, n_envs: int) -> Any:
    """
    Return a JIT-compiled vmapped reset that skips mjx.forward.

    The full _reset calls mjx.forward to update geom positions and sensors
    before returning.  Inside a lax.scan this runs for every env at every
    step — even when no episode is done — because jnp.where always evaluates
    both branches.  Skipping mjx.forward here is safe: the immediately
    following mjx.step handles forward kinematics anyway.

    Savings: ~10× cheaper per call vs the full reset.  Matters because the
    auto-reset inside the rollout scan touches this path every single step.

    Parameters
    ----------
    cfg    : MJXEnvConfig for this individual.
    n_envs : parallel-environment batch size.

    Returns
    -------
    fast_batch_reset(keys) : keys (n_envs, 2) → (batched EnvState, obs batch)
    """
    _require_mjx()

    base_dx     = cfg.base_data
    rest_angles = cfg.rest_angles
    cl_init     = cfg.ctrl_low_init
    ch_init     = cfg.ctrl_high_init
    spawn_h     = jnp.float32(cfg.spawn_height)
    n_joints    = cfg.n_joints

    def _fast_reset(rng_key: Any) -> Tuple[EnvState, Any]:
        _, subkey = jax.random.split(rng_key)
        jitter = jax.random.uniform(
            subkey, (n_joints,), minval=-0.05, maxval=0.05, dtype=jnp.float32
        )
        init_joints = jnp.clip(rest_angles + jitter, cl_init, ch_init)
        qpos = base_dx.qpos.at[2].set(spawn_h)
        qpos = qpos.at[7 : 7 + n_joints].set(init_joints)
        data = base_dx.replace(qpos=qpos)
        # No mjx.forward — the next mjx.step runs full forward kinematics
        state = EnvState(
            data              = data,
            step_idx          = jnp.int32(0),
            sim_time          = jnp.float32(0.0),
            prev_action       = jnp.zeros(n_joints, jnp.float32),
            initial_torso_pos = data.qpos[0:3].astype(jnp.float32),
            fell              = jnp.bool_(False),
        )
        obs = _build_obs(state, n_joints)
        return state, obs

    return jax.jit(jax.vmap(_fast_reset, in_axes=0))


# ---------------------------------------------------------------------------
# Reward-agnostic batch functions (rw_vec as runtime arg — no recompile per individual)
# ---------------------------------------------------------------------------

def make_reward_agnostic_batch_fns(
    cfg:    MJXEnvConfig,
    n_envs: int,
) -> Tuple[Any, Any, Any]:
    """
    Return batch env functions where reward_weights_vec is a RUNTIME argument.

    This is the critical optimisation for multi-individual evolution: because
    reward weights are passed at call time (not baked into the JIT closure),
    the compiled XLA kernel can be reused across ALL individuals.  Without
    this, every individual triggers a fresh 30-60 s JIT recompilation.

    Returns
    -------
    batch_reset(keys)                       → (batched_state, obs)
        Full reset with mjx.forward.  keys shape (n_envs, 2).
    batch_step_rw(states, actions, rw_vec)  → (states, obs, rewards, dones)
        rw_vec : (reward_dim,) float32 — NOT vmapped (broadcast over envs).
    fast_batch_reset(keys)                  → (batched_state, obs)
        Fast reset without mjx.forward.
    """
    _require_mjx()

    mx          = cfg.mjx_model
    base_dx     = cfg.base_data
    foot_gids   = cfg.foot_geom_ids
    rest_angles = cfg.rest_angles
    cl_init     = cfg.ctrl_low_init
    ch_init     = cfg.ctrl_high_init
    ctrl_low    = cfg.ctrl_low
    ctrl_high   = cfg.ctrl_high
    c2q         = cfg.ctrl_to_qpos

    n_joints    = cfg.n_joints
    n_feet      = cfg.n_feet
    spawn_h     = jnp.float32(cfg.spawn_height)
    n_substeps  = cfg.physics_steps_per_action
    max_steps_  = cfg.max_steps
    fall_h      = jnp.float32(cfg.fall_height)
    dscale      = jnp.float32(cfg.delta_scale)
    timestep_   = jnp.float32(cfg.timestep)

    # ---- full reset (with mjx.forward) ------------------------------------
    def _reset(rng_key: Any) -> Tuple[EnvState, Any]:
        key, subkey = jax.random.split(rng_key)
        jitter = jax.random.uniform(
            subkey, (n_joints,), minval=-0.05, maxval=0.05, dtype=jnp.float32
        )
        init_joints = jnp.clip(rest_angles + jitter, cl_init, ch_init)
        qpos = base_dx.qpos.at[2].set(spawn_h)
        qpos = qpos.at[7 : 7 + n_joints].set(init_joints)
        data = base_dx.replace(qpos=qpos)
        data = mjx.forward(mx, data)
        state = EnvState(
            data              = data,
            step_idx          = jnp.int32(0),
            sim_time          = jnp.float32(0.0),
            prev_action       = jnp.zeros(n_joints, jnp.float32),
            initial_torso_pos = data.qpos[0:3].astype(jnp.float32),
            fell              = jnp.bool_(False),
        )
        return state, _build_obs(state, n_joints)

    # ---- fast reset (no mjx.forward) --------------------------------------
    def _fast_reset(rng_key: Any) -> Tuple[EnvState, Any]:
        _, subkey = jax.random.split(rng_key)
        jitter = jax.random.uniform(
            subkey, (n_joints,), minval=-0.05, maxval=0.05, dtype=jnp.float32
        )
        init_joints = jnp.clip(rest_angles + jitter, cl_init, ch_init)
        qpos = base_dx.qpos.at[2].set(spawn_h)
        qpos = qpos.at[7 : 7 + n_joints].set(init_joints)
        data = base_dx.replace(qpos=qpos)
        state = EnvState(
            data              = data,
            step_idx          = jnp.int32(0),
            sim_time          = jnp.float32(0.0),
            prev_action       = jnp.zeros(n_joints, jnp.float32),
            initial_torso_pos = data.qpos[0:3].astype(jnp.float32),
            fell              = jnp.bool_(False),
        )
        return state, _build_obs(state, n_joints)

    # ---- step with rw_vec as explicit arg ---------------------------------
    def _step_rw(
        state:  EnvState,
        action: Any,           # (n_joints,) float32
        rw_vec: Any,           # (reward_dim,) float32 — broadcast, not vmapped
    ) -> Tuple[EnvState, Any, Any, Any]:
        action = jnp.clip(action.astype(jnp.float32), jnp.float32(-1.0), jnp.float32(1.0))
        sensors_pre  = _read_sensors(state.data, n_joints, n_feet, foot_gids)
        target_qpos  = sensors_pre.hip_angles + dscale * action
        target_ctrl  = target_qpos[c2q]
        target_ctrl  = jnp.clip(target_ctrl, ctrl_low, ctrl_high)
        data         = state.data.replace(ctrl=target_ctrl)

        def _substep(d, _):
            return mjx.step(mx, d), None
        data, _ = jax.lax.scan(_substep, data, None, length=n_substeps)

        new_sim_time = state.sim_time + timestep_ * jnp.float32(n_substeps)
        new_step_idx = state.step_idx + jnp.int32(1)
        sensors      = _read_sensors(data, n_joints, n_feet, foot_gids)

        # Physics-divergence guard: if MJX integration blew up (qpos/qvel become
        # non-finite for an unlucky reward/action combo), terminate like a fall
        # so the env auto-resets, and keep reward/obs finite below so the NaN
        # never reaches GAE / the optimiser — one NaN there poisons every param
        # and stalls the GPU on denormals.
        blew_up         = ~(jnp.all(jnp.isfinite(data.qpos))
                            & jnp.all(jnp.isfinite(data.qvel)))
        fell_now        = (sensors.torso_height < fall_h) | blew_up
        fell_transition = fell_now & ~state.fell
        fell            = state.fell | fell_now

        reward = compute_step_reward_jax(
            rw_vec, sensors, action, state.prev_action,
            fell_transition, state.initial_torso_pos,
        )
        reward = jnp.clip(
            jnp.nan_to_num(reward, nan=0.0, posinf=_REWARD_CLIP, neginf=-_REWARD_CLIP),
            -_REWARD_CLIP, _REWARD_CLIP,
        )
        terminated = fell
        truncated  = new_step_idx >= jnp.int32(max_steps_)
        done       = terminated | truncated

        new_state = EnvState(
            data              = data,
            step_idx          = new_step_idx,
            sim_time          = new_sim_time,
            prev_action       = action,
            initial_torso_pos = state.initial_torso_pos,
            fell              = fell,
        )
        obs = jnp.nan_to_num(_build_obs(new_state, n_joints), nan=0.0,
                             posinf=_OBS_CLIP, neginf=-_OBS_CLIP)
        return new_state, obs, reward, done

    # vmap: states and actions over envs, rw_vec broadcast (in_axes=None)
    batch_reset     = jax.jit(jax.vmap(_reset,     in_axes=0))
    batch_step_rw   = jax.jit(jax.vmap(_step_rw,   in_axes=(0, 0, None)))
    fast_batch_reset = jax.jit(jax.vmap(_fast_reset, in_axes=0))

    return batch_reset, batch_step_rw, fast_batch_reset


# ---------------------------------------------------------------------------
# Debug / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  mujoco_env_mjx.py — debug mode")
    print("=" * 60)

    _require_mjx()

    print(f"\n  JAX backend : {jax.default_backend()}")
    print(f"  JAX devices : {jax.devices()}")

    # 1. Build config
    print("\n[1] Build MJXEnvConfig\n")
    cfg = build_env_config(episode_duration=2.0)
    print(f"  n_joints  : {cfg.n_joints}")
    print(f"  n_feet    : {cfg.n_feet}")
    print(f"  nconmax   : {cfg.nconmax}")
    print(f"  n_substeps: {cfg.physics_steps_per_action}")
    print(f"  max_steps : {cfg.max_steps}")
    print(f"  spawn_h   : {cfg.spawn_height:.4f}")
    print(f"  foot_gids : {cfg.foot_geom_ids.tolist()}")
    print(f"  obs_dim   : {cfg.obs_dim}")

    # 2. Make env functions
    print("\n[2] make_env_fns\n")
    reset_fn, step_fn, make_batch_fns = make_env_fns(cfg)
    print("  Functions created: OK")

    # 3. Single reset
    print("\n[3] Single reset (JIT first call)\n")
    key = jax.random.PRNGKey(0)
    state, obs = reset_fn(key)
    print(f"  obs shape : {obs.shape}  (expected {cfg.obs_dim})")
    print(f"  obs finite: {bool(jnp.isfinite(obs).all())}")
    assert obs.shape == (cfg.obs_dim,), f"obs shape mismatch: {obs.shape}"
    assert jnp.isfinite(obs).all(),     "obs contains non-finite values"
    print(f"  step_idx  : {int(state.step_idx)}")
    print(f"  spawn_h   : {float(state.data.qpos[2]):.4f}")
    print("  Reset: OK")

    # 4. Single step
    print("\n[4] Single step (JIT first call)\n")
    action = jnp.zeros(cfg.n_joints, dtype=jnp.float32)
    state2, obs2, reward, done = step_fn(state, action)
    print(f"  obs shape : {obs2.shape}")
    print(f"  reward    : {float(reward):+.4f}")
    print(f"  done      : {bool(done)}")
    print(f"  step_idx  : {int(state2.step_idx)}")
    assert obs2.shape == (cfg.obs_dim,)
    assert jnp.isfinite(obs2).all()
    assert jnp.isfinite(reward)
    print("  Step: OK")

    # 5. Full episode rollout
    print("\n[5] Full episode rollout (100 random-action steps)\n")
    key2 = jax.random.PRNGKey(42)
    state, obs = reset_fn(key2)
    rng = jax.random.PRNGKey(7)
    total_r = 0.0
    n_done  = 0
    for i in range(100):
        rng, subk = jax.random.split(rng)
        action = jax.random.uniform(subk, (cfg.n_joints,), minval=-1.0, maxval=1.0)
        state, obs, r, done = step_fn(state, action)
        total_r += float(r)
        if done:
            n_done += 1
            break
    print(f"  steps ran    : {int(state.step_idx)}")
    print(f"  total reward : {total_r:+.4f}")
    print(f"  fell         : {bool(state.fell)}")
    print("  Episode rollout: OK")

    # 6. Batch reset + step (vmap smoke test)
    print("\n[6] Batch vmap smoke test (n_envs=4)\n")
    batch_reset, batch_step = make_batch_fns(4)
    keys = jax.random.split(jax.random.PRNGKey(99), 4)
    states_b, obs_b = batch_reset(keys)
    print(f"  batch obs shape : {obs_b.shape}   (expected (4, {cfg.obs_dim}))")
    assert obs_b.shape == (4, cfg.obs_dim)

    actions_b = jnp.zeros((4, cfg.n_joints), dtype=jnp.float32)
    states_b2, obs_b2, rews_b, dones_b = batch_step(states_b, actions_b)
    print(f"  batch rews shape: {rews_b.shape}  (expected (4,))")
    print(f"  rewards         : {rews_b.tolist()}")
    assert rews_b.shape == (4,)
    assert jnp.isfinite(rews_b).all()
    print("  Batch vmap: OK")

    print("\nAll mujoco_env_mjx.py checks passed.")
