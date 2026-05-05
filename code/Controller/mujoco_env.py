"""
mujoco_env.py
=============
Gymnasium environment that wraps the static QUADRIPOD body and is
parameterised by a `RewardWeights` vector. Each individual = a fresh env
with its own weight vector, and PPO is trained against the resulting
shaped reward.

Observation space (1-D Box, dtype=float32)
------------------------------------------
    [ sin(ω·t) for ω in (1, 5, 15) ]                     (3,)
    [ hip_angles ]                                        (n_joints,)
    [ hip_velocities ]                                    (n_joints,)
    [ torso_quat (w, x, y, z) ]                           (4,)
    [ torso_lin_vel (3) + torso_ang_vel (3) ]             (6,)

Action space
------------
    Box(low=-1, high=1, shape=(n_joints,), dtype=float32)
The action is a Δ-angle (scaled by `_PREDICTION_FACTOR`) added to the
current hip angles, then clipped to actuator ctrl_range. This mirrors the
convention of `proto/Robot/simple_brain.get_simplebrain_controller`, so
the actuator stiffness / range from `Morphology/morphology.py` makes sense.

Termination
-----------
- truncation : episode time exceeds `episode_duration`
- termination : torso z falls below `fall_height`

Render
------
`render(mode="rgb_array")` returns a numpy frame (H, W, 3). The recorder
in `video_renderer.py` uses this for the rollout MP4.

Debug
-----
Run this file to (1) validate against `gymnasium.utils.env_checker` and
(2) execute a 1000-step random-action episode, printing return + reason
for termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import config as _cfg
from controller_morph import build_model
from reward import RewardWeights, compute_step_reward


# Δ-angle scale: action ∈ [-1, 1] becomes a small target-angle delta.
# We divide by control_frequency below so the per-step delta is reasonable
# regardless of how often the policy fires.
_PREDICTION_FACTOR = -20.0


# ---------------------------------------------------------------------------
# Sensor reading — what compute_step_reward and the obs vector consume
# ---------------------------------------------------------------------------

@dataclass
class RobotSensorReading:
    torso_pos:              np.ndarray   # (3,)
    torso_height:           float
    torso_orientation:      np.ndarray   # (4,)  [w, x, y, z]
    torso_velocity:         np.ndarray   # (3,)
    torso_angular_velocity: np.ndarray   # (3,)
    hip_angles:             np.ndarray   # (n_joints,)
    hip_velocities:         np.ndarray   # (n_joints,)
    n_contacts:             int          # number of feet in ground contact
    n_feet_total:           int          # total number of feet (for airborne fraction)


def _read_sensors(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    n_joints:    int,
    n_feet_total: int = 4,
) -> RobotSensorReading:
    """Extract a `RobotSensorReading` from the raw MjData arrays."""
    # qpos: [3 pos | 4 quat | n_joints hip angles]
    # qvel: [3 lin | 3 ang  | n_joints hip vels]
    n_contacts = 0
    if data.ncon > 0:
        # We treat any contact involving a foot geom as "foot in contact".
        # Foot geoms are named "footN_geom" in Morphology/morphology.py.
        for i in range(data.ncon):
            c = data.contact[i]
            for gid in (c.geom1, c.geom2):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if name and name.startswith("foot") and name.endswith("_geom"):
                    n_contacts += 1
                    break

    return RobotSensorReading(
        torso_pos              = data.qpos[0:3].copy(),
        torso_height           = float(data.qpos[2]),
        torso_orientation      = data.qpos[3:7].copy(),
        torso_velocity         = data.qvel[0:3].copy(),
        torso_angular_velocity = data.qvel[3:6].copy(),
        hip_angles             = data.qpos[7 : 7 + n_joints].copy(),
        hip_velocities         = data.qvel[6 : 6 + n_joints].copy(),
        n_contacts             = n_contacts,
        n_feet_total           = n_feet_total,
    )


# ---------------------------------------------------------------------------
# RobotControllerEnv
# ---------------------------------------------------------------------------

class RobotControllerEnv(gym.Env):
    """
    Gym env wrapping a fixed-morphology MuJoCo robot. PPO sees:

    obs = [3 clock signals, n_joints hip angles, n_joints hip vels,
           4 torso quat, 3 torso lin vel, 3 torso ang vel]
    act = (n_joints,) Δ-angle in [-1, 1]

    Parameters
    ----------
    reward_weights      : the evolved per-step shaped-reward weights.
    seed                : RNG seed for the env's `np_random`.
    episode_duration    : seconds of simulated time per episode.
    control_frequency   : how often the policy outputs an action (Hz).
                          MuJoCo timestep is read from the model.
    fall_height         : torso z below this terminates the episode.
    render_mode         : "rgb_array" enables `render()`. None = no render.
    render_width/height : image size when render() is called.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        reward_weights:    Optional[RewardWeights] = None,
        seed:              Optional[int]           = None,
        episode_duration:  float                   = 5.0,
        control_frequency: int                     = 20,
        fall_height:       float                   = 0.05,
        render_mode:       Optional[str]           = None,
        render_width:      int                     = 192,
        render_height:     int                     = 192,
        cam1_azimuth:      float                   = _cfg.ExperimentConfig.cam1_azimuth,
        cam1_elevation:    float                   = _cfg.ExperimentConfig.cam1_elevation,
        cam1_distance:     float                   = _cfg.ExperimentConfig.cam1_distance,
        cam1_lookat_z:     float                   = _cfg.ExperimentConfig.cam1_lookat_z,
        cam2_azimuth:      float                   = _cfg.ExperimentConfig.cam2_azimuth,
        cam2_elevation:    float                   = _cfg.ExperimentConfig.cam2_elevation,
        cam2_distance:     float                   = _cfg.ExperimentConfig.cam2_distance,
    ):
        super().__init__()
        self.reward_weights    = reward_weights or RewardWeights()
        self.episode_duration  = float(episode_duration)
        self.control_frequency = int(control_frequency)
        self.fall_height       = float(fall_height)
        self.render_mode       = render_mode
        self._render_width     = render_width
        self._render_height    = render_height

        # Build model once — every reset uses fresh MjData on the same model.
        self._model, self._morph = build_model()
        self._data = mujoco.MjData(self._model)

        self._n_joints = self._morph.n_joints
        self._n_feet   = len(self._morph.legs)
        self._timestep = float(self._model.opt.timestep)
        self._physics_steps_per_action = max(
            1, int(round(1.0 / (self.control_frequency * self._timestep)))
        )
        self._delta_scale = _PREDICTION_FACTOR / float(self.control_frequency)
        self._max_steps   = int(self.episode_duration * self.control_frequency)

        # ctrl_range: (n_joints, 2) numpy array from the actuator block
        self._ctrl_low  = self._model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self._model.actuator_ctrlrange[:, 1].copy()

        # ---- Spaces -----------------------------------------------------------
        # Observation = 3 clocks + 2*n_joints + 4 quat + 6 vel
        obs_dim = 3 + 2 * self._n_joints + 4 + 6
        # Loose bounds — SB3 only needs a Box, the actual values are bounded by
        # joint limits / quaternions / a finite physics horizon.
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (self._n_joints,),
            dtype = np.float32,
        )

        # Internal episode state
        self._step_idx      = 0
        self._sim_time      = 0.0
        self._prev_action   = np.zeros(self._n_joints, dtype=np.float32)
        self._renderer:     Optional[mujoco.Renderer] = None
        self._renderer2:    Optional[mujoco.Renderer] = None

        self._render_camera = mujoco.MjvCamera()
        self._render_camera.azimuth   = cam1_azimuth
        self._render_camera.elevation = cam1_elevation
        self._render_camera.distance  = cam1_distance
        self._render_camera.lookat[:] = [0.0, 0.0, cam1_lookat_z]

        self._render_camera2 = mujoco.MjvCamera()
        self._render_camera2.azimuth   = cam2_azimuth
        self._render_camera2.elevation = cam2_elevation
        self._render_camera2.distance  = cam2_distance
        self._render_camera2.lookat[:] = [0.0, 0.0, 0.2]

        # RNG (gym v1 uses np_random)
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._fell         = False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self._model, self._data)
        # Drop the torso at the computed spawn height (rest-pose feet on floor).
        self._data.qpos[2] = self._morph.spawn_height
        # Tiny initial joint noise so PPO sees a non-degenerate start.
        if self._n_joints > 0:
            jitter = self._np_random.uniform(-0.05, 0.05, size=self._n_joints)
            self._data.qpos[7 : 7 + self._n_joints] = jitter
        mujoco.mj_forward(self._model, self._data)

        self._step_idx    = 0
        self._sim_time    = 0.0
        self._fell        = False
        self._prev_action = np.zeros(self._n_joints, dtype=np.float32)
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        prev_action = self._prev_action  # snapshot before physics step

        # Convert action (Δ-angle proxy in [-1, 1]) to a target hip angle.
        sensors_pre = _read_sensors(self._model, self._data, self._n_joints, self._n_feet)
        target = sensors_pre.hip_angles + self._delta_scale * action
        target = np.clip(target, self._ctrl_low, self._ctrl_high)
        self._data.ctrl[:] = target

        # Step physics N substeps so the policy fires at control_frequency.
        for _ in range(self._physics_steps_per_action):
            mujoco.mj_step(self._model, self._data)
        self._sim_time += self._physics_steps_per_action * self._timestep
        self._step_idx += 1

        sensors  = _read_sensors(self._model, self._data, self._n_joints, self._n_feet)
        fell_now = sensors.torso_height < self.fall_height
        # `fell` is the one-step transition into the fallen state, used by
        # compute_step_reward so the fall_penalty fires exactly once.
        fell_transition = fell_now and not self._fell
        self._fell = self._fell or fell_now

        reward = compute_step_reward(
            self.reward_weights, sensors, action, prev_action, fell_transition
        )
        self._prev_action = action

        terminated = bool(self._fell)
        truncated  = self._step_idx >= self._max_steps

        info = {
            "torso_height": sensors.torso_height,
            "fwd_velocity": float(sensors.torso_velocity[0]),
            "fell":         self._fell,
            "sim_time":     self._sim_time,
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model, height=self._render_height, width=self._render_width
            )
        if self._renderer2 is None:
            self._renderer2 = mujoco.Renderer(
                self._model, height=self._render_height, width=self._render_width
            )
        # Both cameras follow the torso along X so the gait stays centred.
        torso_x = float(self._data.qpos[0])

        cam1 = self._render_camera
        if _cfg.ExperimentConfig.camera_track_torso:
            cam1.lookat[0] = torso_x
        self._renderer.update_scene(self._data, camera=cam1)
        frame1 = self._renderer.render()

        cam2 = self._render_camera2
        if _cfg.ExperimentConfig.camera_track_torso:
            cam2.lookat[0] = torso_x
        self._renderer2.update_scene(self._data, camera=cam2)
        frame2 = self._renderer2.render()

        return np.concatenate([frame1, frame2], axis=1)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._renderer2 is not None:
            self._renderer2.close()
            self._renderer2 = None

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        sensors = _read_sensors(self._model, self._data, self._n_joints, self._n_feet)
        t = self._sim_time
        # Same three frequencies as proto.simple_brain.get_input_clocks
        clocks = np.array([np.sin(t * 1 / np.pi),
                           np.sin(t * 5 / np.pi),
                           np.sin(t * 15 / np.pi)], dtype=np.float32)
        return np.concatenate([
            clocks,
            sensors.hip_angles.astype(np.float32),
            sensors.hip_velocities.astype(np.float32),
            sensors.torso_orientation.astype(np.float32),
            sensors.torso_velocity.astype(np.float32),
            sensors.torso_angular_velocity.astype(np.float32),
        ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  mujoco_env.py — debug mode")
    print("=" * 60)

    # 1. Basic build
    print("\n[1] Build env\n")
    env = RobotControllerEnv(seed=0, episode_duration=2.0)
    print(f"  obs space   : {env.observation_space}")
    print(f"  act space   : {env.action_space}")
    print(f"  obs dim     : {env.observation_space.shape}")
    print(f"  n_joints    : {env._n_joints}")
    print(f"  timestep    : {env._timestep}s")
    print(f"  physics_steps_per_action : {env._physics_steps_per_action}")

    obs, info = env.reset(seed=0)
    print(f"  reset obs   : shape={obs.shape}  finite={np.isfinite(obs).all()}")

    # 2. Gymnasium check
    print("\n[2] gymnasium env_checker\n")
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        print("  check_env: OK")
    except Exception as e:
        print(f"  check_env failed: {type(e).__name__}: {e}")

    # 3. Random rollout
    print("\n[3] 1000-step random-action episode\n")
    env2 = RobotControllerEnv(seed=42, episode_duration=10.0)
    obs, _ = env2.reset(seed=42)
    rng = np.random.default_rng(7)
    total_reward = 0.0
    final_info   = {}
    for step in range(1000):
        action = rng.uniform(-1.0, 1.0, size=env2.action_space.shape).astype(np.float32)
        obs, r, terminated, truncated, info = env2.step(action)
        total_reward += r
        final_info = info
        if terminated or truncated:
            print(f"  episode ended at step {step+1}: "
                  f"terminated={terminated} truncated={truncated}")
            break
    print(f"  total reward : {total_reward:+.3f}")
    print(f"  final info   : {final_info}")
    env2.close()

    # 4. Render returns a frame
    print("\n[4] render() smoke test\n")
    env3 = RobotControllerEnv(seed=1, render_mode="rgb_array")
    env3.reset(seed=1)
    frame = env3.render()
    print(f"  frame shape : {frame.shape if frame is not None else None}")
    assert frame is not None and frame.ndim == 3 and frame.shape[2] == 3
    env3.close()

    print("\nAll mujoco_env.py checks passed.")
