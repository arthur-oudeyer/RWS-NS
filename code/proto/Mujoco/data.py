"""
data.py
=======
Tracks and records simulation data for each robot.

Modes:
  "Full"       — records every timestep (enables fell_at_time, avg_speed)
  "StartStop"  — records only first and last timestep per robot (light)

Usage in main_sim.py:
    dm = DataManager(N, mode="StartStop")
    # inside the per-robot loop:
    dm.record(current_time, robot_index, sensors)
    # after the simulation loop:
    dm.print_summary()
    metrics = dm.get_all_metrics()
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import os
import sys

import numpy as np

from control import RobotSensorData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Brain'))
from saver import save_controller

# ---------------------------------------------------------------------------
# Tunable thresholds — adjust to match the robot's actual geometry
# ---------------------------------------------------------------------------
STANDING_HEIGHT_THRESHOLD = 0.08   # metres  — torso z below this → fallen
STANDING_TILT_THRESHOLD   = 0.70   # |quat w| — below this → tipped >45° (NOT USED FOR NOW)


# ---------------------------------------------------------------------------
# Snapshot — one recorded timestep for one robot
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    time:    float
    sensors: RobotSensorData


# ---------------------------------------------------------------------------
# RobotMetrics — performance summary derived from recorded snapshots
# ---------------------------------------------------------------------------
@dataclass
class RobotMetrics:
    robot_index:       int
    is_standing_start: bool
    is_standing_end:   bool
    displacement_xy:   float           # metres travelled from start to end (XY plane)
    avg_speed_xy:      float           # m/s  (meaningful only in Full mode)
    fell_at_time:      Optional[float] # seconds of first fall, None if never (Full mode only)

    def __str__(self) -> str:
        status = "UP    " if self.is_standing_end else "FALLEN"
        fell   = f"{self.fell_at_time:.2f}s" if self.fell_at_time is not None else "never"
        return (
            f"R{self.robot_index:<2}  {status}  "
            f"disp={self.displacement_xy:+.3f}m  "
            f"speed={self.avg_speed_xy:.3f}m/s  "
            f"fell={fell}"
        )


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------
class DataManager:

    MODES = ("Full", "StartStop")

    def __init__(self, n_robots: int, mode: str = "StartStop", controllers=None, save_best: tuple[bool, bool] = (False, False)):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.MODES}")
        self.n_robots    = n_robots
        self.mode        = mode
        self.controllers = controllers   # list[NeuralNetwork] — required when save_best=True
        self.save_best, self.save_best_unique = save_best
        self._data: list[list[Snapshot]] = [[] for _ in range(n_robots)]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(self, current_time: float, robot_index: int, sensors: RobotSensorData):
        snap = Snapshot(time=current_time, sensors=sensors)
        buf  = self._data[robot_index]

        if self.mode == "Full":
            buf.append(snap)

        elif self.mode == "StartStop":
            if not buf:
                buf.append(snap)          # first call: store start
            elif len(buf) == 1:
                buf.append(snap)          # second call: placeholder for end
            else:
                buf[-1] = snap            # keep overwriting until last step

    # ------------------------------------------------------------------
    # Derived checks
    # ------------------------------------------------------------------
    @staticmethod
    def is_standing(sensors: RobotSensorData) -> bool:
        """
        True if the robot is considered upright.
          - torso height above STANDING_HEIGHT_THRESHOLD
        """
        height_ok = sensors.torso_height > STANDING_HEIGHT_THRESHOLD
        return height_ok

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_metrics(self, robot_index: int) -> RobotMetrics:
        buf = self._data[robot_index]
        if not buf:
            raise RuntimeError(f"No data recorded for robot {robot_index}.")

        start = buf[0]
        end   = buf[-1]

        # XY displacement (ignores vertical jumps)
        displacement = float(np.linalg.norm(
            end.sensors.torso_pos[:2] - start.sensors.torso_pos[:2]
        ))

        # Average XY speed
        if self.mode == "Full" and len(buf) > 1:
            avg_speed = float(np.mean([
                np.linalg.norm(s.sensors.torso_velocity[:2]) for s in buf
            ]))
        else:
            dt = end.time - start.time
            avg_speed = displacement / dt if dt > 0 else 0.0

        # First fall (only detectable step-by-step in Full mode)
        fell_at = None
        if self.mode == "Full":
            for snap in buf:
                if not self.is_standing(snap.sensors):
                    fell_at = snap.time
                    break

        return RobotMetrics(
            robot_index       = robot_index,
            is_standing_start = self.is_standing(start.sensors),
            is_standing_end   = self.is_standing(end.sensors),
            displacement_xy   = displacement,
            avg_speed_xy      = avg_speed,
            fell_at_time      = fell_at,
        )

    def get_all_metrics(self) -> list[RobotMetrics]:
        return [self.get_metrics(i) for i in range(self.n_robots)]

    def get_snapshots(self, robot_index: int) -> list[Snapshot]:
        return self._data[robot_index]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def print_summary(self):
        metrics  = self.get_all_metrics()
        standing = sum(1 for m in metrics if m.is_standing_end)
        best     = max(metrics, key=lambda m: m.displacement_xy)

        print(f"\n── Simulation results ({self.mode} mode) ──")
        print(f"  Standing at end : {standing}/{self.n_robots}")
        print(f"  Best traveller  : R{best.robot_index} → {best.displacement_xy:.3f} m")
        print()
        for m in metrics:
            print(f"  {m}")
        print()

        if self.controllers is not None:
            self._save_last_sim()
        if self.save_best:
            self._save_best(best)

    def _save_last_sim(self):
        save_controller(
            networks = self.controllers,
            name     = "last_sim",
            context  = {"n_robots": self.n_robots, "data_mode": self.mode},
        )

    def _save_best(self, best: RobotMetrics):
        if self.controllers is None:
            print("[data] SAVE_BEST is True but no controllers were provided to DataManager — skipping.")
            return
        context = {
            "robot_index":    best.robot_index,
            "displacement_m": round(best.displacement_xy, 4),
            "avg_speed_ms":   round(best.avg_speed_xy, 4),
            "is_standing":    best.is_standing_end,
            "fell_at":        best.fell_at_time,
            "data_mode":      self.mode,
        }
        network = self.controllers[best.robot_index]
        if self.save_best_unique:
            # timestamped archive (never overwritten)
            save_controller(networks=network, name=datetime.now().strftime("best_%Y%m%d_%H%M%S"), context=context)
        # always-current shortcut (overwritten each sim)
        save_controller(networks=network, name="last_best", context=context)