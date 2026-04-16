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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Robot'))
from saver import save_controller, load_controller, clear_save
from morphology import RobotMorphology

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Selection'))
from selector import feature_descriptor, feature_label, selection

# ---------------------------------------------------------------------------
# Tunable thresholds — adjust to match the robot's actual geometry
# ---------------------------------------------------------------------------
STANDING_HEIGHT_THRESHOLD = 0.2   # metres  — torso z below this → fallen
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
    robot_index:          int
    morphology:           RobotMorphology  # static body descriptor — not stored per snapshot
    is_standing_start:    bool
    is_standing_end:      bool
    displacement_xy:      float            # metres from start to end (XY plane, torso)
    avg_speed_xy:         float            # m/s  (Full: mean; StartStop: displacement/dt)
    fell_at_time:         Optional[float]  # seconds of first fall, None if never (Full only)
    max_height:           float            # peak torso Z  (Full: max over all steps; else end)
    mean_height:          float            # mean torso Z  (Full: mean over all steps; else end)
    total_rotation:       float            # integral of ||ω|| over sim  (Full only, else 0)
    mean_rotation_speed:  float            # mean ||ω||  rad/s  (Full only, else 0)
    symmetry_score:       float            # 0..1 — how evenly spaced the legs are (from morph)

    @property
    def nb_legs(self) -> int:
        return len(self.morphology.legs)

    def __str__(self) -> str:
        status = "UP    " if self.is_standing_end else "FALLEN"
        fell   = f"{self.fell_at_time:.2f}s" if self.fell_at_time is not None else "never"
        return (
            f"R{self.robot_index:<2} {self.nb_legs} legs {self.morphology.name:<10} : {status}  "
            f"disp={self.displacement_xy:+.3f}m  speed={self.avg_speed_xy:.3f}m/s  fell={fell}  "
            f"h=[{self.mean_height:.3f}/{self.max_height:.3f}]m  "
            f"rot={self.total_rotation:.2f}rad  ω={self.mean_rotation_speed:.2f}rad/s  "
            f"sym={self.symmetry_score:.2f}"
        )


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------
class DataManager:

    MODES = ("Full", "StartStop")

    def __init__(self, n_robots: int, mode: str = "StartStop", controllers=None, save_best: tuple[bool, bool, bool] = (False, False), morphologies=None):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.MODES}")
        self.n_robots    = n_robots
        self.mode        = mode
        self.controllers  = controllers    # list[NeuralNetwork] — required when save_best=True
        self.morphologies = morphologies   # list[RobotMorphology] — saved alongside networks
        self.save_best, self.save_best_unique, self.clear_archive = save_best
        self._data: list[list[Snapshot]] = [[] for _ in range(n_robots)]
        print(f"Data Manager initialized (n={n_robots}, mode={mode}, save_best={self.save_best}, save_unique={self.save_best_unique})")

        if self.save_best and self.clear_archive:
            clear_save("last_best")

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
        buf   = self._data[robot_index]
        morph = self.morphologies[robot_index]
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

        # Height metrics
        if self.mode == "Full" and len(buf) > 1:
            heights    = [s.sensors.torso_height for s in buf]
            max_height  = float(max(heights))
            mean_height = float(np.mean(heights))
        else:
            max_height  = float(end.sensors.torso_height)
            mean_height = float(end.sensors.torso_height)

        # Rotation metrics
        if self.mode == "Full" and len(buf) > 1:
            dt         = buf[1].time - buf[0].time
            ang_speeds = [float(np.linalg.norm(s.sensors.torso_angular_velocity)) for s in buf]
            mean_rotation_speed = float(np.mean(ang_speeds))
            total_rotation      = float(sum(ang_speeds) * dt)
        else:
            mean_rotation_speed = 0.0
            total_rotation      = 0.0

        # Symmetry score — how uniformly spaced are the legs (pure morphology property)
        angles  = sorted(leg.placement_angle_deg % 360 for leg in morph.legs)
        n       = len(angles)
        gaps    = [(angles[(i + 1) % n] - angles[i]) % 360 for i in range(n)]
        expected = 360.0 / n
        symmetry_score = float(max(0.0, 1.0 - np.std(gaps) / expected)) if n > 1 else 1.0

        return RobotMetrics(
            robot_index         = robot_index,
            morphology          = morph,
            is_standing_start   = self.is_standing(start.sensors),
            is_standing_end     = self.is_standing(end.sensors),
            displacement_xy     = displacement,
            avg_speed_xy        = avg_speed,
            fell_at_time        = fell_at,
            max_height          = max_height,
            mean_height         = mean_height,
            total_rotation      = total_rotation,
            mean_rotation_speed = mean_rotation_speed,
            symmetry_score      = symmetry_score,
        )

    def get_all_metrics(self) -> list[RobotMetrics]:
        return [self.get_metrics(i) for i in range(self.n_robots)]

    def get_snapshots(self, robot_index: int) -> list[Snapshot]:
        return self._data[robot_index]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def print_summary(self):
        metrics       = self.get_all_metrics()
        standing_count = sum(1 for m in metrics if m.is_standing_end)
        elites        = selection(metrics)   # dict: feature_key → RobotMetrics

        print(f"\n── Simulation results ({self.mode} mode) ──")
        print(f"  Standing at end : {standing_count}/{self.n_robots}")
        if elites:
            best_overall = max(elites.values(), key=lambda m: m.displacement_xy)
            print(f"  Best elite      : R{best_overall.robot_index} [{feature_label(feature_descriptor(best_overall))}] → {best_overall.displacement_xy:.3f} m")
            print(f"  Elite cells     : {len(elites)}  {[feature_label(k) for k in sorted(elites)]}")
        else:
            print("  No standing robots — no elites selected.")
        print()
        #for m in metrics:
        #    print(f"  {m}")
        #print()

        if self.controllers is not None:
            self._save_last_sim()
        if self.save_best:
            self._save_best(elites)

    def _save_last_sim(self):
        save_controller(
            networks     = self.controllers,
            name         = "last_sim",
            context      = {"n_robots": self.n_robots, "data_mode": self.mode},
            morphologies = self.morphologies,
        )

    def _save_best(self, elites: dict):
        """
        Merge new elites into the persistent archive (last_best.pkl).

        For each feature cell: keep the robot with the higher displacement_xy.
        Cells absent from the new sim are preserved unchanged from the old archive.
        Only overwrite a cell when the new robot is strictly better.
        """
        if self.controllers is None:
            print("[data] SAVE_BEST is True but no controllers were provided to DataManager — skipping.")
            return
        if not elites:
            print("[data] No elites to save (all robots fell).")
            return

        # --- Load existing archive (if any) ---
        # merged: feature_key → {"network", "morphology", "meta"}
        merged = {}
        try:
            old = load_controller("last_best")
            old_meta_list = old["context"].get("elites", [])
            for i, meta in enumerate(old_meta_list):
                key = tuple(meta["feature_key"])
                merged[key] = {
                    "network":    old["networks"][i],
                    "morphology": old["morphologies"][i] if old["morphologies"] else None,
                    "meta":       meta,
                }
            print(f"[data] Loaded existing archive: {len(merged)} cell(s).")
        except FileNotFoundError:
            pass   # no archive yet — start fresh

        # --- Merge: update a cell only if the new robot scores higher ---
        updated = 0
        for key, m in elites.items():
            new_disp = m.displacement_xy
            if key not in merged or new_disp > merged[key]["meta"]["displacement_m"]:
                merged[key] = {
                    "network":    self.controllers[m.robot_index],
                    "morphology": self.morphologies[m.robot_index] if self.morphologies else None,
                    "meta": {
                        "feature_key":    list(key),
                        "feature_label":  feature_label(key),
                        "robot_index":    m.robot_index,
                        "displacement_m": round(m.displacement_xy, 4),
                        "avg_speed_ms":   round(m.avg_speed_xy, 4),
                        "fell_at":        m.fell_at_time,
                        "data_mode":      self.mode,
                    },
                }
                updated += 1

        kept = len(merged) - updated
        print(f"[data] Archive: {updated} cell(s) updated, {kept} kept from previous archive — {len(merged)} total.")

        # --- Build save lists (order follows sorted keys for determinism) ---
        merged_list = [merged[k] for k in sorted(merged)]
        networks    = [e["network"]    for e in merged_list]
        morphs      = [e["morphology"] for e in merged_list] if any(e["morphology"] is not None for e in merged_list) else None
        context     = {"elites": [e["meta"] for e in merged_list]}

        if self.save_best_unique:
            save_controller(networks=networks, name=datetime.now().strftime("best_%Y%m%d_%H%M%S"), context=context, morphologies=morphs)
        save_controller(networks=networks, name="last_best", context=context, morphologies=morphs)