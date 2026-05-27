"""
archive.py
==========
Population storage for the two evolution strategies, typed for
`ControllerResult`. Mirrors `Morphology/archive.py` line-for-line —
only the result type changes.

Public interface (shared by both archives)
------------------------------------------
    update(results)              → ingest results (μ+λ pool, or λ children)
    best()                       → ControllerResult with highest fitness
    get_parents(n)               → list of (reward_weights_dict, policy_path)
    get_parent_results(n)        → list of full ControllerResult parents
    save(path) / load(path)      → JSON round-trip
    summary()                    → human-readable state print
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from data_handler import ControllerResult, result_to_dict, dict_to_result


# ---------------------------------------------------------------------------
# Per-generation statistics
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    """Aggregate fitness statistics for one generation."""
    generation:         int
    n_evaluated:        int
    best_fitness:       float
    mean_fitness:       float
    std_fitness:        float
    best_individual_id: int
    best_raw_scores:    dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"gen={self.generation:>4}  "
            f"evaluated={self.n_evaluated:>4}  "
            f"best={self.best_fitness:+.4f}  "
            f"mean={self.mean_fitness:+.4f}  "
            f"std={self.std_fitness:.4f}  "
            f"best_id={self.best_individual_id}"
        )


def _make_stats(generation: int, results: list[ControllerResult]) -> GenerationStats:
    import statistics as _stats
    fitnesses = [r.fitness for r in results]
    best      = max(results, key=lambda r: r.fitness)
    return GenerationStats(
        generation         = generation,
        n_evaluated        = len(results),
        best_fitness       = best.fitness,
        mean_fitness       = _stats.mean(fitnesses),
        std_fitness        = _stats.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
        best_individual_id = best.individual_id,
        best_raw_scores    = best.raw_scores,
    )


# ---------------------------------------------------------------------------
# MuLambdaArchive
# ---------------------------------------------------------------------------

class MuLambdaArchive:
    """(μ+λ) population archive of `ControllerResult`s."""

    def __init__(self, mu: int):
        self.mu:         int                       = mu
        self.population: list[ControllerResult]    = []
        self.history:    list[GenerationStats]     = []

    # ---- core --------------------------------------------------------------

    def update(self, results: list[ControllerResult]) -> None:
        if not results:
            return
        generation = results[0].generation
        self.history.append(_make_stats(generation, results))
        sorted_r = sorted(results, key=lambda r: r.fitness, reverse=True)
        self.population = sorted_r[:self.mu]

    def best(self) -> Optional[ControllerResult]:
        if not self.population:
            return None
        return max(self.population, key=lambda r: r.fitness)

    def get_parent_results(self, n: int) -> list[ControllerResult]:
        if not self.population:
            raise RuntimeError("Archive is empty — populate it first.")
        if n <= len(self.population):
            return random.sample(self.population, n)
        return random.choices(self.population, k=n)

    # ---- summary -----------------------------------------------------------

    def summary(self) -> None:
        print(f"\n── MuLambdaArchive  μ={self.mu} ──")
        print(f"  Population size : {len(self.population)}")
        b = self.best()
        if b:
            print(f"  Best individual : id={b.individual_id}  fitness={b.fitness:+.4f}  "
                  f"steps={b.n_train_steps:,}  parent={b.parent_id}")
        if self.history:
            print(f"  Last gen        : {self.history[-1]}")
        print(f"  History length  : {len(self.history)} generation(s)")

    def fitness_history(self) -> tuple[list[float], list[float]]:
        return ([s.best_fitness for s in self.history],
                [s.mean_fitness for s in self.history])

    # ---- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type":       "mu_lambda",
            "mu":         self.mu,
            "population": [result_to_dict(r) for r in self.population],
            "history":    [asdict(s) for s in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MuLambdaArchive":
        a = cls(mu=d["mu"])
        a.population = [dict_to_result(r) for r in d["population"]]
        a.history = [
            GenerationStats(
                generation         = s["generation"],
                n_evaluated        = s["n_evaluated"],
                best_fitness       = s["best_fitness"],
                mean_fitness       = s["mean_fitness"],
                std_fitness        = s["std_fitness"],
                best_individual_id = s["best_individual_id"],
                best_raw_scores    = s.get("best_raw_scores", {}),
            )
            for s in d["history"]
        ]
        return a

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MuLambdaArchive":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# MapEliteArchive
# ---------------------------------------------------------------------------

class MapEliteArchive:
    """
    MAP-Elites grid archive over feature dimensions returned by the VLM
    (`vlm_descriptors`) merged into `ControllerResult.descriptors`.

    Parameters
    ----------
    feature_dims : two descriptor key names that serve as grid axes.
                   Empty/None disables MAP-Elites (single-cell archive that
                   keeps only the best-fitness individual seen).
    feature_bins : dict mapping dim name → list of float bin edges.
                   Dims missing from this dict are treated as discrete ints.
    dim_labels   : optional dict mapping dim name → list of bucket labels.
    """

    def __init__(
        self,
        feature_dims: Optional[list[str]]              = None,
        feature_bins: Optional[dict[str, list[float]]] = None,
        dim_labels:   Optional[dict[str, list[str]]]   = None,
    ):
        self.feature_dims: list[str]              = list(feature_dims) if feature_dims else []
        self.feature_bins: dict[str, list[float]] = feature_bins or {}
        self.dim_labels:   dict[str, list[str]]   = dim_labels   or {}
        self.grid:         dict[tuple, ControllerResult] = {}
        self.history:      list[GenerationStats]         = []

    @staticmethod
    def _bin(value: float, edges: list[float]) -> int:
        for i, edge in enumerate(edges):
            if value < edge:
                return i
        return len(edges)

    def feature_key(self, result: ControllerResult) -> tuple:
        """Map a result to its N-dimensional cell key."""
        if not self.feature_dims:
            return ()
        key = []
        for dim in self.feature_dims:
            val = result.descriptors.get(dim)
            if val is None:
                print(f"[MapEliteArchive] WARNING: descriptor '{dim}' missing for "
                      f"id={result.individual_id} — placing in bin 0. "
                      f"Available keys: {list(result.descriptors.keys())}")
                key.append(0)
                continue
            edges = self.feature_bins.get(dim, [])
            if edges:
                key.append(self._bin(float(val), edges))
            else:
                key.append(int(val))
        return tuple(key)

    # ---- core --------------------------------------------------------------

    def update(self, results: list[ControllerResult]) -> None:
        if not results:
            return
        generation = results[0].generation
        for r in results:
            try:
                key = self.feature_key(r)
            except KeyError:
                continue
            if key not in self.grid or r.fitness > self.grid[key].fitness:
                self.grid[key] = r

        import statistics as _stats
        grid_best = max(self.grid.values(), key=lambda r: r.fitness) if self.grid else results[0]
        fitnesses = [r.fitness for r in results]
        self.history.append(GenerationStats(
            generation         = generation,
            n_evaluated        = len(results),
            best_fitness       = grid_best.fitness,
            mean_fitness       = _stats.mean(fitnesses),
            std_fitness        = _stats.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            best_individual_id = grid_best.individual_id,
            best_raw_scores    = grid_best.raw_scores,
        ))

    def best(self) -> Optional[ControllerResult]:
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda r: r.fitness)

    def get_parent_results(self, n: int) -> list[ControllerResult]:
        if not self.grid:
            raise RuntimeError("Archive grid is empty — populate it first.")
        filled = list(self.grid.values())
        if n <= len(filled):
            return random.sample(filled, n)
        return random.choices(filled, k=n)

    # ---- summary -----------------------------------------------------------

    def feature_label(self, key: tuple) -> str:
        parts = []
        for dim, bucket in zip(self.feature_dims, key):
            labels = self.dim_labels.get(dim, [])
            if labels and bucket < len(labels):
                parts.append(f"{dim}={labels[bucket]}")
            else:
                parts.append(f"{dim}={bucket}")
        return " / ".join(parts) if parts else "single-cell"

    def summary(self) -> None:
        print(f"\n── MapEliteArchive  dims={self.feature_dims} ──")
        print(f"  Filled cells : {len(self.grid)}")
        b = self.best()
        if b:
            print(f"  Best overall : id={b.individual_id}  fitness={b.fitness:+.4f}  "
                  f"cell={self.feature_label(self.feature_key(b))}")
        for key in sorted(self.grid):
            r = self.grid[key]
            print(f"    {self.feature_label(key):<40}  fitness={r.fitness:+.4f}  "
                  f"id={r.individual_id}")
        if self.history:
            print(f"  Last gen     : {self.history[-1]}")

    def fitness_history(self) -> tuple[list[float], list[float]]:
        return ([s.best_fitness for s in self.history],
                [s.mean_fitness for s in self.history])

    # ---- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type":         "map_elite",
            "feature_dims": self.feature_dims,
            "feature_bins": self.feature_bins,
            "dim_labels":   self.dim_labels,
            "grid":    {json.dumps(list(k)): result_to_dict(v) for k, v in self.grid.items()},
            "history": [asdict(s) for s in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MapEliteArchive":
        a = cls(
            feature_dims = d.get("feature_dims", []),
            feature_bins = d.get("feature_bins", {}),
            dim_labels   = d.get("dim_labels", {}),
        )
        for k_str, v in d.get("grid", {}).items():
            key = tuple(json.loads(k_str)) if k_str.startswith("[") else tuple(int(x) for x in k_str.strip("()").split(", ") if x)
            a.grid[key] = dict_to_result(v)
        a.history = [
            GenerationStats(
                generation         = s["generation"],
                n_evaluated        = s["n_evaluated"],
                best_fitness       = s["best_fitness"],
                mean_fitness       = s["mean_fitness"],
                std_fitness        = s["std_fitness"],
                best_individual_id = s["best_individual_id"],
                best_raw_scores    = s.get("best_raw_scores", {}),
            )
            for s in d["history"]
        ]
        return a

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MapEliteArchive":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os, numpy as np

    print("=" * 60)
    print("  archive.py — debug mode")
    print("=" * 60)

    rng = np.random.default_rng(0)

    def fake_result(i, gen, fit, parent=None, descriptors=None):
        return ControllerResult(
            generation     = gen,
            individual_id  = i,
            parent_id      = parent,
            reward_weights = {"forward_velocity": 1.0 + 0.05*i},
            policy_path    = f"/tmp/p{i}.zip",
            video_path     = f"/tmp/v{i}.mp4",
            n_train_steps  = 50_000 if parent is None else 15_000,
            fitness        = fit,
            raw_scores     = {"coherence": fit, "progress": fit, "interest": fit},
            descriptors    = descriptors or {"gait": (i % 3), "speed": (i % 2)},
            grader_method  = "fake",
            prompt_set     = "fake",
        )

    pool = [fake_result(i, 0, float(rng.uniform(0, 1))) for i in range(20)]

    print("\n[1] MuLambdaArchive\n")
    a = MuLambdaArchive(mu=4)
    a.update(pool[:10])
    a.summary()

    g1 = [fake_result(10 + i, 1, float(rng.uniform(0, 1)), parent=p.individual_id)
          for i, p in enumerate(a.population)]
    a.update([fake_result(p.individual_id, 1, p.fitness) for p in a.population] + g1)
    a.summary()

    print("\n[2] MapEliteArchive\n")
    me = MapEliteArchive(feature_dims=["gait", "speed"])
    me.update(pool)
    me.summary()
    me.update([fake_result(999, 1, 0.999, descriptors={"gait": 0, "speed": 0})])
    print("\nAfter inserting super result:")
    me.summary()

    print("\n[3] Round-trip\n")
    with tempfile.TemporaryDirectory() as tmp:
        for arc, name in ((a, "mu.json"), (me, "me.json")):
            p = os.path.join(tmp, name)
            arc.save(p)
            cls = MuLambdaArchive if isinstance(arc, MuLambdaArchive) else MapEliteArchive
            back = cls.load(p)
            print(f"  {name}: OK  ({os.path.getsize(p)} bytes)")

    print("\nAll archive.py checks passed.")
