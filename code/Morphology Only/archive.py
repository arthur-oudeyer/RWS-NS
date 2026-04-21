"""
archive.py
==========
Population storage for the two evolution strategies.

MuLambdaArchive
---------------
Maintains a population of μ parents.  Each generation:
  - update(results) receives μ+λ MorphologyResults (parents + offspring)
  - Keeps the best μ by fitness → new population
  - Records per-generation statistics (fitness mean, std, best)

MapEliteArchive
---------------
Maintains a grid where each cell holds the best-scoring individual
for a combination of feature descriptors.  Each generation:
  - update(results) inserts or replaces cells if the new individual
    scores higher
  - Any filled cell can be selected as a mutation parent

Feature dimensions (MapElite)
------------------------------
  Dimension 0 — n_legs       (discrete integer)
  Dimension 1 — symmetry_bin (0 = asymmetric, 1 = semi, 2 = symmetric)

The bin edges for symmetry are taken from ExperimentConfig.symmetry_bins
and default to [0.5, 0.8].

Both archives share the same public interface:
  update(results)           → updates the archive
  best()                    → MorphologyResult with highest fitness
  get_parents(n)            → list[RobotMorphology] to mutate next gen
  save(path)                → serialise to JSON
  load(path)  (classmethod) → deserialise from JSON
  summary()                 → print a human-readable state summary

Debug
-----
Run this file to test both archive types without needing CLIP.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from morphology import RobotMorphology
from data_handler import MorphologyResult, result_to_dict, dict_to_result


# ---------------------------------------------------------------------------
# Per-generation statistics (stored in archive history)
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
            f"best={self.best_fitness:+.5f}  "
            f"mean={self.mean_fitness:+.5f}  "
            f"std={self.std_fitness:.5f}  "
            f"best_id={self.best_individual_id}"
        )


def _make_stats(generation: int, results: list[MorphologyResult]) -> GenerationStats:
    fitnesses = [r.fitness for r in results]
    best      = max(results, key=lambda r: r.fitness)
    import statistics
    return GenerationStats(
        generation         = generation,
        n_evaluated        = len(results),
        best_fitness       = best.fitness,
        mean_fitness       = statistics.mean(fitnesses),
        std_fitness        = statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
        best_individual_id = best.individual_id,
        best_raw_scores    = best.raw_scores,
    )


# ---------------------------------------------------------------------------
# MuLambdaArchive
# ---------------------------------------------------------------------------

class MuLambdaArchive:
    """
    (μ+λ) population archive.

    Stores the μ best individuals.  History records per-generation stats
    for the full μ+λ pool evaluated in that generation (not just survivors).

    Parameters
    ----------
    mu : number of parents to keep after each selection step.
    """

    def __init__(self, mu: int):
        self.mu:         int                      = mu
        self.population: list[MorphologyResult]   = []   # current μ parents
        self.history:    list[GenerationStats]     = []   # one entry per generation

    # ---- Core operations ---------------------------------------------------

    def update(self, results: list[MorphologyResult]) -> None:
        """
        Receive the μ+λ evaluated results for one generation.
        Keeps the best μ by fitness → becomes the new population.
        Records generation statistics over the full μ+λ pool.
        """
        if not results:
            return

        generation = results[0].generation
        self.history.append(_make_stats(generation, results))

        sorted_results  = sorted(results, key=lambda r: r.fitness, reverse=True)
        self.population = sorted_results[:self.mu]

    def best(self) -> Optional[MorphologyResult]:
        """Individual with the highest fitness in the current population."""
        if not self.population:
            return None
        return max(self.population, key=lambda r: r.fitness)

    def get_parents(self, n: int) -> list[RobotMorphology]:
        """
        Return n morphologies from the current population to be mutated.
        Samples with replacement if n > len(population).
        """
        if not self.population:
            raise RuntimeError("Archive is empty — populate it first.")
        morphs = [r.morphology for r in self.population]
        if n <= len(morphs):
            return random.sample(morphs, n)
        return random.choices(morphs, k=n)

    def get_parent_results(self, n: int) -> list[MorphologyResult]:
        """Return n MorphologyResult objects from the population (with replacement if needed)."""
        if not self.population:
            raise RuntimeError("Archive is empty — populate it first.")
        if n <= len(self.population):
            return random.sample(self.population, n)
        return random.choices(self.population, k=n)

    # ---- Summary -----------------------------------------------------------

    def summary(self) -> None:
        print(f"\n── MuLambdaArchive  μ={self.mu} ──")
        print(f"  Population size : {len(self.population)}")
        b = self.best()
        if b:
            print(f"  Best individual : id={b.individual_id}  fitness={b.fitness:+.5f}  "
                  f"legs={b.descriptors.get('n_legs', '?')}")
        if self.history:
            last = self.history[-1]
            print(f"  Last generation : {last}")
        print(f"  History length  : {len(self.history)} generation(s)")

    def fitness_history(self) -> tuple[list[float], list[float]]:
        """Return (best_per_gen, mean_per_gen) lists for plotting."""
        return (
            [s.best_fitness  for s in self.history],
            [s.mean_fitness  for s in self.history],
        )

    # ---- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type":       "mu_lambda",
            "mu":         self.mu,
            "population": [result_to_dict(r) for r in self.population],
            "history":    [asdict(s) for s in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> MuLambdaArchive:
        archive            = cls(mu=d["mu"])
        archive.population = [dict_to_result(r) for r in d["population"]]
        archive.history    = [
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
        return archive

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> MuLambdaArchive:
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# MapEliteArchive
# ---------------------------------------------------------------------------

class MapEliteArchive:
    """
    MAP-Elites grid archive with configurable feature dimensions.

    Feature space: two dimensions taken from MorphologyResult.descriptors.
    Each dimension is either discrete (integer used as-is) or binned
    (float value mapped to a bucket index via provided bin edges).

    Parameters
    ----------
    feature_dims : list of two descriptor key names, e.g.
                   ["n_legs", "bilateral_symmetry"].
                   The values are read from result.descriptors at update time.
    feature_bins : dict mapping dim name → list of float bin edges.
                   Dims NOT present here are treated as discrete integers.
                   E.g. {"bilateral_symmetry": [4.0, 7.0]} produces 3 buckets.
    dim_labels   : optional dict mapping dim name → list of bucket labels
                   (one per bucket, for display only).

    Legacy parameter
    ----------------
    symmetry_bins : kept for backward-compat when loading old JSON files.
                    Ignored when feature_dims is provided explicitly.
    """

    def __init__(
        self,
        feature_dims: list[str]             = None,
        feature_bins: dict[str, list[float]] = None,
        dim_labels:   dict[str, list[str]]   = None,
        symmetry_bins: list[float]           = None,  # legacy
    ):
        # Legacy fallback: reproduce old (n_legs, symmetry_score) behaviour
        if feature_dims is None:
            bins = symmetry_bins or [0.5, 0.8]
            feature_dims = ["n_legs", "symmetry_score"]
            feature_bins = feature_bins or {"symmetry_score": bins}
            dim_labels   = dim_labels or {
                "symmetry_score": ["asymmetric", "semi-sym", "symmetric"]
            }

        self.feature_dims: list[str]              = feature_dims
        self.feature_bins: dict[str, list[float]] = feature_bins or {}
        self.dim_labels:   dict[str, list[str]]   = dim_labels   or {}
        self.grid:         dict[tuple, MorphologyResult] = {}
        self.history:      list[GenerationStats]         = []

    # ---- Feature key -------------------------------------------------------

    def _bin(self, value: float, edges: list[float]) -> int:
        for i, edge in enumerate(edges):
            if value < edge:
                return i
        return len(edges)

    def feature_key(self, result: MorphologyResult) -> tuple:
        """Map a result to its N-dimensional cell key."""
        key = []
        for dim in self.feature_dims:
            val = result.descriptors.get(dim)
            if val is None:
                raise KeyError(
                    f"[MapEliteArchive] Descriptor '{dim}' missing in result "
                    f"id={result.individual_id}. "
                    f"Available keys: {list(result.descriptors.keys())}"
                )
            edges = self.feature_bins.get(dim, [])
            if edges:
                key.append(self._bin(float(val), edges))
            else:
                key.append(int(val))
        return tuple(key)

    def feature_label(self, key: tuple) -> str:
        """Human-readable cell label, e.g. 'n_legs=4 / bilateral_symmetry=symmetric'."""
        parts = []
        for dim, bucket in zip(self.feature_dims, key):
            labels = self.dim_labels.get(dim, [])
            if labels and bucket < len(labels):
                parts.append(f"{dim}={labels[bucket]}")
            else:
                parts.append(f"{dim}={bucket}")
        return " / ".join(parts)

    # ---- Core operations ---------------------------------------------------

    def update(self, results: list[MorphologyResult]) -> None:
        """
        Insert each result into the grid.
        A cell is updated only when the new individual's fitness is
        strictly higher than the current occupant.
        Records generation statistics: best/id from the full grid (monotonically
        non-decreasing), mean/std from the evaluated batch only.
        """
        if not results:
            return

        generation = results[0].generation

        # Update grid first so stats reflect the post-update state
        for r in results:
            try:
                key = self.feature_key(r)
            except KeyError:
                continue
            if key not in self.grid or r.fitness > self.grid[key].fitness:
                self.grid[key] = r

        # Best fitness = grid-wide best (never decreases)
        grid_best = max(self.grid.values(), key=lambda r: r.fitness) if self.grid else results[0]
        import statistics as _stats
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

    def best(self) -> Optional[MorphologyResult]:
        """Individual with the highest fitness across all grid cells."""
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda r: r.fitness)

    def get_parents(self, n: int) -> list[RobotMorphology]:
        """
        Sample n morphologies from filled grid cells (uniform over cells,
        not over individuals).  Samples with replacement if n > cells.
        """
        if not self.grid:
            raise RuntimeError("Archive grid is empty — populate it first.")
        filled = list(self.grid.values())
        morphs = [r.morphology for r in filled]
        if n <= len(morphs):
            return random.sample(morphs, n)
        return random.choices(morphs, k=n)

    def get_parent_results(self, n: int) -> list[MorphologyResult]:
        """Return n MorphologyResult objects from filled grid cells (with replacement if needed)."""
        if not self.grid:
            raise RuntimeError("Archive grid is empty — populate it first.")
        filled = list(self.grid.values())
        if n <= len(filled):
            return random.sample(filled, n)
        return random.choices(filled, k=n)

    # ---- Summary -----------------------------------------------------------

    def summary(self) -> None:
        print(f"\n── MapEliteArchive  dims={self.feature_dims} ──")
        print(f"  Filled cells : {len(self.grid)}")
        b = self.best()
        if b:
            print(f"  Best overall : id={b.individual_id}  fitness={b.fitness:+.5f}  "
                  f"cell={self.feature_label(self.feature_key(b))}")
        print(f"  Grid contents:")
        for key in sorted(self.grid):
            r = self.grid[key]
            print(f"    {self.feature_label(key):<40}  fitness={r.fitness:+.5f}  "
                  f"id={r.individual_id}")
        if self.history:
            print(f"  Last gen : {self.history[-1]}")

    def fitness_history(self) -> tuple[list[float], list[float]]:
        return (
            [s.best_fitness for s in self.history],
            [s.mean_fitness for s in self.history],
        )

    # ---- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type":         "map_elite",
            "feature_dims": self.feature_dims,
            "feature_bins": self.feature_bins,
            "dim_labels":   self.dim_labels,
            # Grid keys as JSON arrays: "[4, 1]" — unambiguous and parseable
            "grid":    {json.dumps(list(k)): result_to_dict(v) for k, v in self.grid.items()},
            "history": [asdict(s) for s in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> MapEliteArchive:
        # New format: feature_dims present
        if "feature_dims" in d:
            archive = cls(
                feature_dims = d["feature_dims"],
                feature_bins = d.get("feature_bins", {}),
                dim_labels   = d.get("dim_labels", {}),
            )
        else:
            # Legacy format: symmetry_bins only
            archive = cls(symmetry_bins=d.get("symmetry_bins", [0.5, 0.8]))

        for k_str, v in d["grid"].items():
            # New format: "[4, 1]"
            if k_str.startswith("["):
                key = tuple(json.loads(k_str))
            else:
                # Legacy format: "(4, 1)"
                key = tuple(int(x) for x in k_str.strip("()").split(", "))
            archive.grid[key] = dict_to_result(v)

        archive.history = [
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
        return archive

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> MapEliteArchive:
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, tempfile, os
    sys.path.insert(0, str(Path(__file__).parent))
    from morphology import QUADRIPOD, TRIPOD, HEXAPOD, NewMorph, MutateMorphology
    import numpy as np

    print("=" * 60)
    print("  archive.py — debug mode")
    print("=" * 60)

    # Build fake MorphologyResults (no CLIP needed)
    def fake_result(morph, generation, individual_id, fitness):
        return MorphologyResult(
            generation    = generation,
            individual_id = individual_id,
            morphology    = morph,
            fitness       = fitness,
            raw_scores    = {},
            descriptors   = morph.encoding(),
            grader_method = "fake",
            prompt_set    = "test",
        )

    rng  = np.random.default_rng(0)
    pool = [fake_result(NewMorph(), 0, i, rng.uniform(-1, 1)) for i in range(30)]

    # ---- MuLambdaArchive ---------------------------------------------------
    print("\n[1] MuLambdaArchive\n")
    mu_archive = MuLambdaArchive(mu=5)

    # Gen 0: initialise with 10 individuals
    mu_archive.update(pool[:10])
    mu_archive.summary()

    # Gen 1: produce 5 offspring from parents, evaluate 5+5=10 total
    parents = mu_archive.get_parents(5)
    print(f"\n  Parents drawn: {len(parents)} morphologies")
    offspring = [fake_result(MutateMorphology(p, rng=rng), 1, 10 + i, rng.uniform(-1, 1))
                 for i, p in enumerate(parents)]
    gen1_pool = [fake_result(r.morphology, 1, r.individual_id, r.fitness)
                 for r in mu_archive.population] + offspring
    mu_archive.update(gen1_pool)
    mu_archive.summary()

    # ---- MapEliteArchive ---------------------------------------------------
    print("\n[2] MapEliteArchive\n")
    me_archive = MapEliteArchive()
    me_archive.update(pool)
    me_archive.summary()

    # Add a high-fitness result to a specific cell
    super_morph  = HEXAPOD
    super_result = fake_result(super_morph, 1, 999, 0.999)
    me_archive.update([super_result])
    print(f"\n  After inserting super_result (fitness=0.999):")
    me_archive.summary()

    # ---- Serialisation round-trip ------------------------------------------
    print("\n[3] Serialisation round-trip\n")
    with tempfile.TemporaryDirectory() as tmp:
        for archive, name in ((mu_archive, "mu_lambda.json"), (me_archive, "map_elite.json")):
            path = os.path.join(tmp, name)
            archive.save(path)

            if isinstance(archive, MuLambdaArchive):
                loaded = MuLambdaArchive.load(path)
                assert loaded.mu == archive.mu
                assert len(loaded.population) == len(archive.population)
            else:
                loaded = MapEliteArchive.load(path)
                assert len(loaded.grid) == len(archive.grid)
                assert loaded.symmetry_bins == archive.symmetry_bins

            print(f"  {name}: OK  ({os.path.getsize(path)} bytes)")

    # ---- Fitness history ---------------------------------------------------
    print("\n[4] Fitness history\n")
    bests, means = mu_archive.fitness_history()
    for i, (b, m) in enumerate(zip(bests, means)):
        print(f"  gen {i}: best={b:+.4f}  mean={m:+.4f}")

    print("\nAll archive checks passed.")
