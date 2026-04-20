"""
evolution.py
============
Evolution strategy operators for the morphology experiment.

Two strategies are implemented, both sharing the same interface:

MuLambdaEvolution
-----------------
Classical (μ+λ) strategy.
  - initialise(): evaluate a random initial population of μ individuals.
  - step():
      1. Sample λ parents from the archive (random, with replacement).
      2. Mutate each → λ offspring.
      3. Evaluate λ offspring.
      4. Feed back μ (current population) + λ (offspring) to the archive,
         which keeps the best μ.

MapEliteEvolution
-----------------
MAP-Elites strategy.
  - initialise(): evaluate a random initial population of `init_size`
    individuals and insert them into the grid.
  - step():
      1. Sample λ parents from the filled grid cells (uniform over cells).
      2. Mutate each → λ offspring.
      3. Evaluate λ offspring.
      4. Feed offspring to the archive (it replaces cells if fitness improves).

Both strategies read all mutation parameters from ExperimentConfig so
callers never hard-code hyper-parameters.

Shared interface
----------------
  initialise(renderer, grader, id_counter=0) → (results, new_id_counter)
  step(archive, renderer, grader, generation, id_counter)
                                             → (results, new_id_counter)

  results is the list[MorphologyResult] to pass to archive.update().
  id_counter is a monotonically increasing int; callers thread it through
  every generation so individual IDs are unique across the entire run.

Debug
-----
Run this file to test both strategies without a real CLIP grader.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from morphology   import RobotMorphology, NewMorph, MutateMorphology
from data_handler import MorphologyResult, evaluate
from archive      import MuLambdaArchive, MapEliteArchive
from config       import ExperimentConfig


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEvolution(ABC):
    """
    Shared interface for evolution strategies.

    Parameters
    ----------
    cfg : ExperimentConfig — all hyper-parameters live here.
    rng : numpy Generator  — shared RNG for reproducibility.
    """

    def __init__(self, cfg: ExperimentConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mutate_one(self, parent: RobotMorphology) -> RobotMorphology:
        """Apply one mutation step to a parent morphology."""
        return MutateMorphology(
            base                      = parent,
            length_std                = self.cfg.length_std,
            angle_std                 = self.cfg.angle_std,
            rest_angle_std            = self.cfg.rest_angle_std,
            add_remove_prob           = self.cfg.add_remove_prob,
            allow_branching           = self.cfg.allow_branching,
            branching_prob            = self.cfg.branching_prob,
            torso_radius_std          = self.cfg.torso_radius_std,
            torso_height_std          = self.cfg.torso_height_std,
            torso_euler_std           = self.cfg.torso_euler_std,
            add_remove_body_part_prob = self.cfg.add_remove_body_part_prob,
            body_part_radius_std      = self.cfg.body_part_radius_std,
            body_part_height_std      = self.cfg.body_part_height_std,
            body_part_euler_std       = self.cfg.body_part_euler_std,
            body_part_leg_prob        = self.cfg.body_part_leg_prob,
            rng                       = self.rng,
        )

    def _evaluate_batch(
        self,
        morphs:      list[RobotMorphology],
        renderer,
        grader,
        generation:  int,
        id_counter:  int,
        save_renders: bool = False,
        render_dir:   Optional[str] = None,
        parent_ids:   Optional[list[int]] = None,
    ) -> tuple[list[MorphologyResult], int]:
        """
        Evaluate a list of morphologies and return (results, new_id_counter).

        If save_renders is True and render_dir is given, saves rendered PNGs.
        parent_ids, if provided, maps each morph to its parent's individual_id.
        """
        results = []
        for i, morph in enumerate(morphs):
            if save_renders and render_dir:
                Path(render_dir).mkdir(parents=True, exist_ok=True)
                render_path = str(Path(render_dir) / f"gen{generation:04d}_id{id_counter:06d}.png")
            else:
                render_path = None

            r = evaluate(
                morph            = morph,
                renderer         = renderer,
                grader           = grader,
                generation       = generation,
                individual_id    = id_counter,
                render_save_path = render_path,
                parent_id        = parent_ids[i] if parent_ids else None,
            )
            results.append(r)
            id_counter += 1
        return results, id_counter

    def _random_population(self, size: int) -> list[RobotMorphology]:
        """Generate `size` random morphologies using config leg range."""
        morphs = []
        for _ in range(size):
            morph = NewMorph(
                min_init_legs = self.cfg.init_n_legs_min,
                max_init_legs = self.cfg.init_n_legs_max,
            )
            morphs.append(morph)
        return morphs

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialise(
        self,
        renderer,
        grader,
        id_counter:   int  = 0,
        save_renders: bool = False,
        render_dir:   Optional[str] = None,
    ) -> tuple[list[MorphologyResult], int]:
        """
        Evaluate an initial random population.

        Returns (results, new_id_counter).
        Caller passes results to archive.update() immediately after.
        """
        ...

    @abstractmethod
    def step(
        self,
        archive,
        renderer,
        grader,
        generation:   int,
        id_counter:   int,
        save_renders: bool = False,
        render_dir:   Optional[str] = None,
    ) -> tuple[list[MorphologyResult], int]:
        """
        Run one generation step.

        Returns (results_to_feed_archive, new_id_counter).
        Caller passes results to archive.update() immediately after.
        """
        ...


# ---------------------------------------------------------------------------
# (μ+λ) Evolution
# ---------------------------------------------------------------------------

class MuLambdaEvolution(BaseEvolution):
    """
    (μ+λ) evolution strategy.

    Initialisation  : generate and evaluate μ random individuals.
    Each step       :
      1. Sample λ parents from archive.get_parents(λ) (random over population).
      2. Mutate each parent → λ offspring.
      3. Evaluate offspring.
      4. Return μ (current population) + λ (offspring) — the archive will
         keep the best μ from this joint pool.

    Parameters
    ----------
    cfg : ExperimentConfig  — uses mu, lambda_, and all mutation params.
    """

    def initialise(
        self,
        renderer,
        grader,
        id_counter:   int  = 0,
        save_renders: bool = False,
        render_dir:   Optional[str] = None,
    ) -> tuple[list[MorphologyResult], int]:

        morphs = self._random_population(self.cfg.mu)
        return self._evaluate_batch(morphs, renderer, grader,
                                    generation=0,
                                    id_counter=id_counter,
                                    save_renders=save_renders,
                                    render_dir=render_dir)

    def step(
        self,
        archive:      MuLambdaArchive,
        renderer,
        grader,
        generation:   int,
        id_counter:   int,
        save_renders: bool = False,
        render_dir:   Optional[str] = None,
    ) -> tuple[list[MorphologyResult], int]:
        # 1. Sample λ parents from archive (as full results to get IDs)
        sampled_parents = archive.get_parent_results(self.cfg.lambda_)
        parent_ids      = [r.individual_id for r in sampled_parents]

        # 2. Mutate → λ offspring morphologies
        offspring_morphs = [self._mutate_one(r.morphology) for r in sampled_parents]

        # 3. Evaluate offspring, recording which parent each came from
        offspring_results, id_counter = self._evaluate_batch(
            offspring_morphs, renderer, grader,
            generation   = generation,
            id_counter   = id_counter,
            save_renders = save_renders,
            render_dir   = render_dir,
            parent_ids   = parent_ids,
        )

        # 4. Pool = current μ parents (already evaluated, re-tagged for this
        #    generation so stats are consistent) + λ offspring + σ fresh randoms
        parent_results = [
            MorphologyResult(
                generation    = generation,
                individual_id = r.individual_id,
                morphology    = r.morphology,
                fitness       = r.fitness,
                raw_scores    = r.raw_scores,
                descriptors   = r.descriptors,
                grader_method = r.grader_method,
                prompt_set    = r.prompt_set,
                render_path   = r.render_path,
                grader_extra  = r.grader_extra,
                parent_id     = r.parent_id,
            )
            for r in archive.population
        ]

        # σ fresh random morphologies injected each generation (no parent)
        random_results = []
        if self.cfg.sigma > 0:
            random_morphs = self._random_population(self.cfg.sigma)
            random_results, id_counter = self._evaluate_batch(
                random_morphs, renderer, grader,
                generation   = generation,
                id_counter   = id_counter,
                save_renders = save_renders,
                render_dir   = render_dir,
                parent_ids   = None,
            )

        return parent_results + offspring_results + random_results, id_counter


# ---------------------------------------------------------------------------
# MAP-Elites Evolution
# ---------------------------------------------------------------------------

class MapEliteEvolution(BaseEvolution):
    """
    MAP-Elites evolution strategy.

    Initialisation  : generate and evaluate `init_size` random individuals
                      (defaults to max(mu, lambda_) × 2 for good grid coverage).
    Each step       :
      1. Sample λ parents from archive.get_parents(λ) (uniform over filled cells).
      2. Mutate each → λ offspring.
      3. Evaluate offspring.
      4. Return offspring only — the archive inserts/replaces cells.

    Parameters
    ----------
    cfg       : ExperimentConfig — uses lambda_ and all mutation params.
    init_size : number of individuals in the initial random population.
                If None, defaults to max(mu, lambda_) * 2.
    """

    def __init__(
        self,
        cfg:       ExperimentConfig,
        rng:       Optional[np.random.Generator] = None,
        init_size: Optional[int] = None,
    ):
        super().__init__(cfg, rng)
        self.init_size = init_size if init_size is not None else max(cfg.mu, cfg.lambda_) * 2

    def initialise(
        self,
        renderer,
        grader,
        id_counter:   int  = 0,
        save_renders: bool = False,
        render_dir:   Optional[str] = None,
    ) -> tuple[list[MorphologyResult], int]:
        morphs = self._random_population(self.init_size)
        return self._evaluate_batch(morphs, renderer, grader,
                                    generation=0,
                                    id_counter=id_counter,
                                    save_renders=save_renders,
                                    render_dir=render_dir)

    def step(
        self,
        archive:      MapEliteArchive,
        renderer,
        grader,
        generation:   int,
        id_counter:   int,
        save_renders: bool = False,
        render_dir:   Optional[str] = None,
    ) -> tuple[list[MorphologyResult], int]:
        # 1. Sample λ parents from filled grid cells (as full results to get IDs)
        sampled_parents = archive.get_parent_results(self.cfg.lambda_)
        parent_ids      = [r.individual_id for r in sampled_parents]

        # 2. Mutate → λ offspring morphologies
        offspring_morphs = [self._mutate_one(r.morphology) for r in sampled_parents]

        # 3. Evaluate offspring only (MAP-Elites never re-evaluates incumbents)
        return self._evaluate_batch(
            offspring_morphs, renderer, grader,
            generation   = generation,
            id_counter   = id_counter,
            save_renders = save_renders,
            render_dir   = render_dir,
            parent_ids   = parent_ids,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_evolution(cfg: ExperimentConfig,
                   rng: Optional[np.random.Generator] = None) -> BaseEvolution:
    """
    Return the correct evolution strategy for the given config.

    cfg.strategy must be "mu_lambda" or "map_elite".
    """
    if cfg.strategy == "mu_lambda":
        return MuLambdaEvolution(cfg, rng)
    elif cfg.strategy == "map_elite":
        return MapEliteEvolution(cfg, rng)
    else:
        raise ValueError(f"Unknown strategy '{cfg.strategy}'. "
                         f"Expected 'mu_lambda' or 'map_elite'.")


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from morphology import QUADRIPOD, TRIPOD, HEXAPOD, NewMorph, MutateMorphology

    print("=" * 60)
    print("  evolution.py — debug mode (no CLIP)")
    print("=" * 60)

    # Fake grader and renderer that don't need CLIP or MuJoCo
    import numpy as _np

    class _FakeRenderer:
        def render(self, morph, save_path=None, debug=False):
            from PIL import Image
            return Image.new("RGB", (32, 32), color=(128, 128, 128))

    class _FakeGrader:
        def __init__(self, rng):
            self._rng = rng
        @property
        def prompt_set(self):
            class _PS:
                name = "fake"
            return _PS()
        def score(self, image, debug=False):
            from grader import GraderOutput
            return GraderOutput(
                fitness    = float(self._rng.uniform(-1, 1)),
                raw_scores = {},
                method     = "fake",
                prompt_set = "fake",
            )

    rng      = _np.random.default_rng(0)
    renderer = _FakeRenderer()
    grader   = _FakeGrader(rng)

    # ---- Shared config --------------------------------------------------------
    cfg_mu = ExperimentConfig(
        strategy        = "mu_lambda",
        mu              = 5,
        lambda_         = 10,
        n_generations   = 3,
        seed            = 0,
        init_n_legs_min = 2,
        init_n_legs_max = 6,
    )
    cfg_me = ExperimentConfig(
        strategy        = "map_elite",
        n_generations   = 3,
        seed            = 0,
        init_n_legs_min = 2,
        init_n_legs_max = 6,
    )

    # ---- MuLambdaEvolution ----------------------------------------------------
    print("\n[1] MuLambdaEvolution\n")
    evo_mu  = MuLambdaEvolution(cfg_mu, rng=_np.random.default_rng(0))
    archive_mu = MuLambdaArchive(mu=cfg_mu.mu)

    init_results, id_ctr = evo_mu.initialise(renderer, grader)
    archive_mu.update(init_results)
    print(f"  Gen 0: {len(init_results)} evaluated  "
          f"best={archive_mu.best().fitness:+.5f}")

    for gen in range(1, cfg_mu.n_generations + 1):
        results, id_ctr = evo_mu.step(archive_mu, renderer, grader,
                                      generation=gen, id_counter=id_ctr)
        archive_mu.update(results)
        b = archive_mu.best()
        print(f"  Gen {gen}: pool={len(results)}  "
              f"best={b.fitness:+.5f}  id_ctr={id_ctr}")

    archive_mu.summary()

    # ---- MapEliteEvolution ----------------------------------------------------
    print("\n[2] MapEliteEvolution\n")
    from archive import MapEliteArchive
    evo_me  = MapEliteEvolution(cfg_me, rng=_np.random.default_rng(0))
    archive_me = MapEliteArchive(symmetry_bins=cfg_me.symmetry_bins)

    init_results, id_ctr = evo_me.initialise(renderer, grader)
    archive_me.update(init_results)
    print(f"  Gen 0: {len(init_results)} evaluated  "
          f"cells={len(archive_me.grid)}  "
          f"best={archive_me.best().fitness:+.5f}")

    id_ctr = len(init_results)
    for gen in range(1, cfg_me.n_generations + 1):
        results, id_ctr = evo_me.step(archive_me, renderer, grader,
                                      generation=gen, id_counter=id_ctr)
        archive_me.update(results)
        b = archive_me.best()
        print(f"  Gen {gen}: offspring={len(results)}  "
              f"cells={len(archive_me.grid)}  "
              f"best={b.fitness:+.5f}  id_ctr={id_ctr}")

    archive_me.summary()

    # ---- make_evolution factory -----------------------------------------------
    print("\n[3] make_evolution factory\n")
    for strat, cfg in (("mu_lambda", cfg_mu), ("map_elite", cfg_me)):
        evo = make_evolution(cfg)
        print(f"  strategy={strat}  → {type(evo).__name__}  OK")

    try:
        make_evolution(ExperimentConfig(strategy="bad_name"))
        print("  ERROR: should have raised ValueError")
    except ValueError as e:
        print(f" Invalid strategy correctly rejected: {e} → OK")

    print("\nAll evolution checks passed.")
