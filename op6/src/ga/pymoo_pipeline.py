"""pymoo interface for 7N sequence-pair NSGA-II chiplet placement."""

import multiprocessing
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ga.operators import initialize_population, process_offspring_chunk
from ga.pipeline import evaluate_population_parallel


class ChipletPlacementProblem(Problem):
    """Adapts parallel floorplan decoding into a pymoo multi-objective problem."""

    def __init__(
        self,
        num_chiplets: int,
        chiplet_dimensions: np.ndarray,
        min_slack: np.ndarray,
        max_slack: np.ndarray,
        chiplet_edges: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        cfg_opdir: dict,
        slack_steps: int,
        pool,
        batch_size: int
    ):
        super().__init__(
            n_var=7 * num_chiplets,
            n_obj=2,
            n_ieq_constr=2,
            xl=xl,
            xu=xu,
            vtype=int,
        )
        self.num_chiplets = num_chiplets
        self.chiplet_dimensions = chiplet_dimensions
        self.min_slack = min_slack
        self.max_slack = max_slack
        self.chiplet_edges = chiplet_edges
        self.cfg_opdir = cfg_opdir
        self.slack_steps = slack_steps
        self.pool = pool
        self.batch_size = batch_size

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        fitness_matrix, _, constraint_matrix = evaluate_population_parallel(
            population=X.astype(np.int16),
            num_chiplets=self.num_chiplets,
            chiplet_dimensions=self.chiplet_dimensions,
            min_slack=self.min_slack,
            max_slack=self.max_slack,
            slack_steps=self.slack_steps,
            chiplet_edges=self.chiplet_edges,
            cfg_opdir=self.cfg_opdir,
            pool=self.pool,
            batch_size=self.batch_size,
        )
        out["F"] = fitness_matrix
        out["G"] = constraint_matrix  # Pass constraints to pymoo solver


class SequencePairSampling(Sampling):
    """Adapts custom 7N sequence-pair initialization for pymoo."""

    def __init__(self, slack_steps: int):
        super().__init__()
        self.slack_steps = slack_steps

    def _do(self, problem: Problem, n_samples: int, **kwargs) -> np.ndarray:
        return initialize_population(
            pop_size=n_samples,
            num_chiplets=problem.n_var // 7,
            xl=problem.xl,
            xu=problem.xu,
            slack_steps=self.slack_steps
        )


class SequencePairVariation(Crossover):
    """Wraps process_offspring_chunk as a unified variation operator for pymoo."""

    def __init__(self, cfg_ga: dict):
        super().__init__(n_parents=2, n_offsprings=2)
        self.cfg_ga = cfg_ga

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        n_parents, n_matings, n_var = X.shape
        num_chiplets = n_var // 7

        parents_flat = np.empty((n_parents * n_matings, n_var), dtype=np.int16)
        for i in range(n_matings):
            parents_flat[2 * i] = X[0, i]
            parents_flat[2 * i + 1] = X[1, i]

        offspring_flat = process_offspring_chunk(
            parents_chunk=parents_flat,
            num_chiplets=num_chiplets,
            xl=problem.xl,
            xu=problem.xu,
            slack_steps=self.cfg_ga["slack_steps"],
            prob_ox=self.cfg_ga["prob_ox"],
            prob_blx=self.cfg_ga["prob_blx"],
            prob_hux=self.cfg_ga["prob_hux"],
            prob_mut_swap=self.cfg_ga["prob_mut_swap"],
            prob_mut_joint=self.cfg_ga["prob_mut_joint"],
            prob_mut_rot=self.cfg_ga["prob_mut_rot"],
            prob_mut_creep=self.cfg_ga["prob_mut_creep"],
            prob_immigrant=self.cfg_ga["prob_immigrant"],
            alpha_blx=self.cfg_ga["alpha_blx"],
            prob_seq_choice_x=self.cfg_ga["prob_seq_choice_x"],
            prob_positive_creep_shift=self.cfg_ga["prob_positive_creep_shift"]
        )

        Y = np.empty((2, n_matings, n_var), dtype=np.int16)
        for i in range(n_matings):
            Y[0, i] = offspring_flat[2 * i]
            Y[1, i] = offspring_flat[2 * i + 1]

        return Y


class IdentityMutation(Mutation):
    """Dummy mutation pass-through since SequencePairVariation handles both crossover and mutation."""

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        return X


def run_pymoo_nsga2(
    num_chiplets: int,
    chiplet_dimensions: np.ndarray,
    min_slack: np.ndarray,
    max_slack: np.ndarray,
    chiplet_edges: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    cfg_opdir: dict,
    cfg_ga: dict,
    pool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executes NSGA-II and returns full population with Pareto mask."""
    problem = ChipletPlacementProblem(
        num_chiplets=num_chiplets,
        chiplet_dimensions=chiplet_dimensions,
        min_slack=min_slack,
        max_slack=max_slack,
        chiplet_edges=chiplet_edges,
        xl=xl,
        xu=xu,
        cfg_opdir=cfg_opdir,
        slack_steps=cfg_ga["slack_steps"],
        pool=pool,
        batch_size=cfg_opdir["batch_size"],
    )

    algorithm = NSGA2(
        pop_size=cfg_opdir["pop_size"],
        sampling=SequencePairSampling(
            slack_steps=cfg_ga["slack_steps"]
        ),
        crossover=SequencePairVariation(cfg_ga=cfg_ga),
        mutation=IdentityMutation(),
        eliminate_duplicates=False,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg_opdir["max_generations"]),
        seed=1,
        verbose=True,
    )

    # Extract full final population
    pop_X = res.pop.get("X")
    pop_F = res.pop.get("F")

    # Compute Pareto non-dominated mask
    ranks = res.pop.get("rank")
    if ranks is not None and len(ranks) > 0 and ranks[0] is not None:
        is_pareto = ranks.flatten() == 0
    else:
        fronts = NonDominatedSorting().do(pop_F)
        is_pareto = np.zeros(len(pop_F), dtype=bool)
        if len(fronts) > 0:
            is_pareto[fronts[0]] = True

    return pop_X, pop_F, is_pareto