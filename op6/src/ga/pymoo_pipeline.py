"""pymoo interface for 7N sequence-pair NSGA-II chiplet placement.

This module provides custom pymoo wrappers (Problem, Sampling, Crossover,
and Mutation) to evaluate chiplet placements, apply domain-specific
sequence-pair variation operators, and execute multi-objective NSGA-II
optimization.
"""

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
    """Adapts parallel floorplan decoding into a pymoo multi-objective problem.

    Defines a multi-objective optimization problem with two objectives (Area and
    Wirelength) and two inequality constraints (Aspect Ratio min/max limits).

    Attributes:
        num_chiplets (int): Total number of chiplets N.
        chiplet_dimensions (np.ndarray): Array of shape (N, 2) containing chiplet
            base dimensions [width, height].
        min_slack (np.ndarray): Array of shape (4, N) defining lower physical
            whitespace bounds.
        max_slack (np.ndarray): Array of shape (4, N) defining upper physical
            whitespace bounds.
        chiplet_edges (np.ndarray): Array of shape (E, 2) representing netlist
            connectivity edges.
        cfg_opdir (dict): Operational directive configuration parameters.
        slack_steps (int): Total discrete step quantization count K.
        pool (multiprocessing.Pool | None): Worker process pool for parallel
            evaluation.
        batch_size (int): Chunk size per parallel process during evaluation.
    """

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
        batch_size: int,
    ):
        """Initializes ChipletPlacementProblem with system parameters and bounds.

        Args:
            num_chiplets: Total number of chiplets N.
            chiplet_dimensions: Float array of shape (N, 2) with chiplet [width, height].
            min_slack: Float array of shape (4, N) with minimum slack bounds.
            max_slack: Float array of shape (4, N) with maximum slack bounds.
            chiplet_edges: Integer connectivity matrix of shape (E, 2).
            xl: Lower variable bounds vector of shape (7N,).
            xu: Upper variable bounds vector of shape (7N,).
            cfg_opdir: Dictionary containing operational directive options.
            slack_steps: Discrete step quantization limit K.
            pool: Optional multiprocessing worker pool for parallel decoding.
            batch_size: Batch size used for chunked evaluation dispatch.
        """
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
        """Evaluates candidate solutions for objectives and constraints.

        Args:
            X: Population chromosome matrix of shape (pop_size, 7N).
            out: Output dictionary populated with objective matrix 'F' of shape
                (pop_size, 2) and constraint matrix 'G' of shape (pop_size, 2).
            *args: Additional positional arguments passed by pymoo.
            **kwargs: Additional keyword arguments passed by pymoo.
        """
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
        out["G"] = constraint_matrix


class SequencePairSampling(Sampling):
    """Adapts custom 7N sequence-pair initialization for pymoo.

    Generates an initial population of valid 7N integer chromosomes, ensuring
    Sequence X and Sequence Y are valid permutations.

    Attributes:
        slack_steps (int): Maximum discrete step index for whitespace slacks.
    """

    def __init__(self, slack_steps: int):
        """Initializes the sequence pair sampling operator.

        Args:
            slack_steps: Discrete step quantization count K for whitespace slacks.
        """
        super().__init__()
        self.slack_steps = slack_steps

    def _do(self, problem: Problem, n_samples: int, **kwargs) -> np.ndarray:
        """Generates initial sample population.

        Args:
            problem: Target optimization problem instance.
            n_samples: Number of population individuals to sample.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Sampled chromosome matrix of shape (n_samples, 7N)
            with np.int16 dtype.
        """
        return initialize_population(
            pop_size=n_samples,
            num_chiplets=problem.n_var // 7,
            xl=problem.xl,
            xu=problem.xu,
            slack_steps=self.slack_steps,
        )


class SequencePairVariation(Crossover):
    """Wraps process_offspring_chunk as a unified variation operator for pymoo.

    Applies combined crossover (OX, BLX-alpha, HUX) and mutation operators
    to paired parent chromosomes to produce offspring.

    Attributes:
        cfg_ga (dict): Genetic algorithm configuration directives.
    """

    def __init__(self, cfg_ga: dict):
        """Initializes SequencePairVariation with GA configuration.

        Args:
            cfg_ga: Dictionary containing genetic operator probabilities and
                hyperparameters.
        """
        super().__init__(n_parents=2, n_offsprings=2)
        self.cfg_ga = cfg_ga

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        """Performs genetic variation on parent chromosome pairs.

        Args:
            problem: Target optimization problem instance.
            X: Parent array of shape (2, n_matings, n_var).
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Offspring chromosome array of shape (2, n_matings, n_var).
        """
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
            prob_positive_creep_shift=self.cfg_ga["prob_positive_creep_shift"],
        )

        Y = np.empty((2, n_matings, n_var), dtype=np.int16)
        for i in range(n_matings):
            Y[0, i] = offspring_flat[2 * i]
            Y[1, i] = offspring_flat[2 * i + 1]

        return Y


class IdentityMutation(Mutation):
    """Dummy mutation pass-through operator for pymoo.

    Used because SequencePairVariation handles both crossover and mutation
    within a single unified execution pass.
    """

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        """Passes chromosomes through without alteration.

        Args:
            problem: Target optimization problem instance.
            X: Chromosome matrix of shape (pop_size, n_var).
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Unmodified chromosome matrix X.
        """
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
    """Executes NSGA-II multi-objective optimization via pymoo.

    Args:
        num_chiplets: Total chiplet count N.
        chiplet_dimensions: Array of shape (N, 2) containing base [width, height].
        min_slack: Array of shape (4, N) with lower physical slack bounds.
        max_slack: Array of shape (4, N) with upper physical slack bounds.
        chiplet_edges: Array of shape (E, 2) with netlist connectivity edges.
        xl: Array of shape (7N,) defining lower chromosome bounds.
        xu: Array of shape (7N,) defining upper chromosome bounds.
        cfg_opdir: Dictionary of operational directive settings.
        cfg_ga: Dictionary of genetic algorithm operator settings.
        pool: Optional worker pool for parallel evaluation/variation execution.

    Returns:
        A tuple containing:
            - pop_X (np.ndarray): Final population chromosome matrix of shape
              (pop_size, 7N).
            - pop_F (np.ndarray): Final objective value matrix of shape
              (pop_size, 2).
            - is_pareto (np.ndarray): Boolean mask array of shape (pop_size,)
              where True indicates non-dominated Pareto-optimal solutions.
    """
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
        sampling=SequencePairSampling(slack_steps=cfg_ga["slack_steps"]),
        crossover=SequencePairVariation(cfg_ga=cfg_ga),
        mutation=IdentityMutation(),
        eliminate_duplicates=False,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg_opdir["max_generations"]),
        seed=cfg_opdir["seed"],
        verbose=True,
    )

    pop_X = res.pop.get("X")
    pop_F = res.pop.get("F")

    ranks = res.pop.get("rank")
    if ranks is not None and len(ranks) > 0 and ranks[0] is not None:
        is_pareto = ranks.flatten() == 0
    else:
        fronts = NonDominatedSorting().do(pop_F)
        is_pareto = np.zeros(len(pop_F), dtype=bool)
        if len(fronts) > 0:
            is_pareto[fronts[0]] = True

    return pop_X, pop_F, is_pareto