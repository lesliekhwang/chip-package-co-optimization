"""Execution pipeline and parallel driver functions for GA floorplanning."""

import multiprocessing
import numpy as np

from utils.logger import log_get
from rules.transforms import decode_chunk_worker
from ga.operators import initialize_population, process_offspring_chunk

logger = log_get(__name__)


def construct_chromosome_bounds(
    num_chiplets: int,
    slack_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Constructs lower (xl) and upper (xu) bound vectors for 7N integer chromosomes.

    Args:
        num_chiplets: Number of chiplets N in the system.
        slack_steps: Discrete step quantization limit K.

    Returns:
        tuple[np.ndarray, np.ndarray]: Lower and upper bounds of shape (7N,) in int16.
    """
    n = num_chiplets
    xl = np.zeros(7 * n, dtype=np.int16)
    xu = np.zeros(7 * n, dtype=np.int16)

    # 1. Topology Sequences [0 : 2N] -> Permutation indices [0, N - 1]
    xu[: 2 * n] = max(0, n - 1)

    # 2. Whitespace Slacks [2N : 6N] -> Discrete step indices [0, slack_steps]
    xl[2 * n : 6 * n] = 0
    xu[2 * n : 6 * n] = slack_steps

    # 3. Rotations [6N : 7N] -> Boolean bits [0, 1]
    xl[6 * n : 7 * n] = 0
    xu[6 * n : 7 * n] = 1

    return xl, xu


def evaluate_population_parallel(
    population: np.ndarray,
    num_chiplets: int,
    chiplet_dimensions: np.ndarray,
    min_slack: np.ndarray,
    max_slack: np.ndarray,
    slack_steps: int,
    chiplet_edges: np.ndarray,
    cfg_opdir: dict,
    batch_size: int,
    pool
) -> tuple[np.ndarray, np.ndarray]:
    """Decodes chromosomes in parallel and computes multi-objective costs.

    Args:
        population: Matrix of shape (pop_size, 7N) containing int16 chromosomes.
        num_chiplets: Total chiplet count N.
        chiplet_dimensions: (N, 2) array of chiplet [width, height].
        min_slack: (4, N) minimum physical slack bounds.
        max_slack: (4, N) maximum physical slack bounds.
        slack_steps: Discrete step resolution K.
        chiplet_edges: (E, 2) pairwise connectivity matrix.
        cfg_opdir: Directives dictionary containing objective evaluation flags.
        pool: Optional multiprocessing Pool.
        batch_size: Chunk size per parallel process.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - fitness_matrix: (pop_size, M) array of evaluated objective costs.
            - layout_bboxes: (pop_size, N, 4) array of physical bounding boxes.
    """
    pop_size = population.shape[0]

    # 1. Chunk population for worker execution
    chunks = [population[i : i + batch_size] for i in range(0, pop_size, batch_size)]

    worker_args = [
        (chunk, num_chiplets, chiplet_dimensions, min_slack, max_slack, slack_steps)
        for chunk in chunks
    ]

    # 2. Parallel or serial execution of batch decoder
    if pool and pop_size > batch_size:
        decoded_results = pool.starmap(decode_chunk_worker, worker_args)
    else:
        decoded_results = [decode_chunk_worker(*args) for args in worker_args]

    # Reassemble parallel worker chunks
    bbox_batch = np.vstack([res[0] for res in decoded_results])
    total_widths = np.concatenate([res[1] for res in decoded_results])
    total_heights = np.concatenate([res[2] for res in decoded_results])

    # 3. Objective Cost Evaluation
    # Compute area
    area_cost = total_widths * total_heights

    # Compute Total Half-Perimeter Wirelength (HPWL) if connection edges exist
    wirelength_cost = np.zeros(pop_size, dtype=np.float32)
    if chiplet_edges.shape[0] > 0 and cfg_opdir["use_cost_wirelength"]:
        centroids_x = (bbox_batch[:, :, 0] + bbox_batch[:, :, 2]) / 2.0
        centroids_y = (bbox_batch[:, :, 1] + bbox_batch[:, :, 3]) / 2.0

        u_ids = chiplet_edges[:, 0]
        v_ids = chiplet_edges[:, 1]

        dx = np.abs(centroids_x[:, u_ids] - centroids_x[:, v_ids])
        dy = np.abs(centroids_y[:, u_ids] - centroids_y[:, v_ids])
        wirelength_cost = np.sum(dx + dy, axis=1)

    fitness_matrix = np.column_stack([area_cost, wirelength_cost])

    # 4. Aspect Ratio Constraint Evaluation (g <= 0 is feasible)
    ar_min, ar_max = cfg_opdir.get("aspect_ratio_bounds", [0.25, 4.0])
    aspect_ratios = total_heights / np.maximum(total_widths, 1e-6)

    # g1: aspect_ratio - ar_max <= 0
    # g2: ar_min - aspect_ratio <= 0
    g1 = aspect_ratios - ar_max
    g2 = ar_min - aspect_ratios
    constraint_matrix = np.column_stack([g1, g2])

    return fitness_matrix, bbox_batch, constraint_matrix


def generate_offspring_parallel(
    parents: np.ndarray,
    num_chiplets: int,
    xl: np.ndarray,
    xu: np.ndarray,
    cfg_ga: dict,
    batch_size: int,
    pool
) -> np.ndarray:
    """Dispatches parent chunks to process_offspring_chunk across worker pool.

    Args:
        parents: Parent chromosome matrix of shape (pop_size, 7N).
        num_chiplets: Number of chiplets N.
        xl: Lower bounds array (7N,).
        xu: Upper bounds array (7N,).
        cfg_ga: GA operator configuration dictionary.
        pool: Optional multiprocessing Pool.
        batch_size: Chunk size per parallel process.

    Returns:
        np.ndarray: Generated offspring chromosome matrix of shape (pop_size, 7N).
    """
    n_parents = parents.shape[0]
    chunk_step = batch_size if batch_size % 2 == 0 else batch_size + 1
    chunks = [parents[i : i + chunk_step] for i in range(0, n_parents, chunk_step)]

    args = [
        (
            chunk,
            num_chiplets,
            xl,
            xu,
            cfg_ga["slack_steps"],
            cfg_ga["prob_ox"],
            cfg_ga["prob_blx"],
            cfg_ga["prob_hux"],
            cfg_ga["prob_mut_swap"],
            cfg_ga["prob_mut_joint"],
            cfg_ga["prob_mut_rot"],
            cfg_ga["prob_mut_creep"],
            cfg_ga["prob_immigrant"],
            cfg_ga["alpha_blx"],
            cfg_ga["prob_seq_choice_x"],
            cfg_ga["prob_positive_creep_shift"]
        )
        for chunk in chunks
    ]

    if pool and n_parents > batch_size:
        results = pool.starmap(process_offspring_chunk, args)
        return np.vstack(results)

    results = [process_offspring_chunk(*a) for a in args]
    return np.vstack(results)


def run_ga_optimization(
    num_chiplets: int,
    chiplet_dimensions: np.ndarray,
    min_slack: np.ndarray,
    max_slack: np.ndarray,
    chiplet_edges: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    cfg_opdir: dict,
    cfg_ga: dict,
    pool
) -> tuple[np.ndarray, np.ndarray]:
    """Executes the full genetic algorithm generational loop.

    Args:
        num_chiplets: Total chiplet count N.
        chiplet_dimensions: (N, 2) chiplet dimensions array.
        min_slack: (4, N) minimum slack bounds array.
        max_slack: (4, N) maximum slack bounds array.
        chiplet_edges: Pairwise connections array.
        xl: Lower decision variable bounds (7N,).
        xu: Upper decision variable bounds (7N,).
        cfg_opdir: Operational directives dictionary.
        cfg_ga: GA parameters dictionary.
        pool: Optional worker pool.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pareto-optimal population and fitness matrix.
    """
    slack_steps = cfg_ga["slack_steps"]
    batch_size = cfg_opdir.get("batch_size", 100)
    pop_size = cfg_opdir.get("pop_size", 200)
    max_generations = cfg_opdir.get("max_generations", 50)

    # Initialize Population
    population = initialize_population(
        pop_size=pop_size,
        num_chiplets=num_chiplets,
        xl=xl,
        xu=xu,
        slack_steps=slack_steps
    )

    # Initial Fitness Evaluation
    fitness_matrix, _ = evaluate_population_parallel(
        population=population,
        num_chiplets=num_chiplets,
        chiplet_dimensions=chiplet_dimensions,
        min_slack=min_slack,
        max_slack=max_slack,
        slack_steps=slack_steps,
        chiplet_edges=chiplet_edges,
        cfg_opdir=cfg_opdir,
        pool=pool,
        batch_size=batch_size
    )

    logger.info("Starting Genetic Optimization Loop...")

    for gen in range(1, max_generations + 1):
        offspring = generate_offspring_parallel(
            parents=population,
            num_chiplets=num_chiplets,
            xl=xl,
            xu=xu,
            cfg_ga=cfg_ga,
            pool=pool,
            batch_size=batch_size
        )

        offspring_fitness, _ = evaluate_population_parallel(
            population=offspring,
            num_chiplets=num_chiplets,
            chiplet_dimensions=chiplet_dimensions,
            min_slack=min_slack,
            max_slack=max_slack,
            slack_steps=slack_steps,
            chiplet_edges=chiplet_edges,
            cfg_opdir=cfg_opdir,
            pool=pool,
            batch_size=batch_size
        )

        # Environmental Selection (Combine + Truncate)
        combined_pop = np.vstack([population, offspring])
        combined_fitness = np.vstack([fitness_matrix, offspring_fitness])

        best_indices = np.argsort(combined_fitness[:, 0])[:pop_size]
        population = combined_pop[best_indices]
        fitness_matrix = combined_fitness[best_indices]

        logger.info(
            f"Gen [{gen}/{max_generations}] -> Best Area: {fitness_matrix[0, 0]:.2f}, "
            f"Wirelength: {fitness_matrix[0, 1]:.2f}"
        )

    return population, fitness_matrix