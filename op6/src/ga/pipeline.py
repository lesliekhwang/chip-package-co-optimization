"""Execution pipeline and parallel driver functions for GA floorplanning.

This module handles chromosome bound construction, parallel evaluation of candidate
populations, and parallel offspring generation for sequence-pair multi-objective
chiplet placement optimization.
"""

import numpy as np

from utils.logger import log_get
from rules.transforms import decode_chunk_worker
from ga.operators import process_offspring_chunk
from costs.costs import (
    compute_area_cost,
    compute_wirelength_cost,
    compute_aspect_ratio_constraints,
)

logger = log_get(__name__)


def construct_chromosome_bounds(
    num_chiplets: int,
    slack_steps: int,
    place_compact: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Constructs lower (xl) and upper (xu) bound vectors for 7N integer chromosomes.

    Args:
        num_chiplets: Total number of chiplets N in the system.
        slack_steps: Discrete step quantization limit K for whitespace slacks.

    Returns:
        A tuple containing:
            - xl (np.ndarray): Lower variable bounds vector of shape (7N,) with
              np.int16 dtype.
            - xu (np.ndarray): Upper variable bounds vector of shape (7N,) with
              np.int16 dtype.
    """
    n = num_chiplets
    xl = np.zeros(7 * n, dtype=np.int16)
    xu = np.zeros(7 * n, dtype=np.int16)

    xu[: 2 * n] = max(0, n - 1)          # Topology Sequences [0 : 2N]
    xl[2 * n : 6 * n] = 0                # Whitespace Slacks [2N : 6N]
    xu[2 * n : 6 * n] = 0 if place_compact else slack_steps
    xl[6 * n : 7 * n] = 0                # Rotations [6N : 7N]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decodes chromosomes in parallel and computes multi-objective costs and constraints.

    Splits the population into chunked batches, decodes physical layouts across
    worker processes, and evaluates objective costs (Area, Wirelength) and layout
    constraints (Aspect Ratio).

    Args:
        population: Chromosome matrix of shape (pop_size, 7N) containing int16 genes.
        num_chiplets: Total number of chiplets N.
        chiplet_dimensions: Float array of shape (N, 2) with chiplet [width, height].
        min_slack: Float array of shape (4, N) with minimum whitespace slack bounds.
        max_slack: Float array of shape (4, N) with maximum whitespace slack bounds.
        slack_steps: Discrete step resolution K for whitespace slacks.
        chiplet_edges: Integer array of shape (E, 2) containing pairwise connectivity edges.
        cfg_opdir: Dictionary containing operational directive options and settings.
        batch_size: Chunk size per parallel process.
        pool: Optional multiprocessing Pool instance for parallel worker execution.

    Returns:
        A tuple containing:
            - fitness_matrix (np.ndarray): Evaluated objective matrix of shape (pop_size, 2)
              with [Area, Wirelength].
            - bbox_batch (np.ndarray): Decoded physical bounding box matrix of shape
              (pop_size, N, 4) with [X1, Y1, X2, Y2].
            - constraint_matrix (np.ndarray): Aspect ratio constraint matrix of shape
              (pop_size, 2) where values <= 0 indicate feasibility.
    """
    pop_size = population.shape[0]

    chunks = [population[i : i + batch_size] for i in range(0, pop_size, batch_size)]
    worker_args = [
        (chunk, num_chiplets, chiplet_dimensions, min_slack, max_slack, slack_steps)
        for chunk in chunks
    ]

    if pool and pop_size > batch_size:
        decoded_results = pool.starmap(decode_chunk_worker, worker_args)
    else:
        decoded_results = [decode_chunk_worker(*args) for args in worker_args]

    bbox_batch = np.vstack([res[0] for res in decoded_results])
    total_widths = np.concatenate([res[1] for res in decoded_results])
    total_heights = np.concatenate([res[2] for res in decoded_results])

    # Objective Cost Evaluation with Fallbacks
    area_cost = compute_area_cost(total_widths, total_heights)
    wirelength_cost = compute_wirelength_cost(
        bbox_batch=bbox_batch,
        chiplet_edges=chiplet_edges,
        use_wirelength=cfg_opdir.get("use_cost_wirelength", True),
    )
    fitness_matrix = np.column_stack([area_cost, wirelength_cost])

    # Aspect Ratio Constraint Evaluation with Fallbacks
    ar_bounds = cfg_opdir.get("aspect_ratio_bounds", [0.25, 4.0])
    constraint_matrix = compute_aspect_ratio_constraints(
        total_widths=total_widths,
        total_heights=total_heights,
        aspect_ratio_bounds=ar_bounds,
    )

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

    Splits parent chromosomes into chunks and applies genetic variation operators
    (crossovers, mutations, and random immigrant injections) in parallel.

    Args:
        parents: Parent chromosome matrix of shape (pop_size, 7N) with int16 dtype.
        num_chiplets: Total number of chiplets N.
        xl: Lower variable bounds array of shape (7N,).
        xu: Upper variable bounds array of shape (7N,).
        cfg_ga: Dictionary containing genetic algorithm operator parameters.
        batch_size: Chunk size per parallel worker process.
        pool: Optional multiprocessing Pool instance for parallel worker execution.

    Returns:
        np.ndarray: Generated offspring chromosome matrix of shape (pop_size, 7N)
        with np.int16 dtype.
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
            cfg_ga["prob_positive_creep_shift"],
        )
        for chunk in chunks
    ]

    if pool and n_parents > batch_size:
        results = pool.starmap(process_offspring_chunk, args)
        return np.vstack(results)

    results = [process_offspring_chunk(*a) for a in args]
    return np.vstack(results)