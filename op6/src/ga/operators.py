"""Domain-specific genetic operators for 7N sequence pair chromosomes in int16 encoding.

Chromosome Layout (7N) [np.int16]:
- [0 : N]        Sequence X (topology permutation)
- [N : 2N]       Sequence Y (topology permutation)
- [2N : 6N]      Slack Slacks as Discrete Step Indices k in [0, slack_steps]
- [6N : 7N]      Boolean Rotations (0 = 0 deg, 1 = 90 deg)
"""

import numpy as np


# =============================================================================
# POPULATION INITIALIZATION & RANDOM IMMIGRANTS
# =============================================================================

def initialize_population(
    pop_size: int,
    num_chiplets: int,
    xl: np.ndarray,
    xu: np.ndarray,
    slack_steps: int
) -> np.ndarray:
    """Initializes a random population matrix for 7N sequence-pair chromosomes.

    Args:
        pop_size: Total number of individuals in the population.
        num_chiplets: Number of chiplets N in the system.
        xl: Lower bounds array for decision variables.
        xu: Upper bounds array for decision variables.
        slack_steps: Maximum discrete step index for slacks. Defaults to 10.
        prob_rot_true: Probability of initializing a chiplet rotation gene to 1 (90 deg). Defaults to 0.5.

    Returns:
        np.ndarray: Population matrix of shape (pop_size, 7N) with np.int16 dtype.
    """
    decision_vars = 7 * num_chiplets
    pop = np.zeros((pop_size, decision_vars), dtype=np.int16)

    for i in range(pop_size):
        pop[i] = random_immigrant_injection(
            num_chiplets, xl, xu, slack_steps=slack_steps
        )

    return pop


def random_immigrant_injection(
    num_chiplets: int,
    xl: np.ndarray,
    xu: np.ndarray,
    slack_steps: int
) -> np.ndarray:
    """Generates a single, uniformly random 7N int16 chromosome.

    Sequence-pair topology genes (SeqX, SeqY) are sampled as valid permutations,
    whitespace slack genes are sampled as discrete step indices in [0, slack_steps],
    and rotation genes are sampled as 0 or 1.

    Args:
        num_chiplets: Number of chiplets N in the system.
        xl: Lower bounds array for decision variables.
        xu: Upper bounds array for decision variables.
        slack_steps: Maximum discrete step index for slacks.

    Returns:
        np.ndarray: 1D chromosome vector of shape (7N,) with np.int16 dtype.
    """
    n = num_chiplets
    seq_x = np.random.permutation(n).astype(np.int16)
    seq_y = np.random.permutation(n).astype(np.int16)

    # Discrete slack indices k in [0, slack_steps]
    slack_min = int(xl[2 * n]) if len(xl) > 2 * n else 0
    slack_max = int(xu[2 * n]) if len(xu) > 2 * n else slack_steps
    slacks = np.random.randint(slack_min, slack_max + 1, size=4 * n, dtype=np.int16)

    # Boolean rotations sampled via configurable probability
    rotations = (np.random.rand(n) < 0.5).astype(np.int16)

    return np.concatenate([seq_x, seq_y, slacks, rotations]).astype(np.int16)


# =============================================================================
# CROSSOVER OPERATORS
# =============================================================================

def order_crossover_ox(
    parent1: np.ndarray,
    parent2: np.ndarray,
    num_chiplets: int
) -> tuple[np.ndarray, np.ndarray]:
    """Applies Order Crossover (OX) to topology permutation slices (Sequence X and Y).

    Preserves valid sequence-pair permutations without introducing duplicate chiplet IDs.
    Uses a int16 sentinel value (-1) during slice filling to prevent integer underflow.

    Args:
        parent1: First parent chromosome vector of shape (7N,).
        parent2: Second parent chromosome vector of shape (7N,).
        num_chiplets: Number of chiplets N in the system.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two offspring chromosome vectors of shape (7N,) with np.int16 dtype.
    """
    n = num_chiplets
    child1 = parent1.copy()
    child2 = parent2.copy()

    def _ox_permutation(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = len(p1)
        sentinel = np.int16(-1)  # Out-of-bounds marker for int16
        c1 = np.full(size, sentinel, dtype=np.int16)
        c2 = np.full(size, sentinel, dtype=np.int16)

        cut1, cut2 = sorted(np.random.choice(size, size=2, replace=False))
        c1[cut1:cut2] = p1[cut1:cut2]
        c2[cut1:cut2] = p2[cut1:cut2]

        fill_p2 = [item for item in p2 if item not in c1[cut1:cut2]]
        fill_p1 = [item for item in p1 if item not in c2[cut1:cut2]]

        idx1, idx2 = 0, 0
        for i in range(size):
            pos = (cut2 + i) % size
            if c1[pos] == sentinel:
                c1[pos] = fill_p2[idx1]
                idx1 += 1
            if c2[pos] == sentinel:
                c2[pos] = fill_p1[idx2]
                idx2 += 1
        return c1, c2

    # Apply OX to Sequence X (0:N) and Sequence Y (N:2N)
    child1[:n], child2[:n] = _ox_permutation(parent1[:n], parent2[:n])
    child1[n : 2 * n], child2[n : 2 * n] = _ox_permutation(
        parent1[n : 2 * n], parent2[n : 2 * n]
    )

    return child1, child2


def discrete_blend_crossover_blx_alpha(
    parent1: np.ndarray,
    parent2: np.ndarray,
    num_chiplets: int,
    xl: np.ndarray,
    xu: np.ndarray,
    alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Applies discrete BLX-alpha blend crossover to whitespace slack step indices.

    Blends slack step values within an extended interval [min - alpha*I, max + alpha*I],
    rounds to the nearest integer step index, and clips to valid bounds.

    Args:
        parent1: First parent chromosome vector of shape (7N,).
        parent2: Second parent chromosome vector of shape (7N,).
        num_chiplets: Number of chiplets N in the system.
        xl: Lower bounds array for decision variables.
        xu: Upper bounds array for decision variables.
        alpha: BLX alpha expansion factor. Defaults to 0.5.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two offspring chromosome vectors of shape (7N,) with np.int16 dtype.
    """
    n = num_chiplets
    slack_slice = slice(2 * n, 6 * n)

    s1 = parent1[slack_slice].astype(np.float32)
    s2 = parent2[slack_slice].astype(np.float32)

    diff = np.abs(s1 - s2)
    low = np.minimum(s1, s2) - alpha * diff
    high = np.maximum(s1, s2) + alpha * diff

    c1_float = np.random.uniform(low, high)
    c2_float = np.random.uniform(low, high)

    # Round to discrete integer step indices and clip
    c1_slack = np.clip(np.round(c1_float), xl[slack_slice], xu[slack_slice]).astype(np.int16)
    c2_slack = np.clip(np.round(c2_float), xl[slack_slice], xu[slack_slice]).astype(np.int16)

    child1, child2 = parent1.copy(), parent2.copy()
    child1[slack_slice] = c1_slack
    child2[slack_slice] = c2_slack

    return child1, child2


def half_uniform_crossover_hux(
    parent1: np.ndarray,
    parent2: np.ndarray,
    num_chiplets: int
) -> tuple[np.ndarray, np.ndarray]:
    """Applies Half Uniform Crossover (HUX) to boolean chiplet rotation bits.

    Identifies differing rotation genes between parents (0 vs 1) and swaps
    exactly half of them to create offspring.

    Args:
        parent1: First parent chromosome vector of shape (7N,).
        parent2: Second parent chromosome vector of shape (7N,).
        num_chiplets: Number of chiplets N in the system.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two offspring chromosome vectors of shape (7N,) with np.int16 dtype.
    """
    n = num_chiplets
    rot_slice = slice(6 * n, 7 * n)
    child1, child2 = parent1.copy(), parent2.copy()

    r1 = parent1[rot_slice]
    r2 = parent2[rot_slice]

    diff_indices = np.where(r1 != r2)[0]
    if len(diff_indices) == 0:
        return child1, child2

    swap_count = len(diff_indices) // 2
    if swap_count > 0:
        swap_idx = np.random.choice(diff_indices, size=swap_count, replace=False)
        c1_rot, c2_rot = parent1[rot_slice].copy(), parent2[rot_slice].copy()
        c1_rot[swap_idx] = parent2[rot_slice][swap_idx]
        c2_rot[swap_idx] = parent1[rot_slice][swap_idx]

        child1[rot_slice] = c1_rot
        child2[rot_slice] = c2_rot

    return child1, child2


# =============================================================================
# MUTATION OPERATORS
# =============================================================================

def single_chiplet_rotation_mutation(
    individual: np.ndarray,
    num_chiplets: int
) -> np.ndarray:
    """Flips the boolean rotation bit (0 <-> 1) of a randomly selected chiplet.

    Args:
        individual: Chromosome vector of shape (7N,).
        num_chiplets: Number of chiplets N in the system.

    Returns:
        np.ndarray: Mutated chromosome vector of shape (7N,) with np.int16 dtype.
    """
    n = num_chiplets
    mutated = individual.copy()
    chiplet_idx = np.random.randint(0, n)
    rot_pos = 6 * n + chiplet_idx

    # Flip boolean rotation state (0 <-> 1)
    mutated[rot_pos] = 1 - mutated[rot_pos]
    return mutated


def swap_chiplet_mutation(
    individual: np.ndarray,
    num_chiplets: int,
    prob_seq_choice_x: float
) -> np.ndarray:
    """Swaps two distinct chiplet positions in either Sequence X or Sequence Y.

    Args:
        individual: Chromosome vector of shape (7N,).
        num_chiplets: Number of chiplets N in the system.
        prob_seq_choice_x: Probability of targeting Sequence X instead of Sequence Y.

    Returns:
        np.ndarray: Mutated chromosome vector of shape (7N,) with np.int16 dtype.
    """
    n = num_chiplets
    mutated = individual.copy()
    idx1, idx2 = np.random.choice(n, size=2, replace=False)

    if np.random.rand() < prob_seq_choice_x:
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    else:
        mutated[n + idx1], mutated[n + idx2] = mutated[n + idx2], mutated[n + idx1]

    return mutated


def joint_chiplet_swap_mutation(
    individual: np.ndarray,
    num_chiplets: int,
) -> np.ndarray:
    """Swaps two distinct chiplet positions in both Sequence X and Sequence Y simultaneously.

    Args:
        individual: Chromosome vector of shape (7N,).
        num_chiplets: Number of chiplets N in the system.

    Returns:
        np.ndarray: Mutated chromosome vector of shape (7N,) with np.int16 dtype.
    """
    n = num_chiplets
    mutated = individual.copy()
    idx1, idx2 = np.random.choice(n, size=2, replace=False)

    # Swap in Sequence X
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    # Swap in Sequence Y
    mutated[n + idx1], mutated[n + idx2] = mutated[n + idx2], mutated[n + idx1]

    return mutated


def creep_mutation(
    individual: np.ndarray,
    num_chiplets: int,
    xl: np.ndarray,
    xu: np.ndarray,
    prob_positive_shift: float
) -> np.ndarray:
    """Applies discrete random-walk creep mutation (+1 or -1 grid step) to whitespace slacks.

    Converts slacks to signed integers internally to avoid int16 underflow,
    applies direction shifts, and clips results within bounds before returning int16.

    Args:
        individual: Chromosome vector of shape (7N,).
        num_chiplets: Number of chiplets N in the system.
        xl: Lower bounds array for decision variables.
        xu: Upper bounds array for decision variables.
        prob_positive_shift: Probability of shifting a slack step positively (+1 vs -1).

    Returns:
        np.ndarray: Mutated chromosome vector of shape (7N,) with np.int16 dtype.
    """
    n = num_chiplets
    slack_slice = slice(2 * n, 6 * n)
    mutated = individual.copy()

    xl_slack = xl[slack_slice]
    xu_slack = xu[slack_slice]

    # Convert to signed int to prevent int16 underflow during negative shifts
    current_slacks = mutated[slack_slice]
    shifts = np.where(np.random.rand(4 * n) < prob_positive_shift, 1, -1)
    new_slacks = np.clip(current_slacks + shifts, xl_slack, xu_slack)

    mutated[slack_slice] = new_slacks.astype(np.int16)
    return mutated


# =============================================================================
# CHUNK OFFSPRING PROCESSING OPERATOR
# =============================================================================

def process_offspring_chunk(
    parents_chunk: np.ndarray,
    num_chiplets: int,
    xl: np.ndarray,
    xu: np.ndarray,
    slack_steps: int,
    prob_ox: float,
    prob_blx: float,
    prob_hux: float,
    prob_mut_swap: float,
    prob_mut_joint: float,
    prob_mut_rot: float,
    prob_mut_creep: float,
    prob_immigrant: float,
    alpha_blx: float,
    prob_seq_choice_x: float,
    prob_positive_creep_shift: float
) -> np.ndarray:
    """Processes a chunk of parent chromosomes and generates offspring (np.int16).

    Executes crossover (OX, Discrete BLX-alpha, HUX), mutation (Swap, Joint Swap, Rotation,
    Creep), and random immigrant injection sequentially on an array chunk. Designed as a
    pure, picklable target for serial, multiprocessing, Ray, or MPI execution.

    Args:
        parents_chunk: Array of parent chromosome vectors of shape (chunk_size, 7N).
        num_chiplets: Number of chiplets N in the system.
        xl: Lower bounds array for decision variables.
        xu: Upper bounds array for decision variables.
        slack_steps: Maximum discrete step index for slacks.
        prob_ox: Probability of applying Order Crossover (OX).
        prob_blx: Probability of applying Discrete BLX-alpha crossover.
        prob_hux: Probability of applying Half Uniform Crossover (HUX).
        prob_mut_swap: Probability of applying single sequence swap mutation.
        prob_mut_joint: Probability of applying joint sequence swap mutation.
        prob_mut_rot: Probability of applying single chiplet rotation mutation.
        prob_mut_creep: Probability of applying discrete creep mutation.
        prob_immigrant: Probability of replacing an offspring with a random immigrant.
        alpha_blx: Expansion factor alpha for Discrete BLX-alpha crossover.
        prob_seq_choice_x: Probability of selecting Sequence X during swap mutation.
        prob_positive_creep_shift: Probability of positive direction shift in creep mutation.

    Returns:
        np.ndarray: Generated offspring chunk matrix of shape (chunk_size, 7N) with np.int16 dtype.
    """
    n_offspring = parents_chunk.shape[0]
    offspring = parents_chunk.copy()

    # 1. Crossover Phase (Pairwise)
    for i in range(0, n_offspring - 1, 2):
        p1, p2 = offspring[i], offspring[i + 1]

        if np.random.rand() < prob_ox:
            p1, p2 = order_crossover_ox(p1, p2, num_chiplets)

        if np.random.rand() < prob_blx:
            p1, p2 = discrete_blend_crossover_blx_alpha(
                p1, p2, num_chiplets, xl, xu, alpha=alpha_blx
            )

        if np.random.rand() < prob_hux:
            p1, p2 = half_uniform_crossover_hux(p1, p2, num_chiplets)

        offspring[i], offspring[i + 1] = p1, p2

    # 2. Mutation Phase (Per Individual)
    for i in range(n_offspring):
        if np.random.rand() < prob_mut_swap:
            offspring[i] = swap_chiplet_mutation(
                offspring[i], num_chiplets, prob_seq_choice_x=prob_seq_choice_x
            )

        if np.random.rand() < prob_mut_joint:
            offspring[i] = joint_chiplet_swap_mutation(offspring[i], num_chiplets)

        if np.random.rand() < prob_mut_rot:
            offspring[i] = single_chiplet_rotation_mutation(offspring[i], num_chiplets)

        if np.random.rand() < prob_mut_creep:
            offspring[i] = creep_mutation(
                offspring[i],
                num_chiplets,
                xl,
                xu,
                prob_positive_shift=prob_positive_creep_shift,
            )

        if np.random.rand() < prob_immigrant:
            offspring[i] = random_immigrant_injection(
                num_chiplets,
                xl,
                xu,
                slack_steps=slack_steps
            )

    return offspring