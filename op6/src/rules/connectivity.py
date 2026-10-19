"""Module for parsing chiplet graph topologies.

This module consumes chiplet design configurations to: 
1) Assign unique numeric chiplet IDs
2) Extract non-repeating pairwise connection tuples between them.
"""
import numpy as np
from typing import Any

from utils.logger import log_get

logger = log_get(__name__)


def assign_chiplet_ids(
    chiplet_data: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Assigns a unique integer identifier to each chiplet name.

    Args:
        chiplet_data: A dictionary where top-level keys represent unique
            chiplet names mapped to their respective attributes.

    Returns:
        A dictionary mapping each chiplet name to its assigned integer ID.
    """
    chiplet_id_map = {name: idx for idx, name in enumerate(chiplet_data.keys())}
    logger.info("Successfully assigned unique IDs to %d chiplets.", len(chiplet_id_map))
    return chiplet_id_map


def extract_pairwise_connections(
    chiplet_data: dict[str, dict[str, Any]],
    chiplet_id_map: dict[str, int] | None = None,
) -> np.ndarray:
    """Extracts non-repeating pairwise connections between chiplet integer IDs.

    Iterates through adjacency definitions to build undirected edges. Ensures
    edges are non-repeating by storing them in canonical sorted order
    (min_id, max_id) within a set, then returns them as a 2D integer matrix
    formatted for tensor indexing on GPU.

    Args:
        chiplet_data: A dictionary where top-level keys are chiplet names,
            and values contain an 'adjacency' list of neighbor chiplet names.
        chiplet_id_map: Optional pre-computed mapping from chiplet names to integer
            IDs. If None, IDs will be generated automatically.

    Returns:
        An (E, 2) int64 NumPy array where each row represents a connected pair
        [u_id, v_id] ready for direct conversion to PyTorch GPU tensors.
    """
    if chiplet_id_map is None:
        chiplet_id_map = assign_chiplet_ids(chiplet_data)

    unique_edges: set[tuple[int, int]] = set()

    for chiplet_name, info in chiplet_data.items():
        if chiplet_name not in chiplet_id_map:
            logger.error("Chiplet '%s' not found in ID map!", chiplet_name)
            raise ValueError("Chiplet ID not found!")

        u_id = chiplet_id_map[chiplet_name]
        neighbors = info.get("adjacency", [])

        for neighbor_name in neighbors:
            if neighbor_name not in chiplet_id_map:
                logger.error(
                    "Neighbor chiplet '%s' referenced by '%s' not in ID map.",
                    neighbor_name,
                    chiplet_name,
                )
                raise ValueError("Chiplet ID reference not found!")

            v_id = chiplet_id_map[neighbor_name]

            # Store undirected edge in canonical order (smaller ID first) to prevent duplicates
            if u_id != v_id:
                edge = (min(u_id, v_id), max(u_id, v_id))
                unique_edges.add(edge)

    sorted_connections = sorted(unique_edges)
    logger.info(
        "Extracted %d unique pairwise connection(s) from netlist.",
        len(sorted_connections),
    )

    # Convert to (E, 2) uint16 matrix for PyTorch/GPU indexing compatibility
    if not sorted_connections:
        return np.empty((0, 2), dtype=np.uint16)

    return np.array(sorted_connections, dtype=np.uint16)

def parse_slack_bounds(
    bounds_config: dict[str, Any],
    chiplet_id_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Parses per-chiplet slack lower and upper bounds matrices from a dictionary.

    Args:
        bounds_config: Parsed dictionary containing per-chiplet bounds configurations.
        chiplet_id_map: Dictionary mapping chiplet string names to their integer
            column position index (0 to N - 1) in the chromosome space.

    Returns:
        A tuple containing:
            - slack_lower_bounds: A (4, N) float32 NumPy array representing minimum
              N, E, S, W slack slacks for each chiplet.
            - slack_upper_bounds: A (4, N) float32 NumPy array representing maximum
              N, E, S, W slack slacks for each chiplet.

    Raises:
        KeyError: If a chiplet name in chiplet_id_map is missing from bounds_config,
            or if 'min_slack' / 'max_slack' keys are omitted.
        ValueError: If min or max slack vectors do not contain 4 elements.
    """
    num_chiplets = len(chiplet_id_map)
    slack_lower_bounds = np.zeros((4, num_chiplets), dtype=np.float32)
    slack_upper_bounds = np.zeros((4, num_chiplets), dtype=np.float32)

    for chiplet_name, index in chiplet_id_map.items():
        if chiplet_name not in bounds_config:
            logger.error(
                f"Chiplet '{chiplet_name}' in system map missing from bounds config."
            )
            raise KeyError(
                f"Chiplet '{chiplet_name}' not found in system bounds configuration."
            )

        chiplet_cfg = bounds_config[chiplet_name]
        if "min_slack" not in chiplet_cfg or "max_slack" not in chiplet_cfg:
            raise KeyError(
                f"Chiplet '{chiplet_name}' configuration must contain both "
                f"'min_slack' and 'max_slack' fields."
            )

        min_ws = chiplet_cfg["min_slack"]
        max_ws = chiplet_cfg["max_slack"]

        if len(min_ws) != 4 or len(max_ws) != 4:
            raise ValueError(
                f"slack bounds for '{chiplet_name}' must be length-4 lists [N, E, S, W]. "
                f"Got min_len={len(min_ws)}, max_len={len(max_ws)}."
            )

        slack_lower_bounds[:, index] = min_ws
        slack_upper_bounds[:, index] = max_ws

    logger.info(
        f"Successfully parsed (4, {num_chiplets}) slack lower/upper bound matrices."
    )
    return slack_lower_bounds, slack_upper_bounds

def parse_chiplet_dimensions(
    chiplet_data: dict[str, dict[str, Any]],
    chiplet_id_map: dict[str, int],
) -> np.ndarray:
    """Parses per-chiplet dimensions into a structured NumPy array.

    Args:
        chiplet_data: Parsed configuration dictionary containing chiplet properties.
        chiplet_id_map: Dictionary mapping chiplet string names to their assigned 
            integer IDs (which correspond to row indices).

    Returns:
        A (N, 2) float32 NumPy array where the row index matches the chiplet ID,
        and columns represent [width, height].
        
    Raises:
        KeyError: If a chiplet is missing from the data or lacks a 'dim' attribute.
        ValueError: If the 'dim' list does not contain exactly two elements.
    """
    num_chiplets = len(chiplet_id_map)
    dimensions_matrix = np.zeros((num_chiplets, 2), dtype=np.float32)

    for chiplet_name, chiplet_id in chiplet_id_map.items():
        if chiplet_name not in chiplet_data:
            logger.error("Chiplet '%s' in ID map missing from chiplet_data.", chiplet_name)
            raise KeyError(f"Chiplet '{chiplet_name}' not found in system configuration.")
        
        chiplet_cfg = chiplet_data[chiplet_name]
        
        if "dim" not in chiplet_cfg:
            logger.error("Chiplet '%s' missing 'dim' attribute.", chiplet_name)
            raise KeyError(f"Chiplet '{chiplet_name}' configuration must contain a 'dim' field.")
            
        dims = chiplet_cfg["dim"]
        
        if len(dims) != 2:
            raise ValueError(
                f"Dimensions for '{chiplet_name}' must be length-2 [width, height]. "
                f"Got len={len(dims)}."
            )
            
        dimensions_matrix[chiplet_id, :] = dims
        
    logger.info("Successfully parsed (N, 2) dimension matrix for %d chiplets.", num_chiplets)
    return dimensions_matrix