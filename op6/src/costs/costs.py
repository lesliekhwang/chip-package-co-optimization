"""Cost functions and constraint evaluations for chiplet floorplanning."""

import numpy as np


def compute_area_cost(
    total_widths: np.ndarray,
    total_heights: np.ndarray
) -> np.ndarray:
    """Computes total floorplan bounding box area.

    Args:
        total_widths: Float array of shape (batch_size,) representing floorplan widths.
        total_heights: Float array of shape (batch_size,) representing floorplan heights.

    Returns:
        np.ndarray: Area cost array of shape (batch_size,).
    """
    return total_widths * total_heights


def compute_wirelength_cost(
    bbox_batch: np.ndarray,
    chiplet_edges: np.ndarray,
    use_wirelength: bool = True
) -> np.ndarray:
    """Computes Total Half-Perimeter Wirelength (HPWL) based on chiplet centroid distances.

    Args:
        bbox_batch: Float array of shape (batch_size, N, 4) with bounding boxes [X1, Y1, X2, Y2].
        chiplet_edges: Integer array of shape (E, 2) containing pairwise connectivity edges.
        use_wirelength: Boolean flag indicating whether wirelength evaluation is enabled.

    Returns:
        np.ndarray: HPWL cost array of shape (batch_size,).
    """
    pop_size = bbox_batch.shape[0]
    wirelength_cost = np.zeros(pop_size, dtype=np.float32)

    if chiplet_edges.shape[0] > 0 and use_wirelength:
        centroids_x = (bbox_batch[:, :, 0] + bbox_batch[:, :, 2]) / 2.0
        centroids_y = (bbox_batch[:, :, 1] + bbox_batch[:, :, 3]) / 2.0

        u_ids = chiplet_edges[:, 0]
        v_ids = chiplet_edges[:, 1]

        dx = np.abs(centroids_x[:, u_ids] - centroids_x[:, v_ids])
        dy = np.abs(centroids_y[:, u_ids] - centroids_y[:, v_ids])
        wirelength_cost = np.sum(dx + dy, axis=1)

    return wirelength_cost


def compute_aspect_ratio_constraints(
    total_widths: np.ndarray,
    total_heights: np.ndarray,
    aspect_ratio_bounds: list[float] | tuple[float, float] = (0.25, 4.0)
) -> np.ndarray:
    """Evaluates aspect ratio inequality constraints (g <= 0 indicates feasibility).

    Args:
        total_widths: Float array of shape (batch_size,) representing floorplan widths.
        total_heights: Float array of shape (batch_size,) representing floorplan heights.
        aspect_ratio_bounds: Min and max allowed layout aspect ratios [ar_min, ar_max].

    Returns:
        np.ndarray: Constraint matrix of shape (batch_size, 2) where g <= 0 is feasible.
    """
    ar_min, ar_max = aspect_ratio_bounds
    aspect_ratios = total_heights / np.maximum(total_widths, 1e-6)

    # g1: aspect_ratio - ar_max <= 0
    # g2: ar_min - aspect_ratio <= 0
    g1 = aspect_ratios - ar_max
    g2 = ar_min - aspect_ratios
    return np.column_stack([g1, g2])

