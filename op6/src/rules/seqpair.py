"""Batch Sequence Pair Decoder for Chiplet Placement."""

import numpy as np


def decode_sequence_pair_batch(
    chiplet_dims: np.ndarray,
    seq_x_batch: np.ndarray,
    seq_y_batch: np.ndarray,
    slack_batch: np.ndarray,
    rot_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decodes a batch of sequence pairs, slacks, and rotation genes into bounding boxes.

    Computes absolute 2D spatial coordinates for a batch of chiplet placement
    configurations using Longest Common Subsequence (LCS) constraint graphs derived
    from Sequence Pair (X, Y) permutations.

    Args:
        chiplet_dims: Float array of shape (N, 2) containing unrotated base
            dimensions [width, height] for each chiplet ID (0 to N - 1).
        seq_x_batch: Integer array of shape (B, N) containing Sequence X topology
            permutations for B individuals.
        seq_y_batch: Integer array of shape (B, N) containing Sequence Y topology
            permutations for B individuals.
        slack_batch: Float array of shape (B, 4, N) containing physical whitespace
            slacks per chiplet ordered as [North, East, South, West].
        rot_batch: Boolean array of shape (B, N) where True indicates a 90-degree
            chiplet rotation and False indicates an unrotated state.

    Returns:
        A tuple containing:
            - bbox_batch (np.ndarray): Float32 array of shape (B, N, 4) with
              bounding box coordinates [X1, Y1, X2, Y2] for each chiplet.
            - total_widths (np.ndarray): Float32 array of shape (B,) representing
              the total floorplan width per layout.
            - total_heights (np.ndarray): Float32 array of shape (B,) representing
              the total floorplan height per layout.
    """
    batch_size, num_chiplets = seq_x_batch.shape
    bbox_batch = np.zeros((batch_size, num_chiplets, 4), dtype=np.float32)
    total_widths = np.zeros(batch_size, dtype=np.float32)
    total_heights = np.zeros(batch_size, dtype=np.float32)

    for b in range(batch_size):
        seq_x = seq_x_batch[b]
        seq_y = seq_y_batch[b]
        slacks = slack_batch[b]  # Shape (4, N) -> [N, E, S, W]
        rot = rot_batch[b]       # Shape (N,) boolean vector

        base_w = np.where(rot, chiplet_dims[:, 1], chiplet_dims[:, 0])
        base_h = np.where(rot, chiplet_dims[:, 0], chiplet_dims[:, 1])

        # 1. Compute X-Coordinates (Horizontal LCS on X and Y)
        x_coords = np.zeros(num_chiplets, dtype=np.float32)
        pos_in_y = {chiplet_id: idx for idx, chiplet_id in enumerate(seq_y)}

        for i in range(num_chiplets):
            c_i = seq_x[i]
            idx_y_i = pos_in_y[c_i]

            max_x = 0.0
            for j in range(i):
                c_j = seq_x[j]
                if pos_in_y[c_j] < idx_y_i:
                    eff_w_j = base_w[c_j] + slacks[1, c_j] + slacks[3, c_j]
                    max_x = max(max_x, x_coords[c_j] + eff_w_j)
            x_coords[c_i] = max_x

        # 2. Compute Y-Coordinates (Vertical LCS on X_reversed and Y)
        y_coords = np.zeros(num_chiplets, dtype=np.float32)
        seq_x_rev = seq_x[::-1]

        for i in range(num_chiplets):
            c_i = seq_x_rev[i]
            idx_y_i = pos_in_y[c_i]

            max_y = 0.0
            for j in range(i):
                c_j = seq_x_rev[j]
                if pos_in_y[c_j] < idx_y_i:
                    eff_h_j = base_h[c_j] + slacks[0, c_j] + slacks[2, c_j]
                    max_y = max(max_y, y_coords[c_j] + eff_h_j)
            y_coords[c_i] = max_y

        # 3. Vectorized Bounding Box Assembly
        x1 = x_coords + slacks[3]
        y1 = y_coords + slacks[2]
        x2 = x1 + base_w
        y2 = y1 + base_h

        bbox_batch[b] = np.column_stack([x1, y1, x2, y2])
        total_widths[b] = np.max(x2 + slacks[1])
        total_heights[b] = np.max(y2 + slacks[0])

    return bbox_batch, total_widths, total_heights