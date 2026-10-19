import numpy as np

from mappers.seqpair import decode_sequence_pair_batch

def convert_slack_indices_to_floats(
    slack_indices: np.ndarray,
    min_slack: np.ndarray,
    max_slack: np.ndarray,
    slack_steps: int
) -> np.ndarray:
    """Converts discrete integer step indices into physical float32 slack dimensions.

    Args:
        slack_indices: Integer array containing step indices k in [0, slack_steps].
            Supports single individual shape (4, N) or batch shape (B, 4, N).
        min_slack: Float array of shape (4, N) with minimum slack bounds.
        max_slack: Float array of shape (4, N) with maximum slack bounds.
        slack_steps: Total step count K defining grid resolution between min and max bounds.

    Returns:
        Float32 array matching slack_indices shape with physical slack values.
    """
    # Calculate step delta per slack variable (4, N)
    step_size = (max_slack - min_slack) / float(slack_steps)

    # NumPy broadcasting aligns trailing dims: (4, N) automatically broadcasts across (B, 4, N)
    physical_slacks = min_slack + slack_indices * step_size

    return physical_slacks.astype(np.float32)

def decode_chunk_worker(
    chunk_x: np.ndarray,
    num_chiplets: int,
    chiplet_dimensions: np.ndarray,
    min_slack: np.ndarray,
    max_slack: np.ndarray,
    slack_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decodes a chunk of 7N int16 chromosomes into floorplan bounding boxes and metrics.

    Args:
        chunk_x: (Chunk, 7N) array of chromosome genes in np.int16 encoding.
        num_chiplets: Total number of chiplets (N).
        chiplet_dimensions: (N, 2) array of chiplet dimensions [width, height].
        min_slack: (4, N) lower whitespace physical bounds in float.
        max_slack: (4, N) upper whitespace physical bounds in float.
        slack_steps: Total step count defined for grid quantization.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
            - bbox_batch: (Chunk, N, 4) bounding box coordinates [x1, y1, x2, y2].
            - total_widths: (Chunk,) layout bounding widths.
            - total_heights: (Chunk,) layout bounding heights.
    """
    chunk_size = chunk_x.shape[0]
    n = num_chiplets

    # Slice genes from discrete int16 chromosome
    seq_x_batch = chunk_x[:, 0:n]
    seq_y_batch = chunk_x[:, n : 2 * n]
    slack_indices_batch = chunk_x[:, 2 * n : 6 * n].reshape(chunk_size, 4, n)
    rot_batch = chunk_x[:, 6 * n : 7 * n].astype(bool)

    # Convert discrete integer step indices to physical float slacks
    slack_batch = convert_slack_indices_to_floats(
        slack_indices=slack_indices_batch,
        min_slack=min_slack,
        max_slack=max_slack,
        slack_steps=slack_steps
    )

    return decode_sequence_pair_batch(
        seq_x_batch=seq_x_batch,
        seq_y_batch=seq_y_batch,
        slack_batch=slack_batch,
        rot_batch=rot_batch,
        chiplet_dims=chiplet_dimensions
    )