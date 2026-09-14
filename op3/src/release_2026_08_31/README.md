## Overview
The main useful function is `predict_warpage`, which computes a warpage field over a grid and returns it as a 2D tensor. The underlying model evaluates warpage at individual (x, y) points given chiplet bounding boxes as input.

Note that the warpage field is only defined up to a rigid body transformation — after the model makes its prediction over the grid, the plane of best fit is subtracted to remove this degree of freedom.

---

## Files
- `variable_die_full.pt` — model checkpoint
- `test_load.ipynb` — example notebook

To run `test_load.ipynb`, you will need `torch` and `matplotlib` installed. Make sure to update the checkpoint path to match your local setup.

---

## `predict_warpage`

**Inputs:**
- `bboxes` — N × 4 `torch.Tensor` of chiplet bounding boxes, each as `(x1, y1, x2, y2)`
- `grid_size (default = 128)` — The size of the grid that will be returned

**Output:**
- `grid_size × grid_size` `torch.Tensor` of predicted warpage values

Depending on use case, the output may need to be moved back to CPU or detached from the compute graph via `.detach()` or inside a `torch.no_grad():` block.
For visualization, a 128 × 128 grid takes ~0.0015s on GPU (after the first run).

---

## Model

**Inputs (in order, all concatenated into one Nx82 tensor):**
- `x, y` — coordinates of the point to evaluate (2 values)
- `x1, y1, x2, y2` × 20 — bounding boxes of the chiplets up to 20 with the remaining entries 0 padded.  (80 values)

**Output:**
- Predicted warpage value at the given `(x, y)` point