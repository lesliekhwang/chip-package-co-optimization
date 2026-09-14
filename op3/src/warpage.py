import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

# Define device dynamically, fallback to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cube(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x**3 / 6

# Private globals to hold the loaded state in memory across generations
_WARPAGE_MODEL_CACHE = None
_GRID_SIZE = 128

def initialize_model(cfg: dict):
    """
    Initializes the PyTorch model from the configuration dictionary
    provided by main.py. Called exactly once before optimization starts.
    """
    global _WARPAGE_MODEL_CACHE, _GRID_SIZE
    
    model_path = cfg.get("MODEL_PATH", "variable_die_full.pt")
    _GRID_SIZE = cfg.get("GRID_SIZE", 128)
    
    # Load the model directly into memory and lock it into evaluation mode
    _WARPAGE_MODEL_CACHE = torch.load(model_path, weights_only=False).to(device)
    _WARPAGE_MODEL_CACHE.eval()
    
    print(f"[*] Warpage surrogate model loaded successfully from '{model_path}' (Grid Size: {_GRID_SIZE})")

class WarpageEvaluator:
    """Wrapper class providing the .evaluate() method to the cost function."""
    def __init__(self, model, grid_size):
        self.model = model
        self.grid_size = grid_size
        
    def evaluate(self, bboxes, substrate_length):
        """Returns the full 2D tensor grid of predicted warpage."""
        return predict_warpage(self.model, bboxes, substrate_length, self.grid_size)

    def evaluate_rms(self, bboxes, substrate_length, warpagemap=False):
        """
        Computes and returns the Root Mean Square (RMS) of the predicted warpage.
        Returns a single Python float suitable for a scalar cost function.
        """
        w_grid = predict_warpage(self.model, bboxes, substrate_length, self.grid_size)
        
        # Calculate RMS: sqrt(mean(w_grid^2))
        rms_val = torch.sqrt(torch.mean(w_grid ** 2))

        if not warpagemap:
            return rms_val.item()
        return (w_grid.detach().cpu().numpy(), rms_val.item())

def get_model():
    """
    Returns the initialized evaluator for the cost function to use.
    Throws an error if main.py forgot to initialize it.
    """
    global _WARPAGE_MODEL_CACHE, _GRID_SIZE
    
    if _WARPAGE_MODEL_CACHE is None:
        raise RuntimeError("Warpage model is not initialized. 'initialize_model(cfg)' must be called first.")
    
    return WarpageEvaluator(_WARPAGE_MODEL_CACHE, _GRID_SIZE)

def subtract_plane_of_best_fit(w, x, y):
    """
    Subtracts the plane of best fit from w
    """
    A = torch.stack([x.ravel(), y.ravel(), torch.ones_like(x.ravel())], dim=1)  # (N, 3)
    coeff = torch.linalg.lstsq(A, w.ravel().unsqueeze(-1)).solution  # (3, 1)
    a, b, c = coeff[0], coeff[1], coeff[2]
    plane = a * x + b * y + c
    return w - plane

def predict_warpage(model, bboxes, substrate_length, grid_size):
    """
    Predicts warpage for a given set of die bboxes.
    Assumes bounding boxes are already centered around (0, 0).
    """
    # Ensure incoming array is a tensor
    if not isinstance(bboxes, torch.Tensor):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

    # Pad with zeros up to 20 chiplets expected by the ML model input shape
    num_chiplets = bboxes.shape[0]
    if num_chiplets < 20:
        padding = torch.zeros((20 - num_chiplets, 4), dtype=torch.float32)
        bboxes = torch.cat([bboxes, padding], dim=0)
    else:
        bboxes = bboxes[:20]

    # Normalize coordinates into the [-0.5, 0.5] bounds using the max square substrate
    bboxes = bboxes / substrate_length

    # Create the prediction grid
    x_ = torch.linspace(-.5, .5, grid_size)
    y_ = torch.linspace(-.5, .5, grid_size)
    x, y = torch.meshgrid(x_, y_, indexing='ij')
    x, y = x.flatten().to(device), y.flatten().to(device)
    points = torch.stack([x, y], dim=-1)

    # Repeat chiplet features for the entire grid
    bboxes_rep = bboxes.to(device).unsqueeze(0).repeat_interleave(grid_size**2, 0)
    
    # Rapid inference block - no_grad() prevents memory leaks across generations
    with torch.no_grad():
        w_pred = model(torch.cat([points, bboxes_rep.reshape(grid_size**2, -1)], dim=-1)).squeeze(-1)
        
    w_pred = subtract_plane_of_best_fit(w_pred, x, y)

    # Rescale back to final physical bounds
    return w_pred.reshape(grid_size, grid_size) * (substrate_length**2)