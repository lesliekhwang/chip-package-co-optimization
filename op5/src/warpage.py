"""
warpage.py
---------------------------------
Surrogate warpage model prediction utilities.
Loads architecture weights and grid dimensions directly from the [cost] section
of ga_config.toml without fallback defaults or .get() calls.
"""

import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
_DEVICE = "cuda"

# Globally disable gradient calculation for maximum CPU inference performance and memory savings
torch.set_grad_enabled(False)

# Private globals to hold loaded state across generations
_WARPAGE_MODEL_CACHE = None
_WARP_GRID_X = None
_WARP_GRID_Y = None
model = None

class Cube(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x**3 / 6


class DieSetEncoder(nn.Module):
    """
    Permutation-invariant ("deep sets") encoder over the dies in a scenario.
    Each die (xmin, ymin, xmax, ymax) is embedded independently by a shared
    MLP, then SUMMED across dies.
    """
    def __init__(self, embed_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, die_bboxes, mask=None):
        e = self.net(die_bboxes)
        if mask is not None:
            e = e * mask.unsqueeze(-1)
        return e.sum(dim=-2)


class WarpageNet(nn.Module):
    """
    w(x, y) predictor, conditioned on:
      - query point (x, y)
      - permutation-invariant die-scene embedding (DieSetEncoder output)
      - domain shape, as a single scalar log(Lx / Ly)
    """
    def __init__(self, embed_dim=64, trunk_hidden=256):
        super().__init__()
        self.die_encoder = DieSetEncoder(embed_dim=embed_dim)
        in_dim = 2 + embed_dim + 1  # (x, y) + pooled die embedding + log-aspect-ratio
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, 128),
            Cube(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, xy, die_bboxes=None, mask=None, domain_feat=None, pooled_die_embed=None):
        if pooled_die_embed is None:
            assert die_bboxes is not None, "must supply either die_bboxes or pooled_die_embed"
            pooled_die_embed = self.die_encoder(die_bboxes, mask)
        X = torch.cat([xy, pooled_die_embed, domain_feat], dim=-1)
        return self.trunk(X).squeeze(-1)

@torch.inference_mode()
def predict_warpage_batch(bboxes, substrate_width, substrate_height, grid_width=128, grid_height=128):
    """Batched warpage prediction for tensors of shape (B, N, 4)"""
    B = bboxes.shape[0]
    
    # Force incoming tensors to match the internal execution device
    bboxes = bboxes.to(_DEVICE)
    substrate_width = substrate_width.to(_DEVICE)
    substrate_height = substrate_height.to(_DEVICE)
    
    scale = torch.sqrt(substrate_width * substrate_height)
    Lx = substrate_width / scale
    Ly = substrate_height / scale

    x_ = torch.linspace(-0.5, 0.5, grid_width, device=_DEVICE)
    y_ = torch.linspace(-0.5, 0.5, grid_height, device=_DEVICE)
    x, y = torch.meshgrid(x_, y_, indexing='ij')
    x_flat, y_flat = x.flatten(), y.flatten()
    n_points = grid_width * grid_height

    # Both x_flat and Lx are now guaranteed to be on the same device
    x_scaled = x_flat.unsqueeze(0) * Lx.unsqueeze(1)
    y_scaled = y_flat.unsqueeze(0) * Ly.unsqueeze(1)
    points = torch.stack([x_scaled, y_scaled], dim=-1) # (B, n_points, 2)

    # Pad bboxes to maximum expected sequence length (20)
    pad_len = 20 - bboxes.shape[1]
    if pad_len > 0:
        padding = torch.zeros(B, pad_len, 4, device=_DEVICE)
        bboxes = torch.cat([bboxes, padding], dim=1)
        
    bboxes = bboxes / scale.view(B, 1, 1)
    area = (bboxes[:, :, 2] - bboxes[:, :, 0]) * (bboxes[:, :, 3] - bboxes[:, :, 1])
    mask = area > 1e-9

    pooled = model.die_encoder(bboxes, mask)
    pooled_rep = pooled.unsqueeze(1).expand(-1, n_points, -1)

    log_aspect = torch.log(Lx / Ly)
    domain_feat_rep = log_aspect.view(B, 1, 1).expand(-1, n_points, -1)

    w_pred = model(points, pooled_die_embed=pooled_rep, domain_feat=domain_feat_rep).squeeze(-1)

    # Batched plane of best fit subtraction
    A = torch.stack([x_scaled, y_scaled, torch.ones_like(x_scaled)], dim=-1) # (B, n_points, 3)
    coeff = torch.linalg.lstsq(A, w_pred.unsqueeze(-1)).solution # (B, 3, 1)
    plane = A.bmm(coeff).squeeze(-1)
    
    w_pred = w_pred - plane
    return w_pred.view(B, grid_width, grid_height) * (scale.view(B, 1, 1) ** 2)

def initialize_model(cfg: dict):
    """
    Initializes the PyTorch model directly from the [cost] section of ga_config.toml.
    Raises KeyError immediately if required key configurations do not exist.
    """
    global _WARPAGE_MODEL_CACHE, _WARP_GRID_X, _WARP_GRID_Y, model

    # Direct key lookup under [cost]
    cost_cfg = cfg["cost"]
    model_path = cost_cfg["model_path"]
    _WARP_GRID_X = cost_cfg["warp_grid"][0]
    _WARP_GRID_Y = cost_cfg["warp_grid"][1]

    _WARPAGE_MODEL_CACHE = WarpageNet().to(_DEVICE)
    _WARPAGE_MODEL_CACHE.load_state_dict(torch.load(model_path, map_location=_DEVICE))
    _WARPAGE_MODEL_CACHE.eval()

    # Compile the model for optimized inference
    model = _WARPAGE_MODEL_CACHE #torch.compile(_WARPAGE_MODEL_CACHE)

    print(f"[*] Warpage surrogate model loaded and compiled successfully from '{model_path}' (Grid: {_WARP_GRID_X}x{_WARP_GRID_Y})")

class WarpageEvaluator:
    """Wrapper class providing evaluation methods to the cost function."""
    def __init__(self, grid_x, grid_y):
        self.grid_x = grid_x
        self.grid_y = grid_y

    def evaluate_batch(self, bboxes_tensor, sw_tensor, sh_tensor):
        """Returns batched (warp_maps, rms_vals) for population arrays."""
        w_grid = predict_warpage_batch(bboxes_tensor, sw_tensor, sh_tensor, self.grid_x, self.grid_y)
        rms_vals = torch.sqrt(torch.mean(w_grid ** 2, dim=(1, 2)))
        # Transpose maps to match sequential output format (B, grid_y, grid_x)
        return w_grid.cpu().numpy().transpose(0, 2, 1), rms_vals.cpu().numpy()

def get_model():
    """Returns the initialized evaluator. Raises RuntimeError if uninitialized."""
    global _WARPAGE_MODEL_CACHE, _WARP_GRID_X, _WARP_GRID_Y

    if _WARPAGE_MODEL_CACHE is None:
        raise RuntimeError("Warpage model is not initialized. 'initialize_model(cfg)' must be called first.")

    return WarpageEvaluator(_WARP_GRID_X, _WARP_GRID_Y)