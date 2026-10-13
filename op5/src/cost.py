# cost.py
from warpage import get_model, _DEVICE
import numpy as np
import torch
import gc

def area_batch(total_w: np.ndarray, total_h: np.ndarray) -> np.ndarray:
    return total_w * total_h

def hpwl_batch(x_coords: np.ndarray, y_coords: np.ndarray, base_widths: np.ndarray, 
               base_heights: np.ndarray, adj_list: list, slack_matrix: np.ndarray) -> np.ndarray:
    # x_coords/y_coords: (B, N), slack_matrix: (B, 4, N)
    cx = x_coords + slack_matrix[:, 3, :] + (base_widths / 2.0)
    cy = y_coords + slack_matrix[:, 2, :] + (base_heights / 2.0)
    
    B = len(x_coords)
    total_hpwl = np.zeros(B, dtype=float)
    
    computed_edges = set()
    for i, adjs in enumerate(adj_list):
        for adj_id in adjs:
            edge = tuple(sorted([i, adj_id]))
            if edge not in computed_edges:
                computed_edges.add(edge)
                dx = np.abs(cx[:, i] - cx[:, adj_id])
                dy = np.abs(cy[:, i] - cy[:, adj_id])
                total_hpwl += (dx + dy)
                
    return total_hpwl

def warpage_batch(x_coords: np.ndarray, y_coords: np.ndarray, total_w: np.ndarray, total_h: np.ndarray, 
                  base_widths: np.ndarray, base_heights: np.ndarray, slack_matrix: np.ndarray, 
                  warpagescale: float = 1000.0, device: str = _DEVICE, chunk_size: int = 100):
    warpage_model = get_model()
    B, N = x_coords.shape
    
    all_warp_maps = []
    all_rms_vals = []
    
    # Process tensors in chunks
    for i in range(0, B, chunk_size):
        end = min(i + chunk_size, B)
        
        cx = total_w[i:end] / 2.0
        cy = total_h[i:end] / 2.0
        substrate_w = total_w[i:end] / warpagescale
        substrate_h = total_h[i:end] / warpagescale
        
        # Vectorized bbox extraction on the chunk
        min_x = (x_coords[i:end] + slack_matrix[i:end, 3, :]) - cx[:, None]
        min_y = (y_coords[i:end] + slack_matrix[i:end, 2, :]) - cy[:, None]
        max_x = min_x + base_widths[i:end]
        max_y = min_y + base_heights[i:end]
        
        bboxes = np.stack([min_x, min_y, max_x, max_y], axis=-1) / warpagescale
        
        # Move chunk to device
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32, device=device)
        sw_tensor = torch.tensor(substrate_w, dtype=torch.float32, device=device)
        sh_tensor = torch.tensor(substrate_h, dtype=torch.float32, device=device)
        
        warp_maps, rms_vals = warpage_model.evaluate_batch(bboxes_tensor, sw_tensor, sh_tensor)
        
        all_warp_maps.append(warp_maps * warpagescale)
        all_rms_vals.append(rms_vals * warpagescale)
        
        # Free memory and clear cache for the next chunk
        del bboxes_tensor, sw_tensor, sh_tensor, warp_maps, rms_vals
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()
        
    final_warp_maps = np.concatenate(all_warp_maps, axis=0)
    final_rms_vals = np.concatenate(all_rms_vals, axis=0)
    
    return final_warp_maps, final_rms_vals