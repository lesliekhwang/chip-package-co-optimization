"""
cost.py
---------------------------------
Analytical cost functions for chiplet placement.
Handles directional N, E, S, W whitespace halos and undirected adjacency graphs.
"""
from warpage import get_model

def warpage(x_coords: dict, y_coords: dict, total_w: float, total_h: float, 
            base_chiplets: dict, warpagemap: bool = False, warpagescale: float = 1000.0):
    """
    Evaluates the thermo-mechanical warpage of the given interposer layout.
    warpagescale is the scale the warpage model expects for input
    """
    warpage_model = get_model()

    cx = total_w / 2.0
    cy = total_h / 2.0
    
    substrate_length = max(total_w, total_h) / warpagescale 
    
    bboxes = []
    
    for name in sorted(base_chiplets.keys()):
        chip = base_chiplets[name]

        # Calculate absolute centered coordinates in base units (mm)
        min_x_mm = (x_coords[name] + chip.ws_w) - cx
        min_y_mm = (y_coords[name] + chip.ws_s) - cy
        max_x_mm = min_x_mm + chip.w
        max_y_mm = min_y_mm + chip.h
        
        # Scale to meters exactly once for all values
        bboxes.append([
            min_x_mm / warpagescale, 
            min_y_mm / warpagescale, 
            max_x_mm / warpagescale, 
            max_y_mm / warpagescale
        ])
        
    warp, rms = warpage_model.evaluate_rms(bboxes, substrate_length, warpagemap)
    return (warp * warpagescale, rms * warpagescale)


def area(total_w: float, total_h: float) -> float:
    """
    Calculates the total bounding box area of the interposer substrate.
    Assuming the Sequence Pair algorithm calculates total_w and total_h 
    using the padded macro blocks (width + ws_w + ws_e), we can just multiply them.
    """
    return float(total_w * total_h)


def hpwl(x_coords: dict, y_coords: dict, base_chiplets: dict) -> float:
    """
    Calculates the Half-Perimeter Wire Length (HPWL) proxy.
    Sums Manhattan distances strictly using the predefined adjacency list,
    preventing double counting of mutual connections.
    """
    total_hpwl = 0.0
    centers = {}
    
    # 1. Pre-calculate the exact physical centers of every chiplet
    for name, chip in base_chiplets.items():
        # Offset from envelope bottom-left into the true physical center
        # by adding West/South whitespace and half the physical dimension
        cx = x_coords[name] + chip.ws_w + (chip.w / 2.0)
        cy = y_coords[name] + chip.ws_s + (chip.h / 2.0)
        centers[name] = (cx, cy)
        
    # 2. Iterate through adjacency lists and compute edges without double-counting
    computed_edges = set()
    
    for name, chip in base_chiplets.items():
        for adj_name in chip.adj:
            
            # Sort the two nodes alphabetically to create a unique, unidirectional edge signature
            # (e.g., "CCD1" to "IOD1" and "IOD1" to "CCD1" both become ("CCD1", "IOD1"))
            edge = tuple(sorted([name, adj_name]))
            
            # Only compute the distance if we haven't seen this edge yet
            if edge not in computed_edges:
                computed_edges.add(edge)
                
                # Calculate Manhattan distance (L1 norm) between physical centers
                dx = abs(centers[name][0] - centers[adj_name][0])
                dy = abs(centers[name][1] - centers[adj_name][1])
                
                total_hpwl += (dx + dy)
            
    return total_hpwl