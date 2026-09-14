"""
seqpair.py
---------------------------------
Core sequence pair placement engine.
"""

class Chiplet:
    """Represents a physical chiplet block with its dimensions and whitespace properties."""
    def __init__(self, name: str, w: float, h: float,
                 ws_n: float = 0.0, ws_e: float = 0.0,
                 ws_s: float = 0.0, ws_w: float = 0.0,
                 adj: list[str] = None):
        self.name = name
        self.w = w
        self.h = h
        self.ws_n = ws_n
        self.ws_e = ws_e
        self.ws_s = ws_s
        self.ws_w = ws_w
        self.adj = adj if adj is not None else []

def calculate_longest_paths(nodes: list, edges: list, weights: dict) -> dict:
    """Calculates coordinates using a DAG Topological Sort Longest Path algorithm."""
    in_degree = {u: 0 for u in nodes}
    adj = {u: [] for u in nodes}
    
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
        
    queue = [u for u in nodes if in_degree[u] == 0]
    dist = {u: 0 for u in nodes}
    
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            weight = weights[u]
            if dist[u] + weight > dist[v]:
                dist[v] = dist[u] + weight
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    return dist

def sequence_pair_to_placement(X: list, Y: list, chiplets: dict) -> tuple:
    """
    Maps Sequence Pairs to physical coordinates using internal chiplet properties.
    Corrected to process distinct N, E, S, W whitespace halos around each chiplet.
    
    Args:
        X (list): Sequence permutation X containing all chiplet names.
        Y (list): Sequence permutation Y containing all chiplet names.
        chiplets (dict): Dictionary mapping string names to Chiplet objects.
        
    Returns:
        tuple: (x_coords, y_coords, total_width, total_height)
    """
    nodes = list(chiplets.keys())
    gh_edges = []
    gv_edges = []
    
    x_pos = {name: i for i, name in enumerate(X)}
    y_pos = {name: i for i, name in enumerate(Y)}
    
    # 1. Build the Constraint Graphs
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            
            a_before_b_in_x = x_pos[a] < x_pos[b]
            a_before_b_in_y = y_pos[a] < y_pos[b]
            
            if a_before_b_in_x and a_before_b_in_y:
                gh_edges.append((a, b))  
            elif not a_before_b_in_x and not a_before_b_in_y:
                gh_edges.append((b, a))  
            elif a_before_b_in_x and not a_before_b_in_y:
                gv_edges.append((b, a))  
            else:
                gv_edges.append((a, b))  
                
    # 2. Extract structural dimensions including the DIRECTIONAL HALOS
    # Envelope Width = West Whitespace + Physical Width + East Whitespace
    widths = {name: chiplets[name].w + chiplets[name].ws_w + chiplets[name].ws_e for name in nodes}
    # Envelope Height = South Whitespace + Physical Height + North Whitespace
    heights = {name: chiplets[name].h + chiplets[name].ws_s + chiplets[name].ws_n for name in nodes}
    
    # x_coords and y_coords now represent the bottom-left corner of the ENVELOPE
    x_coords = calculate_longest_paths(nodes, gh_edges, widths)
    y_coords = calculate_longest_paths(nodes, gv_edges, heights)
    
    # 3. Calculate the ultimate bounding box of the packed interposer substrate
    # We must use the full padded 'widths' and 'heights' to prevent halos from spilling off the edge
    total_width = max(x_coords[name] + widths[name] for name in nodes)
    total_height = max(y_coords[name] + heights[name] for name in nodes)
    
    return x_coords, y_coords, total_width, total_height