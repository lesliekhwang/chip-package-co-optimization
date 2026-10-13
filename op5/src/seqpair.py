import numpy as np

def calculate_longest_paths(n_nodes: int, edges: list, weights: np.ndarray) -> np.ndarray:
    """Calculates coordinates using a DAG Topological Sort over integer nodes."""
    in_degree = np.zeros(n_nodes, dtype=int)
    adj = [[] for _ in range(n_nodes)]
    
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
        
    queue = [i for i in range(n_nodes) if in_degree[i] == 0]
    dist = np.zeros(n_nodes, dtype=float)
    
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if dist[u] + weights[u] > dist[v]:
                dist[v] = dist[u] + weights[u]
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    return dist

def sequence_pair_to_placement(X: np.ndarray, Y: np.ndarray, base_widths: np.ndarray, 
                               base_heights: np.ndarray, slack_matrix: np.ndarray) -> tuple:
    """
    Maps Sequence Pairs to physical coordinates using ID sequences and a 4xN slack matrix.
    slack_matrix row mapping: 0=North, 1=East, 2=South, 3=West.
    """
    N = len(X)
    
    x_pos = np.argsort(X)
    y_pos = np.argsort(Y)
    
    gh_edges, gv_edges = [], []
    
    for i in range(N):
        for j in range(i + 1, N):
            a_before_b_in_x = x_pos[i] < x_pos[j]
            a_before_b_in_y = y_pos[i] < y_pos[j]
            
            if a_before_b_in_x and a_before_b_in_y:
                gh_edges.append((i, j))  
            elif not a_before_b_in_x and not a_before_b_in_y:
                gh_edges.append((j, i))  
            elif a_before_b_in_x and not a_before_b_in_y:
                gv_edges.append((j, i))  
            else:
                gv_edges.append((i, j))
                
    # West(3) + East(1)
    widths = base_widths + slack_matrix[3] + slack_matrix[1] 
    # North(0) + South(2)
    heights = base_heights + slack_matrix[0] + slack_matrix[2] 
    
    x_coords = calculate_longest_paths(N, gh_edges, widths)
    y_coords = calculate_longest_paths(N, gv_edges, heights)
    
    total_width = np.max(x_coords + widths)
    total_height = np.max(y_coords + heights)
    
    return x_coords, y_coords, float(total_width), float(total_height)