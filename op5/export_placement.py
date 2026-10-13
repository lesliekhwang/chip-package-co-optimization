import sys
import os
import json
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# --- Sequence Pair & DAG Placement Logic ---
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


# --- Helper for Drawing Dashed Rectangles (PIL) ---
def draw_dashed_rect(draw: ImageDraw.ImageDraw, bbox: list, dash_len: int = 4, fill: int = 0):
    x1, y1, x2, y2 = bbox
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    # Top & Bottom edges
    for x in range(min_x, max_x, dash_len * 2):
        draw.line([(x, min_y), (min(x + dash_len, max_x), min_y)], fill=fill)
        draw.line([(x, max_y), (min(x + dash_len, max_x), max_y)], fill=fill)

    # Left & Right edges
    for y in range(min_y, max_y, dash_len * 2):
        draw.line([(min_x, y), (min_x, min(y + dash_len, max_y))], fill=fill)
        draw.line([(max_x, y), (max_x, min(y + dash_len, max_y))], fill=fill)


# --- Main Runner ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python export_placement.py <generation> [individual_idx]")
        print("Example: python export_placement.py 61 0")
        sys.exit(1)

    gen = int(sys.argv[1])
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    json_file = "chiplets.json"
    h5_file = "run.h5"

    # 1. Load base chiplet specifications
    with open(json_file, "r") as f:
        chiplet_spec = json.load(f)

    # 2. Extract genotype, phenotype, and fitness data from run.h5
    with h5py.File(h5_file, "r") as h5f:
        if "metadata/chiplet_id_map" in h5f:
            chiplet_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in h5f["metadata/chiplet_id_map"][:]]
        else:
            chiplet_names = [k for k in chiplet_spec.keys() if k != "interposer"]

        N = len(chiplet_names)

        raw_widths = np.zeros(N, dtype=np.float32)
        raw_heights = np.zeros(N, dtype=np.float32)
        for i, name in enumerate(chiplet_names):
            raw_widths[i] = chiplet_spec[name]["dim"][0]
            raw_heights[i] = chiplet_spec[name]["dim"][1]

        gen_path = f"generations/gen{gen}"
        if gen_path not in h5f:
            raise KeyError(f"Generation '{gen_path}' not found in {h5_file}")

        geno = h5f[f"{gen_path}/genotype"]
        X = geno["X"][idx]
        Y = geno["Y"][idx]
        slack_matrix = geno["slack"][idx]  # 0=North, 1=East, 2=South, 3=West
        rot = geno["rot"][idx]              # Shape (N,)

        # Extract fitness values F = [hpwl, area, warpage]
        fit_group = h5f[f"{gen_path}/fitness"]
        F_vals = fit_group["F"][idx].tolist() if "F" in fit_group else []

        pheno = h5f[f"{gen_path}/phenotype"]
        warpage_map = pheno["warpage_maps"][idx] if "warpage_maps" in pheno else None

    # 3. Apply rotation & compute placement coordinates
    eff_widths = np.where(rot == 1, raw_heights, raw_widths)
    eff_heights = np.where(rot == 1, raw_widths, raw_heights)

    x_coords, y_coords, total_w, total_h = sequence_pair_to_placement(
        X, Y, eff_widths, eff_heights, slack_matrix
    )

    # Offset by West(3) and South(2) slacks for physical bounding box coordinates
    x1 = x_coords + slack_matrix[3]
    y1 = y_coords + slack_matrix[2]
    x2 = x1 + eff_widths
    y2 = y1 + eff_heights

    # Extract N, E, S, W slacks per chiplet
    slack_north = slack_matrix[0]
    slack_east = slack_matrix[1]
    slack_south = slack_matrix[2]
    slack_west = slack_matrix[3]

    # 4. Output Directory Setup
    out_dir = f"gen{gen}_ind{idx}"
    os.makedirs(out_dir, exist_ok=True)

    # 5. Export placement.csv (includes N, E, S, W slacks)
    df = pd.DataFrame({
        "chiplet_name": chiplet_names,
        "x1": np.round(x1, 6),
        "y1": np.round(y1, 6),
        "x2": np.round(x2, 6),
        "y2": np.round(y2, 6),
        "slack_north": np.round(slack_north, 6),
        "slack_east": np.round(slack_east, 6),
        "slack_south": np.round(slack_south, 6),
        "slack_west": np.round(slack_west, 6),
        "width": np.round(eff_widths, 6),
        "height": np.round(eff_heights, 6),
        "rotation": rot
    })
    df.to_csv(os.path.join(out_dir, "placement.csv"), index=False)

    # 6. Export placement.json
    sum_x_ok = np.isclose(np.max(x2 + slack_east), total_w)
    sum_y_ok = np.isclose(np.max(y2 + slack_north), total_h)
    pass_flag = bool(sum_x_ok and sum_y_ok)

    interposer_json_data = {
        "source": {
            "generation": gen,
            "individual_idx": idx
        },
        "interposer": {
            "total_w": float(round(total_w, 6)),
            "total_h": float(round(total_h, 6)),
            "aspect_ratio": float(round(total_w / total_h, 6)) if total_h > 0 else 0.0,
        },
        "cost": {
            "hpwl": float(round(F_vals[0], 6)) if len(F_vals) > 0 else None,
            "area": float(round(F_vals[1], 6)) if len(F_vals) > 1 else None,
            "warpage": float(round(F_vals[2], 6)) if len(F_vals) > 2 else None
        },
        "pass": pass_flag
    }
    with open(os.path.join(out_dir, "placement.json"), "w") as f:
        json.dump(interposer_json_data, f, indent=2)

    # 7. Export warpage.npy
    if warpage_map is not None:
        np.save(os.path.join(out_dir, "warpage.npy"), warpage_map)

    # 8. Export Validation Image (Black & White 1-bit PNG)
    scale = 20  # Pixels per unit coordinate
    padding = 40  # Margin around interposer canvas
    
    img_w = int(total_w * scale) + 2 * padding
    img_h = int(total_h * scale) + 2 * padding

    # 1-bit monochrome image (mode '1': 1 = White, 0 = Black)
    img = Image.new('1', (img_w, img_h), 1)
    draw = ImageDraw.Draw(img)

    def world_to_px(wx, wy):
        px_x = int(padding + wx * scale)
        px_y = int(padding + (total_h - wy) * scale)  # Invert Y axis for images
        return px_x, px_y

    # Outer Interposer bounding box
    ip_x1, ip_y1 = world_to_px(0, total_h)
    ip_x2, ip_y2 = world_to_px(total_w, 0)
    draw.rectangle([ip_x1, ip_y1, ip_x2, ip_y2], fill=1, outline=0, width=2)

    font = ImageFont.load_default()

    # Draw each chiplet and its slacks
    for i in range(N):
        # 1. Dashed line boundary for North, East, South, West whitespace
        sx1 = x1[i] - slack_west[i]
        sy1 = y1[i] - slack_south[i]
        sx2 = x2[i] + slack_east[i]
        sy2 = y2[i] + slack_north[i]

        spx_x1, spx_y1 = world_to_px(sx1, sy2)
        spx_x2, spx_y2 = world_to_px(sx2, sy1)
        draw_dashed_rect(draw, [spx_x1, spx_y1, spx_x2, spx_y2], dash_len=4, fill=0)

        # 2. Chiplet bounding box with interior hatching
        px_x1, px_y1 = world_to_px(x1[i], y2[i])
        px_x2, px_y2 = world_to_px(x2[i], y1[i])
        
        draw.rectangle([px_x1, px_y1, px_x2, px_y2], fill=1, outline=0, width=2)

        # Hatching inside chiplet area
        hatch_spacing = 8
        for offset in range(-img_h, img_w + img_h, hatch_spacing):
            xs = max(px_x1, offset + px_y1)
            xe = min(px_x2, offset + px_y2)
            if xs < xe:
                draw.line([(xs, xs - offset), (xe, xe - offset)], fill=0, width=1)

        # Redraw chiplet outline over hatching
        draw.rectangle([px_x1, px_y1, px_x2, px_y2], outline=0, width=2)

        # 3. Label Badge (White background box + Bold monospace text)
        cx = (px_x1 + px_x2) // 2
        cy = (px_y1 + px_y2) // 2
        
        name_str = chiplet_names[i]
        bbox = draw.textbbox((cx, cy), name_str, font=font, anchor="mm")
        pad_x, pad_y = 5, 3
        badge_box = [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y]

        # Draw solid white badge behind text
        draw.rectangle(badge_box, fill=1, outline=0, width=1)

        # Render bold monospace text
        draw.text((cx, cy), name_str, fill=0, font=font, anchor="mm")
        draw.text((cx + 1, cy), name_str, fill=0, font=font, anchor="mm")

    # Save as 1-bit monochrome PNG
    img.save(os.path.join(out_dir, "validation.png"), optimize=True)

    print(f"Exported files to '{out_dir}/': placement.csv, placement.json, warpage.npy, validation.png")


if __name__ == "__main__":
    main()