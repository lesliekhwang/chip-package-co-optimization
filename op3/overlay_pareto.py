from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "monospace"

def get_2d_pareto_front(points: np.ndarray) -> np.ndarray:
    """Extracts the non-dominated Pareto front from a set of 2D points assuming minimization."""
    if len(points) == 0:
        return np.array([])
        
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]

    pareto_front = []
    min_y = float("inf")

    for point in sorted_points:
        if point[1] < min_y:
            pareto_front.append(point)
            min_y = point[1]

    return np.array(pareto_front)

def map_gen_to_hsv(generations: np.ndarray, target_hue: float) -> np.ndarray:
    """Maps generation age to HSV space: older -> desaturated/pale, younger -> saturated/bold."""
    if len(generations) == 0:
        return np.empty((0, 3))
    
    min_g, max_g = generations.min(), generations.max()
    if max_g > min_g:
        norm_g = (generations - min_g) / (max_g - min_g)
    else:
        norm_g = np.ones_like(generations)
        
    # Saturation: 0.15 (old) to 1.0 (young)
    s = 0.15 + norm_g * 0.85
    # Value: 0.95 (old/pale) to 0.75 (young/deep contrast against white)
    v = 0.95 - norm_g * 0.20
    h = np.full_like(norm_g, target_hue)
    
    hsv = np.stack([h, s, v], axis=-1)
    return mcolors.hsv_to_rgb(hsv)

def generate_overlay_plot(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    name1: str, 
    name2: str, 
    x_col: str, 
    y_col: str, 
    limits: dict
):
    cols_lower1 = {col.lower(): col for col in df1.columns}
    cols_lower2 = {col.lower(): col for col in df2.columns}
    
    target_x1, target_y1 = cols_lower1.get(x_col.lower()), cols_lower1.get(y_col.lower())
    target_x2, target_y2 = cols_lower2.get(x_col.lower()), cols_lower2.get(y_col.lower())
    gen_col1 = cols_lower1.get("generation")
    gen_col2 = cols_lower2.get("generation")

    if not all([target_x1, target_y1, target_x2, target_y2]):
        print(f"[-] Skipping {x_col} vs {y_col}: Missing target objective columns in one or both files.")
        return

    # Filter NaNs
    df1_clean = df1.dropna(subset=[target_x1, target_y1]).copy()
    df2_clean = df2.dropna(subset=[target_x2, target_y2]).copy()

    points1 = df1_clean[[target_x1, target_y1]].to_numpy()
    points2 = df2_clean[[target_x2, target_y2]].to_numpy()

    # Extract Generation values for color mapping
    gen1 = df1_clean[gen_col1].to_numpy() if gen_col1 and gen_col1 in df1_clean.columns else np.zeros(len(df1_clean))
    gen2 = df2_clean[gen_col2].to_numpy() if gen_col2 and gen_col2 in df2_clean.columns else np.zeros(len(df2_clean))

    # Generate HSV RGB arrays (Hue ~0.6 is Blue, Hue ~0.05 is Orange)
    colors1 = map_gen_to_hsv(gen1, target_hue=0.60)
    colors2 = map_gen_to_hsv(gen2, target_hue=0.05)

    # Global Pareto Fronts
    pf1_global = get_2d_pareto_front(points1)
    pf2_global = get_2d_pareto_front(points2)

    # Final Generation Pareto Fronts
    pf1_final = np.array([])
    if gen_col1:
        max_gen1 = df1_clean[gen_col1].max()
        df1_final = df1_clean[df1_clean[gen_col1] == max_gen1]
        pf1_final = get_2d_pareto_front(df1_final[[target_x1, target_y1]].to_numpy())

    pf2_final = np.array([])
    if gen_col2:
        max_gen2 = df2_clean[gen_col2].max()
        df2_final = df2_clean[df2_clean[gen_col2] == max_gen2]
        pf2_final = get_2d_pareto_front(df2_final[[target_x2, target_y2]].to_numpy())

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter Plots using HSV-derived RGB arrays
    ax.scatter(
        points1[:, 0], points1[:, 1], 
        c=colors1, marker="o", s=25, edgecolors="none", label="Opt 1 Space", zorder=1
    )
    
    ax.scatter(
        points2[:, 0], points2[:, 1], 
        c=colors2, marker="^", s=25, edgecolors="none", label="Opt 2 Space", zorder=2
    )

    # Global Fronts (Solid whole lines, cool/warm bases)
    if len(pf1_global) > 0:
        ax.plot(
            pf1_global[:, 0], pf1_global[:, 1], 
            color="navy", linestyle="-", linewidth=2.5, alpha=0.9, 
            label="Opt 1 Global Front", zorder=3,
        )
    if len(pf2_global) > 0:
        ax.plot(
            pf2_global[:, 0], pf2_global[:, 1], 
            color="darkred", linestyle="-", linewidth=2.5, alpha=0.9, 
            label="Opt 2 Global Front", zorder=3,
        )

    # Final Generation Fronts (Dashed lines only, prominent cool/warm highlights)
    if len(pf1_final) > 0:
        ax.plot(
            pf1_final[:, 0], pf1_final[:, 1], 
            color="deepskyblue", linestyle=":", linewidth=3.5, 
            label="Opt 1 Final Gen Front", zorder=4,
            path_effects=[pe.Stroke(linewidth=5.5, foreground='black'), pe.Normal()]
        )

    if len(pf2_final) > 0:
        ax.plot(
            pf2_final[:, 0], pf2_final[:, 1], 
            color="darkorange", linestyle=":", linewidth=3.5, 
            label="Opt 2 Final Gen Front", zorder=4,
            path_effects=[pe.Stroke(linewidth=5.5, foreground='black'), pe.Normal()]
        )

    # Enforce synchronized axis limits across all figures
    if x_col in limits:
        ax.set_xlim(limits[x_col])
    if y_col in limits:
        ax.set_ylim(limits[y_col])

    # Granular Ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=25))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=25))
    ax.grid(True, linestyle=":", alpha=0.7)

    disp_x = x_col.upper() if len(x_col) <= 7 else x_col.title()
    disp_y = y_col.upper() if len(y_col) <= 7 else y_col.title()
    
    ax.set_title(f"Pareto Overlay: {disp_x} vs {disp_y}", fontsize=12, weight="bold")
    if (disp_x == 'WARPAGE'):
        ax.set_xlabel(f"{disp_x} ($\mu$m)", fontsize=10)
    else:
        ax.set_xlabel(f"{disp_x} (mm)", fontsize=10)
    ax.set_ylabel(f"{disp_y} (mm)", fontsize=10)
    
    # Restructure Legend Handles to enforce 2 Distinct Columns (Col 1 = Opt 1, Col 2 = Opt 2)
    handles, labels = ax.get_legend_handles_labels()
    r1_handles, r1_labels = [], []
    r2_handles, r2_labels = [], []
    
    for h, l in zip(handles, labels):
        if "Opt 1" in l:
            r1_handles.append(h)
            r1_labels.append(l)
        elif "Opt 2" in l:
            r2_handles.append(h)
            r2_labels.append(l)
            
    # Pad lists with dummy elements if one run is missing a front to prevent staggered shifting
    dummy_handle = mlines.Line2D([], [], color='none')
    max_len = max(len(r1_handles), len(r2_handles))
    
    r1_handles.extend([dummy_handle] * (max_len - len(r1_handles)))
    r1_labels.extend([""] * (max_len - len(r1_labels)))
    r2_handles.extend([dummy_handle] * (max_len - len(r2_handles)))
    r2_labels.extend([""] * (max_len - len(r2_labels)))
    
    ordered_handles = r1_handles + r2_handles
    ordered_labels = r1_labels + r2_labels
    
    ax.legend(ordered_handles, ordered_labels, loc="upper right", framealpha=1.0, edgecolor="black", fontsize=9, ncol=2)

    out_name = f"pareto_overlay_{x_col}_vs_{y_col}.png"
    plt.tight_layout()
    plt.savefig(out_name, bbox_inches="tight", dpi=300)
    print(f"[*] Saved {out_name}")
    plt.close()


if __name__ == "__main__":
    file1_path = "/home/apc5/Desktop/projects/op3/demos/area+hpwl+warpage/history.parquet"
    file2_path = "/home/apc5/Desktop/projects/op3/demos/area+hpwl/history.parquet"

    p1, p2 = Path(file1_path), Path(file2_path)

    if not p1.exists() or not p2.exists():
        print(f"Error: Could not find '{file1_path}' and/or '{file2_path}'. Ensure paths are correct.")
        exit(1)

    print("\nLoading datasets...")
    df1 = pd.read_parquet(p1, engine="pyarrow")
    df2 = pd.read_parquet(p2, engine="pyarrow")

    # Scale warpage to micrometers globally so limits and Pareto fronts calculate correctly
    for df in [df1, df2]:
        cols_lower = {col.lower(): col for col in df.columns}
        if "warpage" in cols_lower:
            df[cols_lower["warpage"]] *= 1000

    # Reordered plot pairs so warpage is consistently placed on the x-axis
    plot_pairs = [
        ("hpwl", "area"),
        ("warpage", "hpwl"),
        ("warpage", "area")
    ]

    # Pre-calculate global min and max for each objective across both datasets (with 2% padding)
    objectives = ["hpwl", "area", "warpage"]
    global_limits = {}
    
    for obj in objectives:
        vals = []
        for df in [df1, df2]:
            cols_lower = {col.lower(): col for col in df.columns}
            if obj in cols_lower:
                vals.extend(df[cols_lower[obj]].dropna().values)
        if vals:
            min_v, max_v = min(vals), max(vals)
            padding = (max_v - min_v) * 0.02 if max_v > min_v else 0.01
            global_limits[obj] = (min_v - padding, max_v + padding)

    print("Generating plots...")
    for x_col, y_col in plot_pairs:
        generate_overlay_plot(df1, df2, p1.name, p2.name, x_col, y_col, global_limits)
    
    print("Done.")