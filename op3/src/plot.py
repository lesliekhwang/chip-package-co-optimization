"""
plot.py
---------------------------------
Visualization engine for chiplet layouts with intrinsic whitespace zones.
Optimized for high-speed, low-resource rendering with minimal flat colors.
Updated to read binary Parquet logs via PyArrow and load offline .npz warpage tensors.
Handles directional N, E, S, W whitespace halos.
"""

from pathlib import Path
from typing import List, Dict, Any, Union

import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
import copy

import seqpair

# Global Stylistic & Speed Optimizations
plt.style.use("fast")
plt.rcParams["font.family"] = "monospace"
plt.rcParams["path.simplify"] = True


def get_utopian_sample(pf_points: np.ndarray) -> int:
    """Finds the index of the point closest to the ideal utopian origin."""
    if len(pf_points) == 1:
        return 0
    # Normalize objectives to prevent scale mismatch bias
    f_min = pf_points.min(axis=0)
    f_max = pf_points.max(axis=0)

    # Avoid division by zero if all solutions have identical costs
    denom = f_max - f_min
    denom[denom == 0] = 1e-8

    normalized_pf = (pf_points - f_min) / denom
    # Find minimum Euclidean distance from normalized origin (0, 0)
    distances = np.linalg.norm(normalized_pf, axis=1)
    return int(np.argmin(distances))


def get_2d_pareto_front(points: np.ndarray) -> np.ndarray:
    """Extracts the non-dominated Pareto front from a set of 2D points assuming minimization for both objectives."""
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]

    pareto_front = []
    min_y = float("inf")

    for point in sorted_points:
        if point[1] < min_y:
            pareto_front.append(point)
            min_y = point[1]

    return np.array(pareto_front)


def plot_squareness_analysis(parquet_path, filename="squareness_analysis.png"):
    """Enhanced squareness analysis tracking directional aspect ratios, 
    Pareto elites vs. population, and objective correlations over time.
    """
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        return

    df = pd.read_parquet(parquet_file, engine="pyarrow")
    cols = {col.lower(): col for col in df.columns}
    gen_col, w_col, h_col = cols.get("generation"), cols.get("total_w"), cols.get("total_h")
    
    if not all([gen_col, w_col, h_col]):
        print("Error: Missing generation, total_w, or total_h columns.")
        return

    # 1. Compute Metrics
    df["aspect_ratio"] = df[w_col] / df[h_col]  # True W/H ratio
    df["squareness"] = np.minimum(df[w_col], df[h_col]) / np.maximum(df[w_col], df[h_col])

    generations = sorted(df[gen_col].unique())
    pop_mean, pareto_mean, best_area_sq, best_warp_sq = [], [], [], []

    # Objective column resolution
    hpwl_c, area_c, warp_c = cols.get("hpwl"), cols.get("area"), cols.get("warpage")

    for gen in generations:
        df_gen = df[df[gen_col] == gen]
        pop_mean.append(df_gen["squareness"].mean())

        # Extract Pareto Front for current generation
        if hpwl_c and area_c:
            costs = df_gen[[hpwl_c, area_c]].to_numpy()
            pareto_mask = get_pareto_mask(costs)
            df_pareto = df_gen[pareto_mask]
            pareto_mean.append(df_pareto["squareness"].mean())
        
        if area_c:
            best_area_sq.append(df_gen.loc[df_gen[area_c].idxmin()]["squareness"])
        if warp_c:
            best_warp_sq.append(df_gen.loc[df_gen[warp_c].idxmin()]["squareness"])

    # 2. Multi-Panel Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Top Subplot: Squareness Breakdown
    ax1.plot(generations, pop_mean, 'k--', alpha=0.6, label="Population Mean")
    if pareto_mean:
        ax1.plot(generations, pareto_mean, 'c-o', linewidth=2, label="Pareto Front Mean")
    if best_area_sq:
        ax1.plot(generations, best_area_sq, 'g-s', alpha=0.8, label="Best Area Solution")
    if best_warp_sq:
        ax1.plot(generations, best_warp_sq, 'm-^', alpha=0.8, label="Best Warpage Solution")

    ax1.set_title("Layout Squareness Trajectory Across Optimization Objectives", fontsize=11, weight="bold")
    ax1.set_ylabel("Squareness Ratio (Min / Max)", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    # Bottom Subplot: Directional Aspect Ratio Distribution (W / H)
    aspect_means = df.groupby(gen_col)["aspect_ratio"].mean()
    aspect_stds = df.groupby(gen_col)["aspect_ratio"].std().fillna(0)

    ax2.errorbar(generations, aspect_means, yerr=aspect_stds, fmt='b-o', ecolor='gray', capsize=3, label="Aspect Ratio (W / H)")
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7, label="Perfect Square (1.0)")
    ax2.set_title("Directional Bias (W > H vs H > W)", fontsize=10, weight="bold")
    ax2.set_xlabel("Generation", fontsize=10)
    ax2.set_ylabel("Width / Height", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    save_path = Path(parquet_path).parent / filename
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved enhanced squareness analysis plot to: {save_path}")

def plot_pareto_from_parquet(parquet_path, filename=None, x_col="hpwl", y_col="area"):
    """Reads tracking Parquet file to plot search history, final pop, and Pareto fronts."""
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        print(f"Error: Log file '{parquet_path}' not found.")
        return

    out_dir = parquet_file.parent
    df = pd.read_parquet(parquet_file, engine="pyarrow")

    # Case-insensitive column resolution
    cols_lower = {col.lower(): col for col in df.columns}
    target_x = cols_lower.get(x_col.lower())
    target_y = cols_lower.get(y_col.lower())
    gen_col = cols_lower.get("generation")

    if not target_x or not target_y or not gen_col:
        print(f"Error: Columns '{x_col}', '{y_col}', or 'generation' not found in Parquet file.")
        return

    max_gen = df[gen_col].max()
    df_history = df[df[gen_col] < max_gen]

    df_final = df[df[gen_col] == max_gen]
    F_final = df_final[[target_x, target_y]].to_numpy()

    pf_final = get_2d_pareto_front(F_final)
    all_points = df[[target_x, target_y]].to_numpy()
    pf_global = get_2d_pareto_front(all_points)

    utopian_idx = get_utopian_sample(pf_global)
    best_sample = pf_global[utopian_idx]

    nadir_point = pf_global.max(axis=0)

    plt.figure(figsize=(9, 6))

    # Plot history
    if not df_history.empty:
        plt.scatter(
            df_history[target_x],
            df_history[target_y],
            c="blue",
            s=10,
            edgecolors="none",
            label="Search History",
            zorder=1,
            rasterized=True,
        )

    # Plot final generation points
    plt.scatter(
        F_final[:, 0],
        F_final[:, 1],
        c="orange",
        s=20,
        edgecolors="none",
        label=f"Final Gen ({max_gen})",
        zorder=2,
        rasterized=True,
    )

    # Plot fronts using monochrome styles
    plt.plot(
        pf_final[:, 0],
        pf_final[:, 1],
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="Final Gen Front",
        zorder=3,
        rasterized=True,
    )

    plt.plot(
        pf_global[:, 0],
        pf_global[:, 1],
        color="red",
        linewidth=1.5,
        label="Global Front",
        zorder=4,
        rasterized=True,
    )

    # Utopian and Nadir points
    plt.scatter(
        best_sample[0],
        best_sample[1],
        c="yellow",
        marker="*",
        s=120,
        edgecolors="black",
        label="Utopian Point",
        zorder=5,
        rasterized=True,
    )

    plt.scatter(
        nadir_point[0],
        nadir_point[1],
        c="yellow",
        marker="X",
        s=120,
        edgecolors="black",
        label="Nadir Point",
        zorder=5,
        rasterized=True,
    )

    disp_x = target_x.upper() if len(target_x) <= 7 else target_x.title()
    disp_y = target_y.upper() if len(target_y) <= 7 else target_y.title()

    plt.title(f"2D Objective Space: {disp_x} vs {disp_y}")
    plt.xlabel(f"{disp_x} (Minimize)")
    plt.ylabel(f"{disp_y} (Minimize)")

    plt.legend(loc="upper right", framealpha=1.0, edgecolor="black")

    default_name = f"pareto_{target_x.lower()}_vs_{target_y.lower()}.png"
    save_path = out_dir / (filename if filename else default_name)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_placement(
    chiplets,
    x_locs,
    y_locs,
    total_w,
    total_h,
    title="Chiplet Floorplan",
    heatmap_matrix=None,
    rms_val=None,
    hpwl_val=None,
    filename=None
):
    """Generates a high-speed 2D visualization using black hatching patterns for chiplets."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sub_max = max(total_w, total_h)

    if heatmap_matrix is not None:
        im = ax.imshow(
            heatmap_matrix * 1000.0, # Multiply by 1000 for micrometer
            cmap="magma",
            extent=[0, sub_max, 0, sub_max],
            origin="lower",
            zorder=0,
            interpolation="nearest",
            rasterized=True,
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(f"Warpage ($\mu$m)", rotation=270, labelpad=15)

        interposer_face = "none"
    else:
        interposer_face = "#F4F4F4"

    # Dual-Stroke Interposer Boundary
    interposer_color = "#00E5FF" if heatmap_matrix is not None else "black"

    interposer = patches.Rectangle(
        (0, 0),
        total_w,
        total_h,
        linewidth=2.5,
        edgecolor=interposer_color,
        facecolor=interposer_face,
        linestyle="--",  # Dashed line for high visual distinction
        zorder=1.5,
    )
    # Adds a 5px black outline around the 2.5px cyan line
    interposer.set_path_effects([path_effects.withStroke(linewidth=5, foreground="black")])
    ax.add_patch(interposer)

    # Diverse black hatching patterns to uniquely identify distinct chiplets
    hatch_patterns = ["///", "\\\\\\", "|||", "---", "+++", "xxx", "ooo", "***"]

    for i, (name, chiplet) in enumerate(chiplets.items()):
        env_x = x_locs[name]
        env_y = y_locs[name]
        w = chiplet.w
        h = chiplet.h
        
        ws_n = chiplet.ws_n
        ws_e = chiplet.ws_e
        ws_s = chiplet.ws_s
        ws_w = chiplet.ws_w

        hatch_style = hatch_patterns[i % len(hatch_patterns)]

        env_w = w + ws_w + ws_e
        env_h = h + ws_s + ws_n

        silicon_x = env_x + ws_w
        silicon_y = env_y + ws_s

        # Whitespace Envelope (Dashed White Lines)
        if ws_n > 0 or ws_e > 0 or ws_s > 0 or ws_w > 0:
            ws_rect = patches.Rectangle(
                (env_x, env_y),
                env_w,
                env_h,
                linewidth=1.2,
                edgecolor="white",
                facecolor="none",
                linestyle="--",
                zorder=2,
            )
            ax.add_patch(ws_rect)

        # Physical Silicon (Black Hatching, Black Edges, Transparent Fill)
        rect = patches.Rectangle(
            (silicon_x, silicon_y),
            w,
            h,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
            hatch=hatch_style,
            zorder=3,
        )
        ax.add_patch(rect)

        # Labeling (Bold Cyan Text with Black Outline)
        ax.text(
            silicon_x + (w / 2),
            silicon_y + (h / 2),
            str(name),
            color="#00E5FF",
            weight="bold",
            fontsize=8,
            ha="center",
            va="center",
            zorder=4,
        ).set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="black")])

    ax.set_aspect("equal")
    ax.set_xlim(0, sub_max)
    ax.set_ylim(0, sub_max)

    metrics_list = []
    if rms_val is not None:
        metrics_list.append(f"Warpage (RMS): {rms_val:.4f}")
    if hpwl_val is not None:
        metrics_list.append(f"HPWL: {hpwl_val:.2f}")

    metrics_str = " | ".join(metrics_list)
    full_title = f"{title}\nFootprint (mm): {total_w:.2f}x{total_h:.2f}"
    if metrics_str:
        full_title += f" | {metrics_str}"

    ax.set_title(full_title, fontsize=10, weight="bold")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def _plot_single_hist(ax, data, title_text, col_display, color="green"):
    """Helper to plot a single histogram with KDE and stats on a specific axis."""
    if data.empty:
        ax.set_title(f"{title_text}\n(No Data)", fontsize=11, weight="bold")
        return

    mean_val = data.mean()
    std_val = data.std()

    # Histogram
    counts, bins, _ = ax.hist(
        data, bins=15, color=color, edgecolor="black", rwidth=0.9
    )

    # Text above bars
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for count, x in zip(counts, bin_centers):
        if count > 0:
            ax.text(x, count, str(int(count)), ha="center", va="bottom", fontsize=9)

    # KDE (Only plot if there is variance, otherwise gaussian_kde fails)
    if std_val > 1e-6 and len(data.unique()) > 1:
        xmin, xmax = ax.get_xlim()
        x_axis = np.linspace(xmin, xmax, 100)
        kde = gaussian_kde(data)
        pdf_kde = kde.evaluate(x_axis)
        
        bin_width = bins[1] - bins[0]
        scaled_kde = pdf_kde * len(data) * bin_width
        ax.plot(x_axis, scaled_kde, "r-", linewidth=2, alpha=0.8, label="Data Trend (KDE)")

    # Statistics text box
    stats_text = f"N: {len(data)}\nMean: {mean_val:.4f}\nStd: {std_val:.4f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_ylim(0, max(counts) * 1.15 if len(counts) > 0 else 1)
    ax.set_title(title_text, fontsize=11, weight="bold")
    ax.set_xlabel(f"{col_display} Values", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)


def get_pareto_mask(costs: np.ndarray) -> np.ndarray:
    """Calculates the Pareto front mask for a 2D numpy array of minimizing objectives."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep solutions that are NOT strictly dominated by c
            dominated = np.all(costs[is_efficient] >= c, axis=1) & np.any(costs[is_efficient] > c, axis=1)
            is_efficient[is_efficient] = ~dominated
    return is_efficient


def plot_final_generation_histograms(parquet_path):
    """Plots 1x3 comparative distributions (Final Gen | Pareto Front | Entire Space) for each objective."""
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        print(f"Error: Log file '{parquet_path}' not found.")
        return

    out_dir = parquet_file.parent
    df = pd.read_parquet(parquet_file, engine="pyarrow")

    cols_lower = {col.lower(): col for col in df.columns}
    gen_col = cols_lower.get("generation")
    if not gen_col:
        print("Error: 'generation' column not found in Parquet file.")
        return

    max_gen = df[gen_col].max()
    df_final = df[df[gen_col] == max_gen]

    # Resolve objective columns dynamically
    target_obj_cols = ["hpwl", "area", "warpage"]
    real_obj_cols = [cols_lower[c] for c in target_obj_cols if c in cols_lower and cols_lower[c] in df.columns]

    # Calculate global Pareto Front
    if real_obj_cols:
        df_valid = df.dropna(subset=real_obj_cols).copy()
        costs_matrix = df_valid[real_obj_cols].values
        pareto_mask = get_pareto_mask(costs_matrix)
        df_pareto = df_valid[pareto_mask]
    else:
        df_pareto = pd.DataFrame()

    for col_name in target_obj_cols:
        real_col = cols_lower.get(col_name.lower())
        if not real_col or real_col not in df.columns:
            continue

        col_display = real_col.upper() if len(real_col) <= 7 else real_col.title()
        
        # Setup 1x3 Figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Distribution Comparison: {col_display}", fontsize=14, weight="bold", y=1.05)

        # 1) Final Generation
        _plot_single_hist(
            ax=axes[0], 
            data=df_final[real_col].dropna(), 
            title_text=f"Final Gen ({max_gen})", 
            col_display=col_display, 
            color="green"
        )

        # 2) Pareto Front
        _plot_single_hist(
            ax=axes[1], 
            data=df_pareto[real_col].dropna() if not df_pareto.empty else pd.Series(dtype=float), 
            title_text="Global Pareto Front", 
            col_display=col_display, 
            color="orange"
        )

        # 3) Entire Search Space
        _plot_single_hist(
            ax=axes[2], 
            data=df[real_col].dropna(), 
            title_text="Entire Search Space", 
            col_display=col_display, 
            color="dodgerblue"
        )

        plt.tight_layout()
        
        save_path = out_dir / f"distribution_comparison_{real_col.lower()}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def print_metadata(metadata: dict, raw_data: dict) -> None:
    """Print formatted metadata header and chiplet parameters to terminal."""
    print("=" * 75)
    print(f" CONFIGURATION: {metadata.get('name', 'N/A')}")
    print("=" * 75)
    print(f" Version    : {metadata.get('version', 'N/A')}")
    print(f" Author     : {metadata.get('author', 'N/A')}")
    print(f" Created    : {metadata.get('date', 'N/A')}")
    print(f" Remark     : {metadata.get('remark', 'N/A')}")
    print("=" * 75)
    print(" CHIPLET PARAMETERS")
    print("-" * 75)
    print(f" {'Name':<12} | {'Width':<8} | {'Height':<8} | {'WS_N':<6} | {'WS_E':<6} | {'WS_S':<6} | {'WS_W':<6}")
    print("-" * 75)

    for _, props in raw_data.items():
        print(f" {props.name:<12} | {props.w:<8} | {props.h:<8} | {props.ws_n:<6} | {props.ws_e:<6} | {props.ws_s:<6} | {props.ws_w:<6}")

    print("=" * 75)


def print_pareto_table(
    pareto_F: np.ndarray, 
    pop_data: List[Dict[str, Any]] = None
) -> None:
    """Prints a simplified text table of unique Pareto front solutions,
    notating their presence in the final generation and total survival duration.
    """
    if len(pareto_F) == 0:
        print("No Pareto front data provided.")
        return

    # 1. Phenotypic Deduplication 
    rounded_F = np.round(pareto_F, 6)
    unique_F, inverse_indices = np.unique(rounded_F, axis=0, return_inverse=True)

    # 2. Track Lineage & Survival
    max_gen = max((ind.get("generation", 0) for ind in pop_data)) if pop_data else 0
    
    unique_gens = {i: set() for i in range(len(unique_F))}
    if pop_data:
        for idx, unique_idx in enumerate(inverse_indices):
            gen = pop_data[idx].get("generation", 0)
            unique_gens[unique_idx].add(gen)

    # 3. Identify Edge Cases & Balanced Solution
    best_hpwl_idx = np.argmin(unique_F[:, 0])
    best_area_idx = np.argmin(unique_F[:, 1])
    best_warp_idx = np.argmin(unique_F[:, 2])

    mins = np.min(unique_F, axis=0)
    maxs = np.max(unique_F, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1e-9  
    
    normalized_F = (unique_F - mins) / ranges
    distances_to_ideal = np.linalg.norm(normalized_F, axis=1)
    balanced_idx = np.argmin(distances_to_ideal)

    # 4. Print Simplified Table
    header = f"{'Solution':<9} | {'HPWL (Wire)':<12} | {'Area (Size)':<12} | {'Warpage':<10} | {'In Final?':<10} | {'Survived':<9} | {'Note'}"
    divider_len = len(header) + 25 
    title = "PARETO FRONT SOLUTIONS"

    print("\n" + "=" * divider_len)
    print(f"{title:^{divider_len}}")
    print("=" * divider_len)
    print(header)
    print("-" * divider_len)

    for i, row in enumerate(unique_F):
        disp_num = i + 1 
        
        # Calculate survival metrics
        gens_appeared = unique_gens[i]
        if gens_appeared:
            birth_gen = min(gens_appeared)
            in_final = "Yes" if max_gen in gens_appeared else "No"
            survived_gens = max_gen - birth_gen
        else:
            in_final, survived_gens = "N/A", "N/A"
        
        # Flag key notes
        notes = []
        if i == best_hpwl_idx: notes.append("Best HPWL")
        if i == best_area_idx: notes.append("Best Area")
        if i == best_warp_idx: notes.append("Best Warpage")
        if i == balanced_idx:  notes.append("Balanced")

        note_str = ", ".join(notes)

        print(f"#{disp_num:<8} | {row[0]:<12.4f} | {row[1]:<12.4f} | {row[2]:<10.4f} | {in_final:<10} | {str(survived_gens) + ' gens':<9} | {note_str}")

    print("=" * divider_len + "\n")


def save_key_placement_plots(res_X: np.ndarray, res_F: np.ndarray, base_chiplets: dict, parquet_path: str) -> None:
    """Extract key non-dominated solutions and plot physical placement heatmaps loading tensors from .npz files."""
    out_dir = Path(parquet_path).parent
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Calculate worst-case sample (furthest from utopian ideal)
    if len(res_F) > 1:
        f_min = res_F.min(axis=0)
        f_max = res_F.max(axis=0)
        denom = f_max - f_min
        denom[denom == 0] = 1e-8
        normalized_F = (res_F - f_min) / denom
        distances = np.linalg.norm(normalized_F, axis=1)
        worst_idx = int(np.argmax(distances))
    else:
        worst_idx = 0

    indices = {
        "Optimal Wirelength": res_F[:, 0].argmin(),
        "Optimal Footprint": res_F[:, 1].argmin(),
        "Optimal Thermal Warpage": res_F[:, 2].argmin(),
        "Balanced Sample": get_utopian_sample(res_F),
        "Worst-Case Sample": worst_idx,
    }

    unique_configs = {}
    for title_prefix, idx in indices.items():
        if idx not in unique_configs:
            unique_configs[idx] = title_prefix
        else:
            unique_configs[idx] += f", {title_prefix}"

    print(f"\nExtracting and saving key optimal and worst-case placement configurations to {out_dir}...")
    for index, context in unique_configs.items():
        best_individual = res_X[index]
        best_f = res_F[index]
        best_solution = (
            best_individual[0]
            if isinstance(best_individual, (np.ndarray, list))
            else best_individual
        )

        # Look up matching row in Parquet by objective functions
        tol = 1e-4
        mask = (np.abs(df['hpwl'] - best_f[0]) < tol) & \
               (np.abs(df['area'] - best_f[1]) < tol) & \
               (np.abs(df['warpage'] - best_f[2]) < tol)
        
        match = df[mask]
        heatmap_matrix = None
        rms_val = best_f[2]

        if not match.empty:
            if "warpage_tensor_path" in match.columns:
                npz_rel_path = match.iloc[0]['warpage_tensor_path']
                npz_file = out_dir / npz_rel_path
                if npz_file.exists():
                    heatmap_matrix = np.load(npz_file)['warpage']
                    rms_val = float(np.sqrt(np.mean(heatmap_matrix ** 2)))
                else:
                    print(f"Warning: NPZ file not found at {npz_file}")
        else:
            print(f"Warning: Could not find matching Parquet row for objectives {best_f}")

        # Merge Base Whitespace + GA Whitespace
        eval_chiplets = copy.deepcopy(base_chiplets)
        ws_n = best_solution.get("WS_N", {})
        ws_e = best_solution.get("WS_E", {})
        ws_s = best_solution.get("WS_S", {})
        ws_w = best_solution.get("WS_W", {})
        
        for name in eval_chiplets:
            eval_chiplets[name].ws_n += ws_n.get(name, 0.0)
            eval_chiplets[name].ws_e += ws_e.get(name, 0.0)
            eval_chiplets[name].ws_s += ws_s.get(name, 0.0)
            eval_chiplets[name].ws_w += ws_w.get(name, 0.0)

        # Pass eval_chiplets to layout
        x_coords, y_coords, total_w, total_h = seqpair.sequence_pair_to_placement(
            best_solution["X"], best_solution["Y"], eval_chiplets
        )

        raw_filename = f"placement_{context.replace(', ', '_').replace(' ', '_').lower()}.png"
        safe_filepath = str(out_dir / raw_filename)

        plot_placement(
            eval_chiplets,
            x_coords,
            y_coords,
            total_w,
            total_h,
            title=context,
            heatmap_matrix=heatmap_matrix,
            rms_val=rms_val,
            hpwl_val=best_f[0],
            filename=safe_filepath,
        )


def save_batch_placement_plots(
    X: Union[List[Dict], np.ndarray],
    F: np.ndarray,
    base_chiplets: dict,
    target_dir: Path,
    prefix: str,
    npz_paths: Union[List[str], None] = None,
) -> None:
    """Decodes and plots a batch of solutions using GA whitespace, rendering heatmaps directly from paths."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(X)} {prefix.replace('_', ' ')} visualizations to {target_dir}...")

    mapping_records = []

    for i, (individual, obj_vals) in enumerate(zip(X, F)):
        best_solution = (
            individual[0]
            if isinstance(individual, (np.ndarray, list))
            else individual
        )

        gen = best_solution.get("generation") if isinstance(best_solution, dict) else None
        sol_idx = best_solution.get("solution_index") if isinstance(best_solution, dict) else i

        # Retrieve mapped heat tensor path (either from the explicit list or from the sample dictionary)
        npz_file_str = None
        if npz_paths is not None and i < len(npz_paths) and npz_paths[i]:
            npz_file_str = npz_paths[i]
        elif isinstance(best_solution, dict) and best_solution.get("warpage_tensor_path"):
            npz_file_str = best_solution.get("warpage_tensor_path")

        heatmap_matrix = None
        rms_val = obj_vals[2] if len(obj_vals) > 2 else 0.0

        # Load the warpage tensor if path exists
        if npz_file_str:
            npz_file = Path(npz_file_str)
            if npz_file.exists():
                heatmap_matrix = np.load(npz_file)['warpage']
                # Calculate actual RMS from the loaded map
                rms_val = float(np.sqrt(np.mean(heatmap_matrix ** 2)))
            else:
                print(f"[!] Warning: Heatmap file missing at {npz_file}")

        # Merge Base Whitespace + GA Whitespace
        eval_chiplets = copy.deepcopy(base_chiplets)
        ws_n = best_solution.get("WS_N", {})
        ws_e = best_solution.get("WS_E", {})
        ws_s = best_solution.get("WS_S", {})
        ws_w = best_solution.get("WS_W", {})
        
        for name in eval_chiplets:
            eval_chiplets[name].ws_n += ws_n.get(name, 0.0)
            eval_chiplets[name].ws_e += ws_e.get(name, 0.0)
            eval_chiplets[name].ws_s += ws_s.get(name, 0.0)
            eval_chiplets[name].ws_w += ws_w.get(name, 0.0)

        # Pass eval_chiplets to layout
        x_coords, y_coords, total_w, total_h = seqpair.sequence_pair_to_placement(
            best_solution["X"], best_solution["Y"], eval_chiplets
        )

        image_name = f"{prefix}_sol_{i:03d}.png"
        filename = target_dir / image_name
        title = f"{prefix.replace('_', ' ').title()} - Solution {i+1}"

        # Call plot_placement (make sure plot_placement is imported/defined in your file)
        plot_placement(
            eval_chiplets,
            x_coords,
            y_coords,
            total_w,
            total_h,
            title=title,
            heatmap_matrix=heatmap_matrix,
            rms_val=rms_val,
            hpwl_val=obj_vals[0],
            filename=str(filename),
        )

        mapping_records.append({
            "filename": image_name,
            "generation": gen,
            "solution_index": sol_idx
        })

    # Export mapping CSV
    csv_filename = target_dir / f"{prefix}_mapping.csv"
    mapping_df = pd.DataFrame(mapping_records)
    mapping_df.to_csv(csv_filename, index=False)
    print(f"[*] Saved image mapping CSV: {csv_filename}")