"""
animate.py
---------------------------------
High-performance population evolution animation tool using Matplotlib buffer 
rasterization, persistent patch recycling, layout pre-caching, and hardware-accelerated 
OpenCV video streams for maximum rendering speed and compressed output.

Usage:
    python animate.py <scenario.toml> <history.parquet> <interval_ms> <dpi>
"""

import sys
import math
import copy
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Fast non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import json
import cv2

import seqpair
import loader

plt.style.use("fast")
plt.rcParams["font.family"] = "monospace"
plt.rcParams["path.simplify"] = True


def get_eval_chiplets(base_chiplets: dict, ws_dict: dict) -> dict:
    """Fast shallow-copy evaluation chiplet generator."""
    eval_chiplets = {}
    for name, chip in base_chiplets.items():
        c = copy.copy(chip)
        if name in ws_dict:
            vals = ws_dict[name]
            c.ws_n += float(vals.get("ws_n", 0.0))
            c.ws_e += float(vals.get("ws_e", 0.0))
            c.ws_s += float(vals.get("ws_s", 0.0))
            c.ws_w += float(vals.get("ws_w", 0.0))
        eval_chiplets[name] = c
    return eval_chiplets


def precalculate_data(df, generations, gen_col, sol_col, seq_x_col, seq_y_col, ws_col, hpwl_col, warp_col, area_col, base_chiplets):
    """Pre-calculates population metrics and layout patch geometry for high-speed rendering."""
    print("Pre-calculating convergence metrics and pre-caching placement geometries...")
    
    metrics = {
        "hpwl_min": [], "hpwl_mean": [], "hpwl_max": [],
        "warp_min": [], "warp_mean": [], "warp_max": [],
        "area_min": [], "area_mean": [], "area_max": [],
        "seq_std_x": [], "seq_std_y": [],
        "ws_std_x": [], "ws_std_y": []
    }
    
    chiplet_names = list(base_chiplets.keys())
    layout_cache = []

    for gen in generations:
        df_gen = df[df[gen_col] == gen].sort_values(by=sol_col)
        
        # Extract native numpy arrays to eliminate Pandas overhead
        hpwls = df_gen[hpwl_col].to_numpy()
        warps = df_gen[warp_col].to_numpy()
        areas = df_gen[area_col].to_numpy()
        
        metrics["hpwl_min"].append(hpwls.min())
        metrics["hpwl_mean"].append(hpwls.mean())
        metrics["hpwl_max"].append(hpwls.max())
        
        metrics["warp_min"].append(warps.min())
        metrics["warp_mean"].append(warps.mean())
        metrics["warp_max"].append(warps.max())
        
        metrics["area_min"].append(areas.min())
        metrics["area_mean"].append(areas.mean())
        metrics["area_max"].append(areas.max())
        
        seq_x_list = df_gen[seq_x_col].tolist()
        seq_y_list = df_gen[seq_y_col].tolist()
        ws_list = df_gen[ws_col].tolist()
        
        pos_x, pos_y = [], []
        widths, heights = [], []
        gen_layouts = []
        
        for sx, sy, ws_dict in zip(seq_x_list, seq_y_list, ws_list):
            sx_list, sy_list = list(sx), list(sy)
            pos_x.append([sx_list.index(c) for c in chiplet_names])
            pos_y.append([sy_list.index(c) for c in chiplet_names])
            
            eval_chiplets = get_eval_chiplets(base_chiplets, ws_dict)
            x_coords, y_coords, total_w, total_h = seqpair.sequence_pair_to_placement(sx_list, sy_list, eval_chiplets)
            
            widths.append(total_w)
            heights.append(total_h)
            
            # Pre-calculate layout patch bounds per individual
            ind_layout = {
                "total_w": total_w,
                "total_h": total_h,
                "chiplets": []
            }
            
            for name in chiplet_names:
                chiplet = eval_chiplets[name]
                env_x, env_y = x_coords[name], y_coords[name]
                w_c, h_c = chiplet.w, chiplet.h
                ws_n, ws_e, ws_s, ws_w = chiplet.ws_n, chiplet.ws_e, chiplet.ws_s, chiplet.ws_w
                
                ind_layout["chiplets"].append({
                    "env_bounds": (env_x, env_y, w_c + ws_w + ws_e, h_c + ws_s + ws_n),
                    "silicon_bounds": (env_x + ws_w, env_y + ws_s, w_c, h_c),
                    "has_ws": (ws_n > 0 or ws_e > 0 or ws_s > 0 or ws_w > 0)
                })
            
            gen_layouts.append(ind_layout)
            
        metrics["seq_std_x"].append(np.mean(np.std(pos_x, axis=0)))
        metrics["seq_std_y"].append(np.mean(np.std(pos_y, axis=0)))
        metrics["ws_std_x"].append(np.std(widths))
        metrics["ws_std_y"].append(np.std(heights))
        
        layout_cache.append(gen_layouts)

    return metrics, layout_cache


def main():
    if len(sys.argv) != 5:
        print("Usage: python animate.py <scenario.toml> <history.parquet> <interval_ms> <dpi>")
        sys.exit(1)

    try:
        cfg = {
            "SCENARIO_FILE": sys.argv[1],
            "HISTORY_PARQUET": sys.argv[2],
            "INTERVAL_MS": int(sys.argv[3]),
            "DPI": int(sys.argv[4])
        }
    except ValueError:
        print("Error: interval_ms and dpi must be integers.", file=sys.stderr)
        sys.exit(1)

    scenario_path = cfg["SCENARIO_FILE"]
    parquet_path = Path(cfg["HISTORY_PARQUET"])
    interval_ms = cfg["INTERVAL_MS"]
    dpi = cfg["DPI"]

    if not parquet_path.exists():
        print(f"Error: Parquet file '{parquet_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # 1. Load Data
    print(f"Loading scenario from {scenario_path}...")
    _, base_chiplets = loader.load_scenario(scenario_path)

    print(f"Loading historical data from {parquet_path}...")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    
    cols_lower = {col.lower(): col for col in df.columns}
    gen_col = cols_lower.get("generation")
    sol_col = cols_lower.get("solution_index")
    seq_x_col = cols_lower.get("x", cols_lower.get("sequence_x"))
    seq_y_col = cols_lower.get("y", cols_lower.get("sequence_y"))
    ws_col = cols_lower.get("chiplet_whitespace", "chiplet_whitespace")
    
    hpwl_col = cols_lower.get("hpwl", "hpwl")
    warp_col = cols_lower.get("warpage", "warpage")
    area_col = cols_lower.get("area", "area")

    if not all([gen_col, sol_col, seq_x_col, seq_y_col, ws_col]):
        print("Error: Missing required tracking columns in Parquet file.", file=sys.stderr)
        sys.exit(1)

    generations = sorted(df[gen_col].unique())
    pop_size = df[df[gen_col] == generations[0]][sol_col].nunique()
    
    # 2. Pre-Calculate Population Metrics & Pre-Cache Layout Patches
    metrics, layout_cache = precalculate_data(
        df, generations, gen_col, sol_col, seq_x_col, seq_y_col, ws_col, hpwl_col, warp_col, area_col, base_chiplets
    )

    # 3. Grid Setup 
    cols = math.ceil(math.sqrt(pop_size))
    rows = math.ceil(pop_size / cols)

    print(f"Found {len(generations)} generations with population size {pop_size}.")
    
    fig = plt.figure(figsize=((cols * 2.5) + 2, (rows + 2.5) * 2.0), dpi=dpi)
    fig.patch.set_facecolor('#F4F4F4')
    
    # DECOUPLED GRID SPECS
    gs_top = fig.add_gridspec(nrows=1, ncols=5, left=0.07, right=0.97, top=0.92, bottom=0.76, wspace=0.25)
    gs_div = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.97, top=0.74, bottom=0.73)
    gs_bot = fig.add_gridspec(nrows=rows, ncols=cols, left=0.05, right=0.97, top=0.70, bottom=0.02, wspace=0.2, hspace=0.15)
    
    # 4. Initialize Metric Axes & Persistent Lines
    metric_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
    
    for ax in metric_axes:
        ax.set_facecolor('white')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=7, direction='out')
        ax.set_xlim(0, max(generations))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))

    lines = {
        "hpwl_min": metric_axes[0].plot([], [], 'g-', label="Min")[0],
        "hpwl_mean": metric_axes[0].plot([], [], 'b--', label="Mean")[0],
        "hpwl_max": metric_axes[0].plot([], [], 'r:', alpha=0.5, label="Max")[0],
        
        "warp_min": metric_axes[1].plot([], [], 'g-', label="Min")[0],
        "warp_mean": metric_axes[1].plot([], [], 'b--', label="Mean")[0],
        "warp_max": metric_axes[1].plot([], [], 'r:', alpha=0.5, label="Max")[0],
        
        "area_min": metric_axes[2].plot([], [], 'g-', label="Min")[0],
        "area_mean": metric_axes[2].plot([], [], 'b--', label="Mean")[0],
        "area_max": metric_axes[2].plot([], [], 'r:', alpha=0.5, label="Max")[0],
        
        "seq_std_x": metric_axes[3].plot([], [], 'm-', label="Seq X")[0],
        "seq_std_y": metric_axes[3].plot([], [], 'c-', label="Seq Y")[0],
        
        "ws_std_x": metric_axes[4].plot([], [], 'm-', label="Slack X")[0],
        "ws_std_y": metric_axes[4].plot([], [], 'c-', label="Slack Y")[0],
    }

    titles = ["HPWL", "Warpage", "Area", "Sequence Std Dev", "Slack Std Dev"]
    for i, title in enumerate(titles):
        metric_axes[i].set_title(title, fontsize=9, weight='bold')
        metric_axes[i].legend(fontsize=7, loc="upper right")

    metric_axes[0].set_ylim(min(metrics["hpwl_min"]) * 0.95, max(metrics["hpwl_max"]) * 1.05)
    metric_axes[1].set_ylim(min(metrics["warp_min"]) * 0.95, max(metrics["warp_max"]) * 1.05)
    metric_axes[2].set_ylim(min(metrics["area_min"]) * 0.95, max(metrics["area_max"]) * 1.05)
    metric_axes[3].set_ylim(0, max(max(metrics["seq_std_x"]), max(metrics["seq_std_y"]), 1e-9) * 1.1)
    metric_axes[4].set_ylim(0, max(max(metrics["ws_std_x"]), max(metrics["ws_std_y"]), 1e-9) * 1.1)

    divider_ax = fig.add_subplot(gs_div[0, 0])
    divider_ax.axhline(0.5, color='black', linewidth=1.5)
    divider_ax.axis('off')

    # 5. Initialize Population Grid Axes & Persistent Patches ONCE
    axes_flat = []
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(gs_bot[r, c])
            axes_flat.append(ax)

    hatch_patterns = ["///", "\\\\\\", "|||", "---", "+++", "xxx", "ooo", "***"]
    n_chiplets = len(base_chiplets)
    grid_artists = []

    for i, ax in enumerate(axes_flat):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        
        if i >= pop_size:
            ax.axis('off')
            grid_artists.append(None)
            continue
            
        # Outer Bounding Box Patch
        bg_patch = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor="black", facecolor="white")
        ax.add_patch(bg_patch)
        
        # Whitespace and Silicon Patches
        ws_patches = []
        silicon_patches = []
        
        for j in range(n_chiplets):
            ws_p = patches.Rectangle((0, 0), 1, 1, linewidth=0.5, edgecolor="gray", facecolor="none", linestyle="--")
            sil_p = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor="black", facecolor="none", hatch=hatch_patterns[j % len(hatch_patterns)])
            
            ax.add_patch(ws_p)
            ax.add_patch(sil_p)
            
            ws_patches.append(ws_p)
            silicon_patches.append(sil_p)
            
        ax.set_title(f"#{i+1}", fontsize=7)
        
        grid_artists.append({
            "ax": ax,
            "bg": bg_patch,
            "ws": ws_patches,
            "silicon": silicon_patches
        })

    x_gen = generations
    total_frames = len(generations)

    # Initialize Compressed OpenCV Video Writer
    fps = max(1, int(1000 / interval_ms))
    out_file = parquet_path.parent / "population_evolution.mp4"

    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    frame_array = np.asarray(rgba_buffer)
    h, w, _ = frame_array.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Hardware-accelerated MP4 compression
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))

    print(f"\nRendering optimized video stream directly ({fps} FPS)...")

    # 6. Render Frames with In-Place Patch Updates & ETA Tracking
    start_time = time.time()

    for frame_idx, gen in enumerate(generations):
        # Calculate ETA
        if frame_idx > 0:
            elapsed = time.time() - start_time
            avg_sec_per_frame = elapsed / frame_idx
            remaining_frames = total_frames - frame_idx
            eta_seconds = avg_sec_per_frame * remaining_frames
            eta_str = f"{eta_seconds:.1f}s remaining"
        else:
            eta_str = "calculating ETA..."

        current_frame = frame_idx + 1
        print(f"Rendering Frame {current_frame}/{total_frames} | {eta_str}    ", end="\r")
        
        # Update Metric Line Plots
        slice_idx = frame_idx + 1
        x_data = x_gen[:slice_idx]
        for key, line in lines.items():
            line.set_data(x_data, metrics[key][:slice_idx])

        fig.suptitle(f"Generation {gen} | Population Evolution", fontsize=16, weight="bold")
        gen_layouts = layout_cache[frame_idx]

        # Mutate Layout Artists in Place
        for i, layout in enumerate(gen_layouts):
            artists = grid_artists[i]
            if artists is None:
                continue
            
            total_w = layout["total_w"]
            total_h = layout["total_h"]
            
            artists["bg"].set_bounds(0, 0, total_w, total_h)
            artists["ax"].set_xlim(0, total_w)
            artists["ax"].set_ylim(0, total_h)
            
            for j, ch_data in enumerate(layout["chiplets"]):
                env_x, env_y, env_w, env_h = ch_data["env_bounds"]
                sil_x, sil_y, sil_w, sil_h = ch_data["silicon_bounds"]
                
                ws_p = artists["ws"][j]
                sil_p = artists["silicon"][j]
                
                ws_p.set_bounds(env_x, env_y, env_w, env_h)
                ws_p.set_visible(ch_data["has_ws"])
                
                sil_p.set_bounds(sil_x, sil_y, sil_w, sil_h)

        # Draw frame buffer directly to OpenCV Writer
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        writer.write(bgr_frame)

    writer.release()
    plt.close(fig)
    print(f"\nDone! Animation saved to {out_file}")

    # Save configuration to output directory
    config_file_path = parquet_path.parent / "animate_config.json"
    with open(config_file_path, "w") as file:
        json.dump(cfg, file, indent=4)
    print(f"Configuration saved to: {config_file_path}")

if __name__ == "__main__":
    main()