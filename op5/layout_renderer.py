import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

def get_pareto_mask(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=bool)
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not is_pareto[i]:
            continue
        diff = points - points[i]
        dominated_by = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
        if np.any(dominated_by):
            is_pareto[i] = False
    return is_pareto

def plot_objective_scatter(ax, df, x_col, y_col, view_mode, selected_indices=None):
    pts_2d = df[[x_col, y_col]].to_numpy()
    
    if view_mode == "N-Dim Stars (Raw)":
        all_metrics = [m for m in ["hpwl", "area", "warpage"] if m in df.columns]
        nd_pts = df[all_metrics].to_numpy()
        nd_mask = get_pareto_mask(nd_pts)
        
        min_gen = df["generation"].min()
        max_gen = df["generation"].max()
        
        orange_cmap = LinearSegmentedColormap.from_list("custom_oranges", ["#FFE4B5", "#FF8C00"])
        yellow_cmap = LinearSegmentedColormap.from_list("custom_yellows", ["#FFFFE0", "#FFD700"])
        
        bg_mask = ~nd_mask
        if np.any(bg_mask):
            df_bg = df[bg_mask]
            ax.scatter(df_bg[x_col], df_bg[y_col], 
                       c=df_bg["generation"], cmap=orange_cmap, 
                       vmin=min_gen, vmax=max_gen, 
                       s=35, alpha=0.6, edgecolors="none", zorder=1)
            
        if np.any(nd_mask):
            df_nd = df[nd_mask]
            ax.scatter(df_nd[x_col], df_nd[y_col], 
                       c=df_nd["generation"], cmap=yellow_cmap, 
                       vmin=min_gen, vmax=max_gen, 
                       marker="*", s=160, alpha=0.85, edgecolors="black", linewidth=0.6, zorder=4)

        ax.scatter([], [], c="#FF8C00", alpha=0.6, s=35, label="Dominated (Lighter = Older)")
        ax.scatter([], [], c="#FFD700", marker="*", s=160, alpha=0.85, edgecolors="black", linewidth=0.6, label="N-Dim Pareto (Lighter = Older)")

    else:
        max_gen = df["generation"].max()
        is_final_gen = df["generation"] == max_gen
        
        ax.scatter(pts_2d[~is_final_gen, 0], pts_2d[~is_final_gen, 1], c="lightgray", s=15, alpha=0.5, label="Previous Gens", zorder=1)
        
        df_final = df[is_final_gen]
        if not df_final.empty:
            ax.scatter(df_final[x_col], df_final[y_col], c="orange", s=30, alpha=0.8, edgecolors="black", linewidth=0.5, label=f"Final Gen ({max_gen})", zorder=2)
            
            pts_final_2d = df_final[[x_col, y_col]].to_numpy()
            final_mask = get_pareto_mask(pts_final_2d)
            pf_final = pts_final_2d[final_mask]
            pf_final = pf_final[np.argsort(pf_final[:, 0])]
            
            if len(pf_final) > 1:
                ax.step(pf_final[:, 0], pf_final[:, 1], where='post', color="orange", linestyle="--", linewidth=1.5, label="Final Gen Pareto Front", zorder=3)
            elif len(pf_final) == 1:
                ax.plot(pf_final[:, 0], pf_final[:, 1], marker="o", color="orange", linestyle="--", linewidth=1.5, label="Final Gen Pareto Front", zorder=3)
        
        global_mask = get_pareto_mask(pts_2d)
        pf_global = pts_2d[global_mask]
        pf_global = pf_global[np.argsort(pf_global[:, 0])]
        
        if len(pf_global) > 1:
            ax.step(pf_global[:, 0], pf_global[:, 1], where='post', color="red", linestyle="-", linewidth=1.5, zorder=4)
            ax.scatter(pf_global[:, 0], pf_global[:, 1], c="red", marker="D", s=40, edgecolors="black", linewidth=0.5, label="Global 2D Pareto Front", zorder=5)
        elif len(pf_global) == 1:
            ax.scatter(pf_global[:, 0], pf_global[:, 1], c="red", marker="D", s=40, edgecolors="black", linewidth=0.5, label="Global 2D Pareto Front", zorder=5)

        if len(pf_global) > 0:
            utopian_x = pf_global[:, 0].min()
            utopian_y = pf_global[:, 1].min()
            nadir_x = pf_global[:, 0].max()
            nadir_y = pf_global[:, 1].max()

            ax.scatter(utopian_x, utopian_y, c="yellow", marker="*", s=160, edgecolors="black", label="Utopian Point", zorder=6)
            ax.scatter(nadir_x, nadir_y, c="yellow", marker="X", s=110, edgecolors="black", label="Nadir Point", zorder=6)

    if selected_indices:
        df_sel = df.loc[selected_indices]
        for sample_id, idx in enumerate(selected_indices, 1):
            row_sel = df_sel.loc[idx]
            ax.annotate(str(sample_id), (row_sel[x_col], row_sel[y_col]),
                        ha='center', va='center', fontsize=10, weight='bold', 
                        color='black', bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=0.5), zorder=8)

    ax.set_title(f"Objective Space: {x_col.upper()} vs {y_col.upper()}", weight="bold")
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, ncol=1, framealpha=0.9)


def draw_single_placement(ax, fig, row, base_chiplets, show_heatmap, show_hpwl, font_size, is_grid=False):
    f_size = max(5, font_size - (4 if is_grid else 0))
    bboxes_dict = row["bboxes"]
    total_w = row["total_w"]
    total_h = row["total_h"]
    ws_dict = row["chiplet_whitespace"]
    
    heatmap_matrix = row.get("warpage_matrix", None) if show_heatmap else None

    if heatmap_matrix is not None:
        im = ax.imshow(heatmap_matrix * 1000.0, cmap="magma", extent=[0, total_w, 0, total_h], origin="lower", zorder=0, interpolation="nearest")
        if not is_grid and fig is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(r"Warpage ($\mu$m)", rotation=270, labelpad=15, fontsize=f_size)
            cbar.ax.tick_params(labelsize=f_size - 1)
        interposer_face = "none"
    else:
        interposer_face = "#F4F4F4"

    lw = 1.0 if is_grid else 2.0
    interposer = patches.Rectangle((0, 0), total_w, total_h, linewidth=lw, edgecolor="yellow", facecolor=interposer_face, linestyle="--", zorder=1.5)
    ax.add_patch(interposer)

    hatch_patterns = ["///", "\\\\\\", "|||", "---", "+++", "xxx", "ooo", "***"]
    centers = {}

    for i, (name, chiplet) in enumerate(base_chiplets.items()):
        bbox = bboxes_dict.get(name)
        if not bbox: continue
        
        env_x, env_y, env_x2, env_y2 = bbox
        w, h = env_x2 - env_x, env_y2 - env_y

        vals = ws_dict.get(name, {})
        ws_n, ws_e, ws_s, ws_w = float(vals.get("ws_n", 0.0)), float(vals.get("ws_e", 0.0)), float(vals.get("ws_s", 0.0)), float(vals.get("ws_w", 0.0))

        silicon_x, silicon_y = env_x + ws_w, env_y + ws_s
        centers[name] = (silicon_x + (w / 2.0), silicon_y + (h / 2.0))

        if ws_n > 0 or ws_e > 0 or ws_s > 0 or ws_w > 0:
            ws_rect = patches.Rectangle((env_x, env_y), w + ws_w + ws_e, h + ws_s + ws_n, linewidth=0.8, edgecolor="white", facecolor="none", linestyle="--", zorder=2)
            ax.add_patch(ws_rect)

        rect = patches.Rectangle((silicon_x, silicon_y), w, h, linewidth=lw, edgecolor="#00FF88", facecolor="none", hatch=hatch_patterns[i % len(hatch_patterns)], zorder=3)
        ax.add_patch(rect)

        txt = ax.text(centers[name][0], centers[name][1], str(name), color="#00E5FF", weight="bold", fontsize=f_size, ha="center", va="center", zorder=5)
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    if show_hpwl:
        drawn_edges = set()
        for name, chiplet in base_chiplets.items():
            for adj_name in getattr(chiplet, 'adj', []):
                adj_str = str(adj_name)
                edge = tuple(sorted([name, adj_str]))
                if edge not in drawn_edges and name in centers and adj_str in centers:
                    drawn_edges.add(edge)
                    line, = ax.plot([centers[name][0], centers[adj_str][0]], [centers[name][1], centers[adj_str][1]], color="cyan", linestyle="-", linewidth=lw, alpha=0.9, zorder=4)
                    line.set_path_effects([path_effects.withStroke(linewidth=lw+1.5, foreground='black')])

    ax.set_aspect("equal")
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    if not is_grid:
        ax.set_xlabel("X (mm)", fontsize=f_size)
        ax.set_ylabel("Y (mm)", fontsize=f_size)
    ax.tick_params(axis='both', which='major', labelsize=f_size - 1)