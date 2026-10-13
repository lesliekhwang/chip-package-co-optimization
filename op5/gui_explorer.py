"""
gui_explorer.py
---------------------------------
High-Performance Interactive GUI Explorer for Chiplet Placement Optimization.
"""

import sys
import json
import itertools
import threading
import queue
from pathlib import Path
import h5py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath

from layout_renderer import plot_objective_scatter, draw_single_placement

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_MODULE_PATH = PROJECT_ROOT / "src"
if SRC_MODULE_PATH.exists() and str(SRC_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_MODULE_PATH))

def nsga2_fast_non_dominated_sort(fitness: np.ndarray) -> np.ndarray:
    n_samples = fitness.shape[0]
    ranks = np.zeros(n_samples, dtype=int)
    dominated_counts = np.zeros(n_samples, dtype=int)
    dominates_list = [[] for _ in range(n_samples)]

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                dominates_list[i].append(j)
                dominated_counts[j] += 1
            elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                dominates_list[j].append(i)
                dominated_counts[i] += 1

    current_rank = 0
    current_front = [i for i in range(n_samples) if dominated_counts[i] == 0]

    while current_front:
        next_front = []
        for i in current_front:
            ranks[i] = current_rank
            for j in dominates_list[i]:
                dominated_counts[j] -= 1
                if dominated_counts[j] == 0:
                    next_front.append(j)
        current_rank += 1
        current_front = next_front

    return ranks

class ChipletInfo:
    def __init__(self, name: str, w: float, h: float, adj: list = None):
        self.name = str(name)
        self.w = float(w)
        self.h = float(h)
        self.adj = adj if adj is not None else []

class ChipletExplorerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chiplet Placement Interactive Explorer (Directory Mode)")
        self.root.geometry("1450x950")
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.bind("<Escape>", self.clear_selections)

        self.df = None
        self.base_chiplets = None
        self.dir_path = None
        self.ga_config = None
        
        self.selected_indices = []
        self.lasso = None
        self.current_page = 0
        self.load_queue = queue.Queue()

        self._build_controls()
        self._build_panels()

        if len(sys.argv) >= 2:
            self.root.after(100, lambda: self._start_load_directory(sys.argv[1]))
        else:
            self.root.after(100, self._browse_directory)

    def _on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def _build_controls(self):
        control_frame = ttk.LabelFrame(self.root, text=" Controls & Axes ", padding=8)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Load Run Directory", command=self._browse_directory).grid(row=0, column=0, padx=5)
        self.lbl_dir = ttk.Label(control_frame, text="No directory loaded", foreground="red")
        self.lbl_dir.grid(row=0, column=1, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).grid(row=0, column=2, sticky="ns", padx=10)

        ttk.Label(control_frame, text="X-Axis:").grid(row=0, column=3, padx=2)
        self.combo_x = ttk.Combobox(control_frame, state="readonly", width=10)
        self.combo_x.grid(row=0, column=4, padx=5)

        ttk.Label(control_frame, text="Y-Axis:").grid(row=0, column=5, padx=2)
        self.combo_y = ttk.Combobox(control_frame, state="readonly", width=10)
        self.combo_y.grid(row=0, column=6, padx=5)
        
        ttk.Label(control_frame, text="View:").grid(row=0, column=7, padx=2)
        self.view_mode = tk.StringVar(value="Final Gen + 2D Front")
        self.combo_view = ttk.Combobox(control_frame, textvariable=self.view_mode, state="readonly", width=22, 
                                       values=["Final Gen + 2D Front", "N-Dim Stars (Raw)"])
        self.combo_view.grid(row=0, column=8, padx=5)
        self.combo_view.bind("<<ComboboxSelected>>", lambda e: self.update_active_tab())

        self.btn_plot = ttk.Button(control_frame, text="Update Plots", command=self.update_active_tab, state=tk.DISABLED)
        self.btn_plot.grid(row=0, column=9, padx=10)

        ttk.Button(control_frame, text="Clear Selections", command=self.clear_selections).grid(row=0, column=10, padx=5)
        ttk.Button(control_frame, text="Export Plots & Layouts", command=self.export_plots).grid(row=0, column=11, padx=5)

    def _build_panels(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        self.fig_scatter, self.ax_scatter = plt.subplots(figsize=(6, 6))
        self.canvas_scatter = FigureCanvasTkAgg(self.fig_scatter, master=left_frame)
        self.canvas_scatter.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_s = NavigationToolbar2Tk(self.canvas_scatter, left_frame)
        toolbar_s.update()

        mid_frame = ttk.LabelFrame(paned, text=" Physical Placement Sample(s) ", padding=5)
        paned.add(mid_frame, weight=1)

        self.lbl_info = ttk.Label(mid_frame, text="Left Click Scatter: Select. Click & Drag: Lasso multiple. Escape: Clear.", font=("monospace", 9))
        self.lbl_info.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        view_frame = ttk.Frame(mid_frame)
        view_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.show_heatmap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_frame, text="Heatmap", variable=self.show_heatmap_var, command=self._re_render).pack(side=tk.LEFT, padx=5)
        
        self.show_hpwl_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_frame, text="HPWL", variable=self.show_hpwl_var, command=self._re_render).pack(side=tk.LEFT, padx=5)

        ttk.Label(view_frame, text="Font:").pack(side=tk.LEFT, padx=2)
        self.font_size_var = tk.IntVar(value=12)
        ttk.Spinbox(view_frame, from_=6, to=24, textvariable=self.font_size_var, width=3, command=self._re_render).pack(side=tk.LEFT)

        ttk.Label(view_frame, text="Grid Size (NxN):").pack(side=tk.LEFT, padx=(10, 2))
        self.grid_size_var = tk.IntVar(value=3)
        ttk.Spinbox(view_frame, from_=1, to=10, textvariable=self.grid_size_var, width=3, command=self._re_render_reset_page).pack(side=tk.LEFT)

        page_frame = ttk.Frame(mid_frame)
        page_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.btn_prev = ttk.Button(page_frame, text="< Prev", command=self._prev_page, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        
        self.lbl_page = ttk.Label(page_frame, text="Page 1 of 1 (0 items)")
        self.lbl_page.pack(side=tk.LEFT, padx=10)
        
        self.btn_next = ttk.Button(page_frame, text="Next >", command=self._next_page, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(page_frame, orient="horizontal", mode="determinate")
        self.fig_place = plt.figure(figsize=(6, 7))
        self.canvas_place = FigureCanvasTkAgg(self.fig_place, master=mid_frame)
        self.canvas_place.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _re_render_reset_page(self, *args):
        self.current_page = 0
        self.render_placement()

    def _re_render(self, *args):
        self.render_placement()

    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.render_placement()

    def _next_page(self):
        max_page = (len(self.selected_indices) - 1) // (self.grid_size_var.get() ** 2)
        if self.current_page < max_page:
            self.current_page += 1
            self.render_placement()

    def clear_selections(self, event=None):
        self.selected_indices.clear()
        self.current_page = 0
        self.update_active_tab()
        self.render_placement()

    def _browse_directory(self):
        out_dir = PROJECT_ROOT / "runs"
        path = filedialog.askdirectory(title="Select Run Directory", initialdir=out_dir if out_dir.exists() else PROJECT_ROOT)
        if path: 
            self._start_load_directory(path)

    def _start_load_directory(self, dir_path: str):
        self.dir_path = Path(dir_path)
        h5_path = self.dir_path / "run.h5"
        chiplets_path = self.dir_path / "chiplets.json"
        gaconfig_path = self.dir_path / "gaconfig.json"

        self.lbl_dir.config(text="Loading data... please wait", foreground="blue")
        self.btn_plot.config(state=tk.DISABLED)
        self.root.config(cursor="watch")
        
        thread = threading.Thread(target=self._load_directory_worker, args=(h5_path, chiplets_path, gaconfig_path))
        thread.daemon = True
        thread.start()
        
        self.root.after(100, self._check_load_queue)

    def _load_directory_worker(self, h5_path, chiplets_path, gaconfig_path):
        try:
            with open(gaconfig_path, "r") as f: ga_config = json.load(f)
            with open(chiplets_path, "r") as f: chiplet_cfg = json.load(f)
            
            id_map = [k for k in chiplet_cfg.keys() if k.lower() != "interposer"]
            base_chiplets = {}
            for name in id_map:
                c_props = chiplet_cfg[name]
                base_chiplets[name] = ChipletInfo(
                    name, 
                    float(c_props["dim"][0]), 
                    float(c_props["dim"][1]), 
                    c_props["adjacency"]
                )

            records = []
            with h5py.File(h5_path, "r") as h5f:
                gen_keys = sorted([k for k in h5f["generations"].keys() if k.startswith("gen") and k[3:].isdigit()], key=lambda x: int(x.replace("gen", "")))
                for gen_key in gen_keys:
                    gen_grp = h5f["generations"][gen_key]
                    fit_grp = gen_grp["fitness"]
                    pheno_grp = gen_grp["phenotype"]
                    geno_grp = gen_grp["genotype"]
                    
                    fitness = fit_grp["F"][:]
                    pop_size = len(fitness)

                    ranks_arr = nsga2_fast_non_dominated_sort(fitness)

                    bboxes_arr = pheno_grp["bboxes"][:]
                    total_w_arr = pheno_grp["total_w"][:]
                    total_h_arr = pheno_grp["total_h"][:]
                    slack_arr = geno_grp["slack"][:]

                    for sol_idx in range(pop_size):
                        sol_bboxes = {id_map[i]: bboxes_arr[sol_idx, i].tolist() for i in range(len(id_map))}
                        
                        slack_mat = slack_arr[sol_idx]
                        ws_dict = {
                            name: {
                                "ws_n": float(slack_mat[0, i_c]), 
                                "ws_e": float(slack_mat[1, i_c]), 
                                "ws_s": float(slack_mat[2, i_c]), 
                                "ws_w": float(slack_mat[3, i_c])
                            } for i_c, name in enumerate(id_map)
                        }

                        records.append({
                            "generation": int(gen_key.replace("gen", "")),
                            "solution_index": sol_idx,
                            "hpwl": float(fitness[sol_idx, 0]),
                            "area": float(fitness[sol_idx, 1]),
                            "warpage": float(fitness[sol_idx, 2]),
                            "pareto_rank": int(ranks_arr[sol_idx]),
                            "chiplet_whitespace": ws_dict,
                            "bboxes": sol_bboxes,
                            "total_w": float(total_w_arr[sol_idx]),
                            "total_h": float(total_h_arr[sol_idx]),
                        })
            
            df = pd.DataFrame(records)
            self.load_queue.put({"success": True, "df": df, "base_chiplets": base_chiplets, "ga_config": ga_config})
        except Exception as e:
            self.load_queue.put({"success": False, "error": str(e)})

    def _check_load_queue(self):
        try:
            msg = self.load_queue.get_nowait()
            self.root.config(cursor="")
            if msg["success"]:
                self.df = msg["df"]
                self.base_chiplets = msg["base_chiplets"]
                self.ga_config = msg["ga_config"]
                
                self.lbl_dir.config(text=self.dir_path.name, foreground="green")
                
                valid_metrics = [col for col in ["hpwl", "area", "warpage"] if col in self.df.columns]
                self.combo_x["values"] = self.combo_y["values"] = valid_metrics
                self.combo_x.set(valid_metrics[0])
                self.combo_y.set(valid_metrics[1])
                
                self.btn_plot.config(state=tk.NORMAL)
                self.update_active_tab()
            else:
                self.lbl_dir.config(text="Load failed", foreground="red")
                messagebox.showerror("Load Error", f"Failed to load data:\n{msg['error']}")
        except queue.Empty:
            self.root.after(100, self._check_load_queue)

    def update_active_tab(self):
        if self.df is None or self.df.empty: return
        self.update_scatter_plot()

    def _on_lasso_select(self, verts):
        x_col, y_col = self.combo_x.get(), self.combo_y.get()
        points = self.df[[x_col, y_col]].to_numpy()
        xs, ys = zip(*verts)
        dx, dy = max(xs) - min(xs), max(ys) - min(ys)
        
        if dx < 1e-5 and dy < 1e-5:
            click_x, click_y = xs[0], ys[0]
            x_range = (points[:, 0].max() - points[:, 0].min()) or 1.0
            y_range = (points[:, 1].max() - points[:, 1].min()) or 1.0
            norm_dist = np.sqrt(((points[:, 0] - click_x) / x_range) ** 2 + ((points[:, 1] - click_y) / y_range) ** 2)
            nearest_idx = np.argmin(norm_dist)
            
            if norm_dist[nearest_idx] > 0.05:
                self.clear_selections()
                return
            abs_index = self.df.index[nearest_idx]
            self.selected_indices.remove(abs_index) if abs_index in self.selected_indices else self.selected_indices.append(abs_index)
        else:
            new_indices = self.df.index[MplPath(verts).contains_points(points)].tolist()
            current_set = set(self.selected_indices)
            for idx in new_indices:
                if idx not in current_set: self.selected_indices.append(idx)

        self.current_page = 0
        self.update_active_tab()
        self.render_placement()

    def update_scatter_plot(self):
        x_col, y_col = self.combo_x.get(), self.combo_y.get()
        if not x_col or not y_col or self.df is None or self.df.empty: return
        if self.lasso: self.lasso.disconnect_events()
        
        self.ax_scatter.clear()
        plot_objective_scatter(self.ax_scatter, self.df, x_col, y_col, self.view_mode.get(), self.selected_indices)
        self.fig_scatter.tight_layout()
        self.canvas_scatter.draw()
        self.lasso = LassoSelector(self.ax_scatter, onselect=self._on_lasso_select, useblit=True, button=1)

    def render_placement(self):
        self.fig_place.clear()
        n_total = len(self.selected_indices)
        if n_total == 0:
            self.lbl_info.config(text="No points selected."); self.lbl_page.config(text="Page 1 of 1 (0 items)")
            self.canvas_place.draw()
            return

        grid_dim = max(1, self.grid_size_var.get())
        items_per_page = grid_dim * grid_dim
        max_page = (n_total - 1) // items_per_page
        self.current_page = min(self.current_page, max_page)
        start_idx = self.current_page * items_per_page
        
        self.lbl_info.config(text=f"Total Selected: {n_total}")
        self.lbl_page.config(text=f"Page {self.current_page + 1} of {max_page + 1}")
        self.btn_prev.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_page < max_page else tk.DISABLED)
        
        page_indices = self.selected_indices[start_idx:min(start_idx + items_per_page, n_total)]
        self.progress["maximum"] = len(page_indices); self.progress["value"] = 0
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.root.update()

        warpage_cache = {}
        if self.show_heatmap_var.get():
            with h5py.File(self.dir_path / "run.h5", "r") as h5f:
                for idx in page_indices:
                    row = self.df.loc[idx]
                    gen, sol = int(row["generation"]), int(row["solution_index"])
                    warpage_cache[idx] = h5f["generations"][f"gen{gen}"]["phenotype"]["warpage_maps"][sol]

        try:
            axs = self.fig_place.subplots(grid_dim, grid_dim, squeeze=False).flatten()
            for i in range(len(axs)):
                if i < len(page_indices):
                    idx = page_indices[i]
                    row = self.df.loc[idx]
                    
                    row_dict = row.to_dict()
                    row_dict["warpage_matrix"] = warpage_cache.get(idx, None)
                    
                    draw_single_placement(
                        axs[i], self.fig_place, row_dict, self.base_chiplets, 
                        self.show_heatmap_var.get(), self.show_hpwl_var.get(), 
                        self.font_size_var.get(), is_grid=(grid_dim > 1)
                    )
                    
                    gen, sol, rank, hpwl, area, warp = row.get("generation"), row.get("solution_index"), row.get("pareto_rank"), row.get("hpwl"), row.get("area"), row.get("warpage")
                    title_text = f"Gen {gen} | #{sol} (R{rank})\nH:{hpwl:.0f} | A:{area:.0f} | W:{warp:.2f}" if grid_dim > 3 else f"Gen: {gen} | Ind: #{sol} (Rank {rank})\nHPWL: {hpwl:.2f} | Area: {area:.2f} | Warp: {warp:.4f}"
                    axs[i].set_title(title_text, fontsize=max(5, self.font_size_var.get() - (4 if grid_dim > 1 else 0)), weight="bold")
                    
                    self.progress["value"] = i + 1
                    self.root.update()
                else:
                    axs[i].axis('off')
            
            self.fig_place.tight_layout()
            self.canvas_place.draw()
        finally:
            self.progress.pack_forget()

    def export_plots(self):
        if self.df is None or self.df.empty: return
        dpi_val = tk.simpledialog.askinteger("Export DPI", "Enter DPI for exported plots:", initialvalue=300)
        if not dpi_val: return
        dir_path = filedialog.askdirectory(title="Select Directory to Save Exports")
        if not dir_path: return
            
        base_path = Path(dir_path)
        metrics = ["hpwl", "area", "warpage"]
        combos = list(itertools.combinations(metrics, 2))
        
        for x_m, y_m in combos:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_objective_scatter(ax, self.df, x_m, y_m, self.view_mode.get(), self.selected_indices)
            fig.tight_layout()
            fig.savefig(base_path / f"pareto_{x_m}_{y_m}.png", dpi=dpi_val)
            plt.close(fig)

        if not self.selected_indices:
            messagebox.showinfo("Export Complete", f"Saved {len(combos)} Pareto plots.")
            return

        with h5py.File(self.dir_path / "run.h5", "r") as h5f:
            for sample_id, idx in enumerate(self.selected_indices, 1):
                row = self.df.loc[idx]
                row_dict = row.to_dict()
                
                if self.show_heatmap_var.get():
                    gen, sol = int(row["generation"]), int(row["solution_index"])
                    row_dict["warpage_matrix"] = h5f["generations"][f"gen{gen}"]["phenotype"]["warpage_maps"][sol]
                else:
                    row_dict["warpage_matrix"] = None

                fig_temp, ax_temp = plt.subplots(figsize=(6, 5))
                draw_single_placement(
                    ax_temp, fig_temp, row_dict, self.base_chiplets, 
                    self.show_heatmap_var.get(), self.show_hpwl_var.get(), 
                    self.font_size_var.get(), is_grid=False
                )
                
                gen = row["generation"]
                sol = row["solution_index"]
                rank = row["pareto_rank"]
                hpwl = row["hpwl"]
                area = row["area"]
                warp = row["warpage"]
                
                title_text = (f"Gen: {gen} | Ind: #{sol} (Rank {rank})\n"
                              f"HPWL: {hpwl:.2f} | Area: {area:.2f} | Warp: {warp:.4f}")
                
                ax_temp.set_title(title_text, weight="bold", fontsize=max(8, self.font_size_var.get()))
                
                fig_temp.tight_layout()
                fig_temp.savefig(base_path / f"placement_sample_{sample_id}_sol_{sol}.png", dpi=dpi_val, bbox_inches="tight")
                plt.close(fig_temp)
                
                self.root.update()

        messagebox.showinfo("Success", f"Exported {len(combos)} Pareto plots and {len(self.selected_indices)} layouts.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChipletExplorerGUI(root)
    root.mainloop()