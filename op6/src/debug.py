"""Interactive GUI debug viewer for full population and Pareto front exploration."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RadioButtons
import numpy as np

try:
    from transforms import convert_slack_indices_to_floats
except ImportError:
    from rules.transforms import convert_slack_indices_to_floats

from rules.seqpair import decode_sequence_pair_batch


def visualize_interactive_population(
    pop_x: np.ndarray,
    pop_f: np.ndarray,
    is_pareto: np.ndarray,
    chiplet_id_map: dict[str, int],
    chiplet_dimensions: np.ndarray,
    min_slack: np.ndarray,
    max_slack: np.ndarray,
    slack_steps: int,
    chiplet_edges: np.ndarray,
    objective_names: list[str] | None = None,
) -> None:
    """Opens an interactive GUI plotting all population individuals with Pareto highlighting."""
    if pop_x is None or len(pop_x) == 0:
        print("[DEBUG] No solutions in population to visualize.")
        return

    n_solutions, _ = pop_x.shape
    num_chiplets = len(chiplet_id_map)
    n_objs = pop_f.shape[1]

    if objective_names is None:
        objective_names = [f"Objective {i+1}" for i in range(n_objs)]

    id_to_name = {v: k for k, v in chiplet_id_map.items()}

    # State Variables
    state = {
        "selected_idx": 0,
        "x_obj_idx": 0,
        "y_obj_idx": 1 if n_objs > 1 else 0,
        "scatter_artist": None,
    }

    # Setup Main Figure Window
    fig = plt.figure(figsize=(16, 8.5))
    fig.canvas.manager.set_window_title("NSGA-II Chiplet Population Explorer")

    # Grid Layout: Left (Scatter + Radio Widgets), Right (Floorplan View)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3], wspace=0.25)
    gs_left = gs[0].subgridspec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1], hspace=0.3)

    ax_scatter = fig.add_subplot(gs_left[0, :])
    ax_radio_x = fig.add_subplot(gs_left[1, 0])
    ax_radio_y = fig.add_subplot(gs_left[1, 1])
    ax_layout = fig.add_subplot(gs[1])

    # Radio Button Selectors
    radio_x = RadioButtons(ax_radio_x, objective_names, active=state["x_obj_idx"])
    radio_y = RadioButtons(ax_radio_y, objective_names, active=state["y_obj_idx"])
    ax_radio_x.set_title("X Axis Objective", fontsize=9, fontweight="bold")
    ax_radio_y.set_title("Y Axis Objective", fontsize=9, fontweight="bold")

    cmap = plt.get_cmap("tab20")

    def draw_layout(solution_idx: int) -> None:
        """Decodes and draws the selected solution's floorplan in the right panel."""
        ax_layout.clear()

        chrom = np.atleast_2d(pop_x[solution_idx])
        seq_x_batch = chrom[:, 0:num_chiplets]
        seq_y_batch = chrom[:, num_chiplets : 2 * num_chiplets]
        slack_indices_batch = chrom[:, 2 * num_chiplets : 6 * num_chiplets].reshape(
            1, 4, num_chiplets
        )
        rot_batch = chrom[:, 6 * num_chiplets : 7 * num_chiplets].astype(bool)

        slack_batch = convert_slack_indices_to_floats(
            slack_indices=slack_indices_batch,
            min_slack=min_slack,
            max_slack=max_slack,
            slack_steps=slack_steps,
        )

        bbox_batch, total_w, total_h = decode_sequence_pair_batch(
            chiplet_dims=chiplet_dimensions,
            seq_x_batch=seq_x_batch,
            seq_y_batch=seq_y_batch,
            slack_batch=slack_batch,
            rot_batch=rot_batch,
        )

        bboxes = bbox_batch[0]
        slacks = slack_batch[0]
        centroids = np.zeros((num_chiplets, 2), dtype=np.float32)

        # 1. Render Outer Whitespace and Inner Silicon Core
        for c in range(num_chiplets):
            chiplet_name = id_to_name.get(c, f"C_{c}")
            x1, y1, x2, y2 = bboxes[c]
            w, h = x2 - x1, y2 - y1
            centroids[c] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

            n_s, e_s, s_s, w_s = slacks[:, c]
            ox1, oy1 = x1 - w_s, y1 - s_s
            ow, oh = w + e_s + w_s, h + n_s + s_s

            # Outer Whitespace Boundary
            outer_rect = patches.Rectangle(
                (ox1, oy1),
                ow,
                oh,
                linewidth=1.2,
                edgecolor="gray",
                facecolor="lightgray",
                linestyle="--",
                alpha=0.3,
                label="Whitespace Bounds" if c == 0 else "",
            )
            ax_layout.add_patch(outer_rect)

            # Inner Solid Chiplet
            color = cmap(c % 20)
            inner_rect = patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=1.5,
                edgecolor="black",
                facecolor=color,
                alpha=0.85,
                label="Chiplet Core" if c == 0 else "",
            )
            ax_layout.add_patch(inner_rect)

            ax_layout.text(
                centroids[c, 0],
                centroids[c, 1],
                f"{chiplet_name}\n({w:.1f}x{h:.1f})",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
            )

        # 2. Render HPWL Connectivity Netlist
        if chiplet_edges.shape[0] > 0:
            for idx, (u, v) in enumerate(chiplet_edges):
                xu, yu = centroids[u]
                xv, yv = centroids[v]
                ax_layout.plot(
                    [xu, xv],
                    [yu, yv],
                    color="crimson",
                    linewidth=1.8,
                    linestyle="-",
                    marker="o",
                    markersize=4,
                    alpha=0.75,
                    label="HPWL Connection" if idx == 0 else "",
                )

        ax_layout.set_xlim(-1.0, total_w[0] + 1.0)
        ax_layout.set_ylim(-1.0, total_h[0] + 1.0)
        ax_layout.set_aspect("equal")
        ax_layout.set_xlabel("X Dimension (mm)", fontsize=10)
        ax_layout.set_ylabel("Y Dimension (mm)", fontsize=10)

        status_text = "PARETO OPTIMAL" if is_pareto[solution_idx] else "Suboptimal"
        cost_info = " | ".join(
            [f"{objective_names[i]}: {pop_f[solution_idx, i]:.2f}" for i in range(n_objs)]
        )
        ax_layout.set_title(
            f"Layout View (Solution #{solution_idx + 1}/{n_solutions} - [{status_text}])\n{cost_info}\nTotal Dimensions: {total_w[0]:.2f} x {total_h[0]:.2f} mm",
            fontsize=11,
            fontweight="bold",
        )
        ax_layout.grid(True, linestyle=":", alpha=0.5)

        handles, labels = ax_layout.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_layout.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    def update_scatter() -> None:
        """Redraws scatter plot with Pareto solutions in crimson and suboptimal solutions in gray."""
        ax_scatter.clear()

        x_data = pop_f[:, state["x_obj_idx"]]
        y_data = pop_f[:, state["y_obj_idx"]]

        # Per-point style mapping
        colors = ["crimson" if p else "slategray" for p in is_pareto]
        sizes = [90 if p else 35 for p in is_pareto]
        alphas = [0.85 if p else 0.45 for p in is_pareto]

        # Single picker artist
        state["scatter_artist"] = ax_scatter.scatter(
            x_data,
            y_data,
            c=colors,
            s=sizes,
            alpha=alphas,
            picker=True,
            pickradius=8,
            zorder=3,
        )

        # Legend Dummies
        ax_scatter.scatter(
            [], [], c="crimson", s=80, alpha=0.85, label=f"Pareto Front ({np.sum(is_pareto)})"
        )
        ax_scatter.scatter(
            [], [], c="slategray", s=35, alpha=0.45, label=f"Suboptimal ({np.sum(~is_pareto)})"
        )

        # Highlight active selection
        sel_idx = state["selected_idx"]
        ax_scatter.scatter(
            x_data[sel_idx],
            y_data[sel_idx],
            c="gold",
            edgecolors="black",
            s=160,
            linewidths=2.0,
            zorder=4,
            label="Active Selection",
        )

        ax_scatter.set_xlabel(objective_names[state["x_obj_idx"]], fontsize=10, fontweight="bold")
        ax_scatter.set_ylabel(objective_names[state["y_obj_idx"]], fontsize=10, fontweight="bold")
        ax_scatter.set_title(
            f"Population Objective Space ({n_solutions} Total Individuals)",
            fontsize=11,
            fontweight="bold",
        )
        ax_scatter.grid(True, linestyle=":", alpha=0.6)
        ax_scatter.legend(loc="upper right", fontsize=8)

    # Event Callbacks
    def on_x_change(label: str) -> None:
        state["x_obj_idx"] = objective_names.index(label)
        update_scatter()
        fig.canvas.draw_idle()

    def on_y_change(label: str) -> None:
        state["y_obj_idx"] = objective_names.index(label)
        update_scatter()
        fig.canvas.draw_idle()

    def on_pick(event) -> None:
        if event.artist != state["scatter_artist"]:
            return
        if len(event.ind) > 0:
            state["selected_idx"] = event.ind[0]
            update_scatter()
            draw_layout(state["selected_idx"])
            fig.canvas.draw_idle()

    radio_x.on_clicked(on_x_change)
    radio_y.on_clicked(on_y_change)
    fig.canvas.mpl_connect("pick_event", on_pick)

    # Launch Window
    update_scatter()
    draw_layout(state["selected_idx"])
    plt.tight_layout()
    plt.show()