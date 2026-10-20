"""Unified Live Chiplet Floorplan Coverage Engine and Interactive GUI Dashboard.

Executes single sequence-pair layout computations on demand and provides real-time
GUI controls for whitespace slacks (including independent per-edge/per-chiplet random
slacks), global random seeding, Tkinter spinbox controls with crash-safe defaulting,
configurable frame delays, canvas accumulation rules, persistence memory management,
centered layout viewports with adaptive dynamic mm axis bounds and clean step growth,
dynamic top-canvas metadata titles in monospace font, system default GUI fonts,
3-column layer controls (Show / Persist / Fill Mask), structured CSV dataset export,
and binary B&W PNG/GIF exports.

Usage:
    python study.py [sys.json]
"""

import argparse
import random
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from utils.io_utils import load_file_json
from rules.connectivity import (
    assign_chiplet_ids,
    parse_chiplet_dimensions,
)
from rules.seqpair import decode_sequence_pair_batch


def sample_single_layout(
    num_chiplets: int,
    chiplet_dims: np.ndarray,
    max_slack: float,
    slack_mode: str = "Random [0, Max]",
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Computes a single sequence-pair layout sample for a given whitespace strategy.

    Args:
        num_chiplets: Total number of chiplets N in the system.
        chiplet_dims: Float array of shape (N, 2) with chiplet [width, height].
        max_slack: Maximum physical whitespace slack limit (mm).
        slack_mode: Sampling strategy ('Random [0, Max]', 'Independent Per-Edge [0, Max]',
                    'At Max Whitespace', 'Zero Slack (0.0)').
        seed: Optional integer seed for reproducible random sampling.

    Returns:
        dict containing:
            - "chiplet_bboxes": Array of shape (N, 4) with [X1, Y1, X2, Y2].
            - "chiplet_centers": Array of shape (N, 2) with centroids [CX, CY].
            - "whitespace_bboxes": Array of shape (N, 4) with [WX1, WY1, WX2, WY2].
            - "substrate_bbox": Array of shape (4,) with [SX1, SY1, SX2, SY2].
            - "slack_batch": Array of shape (1, 4, N) with direction slacks [N, E, S, W].
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if slack_mode == "At Max Whitespace":
        slack_batch = np.full((1, 4, num_chiplets), fill_value=max_slack, dtype=np.float32)
    elif slack_mode == "Zero Slack (0.0)":
        slack_batch = np.zeros((1, 4, num_chiplets), dtype=np.float32)
    elif slack_mode == "Independent Per-Edge [0, Max]":
        slack_batch = np.random.uniform(0.0, max_slack, size=(1, 4, num_chiplets)).astype(np.float32)
    else:  # "Random [0, Max]"
        s_val = float(np.random.uniform(0.0, max_slack))
        slack_batch = np.full((1, 4, num_chiplets), fill_value=s_val, dtype=np.float32)

    seq_x_batch = np.array([np.random.permutation(num_chiplets)], dtype=np.int16)
    seq_y_batch = np.array([np.random.permutation(num_chiplets)], dtype=np.int16)
    rot_batch = np.random.rand(1, num_chiplets) < 0.5

    bbox_batch, _, _ = decode_sequence_pair_batch(
        chiplet_dims=chiplet_dims,
        seq_x_batch=seq_x_batch,
        seq_y_batch=seq_y_batch,
        slack_batch=slack_batch,
        rot_batch=rot_batch,
    )

    chiplet_bboxes = bbox_batch[0].astype(np.float32)

    centers_x = (chiplet_bboxes[:, 0] + chiplet_bboxes[:, 2]) / 2.0
    centers_y = (chiplet_bboxes[:, 1] + chiplet_bboxes[:, 3]) / 2.0
    chiplet_centers = np.stack([centers_x, centers_y], axis=-1).astype(np.float32)

    s_N = slack_batch[0, 0, :]
    s_E = slack_batch[0, 1, :]
    s_S = slack_batch[0, 2, :]
    s_W = slack_batch[0, 3, :]

    ws_x1 = chiplet_bboxes[:, 0] - s_W
    ws_y1 = chiplet_bboxes[:, 1] - s_S
    ws_x2 = chiplet_bboxes[:, 2] + s_E
    ws_y2 = chiplet_bboxes[:, 3] + s_N
    whitespace_bboxes = np.stack([ws_x1, ws_y1, ws_x2, ws_y2], axis=-1).astype(np.float32)

    sub_x1 = float(np.min(ws_x1))
    sub_y1 = float(np.min(ws_y1))
    sub_x2 = float(np.max(ws_x2))
    sub_y2 = float(np.max(ws_y2))
    substrate_bbox = np.array([sub_x1, sub_y1, sub_x2, sub_y2], dtype=np.float32)

    return {
        "chiplet_bboxes": chiplet_bboxes,
        "chiplet_centers": chiplet_centers,
        "whitespace_bboxes": whitespace_bboxes,
        "substrate_bbox": substrate_bbox,
        "slack_batch": slack_batch,
    }


def compute_convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Computes the 2D convex hull polygon using Andrew's Monotone Chain algorithm."""
    sorted_pts = sorted(set(points))
    if len(sorted_pts) <= 1:
        return sorted_pts

    def cross_product(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


class LiveStudyDashboardGUI:
    """Tkinter GUI Dashboard for driving live floorplan coverage studies."""

    def __init__(self, root: tk.Tk, sys_json_path: str | None = None) -> None:
        self.root = root
        self.root.title("Live Chiplet Floorplan Coverage Dashboard Engine")
        self.root.geometry("1240x900")
        self.root.configure(bg="#F4F4F4")

        self.num_chiplets: int = 0
        self.chiplet_dims: np.ndarray | None = None
        self.chiplet_ids: dict[str, int] = {}
        self.loaded_sys_filename: str = "None"

        self.is_running: bool = False
        self.total_samples_generated: int = 0
        self.active_sample: dict[str, np.ndarray] | None = None

        self.accumulated_centers: list[tuple[float, float]] = []
        self.accumulated_chiplet_boxes: list[np.ndarray] = []
        self.accumulated_whitespace_boxes: list[np.ndarray] = []
        self.accumulated_substrate_boxes: list[np.ndarray] = []
        self.accumulated_polygons: list[list[tuple[float, float]]] = []

        # Control Variables (With Spinners)
        self.var_random_seed = tk.IntVar(value=42)
        self.var_max_slack = tk.DoubleVar(value=5.0)
        self.var_slack_mode = tk.StringVar(value="Random [0, Max]")
        self.var_frame_delay_ms = tk.IntVar(value=20)
        self.var_clear_mode = tk.StringVar(value="Clear Each Frame")
        self.var_clear_n_count = tk.IntVar(value=50)

        self.var_data_run_samples = tk.IntVar(value=100)
        self.var_gif_frames = tk.IntVar(value=50)

        # Layer Control Triples (Show, Persist, Fill Mask)
        self.var_show_centers = tk.BooleanVar(value=True)
        self.var_persist_centers = tk.BooleanVar(value=True)
        self.var_fill_centers = tk.BooleanVar(value=False)

        self.var_show_chiplet_boxes = tk.BooleanVar(value=True)
        self.var_persist_chiplet_boxes = tk.BooleanVar(value=False)
        self.var_fill_chiplet_boxes = tk.BooleanVar(value=False)

        self.var_show_whitespace_boxes = tk.BooleanVar(value=True)
        self.var_persist_whitespace_boxes = tk.BooleanVar(value=False)
        self.var_fill_whitespace_boxes = tk.BooleanVar(value=False)

        self.var_show_substrate = tk.BooleanVar(value=True)
        self.var_persist_substrate = tk.BooleanVar(value=False)
        self.var_fill_substrate = tk.BooleanVar(value=False)

        self.var_show_polygon = tk.BooleanVar(value=True)
        self.var_persist_polygon = tk.BooleanVar(value=False)
        self.var_fill_polygon = tk.BooleanVar(value=False)

        self._build_ui()

        if sys_json_path:
            self._load_system_json(sys_json_path)

    def _safe_get_float(self, var: tk.Variable, default: float = 0.0) -> float:
        """Safely extracts a float value from a Tkinter variable, defaulting to 0.0 on error."""
        try:
            val = var.get()
            return float(val) if val != "" else default
        except (tk.TclError, ValueError):
            return default

    def _safe_get_int(self, var: tk.Variable, default: int = 0) -> int:
        """Safely extracts an int value from a Tkinter variable, defaulting to 0 on error."""
        try:
            val = var.get()
            return int(val) if val != "" else default
        except (tk.TclError, ValueError):
            return default

    def _build_ui(self) -> None:
        """Constructs sidebar controls (using default GUI font) and main rendering view."""
        self.sidebar = tk.Frame(self.root, width=360, bg="#E8E8E8", padx=12, pady=12)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_frame = tk.Frame(self.root, bg="white")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. System Topology
        lbl_sys_section = tk.Label(
            self.sidebar, text="System Topology", font=("Helvetica", 11, "bold"), bg="#E8E8E8"
        )
        lbl_sys_section.pack(anchor=tk.W, pady=(0, 4))

        btn_load_sys = tk.Button(
            self.sidebar,
            text="Load System Topology (sys.json)",
            command=self._browse_system_json,
            bg="#333333",
            fg="white",
            font=("Helvetica", 9, "bold"),
        )
        btn_load_sys.pack(fill=tk.X, pady=(0, 4))

        self.lbl_sys_info = tk.Label(
            self.sidebar, text="No system topology loaded", font=("Helvetica", 9), bg="#E8E8E8", fg="#666666"
        )
        self.lbl_sys_info.pack(anchor=tk.W, pady=(0, 6))

        tk.Frame(self.sidebar, height=1, bg="#CCCCCC").pack(fill=tk.X, pady=4)

        # 2. Global Seed & Whitespace Controls (Spinners)
        lbl_slack_title = tk.Label(
            self.sidebar, text="Seed, Whitespace & Timing Controls", font=("Helvetica", 11, "bold"), bg="#E8E8E8"
        )
        lbl_slack_title.pack(anchor=tk.W, pady=(0, 4))

        frame_seed_val = tk.Frame(self.sidebar, bg="#E8E8E8")
        frame_seed_val.pack(fill=tk.X, pady=2)
        tk.Label(frame_seed_val, text="Global Random Seed:", bg="#E8E8E8", font=("Helvetica", 9)).pack(side=tk.LEFT)
        spn_seed = tk.Spinbox(
            frame_seed_val,
            textvariable=self.var_random_seed,
            from_=0,
            to=999999,
            increment=1,
            width=8,
            command=self.render,
        )
        spn_seed.pack(side=tk.RIGHT)

        frame_slack_val = tk.Frame(self.sidebar, bg="#E8E8E8")
        frame_slack_val.pack(fill=tk.X, pady=2)
        tk.Label(frame_slack_val, text="Max Whitespace (mm):", bg="#E8E8E8", font=("Helvetica", 9)).pack(side=tk.LEFT)
        spn_max_slack = tk.Spinbox(
            frame_slack_val,
            textvariable=self.var_max_slack,
            from_=0.0,
            to=1000.0,
            increment=0.5,
            width=8,
            command=self.render,
        )
        spn_max_slack.pack(side=tk.RIGHT)

        frame_delay_val = tk.Frame(self.sidebar, bg="#E8E8E8")
        frame_delay_val.pack(fill=tk.X, pady=2)
        tk.Label(frame_delay_val, text="Frame Delay (ms):", bg="#E8E8E8", font=("Helvetica", 9)).pack(side=tk.LEFT)
        spn_delay = tk.Spinbox(
            frame_delay_val,
            textvariable=self.var_frame_delay_ms,
            from_=0,
            to=5000,
            increment=5,
            width=8,
        )
        spn_delay.pack(side=tk.RIGHT)

        tk.Label(self.sidebar, text="Slack Sampling Strategy:", bg="#E8E8E8", font=("Helvetica", 9)).pack(anchor=tk.W, pady=(4, 2))
        opt_slack_mode = tk.OptionMenu(
            self.sidebar,
            self.var_slack_mode,
            "Random [0, Max]",
            "Independent Per-Edge [0, Max]",
            "At Max Whitespace",
            "Zero Slack (0.0)",
            command=lambda _: self.render(),
        )
        opt_slack_mode.pack(fill=tk.X, pady=(0, 6))

        tk.Frame(self.sidebar, height=1, bg="#CCCCCC").pack(fill=tk.X, pady=4)

        # 3. Canvas Accumulation Settings (Spinner)
        lbl_clear_title = tk.Label(
            self.sidebar, text="Canvas Accumulation Settings", font=("Helvetica", 11, "bold"), bg="#E8E8E8"
        )
        lbl_clear_title.pack(anchor=tk.W, pady=(0, 4))

        opt_clear_mode = tk.OptionMenu(
            self.sidebar,
            self.var_clear_mode,
            "Clear Each Frame",
            "Accumulate Indefinitely",
            "Clear Every N Samples",
        )
        opt_clear_mode.pack(fill=tk.X, pady=(0, 4))

        frame_n_count = tk.Frame(self.sidebar, bg="#E8E8E8")
        frame_n_count.pack(fill=tk.X, pady=2)
        tk.Label(frame_n_count, text="N Samples Threshold:", bg="#E8E8E8", font=("Helvetica", 9)).pack(side=tk.LEFT)
        spn_n_count = tk.Spinbox(
            frame_n_count,
            textvariable=self.var_clear_n_count,
            from_=0,
            to=100000,
            increment=10,
            width=8,
        )
        spn_n_count.pack(side=tk.RIGHT)

        tk.Frame(self.sidebar, height=1, bg="#CCCCCC").pack(fill=tk.X, pady=4)

        # 4. Checkbox Table: Display, Persistence, & Fill Mask
        lbl_toggles = tk.Label(
            self.sidebar, text="Display / Persist / Fill Matrix", font=("Helvetica", 11, "bold"), bg="#E8E8E8"
        )
        lbl_toggles.pack(anchor=tk.W, pady=(0, 4))

        table_frame = tk.Frame(self.sidebar, bg="#E8E8E8")
        table_frame.pack(fill=tk.X, pady=(2, 4))

        tk.Label(table_frame, text="Layer Name", font=("Helvetica", 9, "bold"), bg="#E8E8E8", anchor=tk.W).grid(row=0, column=0, sticky="w", padx=(0, 6))
        tk.Label(table_frame, text="Show", font=("Helvetica", 9, "bold"), bg="#E8E8E8").grid(row=0, column=1, padx=2)
        tk.Label(table_frame, text="Persist", font=("Helvetica", 9, "bold"), bg="#E8E8E8").grid(row=0, column=2, padx=2)
        tk.Label(table_frame, text="Fill", font=("Helvetica", 9, "bold"), bg="#E8E8E8").grid(row=0, column=3, padx=2)

        layer_rows = [
            ("Center 'X's", self.var_show_centers, self.var_persist_centers, self.var_fill_centers),
            ("Solid Boxes", self.var_show_chiplet_boxes, self.var_persist_chiplet_boxes, self.var_fill_chiplet_boxes),
            ("Whitespace", self.var_show_whitespace_boxes, self.var_persist_whitespace_boxes, self.var_fill_whitespace_boxes),
            ("Substrate Box", self.var_show_substrate, self.var_persist_substrate, self.var_fill_substrate),
            ("Boundary Poly", self.var_show_polygon, self.var_persist_polygon, self.var_fill_polygon),
        ]

        for idx, (label_text, var_show, var_persist, var_fill) in enumerate(layer_rows, start=1):
            tk.Label(table_frame, text=label_text, bg="#E8E8E8", font=("Helvetica", 9), anchor=tk.W).grid(row=idx, column=0, sticky="w", pady=1)
            tk.Checkbutton(table_frame, variable=var_show, command=self.render, bg="#E8E8E8").grid(row=idx, column=1, padx=2)
            tk.Checkbutton(table_frame, variable=var_persist, command=self._on_persistence_changed, bg="#E8E8E8").grid(row=idx, column=2, padx=2)
            tk.Checkbutton(table_frame, variable=var_fill, command=self.render, bg="#E8E8E8").grid(row=idx, column=3, padx=2)

        tk.Frame(self.sidebar, height=1, bg="#CCCCCC").pack(fill=tk.X, pady=4)

        # 5. Engine Controls
        lbl_exec = tk.Label(
            self.sidebar, text="Live Engine Controls", font=("Helvetica", 11, "bold"), bg="#E8E8E8"
        )
        lbl_exec.pack(anchor=tk.W, pady=(0, 4))

        self.btn_toggle_run = tk.Button(
            self.sidebar, text="▶ Start Live Study", command=self._toggle_execution, bg="#107C41", fg="white", font=("Helvetica", 9, "bold")
        )
        self.btn_toggle_run.pack(fill=tk.X, pady=2)

        btn_step = tk.Button(
            self.sidebar, text="Step Single Sample (+1)", command=self._step_once, bg="#0078D4", fg="white", font=("Helvetica", 9)
        )
        btn_step.pack(fill=tk.X, pady=2)

        btn_reset = tk.Button(
            self.sidebar, text="Clear Canvas & Reset Memory", command=self._reset_canvas, bg="#D13438", fg="white", font=("Helvetica", 9)
        )
        btn_reset.pack(fill=tk.X, pady=2)

        tk.Frame(self.sidebar, height=1, bg="#CCCCCC").pack(fill=tk.X, pady=4)

        # 6. Exports Panel (Single-frame PNG placed at top)
        lbl_export = tk.Label(
            self.sidebar, text="Media & Data Exports", font=("Helvetica", 11, "bold"), bg="#E8E8E8"
        )
        lbl_export.pack(anchor=tk.W, pady=(0, 4))

        btn_export_png = tk.Button(
            self.sidebar, text="🖼 Export B&W PNG Image (Current Frame)", command=self._export_bw_png, bg="#333333", fg="white", font=("Helvetica", 9, "bold")
        )
        btn_export_png.pack(fill=tk.X, pady=2)

        tk.Frame(self.sidebar, height=1, bg="#CCCCCC").pack(fill=tk.X, pady=4)

        frame_data_run = tk.Frame(self.sidebar, bg="#E8E8E8")
        frame_data_run.pack(fill=tk.X, pady=2)
        tk.Label(frame_data_run, text="Run Sample Count:", bg="#E8E8E8", font=("Helvetica", 9)).pack(side=tk.LEFT)
        spn_data_run = tk.Spinbox(
            frame_data_run,
            textvariable=self.var_data_run_samples,
            from_=0,
            to=100000,
            increment=50,
            width=8,
        )
        spn_data_run.pack(side=tk.RIGHT)

        btn_run_export_data = tk.Button(
            self.sidebar, text="📊 Execute Data Run & Export (.csv)", command=self._execute_data_run, bg="#4A2545", fg="white", font=("Helvetica", 9, "bold")
        )
        btn_run_export_data.pack(fill=tk.X, pady=2)

        tk.Frame(self.sidebar, height=1, bg="#CCCCCC").pack(fill=tk.X, pady=4)

        frame_gif_cfg = tk.Frame(self.sidebar, bg="#E8E8E8")
        frame_gif_cfg.pack(fill=tk.X, pady=2)
        tk.Label(frame_gif_cfg, text="GIF Frame Count:", bg="#E8E8E8", font=("Helvetica", 9)).pack(side=tk.LEFT)
        spn_gif_frames = tk.Spinbox(
            frame_gif_cfg,
            textvariable=self.var_gif_frames,
            from_=0,
            to=10000,
            increment=5,
            width=8,
        )
        spn_gif_frames.pack(side=tk.RIGHT)

        btn_export_gif = tk.Button(
            self.sidebar, text="🎬 Record & Export B&W Animated GIF", command=self._export_bw_gif, bg="#005A9E", fg="white", font=("Helvetica", 9, "bold")
        )
        btn_export_gif.pack(fill=tk.X, pady=2)

        self.lbl_stats = tk.Label(
            self.sidebar, text="Total Samples: 0", font=("Helvetica", 10, "bold"), bg="#E8E8E8", fg="#333333"
        )
        self.lbl_stats.pack(anchor=tk.W, pady=(6, 0))

        self.canvas = tk.Canvas(self.canvas_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())

        self.tk_img_ref = None

    def _on_persistence_changed(self) -> None:
        """Memory Safety: Purges long-term layer buffers when persistence is unchecked."""
        if not self.var_persist_centers.get():
            self.accumulated_centers.clear()
        if not self.var_persist_chiplet_boxes.get():
            self.accumulated_chiplet_boxes.clear()
        if not self.var_persist_whitespace_boxes.get():
            self.accumulated_whitespace_boxes.clear()
        if not self.var_persist_substrate.get():
            self.accumulated_substrate_boxes.clear()
        if not self.var_persist_polygon.get():
            self.accumulated_polygons.clear()
        self.render()

    def _browse_system_json(self) -> None:
        """Opens file dialog to select sys.json file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            self._load_system_json(filepath)

    def _load_system_json(self, filepath: str) -> None:
        """Loads system topology and initializes engine parameters."""
        try:
            cfg_sys = load_file_json(filepath)
            self.chiplet_ids = assign_chiplet_ids(cfg_sys)
            self.num_chiplets = len(self.chiplet_ids)
            self.chiplet_dims = parse_chiplet_dimensions(cfg_sys, self.chiplet_ids)

            self.loaded_sys_filename = filepath.replace("\\", "/").split("/")[-1]
            self.lbl_sys_info.config(
                text=f"Loaded: {self.num_chiplets} Chiplets\nFile: {self.loaded_sys_filename}"
            )
            self._reset_canvas()

        except Exception as e:
            messagebox.showerror("System Load Error", f"Failed to parse sys.json:\n{str(e)}")

    def _toggle_execution(self) -> None:
        """Toggles between running and paused states."""
        if self.chiplet_dims is None:
            messagebox.showwarning("Missing Topology", "Please load a sys.json topology file first.")
            return

        self.is_running = not self.is_running
        if self.is_running:
            self.btn_toggle_run.config(text="⏸ Pause Study", bg="#C8C8C8", fg="black")
            self._run_loop()
        else:
            self.btn_toggle_run.config(text="▶ Resume Live Study", bg="#107C41", fg="white")

    def _step_once(self) -> None:
        """Generates and renders a single sample on demand."""
        if self.chiplet_dims is None:
            messagebox.showwarning("Missing Topology", "Please load a sys.json topology file first.")
            return
        self._generate_next_sample()
        self.render()

    def _reset_canvas(self) -> None:
        """Clears all long-term accumulated buffers and resets statistics."""
        self.accumulated_centers.clear()
        self.accumulated_chiplet_boxes.clear()
        self.accumulated_whitespace_boxes.clear()
        self.accumulated_substrate_boxes.clear()
        self.accumulated_polygons.clear()

        self.active_sample = None
        self.total_samples_generated = 0
        self.lbl_stats.config(text="Total Samples: 0")
        self.canvas.delete("all")
        self.render()

    def _generate_next_sample(self) -> None:
        """Requests single layout sample from engine using current global random seed."""
        max_s = self._safe_get_float(self.var_max_slack, 0.0)
        mode = self.var_slack_mode.get()
        base_seed = self._safe_get_int(self.var_random_seed, 0)

        # Advance seed deterministically per sample
        frame_seed = (base_seed + self.total_samples_generated) % (2**31)

        sample_dict = sample_single_layout(
            self.num_chiplets, self.chiplet_dims, max_s, slack_mode=mode, seed=frame_seed
        )
        self.total_samples_generated += 1
        self.active_sample = sample_dict

        centers = sample_dict["chiplet_centers"]
        chiplet_bboxes = sample_dict["chiplet_bboxes"]
        whitespace_bboxes = sample_dict["whitespace_bboxes"]
        substrate_bbox = sample_dict["substrate_bbox"]

        active_centers_pts = [(float(centers[c, 0]), float(centers[c, 1])) for c in range(self.num_chiplets)]
        active_poly_pts = compute_convex_hull(active_centers_pts) if len(active_centers_pts) >= 3 else []

        clear_mode = self.var_clear_mode.get()
        should_clear = False
        if clear_mode == "Clear Each Frame":
            should_clear = True
        elif clear_mode == "Clear Every N Samples":
            n_thresh = self._safe_get_int(self.var_clear_n_count, 0)
            if n_thresh > 0 and self.total_samples_generated % n_thresh == 0:
                should_clear = True

        if should_clear:
            self.accumulated_centers.clear()
            self.accumulated_chiplet_boxes.clear()
            self.accumulated_whitespace_boxes.clear()
            self.accumulated_substrate_boxes.clear()
            self.accumulated_polygons.clear()

        if self.var_persist_centers.get():
            self.accumulated_centers.extend(active_centers_pts)

        if self.var_persist_chiplet_boxes.get():
            self.accumulated_chiplet_boxes.append(chiplet_bboxes)

        if self.var_persist_whitespace_boxes.get():
            self.accumulated_whitespace_boxes.append(whitespace_bboxes)

        if self.var_persist_substrate.get():
            self.accumulated_substrate_boxes.append(substrate_bbox)

        if self.var_persist_polygon.get() and active_poly_pts:
            self.accumulated_polygons.append(active_poly_pts)

        self.lbl_stats.config(text=f"Total Samples: {self.total_samples_generated}")

    def _execute_data_run(self) -> None:
        """Executes a structured data run for N samples, exports as CSV, and clears memory."""
        if self.chiplet_dims is None:
            messagebox.showwarning("Missing Topology", "Please load a sys.json topology file first.")
            return

        target_n = self._safe_get_int(self.var_data_run_samples, 0)
        if target_n <= 0:
            messagebox.showwarning("Invalid Sample Count", "Sample count must be greater than 0.")
            return

        out_filepath = filedialog.asksaveasfilename(
            title="Save Structured CSV Coverage Dataset",
            defaultextension=".csv",
            filetypes=[("CSV Matrix", "*.csv"), ("All Files", "*.*")],
        )
        if not out_filepath:
            return

        if self.is_running:
            self._toggle_execution()

        max_s = self._safe_get_float(self.var_max_slack, 0.0)
        mode = self.var_slack_mode.get()
        base_seed = self._safe_get_int(self.var_random_seed, 0)

        rows = []
        for run_idx in range(target_n):
            frame_seed = (base_seed + run_idx) % (2**31)
            sample = sample_single_layout(
                self.num_chiplets, self.chiplet_dims, max_s, slack_mode=mode, seed=frame_seed
            )
            bboxes_flat = sample["chiplet_bboxes"].reshape(-1)
            slacks_flat = sample["slack_batch"][0].T.reshape(-1)

            row = np.concatenate([[float(run_idx)], bboxes_flat, slacks_flat])
            rows.append(row)

            self.active_sample = sample
            self.total_samples_generated += 1

        dataset_matrix = np.vstack(rows)

        headers = ["run_id"]
        for c in range(self.num_chiplets):
            headers.extend([f"c{c}_x1", f"c{c}_y1", f"c{c}_x2", f"c{c}_y2"])
        for c in range(self.num_chiplets):
            headers.extend([f"c{c}_slack_N", f"c{c}_slack_E", f"c{c}_slack_S", f"c{c}_slack_W"])

        header_str = ",".join(headers)
        np.savetxt(out_filepath, dataset_matrix, delimiter=",", header=header_str, comments="")

        self._reset_canvas()

        messagebox.showinfo(
            "Data Run Complete",
            f"Successfully collected and exported {target_n} samples to CSV:\n{out_filepath}\n\nCanvas and memory cleared.",
        )

    def _get_monospace_font(self, size: int = 12) -> ImageFont.ImageFont:
        """Loads a monospace font for canvas rendering with fallbacks."""
        for font_name in ["cour.ttf", "Courier", "Consolas", "LiberationMono-Regular.ttf"]:
            try:
                return ImageFont.truetype(font_name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _generate_pil_frame(self) -> Image.Image:
        """Renders visual state with centered view, dynamic mm rulers, metadata header, and shape fills in Monospace font."""
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        img = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
        if self.chiplet_dims is None or self.active_sample is None:
            return img

        draw = ImageDraw.Draw(img)
        font_mono = self._get_monospace_font(12)

        # 1. Determine Actual Dynamic Coordinate Extent from Visible & Accumulated Elements
        max_x = 10.0
        max_y = 10.0

        if self.active_sample is not None:
            sub = self.active_sample["substrate_bbox"]
            max_x = max(max_x, float(sub[2]))
            max_y = max(max_y, float(sub[3]))

        if self.var_persist_substrate.get() and self.accumulated_substrate_boxes:
            for sub_box in self.accumulated_substrate_boxes:
                max_x = max(max_x, float(sub_box[2]))
                max_y = max(max_y, float(sub_box[3]))

        if self.var_persist_whitespace_boxes.get() and self.accumulated_whitespace_boxes:
            for box_set in self.accumulated_whitespace_boxes:
                max_x = max(max_x, float(np.max(box_set[:, 2])))
                max_y = max(max_y, float(np.max(box_set[:, 3])))

        if self.var_persist_chiplet_boxes.get() and self.accumulated_chiplet_boxes:
            for box_set in self.accumulated_chiplet_boxes:
                max_x = max(max_x, float(np.max(box_set[:, 2])))
                max_y = max(max_y, float(np.max(box_set[:, 3])))

        if self.var_persist_centers.get() and self.accumulated_centers:
            for cx, cy in self.accumulated_centers:
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)

        raw_max = max(max_x, max_y)

        # Clean step increment logic for dynamic growth
        if raw_max <= 20:
            step_unit = 5
        elif raw_max <= 50:
            step_unit = 10
        elif raw_max <= 100:
            step_unit = 20
        elif raw_max <= 200:
            step_unit = 25
        elif raw_max <= 500:
            step_unit = 50
        else:
            step_unit = 100

        padded_max = raw_max * 1.05
        current_extent = float(np.ceil(padded_max / step_unit) * step_unit)
        if current_extent < step_unit:
            current_extent = float(step_unit)

        # 2. Define Margins for Axis Rulers and Monospace Title Header
        margin_left, margin_right = 60, 40
        margin_bottom, margin_top = 50, 60

        draw_w = max(10, canvas_w - margin_left - margin_right)
        draw_h = max(10, canvas_h - margin_bottom - margin_top)

        max_slack = self._safe_get_float(self.var_max_slack, 0.0)
        base_seed = self._safe_get_int(self.var_random_seed, 0)

        # 3. Render Canvas Dynamic Metadata Title in Monospace Font
        slack_strategy = self.var_slack_mode.get()
        title_text = (
            f"Topology: {self.loaded_sys_filename}  |  "
            f"Seed: {base_seed}  |  "
            f"Max Whitespace: {max_slack:.2f} mm  |  "
            f"Strategy: {slack_strategy}"
        )
        draw.text((margin_left, 15), title_text, fill=(0, 0, 0, 255), font=font_mono)

        # 4. Scale & Centering Offsets (Square Aspect Ratio)
        scale = min(draw_w / current_extent, draw_h / current_extent)
        offset_x = margin_left + (draw_w - current_extent * scale) / 2.0
        offset_y = margin_bottom + (draw_h - current_extent * scale) / 2.0

        def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
            px_x = int(offset_x + x_mm * scale)
            px_y = int(canvas_h - (offset_y + y_mm * scale))
            return px_x, px_y

        # 5. Render Axis Line & Ticks (mm Rulers with Monospace Labels)
        target_pixel_step = 40.0
        possible_steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500]
        step_mm = possible_steps[-1]
        for st in possible_steps:
            if st * scale >= target_pixel_step:
                step_mm = st
                break

        origin_px_x, origin_px_y = to_px(0, 0)
        end_x_px, _ = to_px(current_extent, 0)
        _, end_y_px = to_px(0, current_extent)

        # Axis Baseline Lines
        draw.line([origin_px_x, origin_px_y, end_x_px, origin_px_y], fill=(0, 0, 0, 255), width=1)
        draw.line([origin_px_x, origin_px_y, origin_px_x, end_y_px], fill=(0, 0, 0, 255), width=1)

        # X-Axis Ticks & Monospace Labels
        curr_x = 0.0
        while curr_x <= current_extent:
            tx, ty = to_px(curr_x, 0)
            draw.line([tx, ty, tx, ty + 5], fill=(0, 0, 0, 255), width=1)
            draw.text((tx - 8, ty + 7), f"{int(curr_x)}", fill=(0, 0, 0, 255), font=font_mono)
            curr_x += step_mm

        # Y-Axis Ticks & Monospace Labels
        curr_y = 0.0
        while curr_y <= current_extent:
            tx, ty = to_px(0, curr_y)
            draw.line([tx, ty, tx - 5, ty], fill=(0, 0, 0, 255), width=1)
            draw.text((tx - 35, ty - 6), f"{int(curr_y)}", fill=(0, 0, 0, 255), font=font_mono)
            curr_y += step_mm

        # Axis Titles
        draw.text((end_x_px + 8, origin_px_y - 6), "X (mm)", fill=(0, 0, 0, 255), font=font_mono)
        draw.text((origin_px_x - 15, end_y_px - 20), "Y (mm)", fill=(0, 0, 0, 255), font=font_mono)

        # 6. Render Layer: Substrate Box
        if self.var_show_substrate.get():
            fill_sub = (0, 0, 0, 255) if self.var_fill_substrate.get() else None
            sub_boxes = self.accumulated_substrate_boxes if self.var_persist_substrate.get() else [self.active_sample["substrate_bbox"]]
            for sub_box in sub_boxes:
                sx1, sy1, sx2, sy2 = sub_box
                px1, py2 = to_px(sx1, sy2)
                px2, py1 = to_px(sx2, sy1)
                draw.rectangle([px1, py2, px2, py1], outline=(0, 0, 0, 255), fill=fill_sub, width=2)

        # 7. Render Layer: Dashed Whitespace Boxes
        if self.var_show_whitespace_boxes.get():
            fill_ws = (0, 0, 0, 255) if self.var_fill_whitespace_boxes.get() else None
            boxes_to_draw = self.accumulated_whitespace_boxes if self.var_persist_whitespace_boxes.get() else [self.active_sample["whitespace_bboxes"]]
            for box_set in boxes_to_draw:
                for c in range(self.num_chiplets):
                    wx1, wy1, wx2, wy2 = box_set[c]
                    px1, py2 = to_px(wx1, wy2)
                    px2, py1 = to_px(wx2, wy1)
                    draw.rectangle([px1, py2, px2, py1], outline=(0, 0, 0, 255), fill=fill_ws, width=1)

        # 8. Render Layer: Solid Chiplet Bounding Boxes
        if self.var_show_chiplet_boxes.get():
            fill_chiplet = (0, 0, 0, 255) if self.var_fill_chiplet_boxes.get() else None
            boxes_to_draw = self.accumulated_chiplet_boxes if self.var_persist_chiplet_boxes.get() else [self.active_sample["chiplet_bboxes"]]
            for box_set in boxes_to_draw:
                for c in range(self.num_chiplets):
                    x1, y1, x2, y2 = box_set[c]
                    px1, py2 = to_px(x1, y2)
                    px2, py1 = to_px(x2, y1)
                    draw.rectangle([px1, py2, px2, py1], outline=(0, 0, 0, 255), fill=fill_chiplet, width=1)

        # 9. Render Layer: Centroid 'X' Markers / Filled Center Squares
        if self.var_show_centers.get():
            r = 2
            centers_to_draw = self.accumulated_centers if self.var_persist_centers.get() else [(float(c[0]), float(c[1])) for c in self.active_sample["chiplet_centers"]]
            is_filled_centers = self.var_fill_centers.get()
            for cx, cy in centers_to_draw:
                px_cx, px_cy = to_px(cx, cy)
                if is_filled_centers:
                    draw.rectangle([px_cx - r, px_cy - r, px_cx + r, px_cy + r], fill=(0, 0, 0, 255), outline=(0, 0, 0, 255))
                else:
                    draw.line([px_cx - r, px_cy - r, px_cx + r, px_cy + r], fill=(0, 0, 0, 255), width=1)
                    draw.line([px_cx - r, px_cy + r, px_cx + r, px_cy - r], fill=(0, 0, 0, 255), width=1)

        # 10. Render Layer: Boundary Polygon
        if self.var_show_polygon.get():
            fill_poly = (0, 0, 0, 255) if self.var_fill_polygon.get() else None
            if self.var_persist_polygon.get():
                for poly_pts in self.accumulated_polygons:
                    if len(poly_pts) >= 3:
                        poly_px = [to_px(vx, vy) for vx, vy in poly_pts]
                        draw.polygon(poly_px, outline=(0, 0, 0, 255), fill=fill_poly, width=2)
            else:
                active_centers_pts = [(float(c[0]), float(c[1])) for c in self.active_sample["chiplet_centers"]]
                active_poly_pts = compute_convex_hull(active_centers_pts)
                if len(active_poly_pts) >= 3:
                    poly_px = [to_px(vx, vy) for vx, vy in active_poly_pts]
                    draw.polygon(poly_px, outline=(0, 0, 0, 255), fill=fill_poly, width=2)

        return img

    def _get_binary_pil_image(self) -> Image.Image:
        """Converts current frame into a pure 1-bit monochrome (binary black & white) image."""
        raw_rgba = self._generate_pil_frame()
        grayscale = raw_rgba.convert("L")
        return grayscale.point(lambda p: 255 if p > 128 else 0, mode="1")

    def _export_bw_png(self) -> None:
        """Exports the active canvas frame as a pure binary black and white PNG file."""
        if self.active_sample is None:
            messagebox.showwarning("No Data", "Generate a sample layout before exporting PNG.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Binary B&W PNG Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
        )
        if filepath:
            bw_img = self._get_binary_pil_image()
            bw_img.save(filepath)
            messagebox.showinfo("Export Successful", f"Saved B&W PNG image to:\n{filepath}")

    def _export_bw_gif(self) -> None:
        """Records M frames of the live study and exports a motion binary B&W animated GIF."""
        if self.chiplet_dims is None:
            messagebox.showwarning("Missing Topology", "Please load a sys.json topology file first.")
            return

        frame_count = self._safe_get_int(self.var_gif_frames, 0)
        if frame_count <= 0:
            messagebox.showwarning("Invalid Frame Count", "GIF frame count must be greater than 0.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Binary B&W Animated GIF",
            defaultextension=".gif",
            filetypes=[("GIF Motion Image", "*.gif")],
        )
        if not filepath:
            return

        if self.is_running:
            self._toggle_execution()

        frames: list[Image.Image] = []
        delay_ms = max(10, self._safe_get_int(self.var_frame_delay_ms, 20))

        for _ in range(frame_count):
            self._generate_next_sample()
            self.render()
            frames.append(self._get_binary_pil_image())

        if frames:
            frames[0].save(
                filepath,
                save_all=True,
                append_images=frames[1:],
                duration=delay_ms,
                loop=0,
            )
            messagebox.showinfo("GIF Export Complete", f"Saved {frame_count}-frame motion B&W GIF to:\n{filepath}")

    def _run_loop(self) -> None:
        """Drives real-time execution loop with user-defined frame delay."""
        if self.is_running:
            self._generate_next_sample()
            self.render()
            delay = max(1, self._safe_get_int(self.var_frame_delay_ms, 20))
            self.root.after(delay, self._run_loop)

    def render(self) -> None:
        """Renders live state onto canvas viewport using Pillow."""
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        if self.chiplet_dims is None or self.active_sample is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                canvas_w // 2, canvas_h // 2,
                text="Load sys.json and start engine to view live coverage",
                font=("Courier", 12), fill="#888888"
            )
            return

        img = self._generate_pil_frame()
        self.tk_img_ref = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img_ref)


def main() -> None:
    """Script entry point."""
    parser = argparse.ArgumentParser(description="Live Chiplet Floorplan Coverage Study Dashboard Engine.")
    parser.add_argument(
        "sys_json", type=str, nargs="?", default=None, help="Optional path to sys.json topology file"
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = LiveStudyDashboardGUI(root, sys_json_path=args.sys_json)
    root.mainloop()


if __name__ == "__main__":
    main()