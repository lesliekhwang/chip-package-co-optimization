"""
analyze.py
---------------------------------
Post-processing analysis tool for reading optimization logs (.parquet) 
and rendering convergence, Pareto, placement visualization, and worst-case plots.

Usage:
    python analyze.py <scenario.toml> <output_path> <history_fname> <model_path> <grid_size> <dpi> <export_csv>
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import json

import plot
import loader
import warpage
from warpage import Cube


def extract_chromosomes_from_parquet_rows(df: pd.DataFrame, base_dir: Path = None) -> list[dict]:
    """Reconstructs PyMoo chromosome dicts from Parquet DataFrame records,
    preserving generation, solution index metadata, and warpage map paths.
    """
    chromosomes = []
    for idx, row in df.iterrows():
        seq_x = list(row["sequence_x"])
        seq_y = list(row["sequence_y"])
        ws_dict = row["chiplet_whitespace"]

        # Keep WS as dictionaries mapped by die name for NESW directions
        ws_n = {die: float(vals.get("ws_n", 0.0)) for die, vals in ws_dict.items()}
        ws_e = {die: float(vals.get("ws_e", 0.0)) for die, vals in ws_dict.items()}
        ws_s = {die: float(vals.get("ws_s", 0.0)) for die, vals in ws_dict.items()}
        ws_w = {die: float(vals.get("ws_w", 0.0)) for die, vals in ws_dict.items()}

        # Read solution_index if present in Parquet row, else default to DataFrame index
        sol_idx = row["solution_index"] if "solution_index" in row else idx

        # Extract and resolve the warpage tensor path
        warpage_path = row.get("warpage_tensor_path", None)
        if base_dir and warpage_path and pd.notna(warpage_path):
            warpage_path = str(Path(base_dir) / warpage_path)
        else:
            warpage_path = None

        chromosomes.append({
            "X": seq_x,
            "Y": seq_y,
            "WS_N": ws_n,
            "WS_E": ws_e,
            "WS_S": ws_s,
            "WS_W": ws_w,
            "generation": row.get("generation", None),
            "solution_index": sol_idx,
            "sequence_x": seq_x,
            "sequence_y": seq_y,
            "chiplet_whitespace": ws_dict,
            "warpage_tensor_path": warpage_path,  # <-- Path saved directly to chromosome
        })
    return chromosomes


def export_flattened_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Flattens complex nested structures (lists, dicts) into atomic columns and saves to CSV."""
    print(f"[*] Flattening data and exporting to CSV: {csv_path}")
    flat_records = []
    
    for _, row in df.iterrows():
        flat_row = {}
        for col_name in df.columns:
            val = row[col_name]
            
            # 1. Flatten sequence arrays (e.g., sequence_x_0, sequence_x_1...)
            if col_name in ("sequence_x", "sequence_y") or isinstance(val, (list, np.ndarray)):
                for i, item in enumerate(val):
                    flat_row[f"{col_name}_{i}"] = item
                    
            # 2. Flatten chiplet whitespace dictionary for directional NESW margins
            elif col_name == "chiplet_whitespace" or isinstance(val, dict):
                val_dict = dict(val) 
                for die_name, ws_vals in val_dict.items():
                    ws_vals_dict = dict(ws_vals)
                    flat_row[f"ws_n_{die_name}"] = float(ws_vals_dict.get("ws_n", 0.0))
                    flat_row[f"ws_e_{die_name}"] = float(ws_vals_dict.get("ws_e", 0.0))
                    flat_row[f"ws_s_{die_name}"] = float(ws_vals_dict.get("ws_s", 0.0))
                    flat_row[f"ws_w_{die_name}"] = float(ws_vals_dict.get("ws_w", 0.0))
                    
            # 3. Keep scalar values as is (hpwl, area, warpage, generation)
            else:
                flat_row[col_name] = val
                
        flat_records.append(flat_row)
        
    flat_df = pd.DataFrame(flat_records)
    flat_df.to_csv(csv_path, index=False)
    print(f"[*] Successfully saved {len(flat_df)} flattened rows to CSV.")


# ==============================================================================
# MAIN ANALYSIS ROUTINE
# ==============================================================================

def main() -> None:
    # 1. Parse Command Line Arguments
    if len(sys.argv) != 8:
        print("Error: Missing or incorrect number of arguments!", file=sys.stderr)
        print("Usage: python analyze.py <scenario.toml> <output_path> <history.parquet> <model_path> <grid_size> <dpi> <export_csv>", file=sys.stderr)
        sys.exit(1)

    # Reconstruct the configuration dictionary dynamically from sys.argv
    cfg = {
        "SCENARIO_FILE": sys.argv[1],
        "OUTPUT_PATH": sys.argv[2],
        "HISTORY_PARQUET": sys.argv[3],
        "MODEL_PATH": sys.argv[4],
        "GRID_SIZE": int(sys.argv[5]),
        "DPI": int(sys.argv[6]),
        "EXPORT_CSV": sys.argv[7].lower() in ["true", "1", "yes", "csv"]
    }

    # 2. Load Inputs
    metadata, base_chiplets = loader.load_scenario(cfg["SCENARIO_FILE"])
    plot.plt.rcParams['savefig.dpi'] = cfg["DPI"]

    # Initialize the ML surrogate model for warpage heatmaps/cost checks
    warpage.initialize_model(cfg)

    plot.print_metadata(metadata, base_chiplets)

    log_path = Path(cfg["OUTPUT_PATH"])
    log_path.mkdir(parents=True, exist_ok=True)
    
    history_fname = cfg["HISTORY_PARQUET"]
    history_parquet = (log_path / history_fname).with_suffix(".parquet")

    if not history_parquet.is_file():
        print(f"Error: Parquet history file not found at '{history_parquet}'.", file=sys.stderr)
        print("Please run optimization first.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Reading optimization history from: {history_parquet}")
    df = pd.read_parquet(history_parquet, engine="pyarrow")

    # Handle CSV Export if flag is true
    if cfg["EXPORT_CSV"]:
        csv_path = history_parquet.with_suffix(".csv")
        export_flattened_csv(df, csv_path)

    # 3. Extract Final Generation and Pareto Front
    max_gen = df["generation"].max()
    final_gen_df = df[df["generation"] == max_gen].copy()

    # Calculate Pareto front on full dataset without dropping duplicates
    objs = df[["hpwl", "area", "warpage"]].values
    is_pareto = np.ones(objs.shape[0], dtype=bool)
    for i, c in enumerate(objs):
        if is_pareto[i]:
            is_pareto[is_pareto] = ~np.all(objs[is_pareto] >= c, axis=1) | np.all(objs[is_pareto] == c, axis=1)
            is_pareto[i] = True

    pareto_df = df[is_pareto].copy()
    pareto_F = pareto_df[["hpwl", "area", "warpage"]].values

    # Extract Chromosomes for Metadata/Plotting (Passing base_dir=log_path)
    pareto_X = extract_chromosomes_from_parquet_rows(pareto_df, base_dir=log_path)
    final_X = extract_chromosomes_from_parquet_rows(final_gen_df, base_dir=log_path)
    final_F = final_gen_df[["hpwl", "area", "warpage"]].values

    # Print Summary Table with Survival & Final Generation Annotations
    plot.print_pareto_table(pareto_F, pop_data=pareto_X)

    # 4. Generate Convergence, Distribution, Squareness & Redundancy Plots
    print("\nGenerating and saving Pareto front objective plots, squareness, & distribution histograms to disk...")
    plot.plot_pareto_from_parquet(parquet_path=history_parquet, x_col="hpwl", y_col="area")
    plot.plot_pareto_from_parquet(parquet_path=history_parquet, x_col="area", y_col="warpage")
    plot.plot_pareto_from_parquet(parquet_path=history_parquet, x_col="hpwl", y_col="warpage")
    plot.plot_final_generation_histograms(history_parquet)
    plot.plot_squareness_analysis(history_parquet)

    # 5. Save Placement Plots
    # Save key extreme layouts (Ideal/Balanced)
    plot.save_key_placement_plots(pareto_X, pareto_F, base_chiplets, parquet_path=history_parquet)

    # Save visualizations for every individual in the final generation
    plot.save_batch_placement_plots(
        final_X, final_F, base_chiplets, log_path / "final_gen_vis", "final_gen"
    )

    # 6. Extract and Save Worst-Case Samples Across All Generations
    worst_vis_dir = log_path / "worst_case_vis"
    worst_vis_dir.mkdir(parents=True, exist_ok=True)

    # Identify row indices for the worst (maximum) values of each objective metric across history
    worst_indices = list(dict.fromkeys([
        df["hpwl"].idxmax(),
        df["area"].idxmax(),
        df["warpage"].idxmax()
    ]))

    worst_df = df.loc[worst_indices].copy()
    worst_X = extract_chromosomes_from_parquet_rows(worst_df, base_dir=log_path)
    worst_F = worst_df[["hpwl", "area", "warpage"]].values

    print(f"\n[*] Extracted {len(worst_X)} unique worst-case samples across all generations. Saving plots to: {worst_vis_dir}")
    plot.save_batch_placement_plots(
        worst_X, worst_F, base_chiplets, worst_vis_dir, "worst_case"
    )

    # 7. Extract and Save Every Sample from the First Generation
    min_gen = df["generation"].min()
    first_gen_df = df[df["generation"] == min_gen].copy()
    first_X = extract_chromosomes_from_parquet_rows(first_gen_df, base_dir=log_path)
    first_F = first_gen_df[["hpwl", "area", "warpage"]].values

    first_vis_dir = log_path / "first_gen_vis"
    first_vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Extracted {len(first_X)} samples from the first generation (Gen {min_gen}). Saving plots to: {first_vis_dir}")
    plot.save_batch_placement_plots(
        first_X, first_F, base_chiplets, first_vis_dir, "first_gen"
    )

    print("\nAnalysis and visual plotting successfully completed.")

    # Save configuration to the output directory
    config_file_path = log_path / "analyze_config.json"
    with open(config_file_path, "w") as file:
        json.dump(cfg, file, indent=4)
    print(f"Configuration saved to: {config_file_path}")


if __name__ == "__main__":
    main()