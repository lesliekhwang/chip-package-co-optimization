import sys
import os

# Append the src directory to the python path so absolute imports inside src/ work
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import argparse
import pandas as pd
import numpy as np
import loader
import warpage
from cost import hpwl_batch, area_batch, warpage_batch

def compute_placement_costs(scenario_path: str, csv_path: str, model_path: str, interposer_x: float, interposer_y: float, output_csv: str):
    # Load scenario to extract adjacencies and chiplet name mappings
    id_to_name, _, _, adj_list, _ = loader.load_scenario(scenario_path)
    name_to_id = {v: k for k, v in id_to_name.items()}
    N = len(id_to_name)

    # Load placement CSV
    df = pd.read_csv(csv_path)
    
    # Standardize column names to match what the script expects
    column_mapping = {
        'Chiplet_Name': 'chiplet_name',
        'X1_mm': 'min_x',
        'Y1_mm': 'min_y',
        'X2_mm': 'max_x',
        'Y2_mm': 'max_y'
    }
    df = df.rename(columns=column_mapping)
    
    # If there's no placement_id (single placement CSV), add a default one
    if 'placement_id' not in df.columns:
        df['placement_id'] = 0

    # Filter out any chiplets not in the JSON scenario
    df = df[df['chiplet_name'].isin(name_to_id.keys())] 
    
    placement_ids = df['placement_id'].unique()
    B = len(placement_ids)

    # Initialize evaluation arrays
    x_coords = np.zeros((B, N))
    y_coords = np.zeros((B, N))
    base_widths = np.zeros((B, N))
    base_heights = np.zeros((B, N))
    slack_matrix = np.zeros((B, 4, N)) # Pre-calculated bounding boxes have 0 slack applied here
    
    # Apply user-defined fixed interposer area constraints
    total_w = np.full(B, float(interposer_x))
    total_h = np.full(B, float(interposer_y))

    # Populate batched coordinate and dimension tensors
    for i, pid in enumerate(placement_ids):
        p_df = df[df['placement_id'] == pid]
        for _, row in p_df.iterrows():
            cid = name_to_id[row['chiplet_name']]
            x_coords[i, cid] = row['min_x']
            y_coords[i, cid] = row['min_y']
            base_widths[i, cid] = row['max_x'] - row['min_x']
            base_heights[i, cid] = row['max_y'] - row['min_y']

    # Evaluate physical placement costs
    hpwl_vals = hpwl_batch(x_coords, y_coords, base_widths, base_heights, adj_list, slack_matrix)
    area_vals = area_batch(total_w, total_h)

    # Initialize surrogate warpage model and evaluate
    ga_config_mock = {
        "cost": {
            "model_path": model_path,
            "warp_grid": [128, 128] # Grid matching the standard trained surrogate dimensions
        }
    }
    warpage.initialize_model(ga_config_mock)
    _, rms_vals = warpage_batch(x_coords, y_coords, total_w, total_h, base_widths, base_heights, slack_matrix)

    # Structure final dataframe per the requested column ordering
    results_df = pd.DataFrame({
        'source_csv': [csv_path] * B,
        'placement_id': placement_ids,
        'hpwl': hpwl_vals,
        'warpage': rms_vals,
        'area': area_vals
    })
    
    results_df.to_csv(output_csv, index=False)
    print(f"Computed metrics for {B} placements. Output saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chiplet placements from a CSV file.")
    parser.add_argument("--scenario", type=str, default="scenario/multigpu.json", help="Path to the JSON scenario definition")
    parser.add_argument("--input_csv", type=str, default="others/case1_placements.csv", help="Path to the placements CSV")
    parser.add_argument("--model", type=str, default="src/release_2026_09_13/variable_die_domain_model.pt", help="Path to the surrogate warpage model weights")
    parser.add_argument("--interposer_x", type=float, default=45, help="Fixed interposer width (X dimension in mm)")
    parser.add_argument("--interposer_y", type=float, default=45, help="Fixed interposer height (Y dimension in mm)")
    parser.add_argument("--output_csv", type=str, default="placement_metrics.csv", help="Path to save the computed metrics")
    
    args = parser.parse_args()

    compute_placement_costs(
        scenario_path=args.scenario,
        csv_path=args.input_csv,
        model_path=args.model,
        interposer_x=args.interposer_x,
        interposer_y=args.interposer_y,
        output_csv=args.output_csv
    )
