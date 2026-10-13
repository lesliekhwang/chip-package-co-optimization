import os
import argparse
import json
import h5py
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def extract_run_summary(h5_path: str) -> dict:
    """Parses an HDF5 file alongside adjacent JSON files to extract a comprehensive summary."""
    run_dir = os.path.dirname(os.path.abspath(h5_path))
    summary = {"run": run_dir}
    
    try:
        # 1. Extract from adjacent config JSONs (Assumes strict presence in run directory)
        ga_path = os.path.join(run_dir, "gaconfig.json")
        with open(ga_path, "r") as f:
            ga_config = json.load(f)
            
            # Direct bracket notation referencing the algorithm block
            summary["ga_pop_size"] = ga_config["algorithm"]["population_size"]
            summary["ga_n_gen"] = ga_config["algorithm"]["max_generations"]
            summary["ga_chunk_size"] = ga_config["algorithm"]["chunk"]

        chiplets_path = os.path.join(run_dir, "chiplets.json")
        with open(chiplets_path, "r") as f:
            chiplets = json.load(f)
            
            # Ignore system-level interposer key to get raw chiplet count
            active_chiplets = [k for k in chiplets.keys() if k.lower() != "interposer"]
            summary["num_chiplets"] = len(active_chiplets)

        # 2. Extract from HDF5 file (Strictly new native structure)
        with h5py.File(h5_path, "r") as h5f:
            summary["seed"] = int(h5f["metadata/seed"][()])

            # Extract from Summary Tables (Aggregated)
            generations_group = h5f["generations"]
            
            # Runtime aggregation
            times_table = generations_group["generation_times"][:]
            summary["total_runtime_minutes"] = float(np.sum(times_table['wall_clock_sec']) / 60.0)
            summary["completed_generations"] = int(len(times_table))
                
            # Mutation & Operator aggregation
            mut_table = generations_group["generations_mutation"][:]
            num_gens = max(1, len(mut_table))
            
            # Dynamically average all operator columns (topo_, slack_, rot_, crossover_, etc.)
            for col_name in mut_table.dtype.names:
                if col_name != "generation":
                    summary[f"avg_{col_name}_per_gen"] = float(np.sum(mut_table[col_name]) / num_gens)

            # 3. Genotype Rotation Analysis
            gen_keys = [k for k in generations_group.keys() if k.startswith("gen") and k[3:].isdigit()]
            
            rot_ratios = []
            for gen_key in gen_keys:
                if "genotype/rot" in generations_group[gen_key]:
                    rot_mat = generations_group[gen_key]["genotype/rot"][:]
                    rot_ratios.append(np.mean(rot_mat))
            
            if rot_ratios:
                summary["avg_rotated_chiplet_ratio"] = float(np.mean(rot_ratios))

            # 4. Global Pareto & Utopian Calculations
            all_F_matrices = []
            for gen_key in gen_keys:
                all_F_matrices.append(generations_group[gen_key]["fitness/F"][:])
                    
            if all_F_matrices:
                F_combined = np.vstack(all_F_matrices)
                F_unique = np.unique(F_combined, axis=0)
                
                fronts = NonDominatedSorting().do(F_unique)
                rank_0_F = F_unique[fronts[0]]
                
                summary["num_unique_pareto_solutions"] = int(len(rank_0_F))
                
                ideal_point = np.min(rank_0_F, axis=0)
                nadir_point = np.max(rank_0_F, axis=0)
                
                # Cast to standard python floats for JSON serialization
                summary["ideal_HPWL"] = float(ideal_point[0])
                summary["ideal_Area"] = float(ideal_point[1])
                summary["ideal_Warpage"] = float(ideal_point[2])
                
                summary["nadir_HPWL"] = float(nadir_point[0])
                summary["nadir_Area"] = float(nadir_point[1])
                summary["nadir_Warpage"] = float(nadir_point[2])

                # --- N-Dim Utopian (0,0,0) Distance Analysis ---
                denom = nadir_point - ideal_point
                denom[denom == 0] = 1e-6  # Prevent division by zero
                
                F_norm = (rank_0_F - ideal_point) / denom
                norm_distances = np.linalg.norm(F_norm, axis=1)
                
                min_norm_dist = np.min(norm_distances)
                best_idx = np.argmin(norm_distances)
                best_sample = rank_0_F[best_idx]

                summary["utopian_best_HPWL"] = float(best_sample[0])
                summary["utopian_best_Area"] = float(best_sample[1])
                summary["utopian_best_Warpage"] = float(best_sample[2])
                summary["utopian_min_norm_distance"] = float(min_norm_dist)

                exact_ties = int(np.sum(np.abs(norm_distances - min_norm_dist) <= 1e-4))
                summary["count_exact_closest_utopian"] = exact_ties

                near_utopian_count = int(np.sum(norm_distances <= (min_norm_dist * 1.05)))
                summary["count_near_utopian_5pct"] = near_utopian_count
                    
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")
        summary["error"] = str(e)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract run summary from an HDF5 file and adjacent config JSONs.")
    parser.add_argument("--input", required=True, help="Path to the input HDF5 file (e.g., run.h5)")
    parser.add_argument("--output", required=True, help="Path to output JSON summary file (e.g., summary.json)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        exit(1)

    print(f"Extracting summary from {args.input} and checking directory for configs...")
    summary_data = extract_run_summary(args.input)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)

    print(f"Summary successfully generated at {args.output}")