"""
main.py
---------------------------------
Optimization entry point for executing the NSGA-II chiplet placement search.
Logs generation-by-generation results directly into a Parquet binary file.
Fully drives the 4-directional (N, E, S, W) whitespace gene model via the ga module.

Usage:
    python main.py <scenario.toml> <output_path> <pop_size> <n_gens> <seed> <history_fname> <mut_x> <mut_y> <mut_ws> <ws_max> <model_path> <grid_size>
"""

from pathlib import Path
import loader
import random
import sys
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import json

import ga
import warpage
from warpage import Cube
import plot

# ==============================================================================
# MAIN OPTIMIZATION ROUTINE
# ==============================================================================

def main() -> None:
    # 1. Parse Command Line Arguments
    if len(sys.argv) != 13:
        print("Error: Missing or incorrect number of arguments!", file=sys.stderr)
        print("Usage: python main.py <scenario.toml> <output_path> <pop_size> <n_gens> <seed> <history.parquet> <mut_x> <mut_y> <mut_ws> <ws_max> <model_path> <grid_size>", file=sys.stderr)
        sys.exit(1)

    scenario_file = sys.argv[1]
    output_path = Path(sys.argv[2])

    # Reconstruct the configuration dictionary dynamically from sys.argv
    # Types must be explicitly cast since sys.argv only provides strings
    cfg = {
        "SCENARIO_FILE": sys.argv[1],
        "OUTPUT_PATH": sys.argv[2],
        "POPULATION_SIZE": int(sys.argv[3]),
        "N_GENERATIONS": int(sys.argv[4]),
        "RANDOM_SEED": int(sys.argv[5]),
        "HISTORY_PARQUET": sys.argv[6],
        "MUT_PROB_SEQ_X": float(sys.argv[7]),
        "MUT_PROB_SEQ_Y": float(sys.argv[8]),
        "MUT_PROB_WS": float(sys.argv[9]),
        "WS_MAX": float(sys.argv[10]),
        "MODEL_PATH": sys.argv[11],
        "GRID_SIZE": int(sys.argv[12]),
    }

    # 2. Load Inputs
    metadata, base_chiplets = loader.load_scenario(scenario_file)
    chiplet_names = list(base_chiplets.keys())

    plot.print_metadata(metadata, base_chiplets)

    # 3. Resolve and Set Global Seeds
    active_seed = cfg.get("RANDOM_SEED")
    
    np.random.seed(active_seed)
    random.seed(active_seed)
    print(f"[*] Global RNG Seed Set To: {active_seed}")

    # Initialize the ML surrogate model globally before GA starts
    warpage.initialize_model(cfg)

    # 4. Configure Problem, Algorithm, and Parquet Tracker Callback
    problem = ga.ChipletPlacementProblem(base_chiplets,
                                        use_hpwl    = True, 
                                        use_area    = True, 
                                        use_warpage = False)
    algorithm = NSGA2(
        pop_size=cfg["POPULATION_SIZE"],
        sampling=ga.PermutationPlacementSampling(chiplet_names, cfg["WS_MAX"]),
        crossover=ga.PermutationPlacementCrossover(),
        mutation=ga.PermutationPlacementMutation(
            names=chiplet_names,
            mut_prob_x=cfg["MUT_PROB_SEQ_X"],
            mut_prob_y=cfg["MUT_PROB_SEQ_Y"],
            mut_prob_ws=cfg["MUT_PROB_WS"],
            ws_max=cfg["WS_MAX"]
        ),
        eliminate_duplicates=False
    )

    output_path.mkdir(parents=True, exist_ok=True)
    
    # Target Parquet history file
    history_fname = cfg.get("HISTORY_PARQUET")
    history_parquet = (output_path / history_fname)
    
    tracker = ga.ParetoFrontTrackerCallback(history_parquet)

    # 5. Run Optimization Search
    print(f"Starting NSGA-II optimization ({cfg['N_GENERATIONS']} gens, pop {cfg['POPULATION_SIZE']})...")
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", cfg["N_GENERATIONS"]),
        seed=active_seed,
        verbose=True,
        callback=tracker
    )

    # Guarantee stream buffer is fully flushed to disk
    tracker.close()

    if res.F is None:
        print("Optimization completed with no solutions found.")
        return

    print(f"\nOptimization run complete.")
    print(f"Complete population history saved to: {history_parquet}")

    # Save configuration to the output directory
    config_file_path = output_path / "main_config.json"
    with open(config_file_path, "w") as file:
        json.dump(cfg, file, indent=4)
    print(f"Configuration saved to: {config_file_path}")

if __name__ == "__main__":
    main()