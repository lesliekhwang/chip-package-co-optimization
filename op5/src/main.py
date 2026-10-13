# main.py
import sys
import json
import os
import numpy as np
import random
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import ga
import loader
import warpage

def main() -> None:
    # 1. Parse Command Line Arguments
    if len(sys.argv) not in (5, 6):
        raise ValueError(
            f"Usage: python {sys.argv[0]} <scenario_file> <ga_config_file> <model_file> <seed> [h5_output_file]"
        )

    scenario_file = sys.argv[1]
    ga_config_file = sys.argv[2]
    model_file = sys.argv[3]
    
    try:
        cli_seed = int(sys.argv[4])
    except ValueError as e:
        raise ValueError(f"Seed argument must be an integer. Received: {sys.argv[4]}") from e

    # Receive explicit output file path from run.sh or fallback to default
    h5_output_file = sys.argv[5] if len(sys.argv) == 6 else "history.h5"

    # Ensure output destination directory exists
    output_dir = os.path.dirname(os.path.abspath(h5_output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.random.seed(cli_seed)
    random.seed(cli_seed)

    # 2. Load Raw JSON Contexts for Callback
    with open(scenario_file, "r") as f:
        chiplet_cfg = json.load(f)

    id_to_name, base_dims, slack_bounds, adj_list, interposer_bounds = loader.load_scenario(scenario_file)
    ga_cfg = loader.load_config(ga_config_file)
    N = len(id_to_name)

    # 3. Load Algorithm Settings
    try:
        alg_cfg = ga_cfg["algorithm"]
        pop_size = alg_cfg["population_size"]
        n_gens = alg_cfg["max_generations"]
        chunk_size = alg_cfg.get("chunk", pop_size)  # Extract chunk size, defaulting to pop_size
            
        cost_cfg = ga_cfg["cost"]
        cost_cfg["model_path"] = model_file
        ga_cfg["cost"] = cost_cfg
        
        ar_bounds = interposer_bounds["aspect_ratio"]
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}") from e

    warpage.initialize_model(ga_cfg)

    # 4. Configure Vectorized Problem
    problem = ga.ChipletPlacementProblem(
        N=N,
        base_dims=base_dims,
        slack_bounds=slack_bounds,
        adj_list=adj_list,
        ar_bounds=ar_bounds,
        use_hpwl=cost_cfg["use_hpwl"],
        use_area=cost_cfg["use_area"],
        use_warpage=cost_cfg["use_warpage"],
        chunk_size=chunk_size  # Pass to problem definition
    )

    mutation_op = ga.PermutationPlacementMutation(N, ga_cfg)

    # 5. Initialize Callback with Structured HDF5 Logging
    callback = ga.HDF5HistoryCallback(
        mutation_operator=mutation_op,
        filename=h5_output_file,
        seed=cli_seed,
        id_to_name=id_to_name,
        adj_list=adj_list,
        model_file=model_file
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=ga.PermutationPlacementSampling(N, slack_bounds),
        crossover=ga.PermutationPlacementCrossover(ga_cfg),
        mutation=mutation_op,
        eliminate_duplicates=False
    )

    minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gens),
        seed=cli_seed,
        callback=callback,
        verbose=True
    )

if __name__ == "__main__":
    main()