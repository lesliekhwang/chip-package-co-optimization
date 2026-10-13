import os
import sys
import json
import random
import subprocess
import argparse
from pathlib import Path

# =============================================================================
# HUMAN-EDITABLE SWEEP CONFIGURATION TEMPLATE
# -----------------------------------------------------------------------------
# Instructions:
# - For a FIXED value, provide the exact type (e.g., 100, 0.85, [128, 128], True).
# - For a RANDOM RANGE, provide a tuple:
#   * 2-Tuple (min, max): Continuous uniform sampling.
#   * 3-Tuple (min, max, step): Stepped sampling in increments of `step` (inclusive).
#   * If bounds are ints, samples integers; if floats, samples floats.
# - Mutation weights will be automatically normalized to sum to 1.0.
# =============================================================================

SWEEP_TEMPLATE = {
    "immigrant_probability": (0.01, 0.15),  # Continuous float for random immigrant injection
    "algorithm": {
        "population_size": (100, 300, 50),    # Multiples of 20 from 20 to 200 inclusive
        "max_generations": (100, 300, 20),    # Multiples of 20 from 20 to 200 inclusive
        "chunk": 100,
        "device": "cuda"
    },
    "topology": {
        "crossover": {
            "probability": (0.6, 0.95)      # Continuous float between 0.6 and 0.95
        },
        "mutation": {
            "probability": (0.6, 0.95),
            "weights": {
                "swap": (0.1, 1.0),
                "insert": (0.1, 1.0),
                "inversion": (0.1, 1.0),
                "joint_sequence_swap": (0.1, 1.0),
                "scramble": (0.1, 1.0)      # Added scramble mutation
            }
        }
    },
    "slack": {
        "crossover": {
            "alpha": (0.1, 0.9),
            "probability": (0.5, 0.9)
        },
        "mutation": {
            "probability": (0.6, 0.95),
            "weights": {
                "noise": (0.1, 1.0),
                "uniform_reset": (0.1, 1.0)
            },
            "params": {
                "scale": (0.05, 0.2)
            }
        }
    },
    "rotation": {
        "crossover": {
            "probability": (0.1, 0.95)
        },
        "mutation": {
            "probability": (0.01, 0.2)      # Reduced range for per-chiplet independent flip
        }
    },
    "cost": {
        "warp_grid": [128, 128],            # Fixed values
        "use_hpwl": True,
        "use_area": True,
        "use_warpage": True
    }
}

# =============================================================================

def sample_config(template):
    """Recursively parses the template to sample values or keep fixed values."""
    if isinstance(template, dict):
        return {k: sample_config(v) for k, v in template.items()}
    elif isinstance(template, tuple):
        if len(template) == 3:
            min_v, max_v, step = template
            steps = int(round((max_v - min_v) / step))
            sampled_val = min_v + random.randint(0, steps) * step
            
            # Return integer if all parts are int, otherwise round float
            if isinstance(min_v, int) and isinstance(max_v, int) and isinstance(step, int):
                return int(sampled_val)
            return round(float(sampled_val), 4)

        elif len(template) == 2:
            min_v, max_v = template
            if isinstance(min_v, int) and isinstance(max_v, int):
                return random.randint(min_v, max_v)
            else:
                return round(random.uniform(min_v, max_v), 4)

    # Return fixed values (lists, strings, bools, standalone numbers)
    return template

def normalize_dict_values(d: dict) -> dict:
    """Normalizes floating point values in a dictionary so they sum to 1.0."""
    total = sum(d.values())
    if total <= 0:
        uniform_val = 1.0 / len(d)
        return {k: round(uniform_val, 4) for k in d.keys()}
    return {k: round(v / total, 4) for k, v in d.items()}

def generate_random_config(template: dict) -> dict:
    """Samples the template and normalizes required weight dictionaries."""
    cfg = sample_config(template)

    # Normalize weights post-sampling
    if "topology" in cfg and "mutation" in cfg["topology"] and "weights" in cfg["topology"]["mutation"]:
        cfg["topology"]["mutation"]["weights"] = normalize_dict_values(cfg["topology"]["mutation"]["weights"])
        
    if "slack" in cfg and "mutation" in cfg["slack"] and "weights" in cfg["slack"]["mutation"]:
        cfg["slack"]["mutation"]["weights"] = normalize_dict_values(cfg["slack"]["mutation"]["weights"])

    return cfg

def run_sweeps(n_runs: int, run_script: str, tmp_config_path: str, base_scenario: str):
    run_script_path = Path(run_script).resolve()
    tmp_config_file = Path(tmp_config_path).resolve()

    if not run_script_path.exists():
        sys.exit(f"Error: Executable script '{run_script_path}' not found.")

    print("==================================================")
    print(f" Starting Hyperparameter Sweep for {n_runs} Runs")
    print(f" Executable:   {run_script_path}")
    print(f" Tmp Config:   {tmp_config_file}")
    print("==================================================\n")

    for i in range(1, n_runs + 1):
        # Generate and save temporary GA configuration
        cfg = generate_random_config(SWEEP_TEMPLATE)
        
        tmp_config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_config_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        print(f"[{i}/{n_runs}] Executing run with sampled configuration:")
        print(f"  Pop Size:      {cfg['algorithm']['population_size']}")
        print(f"  Max Gens:      {cfg['algorithm']['max_generations']}")
        print(f"  Immigrant Prob:{cfg['immigrant_probability']}")
        print(f"  Topo Mut Prob: {cfg['topology']['mutation']['probability']} | Weights: {cfg['topology']['mutation']['weights']}")
        print(f"  Slack Mut Prob:{cfg['slack']['mutation']['probability']} | Weights: {cfg['slack']['mutation']['weights']}")
        print(f"  Rot Cross Prob:{cfg['rotation']['crossover']['probability']} | Mut Prob: {cfg['rotation']['mutation']['probability']}")

        # Build execution command
        cmd = [
            str(run_script_path),
            "--ga-config", str(tmp_config_file),
            "--random-seed"
        ]

        if base_scenario:
            cmd.extend(["--scenario", base_scenario])
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[{i}/{n_runs}] Completed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{n_runs}] Error executing run.sh: {e}\n")
        except KeyboardInterrupt:
            print("\nSweep interrupted by user. Exiting.")
            sys.exit(1)

    # Cleanup temporary configuration
    if tmp_config_file.exists():
        tmp_config_file.unlink()

    print("==================================================")
    print(f" All {n_runs} sweep runs finished!")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep GA hyperparameters using a predefined template.")
    parser.add_argument("-n", "--runs", type=int, default=10, help="Number of runs to execute (default: 10)")
    parser.add_argument("--script", default="./run.sh", help="Path to execution bash script (default: ./run.sh)")
    parser.add_argument("--tmp-config", default="./scenario/tmp_sweep_gaconfig.json", help="Path for temporary json output")
    parser.add_argument("--scenario", default="", help="Path to scenario config file (optional pass-through)")

    args = parser.parse_args()
    run_sweeps(args.runs, args.script, args.tmp_config, args.scenario)