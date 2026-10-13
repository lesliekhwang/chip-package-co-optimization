"""
loader.py
---------------------------------
JSON configuration and scenario loaders for chiplet placement.
Strictly parses scenario structure, extracts interposer constraints, and outputs 
vectorized dimensions, slacks, and adjacencies.
"""

import json
from pathlib import Path
import numpy as np

def load_scenario(scenario_path: str) -> tuple[dict, np.ndarray, np.ndarray, list, dict]:
    """
    Loads scenario JSON and returns purely vectorized data representations.
    Returns:
        - id_to_name: dict mapping int ID to string name.
        - base_dims: 2xN np.ndarray (row 0: widths, row 1: heights).
        - slack_bounds: 2xN np.ndarray (row 0: min slacks, row 1: max slacks).
        - adj_list: list of lists containing integer adjacencies.
        - interposer_bounds: dict of constraint boundaries.
    """
    path = Path(scenario_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    try:
        with open(path, "r") as f:
            raw_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse scenario JSON file '{scenario_path}': {e}") from e

    if "interposer" not in raw_data:
        raise KeyError(f"Missing required key 'interposer' in scenario file: {scenario_path}")

    interposer_bounds = raw_data.pop("interposer")

    chiplet_names = list(raw_data.keys())
    N = len(chiplet_names)
    
    name_to_id = {name: i for i, name in enumerate(chiplet_names)}
    id_to_name = {i: name for i, name in enumerate(chiplet_names)}

    base_dims = np.zeros((2, N))
    slack_bounds = np.zeros((2, N))
    adj_list = [[] for _ in range(N)]

    for i, name in enumerate(chiplet_names):
        props = raw_data[name]
        try:
            base_dims[0, i] = props["dim"][0]
            base_dims[1, i] = props["dim"][1]
            slack_bounds[0, i] = 0.0 #props["slack"][0]
            slack_bounds[1, i] = 0.0 #props["slack"][1]
            
            if "adjacency" in props:
                adj_list[i] = [name_to_id[adj] for adj in props.get("adjacency", []) if adj in name_to_id]
        except KeyError as e:
            raise KeyError(f"Missing required property {e} in chiplet '{name}' inside {scenario_path}") from e

    return id_to_name, base_dims, slack_bounds, adj_list, interposer_bounds


def load_config(config_path: str) -> dict:
    """Load system configuration JSON file and return the nested dictionary."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse config JSON file '{config_path}': {e}") from e