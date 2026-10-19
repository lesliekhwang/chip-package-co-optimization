"""Generator script for creating chiplet modification (*_mod.json) files.

Scans a case directory for base chiplet system JSON configurations, calculates
whitespace slack bounds using physical dimensions into a 'slacks' dictionary,
and generates polynomial cost coefficient vectors for unique edge pairs into an
'edges' dictionary based on frequency-dependent distance thresholds.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

# =============================================================================
# TOP-LEVEL SCRIPT CONSTANTS & RULE CONFIGURATIONS
# =============================================================================

# Slack Calculation Constants
DEFAULT_SLACK_RULE: str = "perimeter"  # Metric type: 'perimeter' or 'area'
MIN_SLACK_SCALE: float = 0.003         # Scaling factor applied to chiplet metric
MAX_SLACK_MULTIPLIER: float = 20.0     # Multiplier to derive max slack from min slack
MIN_SLACK_FLOOR: float = 0.05          # Absolute minimum bound for min slack (mm)
MAX_SLACK_FLOOR: float = 0.5           # Absolute minimum bound for max slack (mm)

# Frequency & Polynomial Cost Modeling Constants
DEFAULT_TARGET_FREQ_GHZ: float = 2.0   # Base operating frequency (GHz)
BASE_REACH_THRESHOLD_MM: float = 6.0   # Critical interconnect reach at 1.0 GHz (mm)
BASE_LINEAR_COST_COEFF: float = 1.0    # Linear cost coefficient c1 for short lengths
CUBIC_PENALTY_WEIGHT: float = 10.0     # Penalty weight W applied when reach threshold is breached
POLYNOMIAL_DEGREE: int = 3             # Polynomial degree (order 3 yields [c0, c1, c2, c3])


# =============================================================================
# CORE HELPER FUNCTIONS
# =============================================================================

def calculate_slack_bounds(
    dim: list[float],
    rule: str = DEFAULT_SLACK_RULE,
) -> tuple[list[float], list[float]]:
    """Calculates [N, E, S, W] min and max slacks based on chiplet dimensions.

    Args:
        dim: List of [width, height] dimensions for a chiplet.
        rule: Scaling metric to apply ('perimeter' or 'area').

    Returns:
        A tuple containing two length-4 lists: (min_slack, max_slack).
    """
    w, h = dim[0], dim[1]

    if rule == "area":
        metric = math.sqrt(w * h)
    else:
        metric = 2.0 * (w + h)

    min_val = max(MIN_SLACK_FLOOR, round(metric * MIN_SLACK_SCALE, 2))
    max_val = max(MAX_SLACK_FLOOR, round(min_val * MAX_SLACK_MULTIPLIER, 2))

    min_slack = [min_val] * 4
    max_slack = [max_val] * 4

    return min_slack, max_slack


def calculate_wirelength_polynomial(
    src_info: dict[str, Any],
    dst_info: dict[str, Any],
    target_freq_ghz: float = DEFAULT_TARGET_FREQ_GHZ,
) -> list[float]:
    """Generates polynomial cost coefficients for wirelength between two chiplets.

    Models wirelength cost as a polynomial:
        Cost(L) = c0 + c1*L + c2*L^2 + c3*L^3 ...

    The critical reach threshold d_thresh scales inversely with operating frequency
    and chiplet power density. For L <= d_thresh, cost scales linearly with c1.
    For L > d_thresh, the cubic term c3 causes a steep cost increase.

    Args:
        src_info: Attribute dictionary for the source chiplet.
        dst_info: Attribute dictionary for the target/adjacent chiplet.
        target_freq_ghz: System interconnect operating frequency in GHz.

    Returns:
        A list of float polynomial coefficients [c0, c1, c2, c3].
    """
    src_power = src_info.get("power", 10.0)
    dst_power = dst_info.get("power", 10.0)
    power_factor = math.sqrt((src_power + dst_power) / 20.0)
    power_factor = max(0.5, min(power_factor, 3.0))

    d_thresh = BASE_REACH_THRESHOLD_MM / (target_freq_ghz * power_factor)
    d_thresh = max(1.0, d_thresh)

    c0 = 0.0
    c1 = BASE_LINEAR_COST_COEFF
    c2 = 0.0
    c3 = round(CUBIC_PENALTY_WEIGHT / (d_thresh ** 3), 6)

    return [c0, c1, c2, c3]


def build_mod_config(
    base_config: dict[str, dict[str, Any]],
    slack_rule: str = DEFAULT_SLACK_RULE,
    target_freq_ghz: float = DEFAULT_TARGET_FREQ_GHZ,
) -> dict[str, Any]:
    """Transforms base chiplet system configuration into a _mod configuration.

    Constructs a top-level 'slacks' dictionary containing per-chiplet min/max
    slacks and a top-level 'edges' dictionary containing polynomial cost vectors
    for each unique undirected connection pair.

    Args:
        base_config: Parsed base JSON dictionary containing chiplet definitions.
        slack_rule: Dimension metric ('perimeter' or 'area') used for slack calculation.
        target_freq_ghz: Target operating frequency for polynomial cost generation.

    Returns:
        Dictionary containing 'slacks' and 'edges' top-level keys.
    """
    slacks_dict: dict[str, dict[str, list[float]]] = {}
    edges_dict: dict[str, list[float]] = {}
    processed_edges: set[tuple[str, str]] = set()

    # 1. Build per-chiplet slacks dictionary
    for chiplet_name, chiplet_info in base_config.items():
        dims = chiplet_info.get("dim", [10.0, 10.0])
        min_slack, max_slack = calculate_slack_bounds(dims, rule=slack_rule)
        slacks_dict[chiplet_name] = {
            "min_slack": min_slack,
            "max_slack": max_slack,
        }

    # 2. Build unique edges connectivity dictionary with polynomial coefficients
    for src_name, src_info in base_config.items():
        adjacencies = src_info.get("adjacency", [])

        for dst_name in adjacencies:
            if dst_name not in base_config:
                continue

            # Store undirected edges in canonical sorted order to prevent duplicates
            edge_pair = tuple(sorted([src_name, dst_name]))
            if edge_pair not in processed_edges:
                processed_edges.add(edge_pair)

                dst_info = base_config[dst_name]
                poly_coeffs = calculate_wirelength_polynomial(
                    src_info=src_info,
                    dst_info=dst_info,
                    target_freq_ghz=target_freq_ghz,
                )
                edge_key = f"{edge_pair[0]}+{edge_pair[1]}"
                edges_dict[edge_key] = poly_coeffs

    return {
        "slacks": slacks_dict,
        "edges": edges_dict,
    }


# =============================================================================
# DIRECTORY PROCESSING & CLI
# =============================================================================

def process_case_directory(
    case_dir: str | Path,
    mod_dir: str | Path,
    slack_rule: str = DEFAULT_SLACK_RULE,
    target_freq_ghz: float = DEFAULT_TARGET_FREQ_GHZ,
    overwrite: bool = True,
) -> None:
    """Scans directory and generates corresponding _mod.json files in mod_dir.

    Args:
        case_dir: Directory path containing base case JSON files.
        mod_dir: Output directory path where _mod.json files will be saved.
        slack_rule: Metric rule ('perimeter' or 'area') for slack derivation.
        target_freq_ghz: Interconnect operating frequency for threshold scaling.
        overwrite: Whether to overwrite existing _mod.json files.

    Raises:
        ValueError: If case_dir does not exist or is not a directory.
    """
    directory = Path(case_dir)
    target_mod_dir = Path(mod_dir)

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Target directory '{case_dir}' does not exist or is not a directory.")

    target_mod_dir.mkdir(parents=True, exist_ok=True)

    json_files = [
        f for f in directory.glob("*.json") if not f.name.endswith("_mod.json")
    ]

    print(f"Found {len(json_files)} base configuration file(s) in '{directory}'")
    print(f"Output directory: '{target_mod_dir.resolve()}'")

    for json_file in json_files:
        mod_filename = json_file.stem + "_mod.json"
        mod_filepath = target_mod_dir / mod_filename

        if mod_filepath.exists() and not overwrite:
            print(f"  Skipping '{json_file.name}' (_mod file already exists)")
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            base_config = json.load(f)

        mod_config = build_mod_config(
            base_config,
            slack_rule=slack_rule,
            target_freq_ghz=target_freq_ghz,
        )

        with open(mod_filepath, "w", encoding="utf-8") as f:
            json.dump(mod_config, f, indent=4)

        print(f"  Generated -> {mod_filepath.name}")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Generates _mod.json configurations with rules-based slacks and wirelength cost polynomials."
    )
    parser.add_argument(
        "case_dir",
        type=str,
        nargs="?",
        default="./cases",
        help="Path to directory containing case JSON files (default: ./cases)",
    )
    parser.add_argument(
        "--mod-dir",
        type=str,
        default="./mods",
        help="Path to directory where _mod.json files will be saved (default: ./mods)",
    )
    parser.add_argument(
        "--rule",
        choices=["perimeter", "area"],
        default=DEFAULT_SLACK_RULE,
        help=f"Rule dimension metric for computing slack bounds (default: {DEFAULT_SLACK_RULE})",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=DEFAULT_TARGET_FREQ_GHZ,
        help=f"Target operating frequency in GHz (default: {DEFAULT_TARGET_FREQ_GHZ})",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing _mod.json files if present",
    )

    args = parser.parse_args()
    process_case_directory(
        case_dir=args.case_dir,
        mod_dir=args.mod_dir,
        slack_rule=args.rule,
        target_freq_ghz=args.freq,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()