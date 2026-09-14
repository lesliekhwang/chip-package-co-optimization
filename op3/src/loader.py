import tomllib
import seqpair
import sys
from pathlib import Path

# ==============================================================================
# CONFIGURATION & SCENARIO LOADERS
# ==============================================================================

def load_scenario(scenario_path: str) -> tuple[dict, dict]:
    """Load the scenario TOML file and return metadata and chiplet definitions."""
    path = Path(scenario_path)
    if not path.is_file():
        print(f"Error: Scenario file '{scenario_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "rb") as f:
            raw_data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Error: Failed to parse TOML file '{scenario_path}': {e}", file=sys.stderr)
        sys.exit(1)

    metadata = raw_data.pop("metadata", {})
    
    base_chiplets = {
        name: seqpair.Chiplet(
            name=name,
            w=props["dmx"],
            h=props["dmy"],
            ws_n=props.get("wsn", 0.0),
            ws_e=props.get("wse", 0.0),
            ws_s=props.get("wss", 0.0),
            ws_w=props.get("wsw", 0.0),
            adj=props.get("adj", [])
        )
        for name, props in raw_data.items()
    }

    return metadata, base_chiplets

def load_config(config_path: str) -> dict:
    """Load system configuration and return an uppercase-normalized dictionary."""
    path = Path(config_path)
    if not path.is_file():
        print(f"Error: Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "rb") as f:
            config_data = tomllib.load(f)
    except Exception as e:
        print(f"Error loading config file '{config_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Flatten nested sections and convert keys to uppercase constants
    flattened_config = {}
    for section in config_data.values():
        if isinstance(section, dict):
            for key, val in section.items():
                flattened_config[key.upper()] = val

    return flattened_config