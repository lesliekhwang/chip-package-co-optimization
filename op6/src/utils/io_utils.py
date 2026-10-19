"""Provides read/write utility functions for files.

This module handles file system interactions, including loading JSON configuration
files, handling file path conversions, and saving output datasets or layout results.
"""
import json
import numpy as np
from pathlib import Path
from typing import Any

from utils.logger import log_get

logger = log_get(__name__)

def load_file_json(file_path: str | Path) -> dict[str, Any] | list[Any]:
    """Loads and parses a JSON file from disk.

    Args:
        file_path: The file system path to the JSON file.

    Returns:
        The decoded Python object (typically a dictionary or list).

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the file contains invalid JSON syntax.
    """
    path = Path(file_path)

    if not path.is_file():
        logger.error(f"Target JSON file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as file_stream:
            data = json.load(file_stream)
            logger.info(f"Successfully loaded JSON from {path.name}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Syntax error in JSON file '{path}' on line {e.lineno}: {e.msg}")
        raise ValueError(f"Failed to parse JSON file '{path}': {e.msg}") from e