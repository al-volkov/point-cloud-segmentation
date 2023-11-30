from typing import Any, Dict

import yaml


def read_config(config_path: str) -> Dict[str, Any]:
    """
    Read and parse a configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Config file not found at '{config_path}'")
