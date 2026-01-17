"""YAML configuration loader utility."""
import os
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_yaml(config: Dict[str, Any], path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
