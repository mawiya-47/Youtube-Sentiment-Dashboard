"""
config.py
---------
Loads config/config.yaml into a plain dict so every module reads settings
from one place instead of hardcoding paths/hyperparameters.
"""

import os
import yaml

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
