"""Utility functions for manage the configuration file."""

import yaml
from loguru import logger

from src.cifar_classifier import CONFIG_DIR


def load_config() -> dict:
    """Load the configuration file."""
    config_path = CONFIG_DIR / "config.yml"

    try:
        with config_path.open() as file:
            config = yaml.safe_load(file)
            logger.debug(f"Configuration file loaded from {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise

    return config
