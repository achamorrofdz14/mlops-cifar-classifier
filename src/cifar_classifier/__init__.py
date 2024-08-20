import os
from pathlib import Path

# Base path of the project
BASEPATH = Path(__file__).parent
WORKDIR = Path(os.getenv("WORKDIR", BASEPATH.parent))

# Data and model directories
DATA_DIR = WORKDIR / "data"
MODEL_DIR = WORKDIR / "models"
CONFIG_DIR = WORKDIR / "config"

# Datasets name
TRAIN = "train"
TEST = "test"
