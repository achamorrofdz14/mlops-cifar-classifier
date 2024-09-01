"""cifar_classifier package.

It contains modules for data processing, model training, and evaluation.
"""

import os
from pathlib import Path

# Base path of the project
BASEPATH = Path(__file__).parent
WORKDIR = Path(os.getenv("WORKDIR", BASEPATH.parent))

# Data and model directories
DATA_DIR = WORKDIR / "data"
MODEL_DIR = WORKDIR / "models"
CONFIG_DIR = WORKDIR / "config"

# Dataset paths
TRAIN_PATH = DATA_DIR / "train_dataset.pickle"
VAL_PATH = DATA_DIR / "val_dataset.pickle"

# Dataset categories name
TRAIN = "train"
VAL = "val"
