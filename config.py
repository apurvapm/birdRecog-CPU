"""Hyperparameters and paths for the CUB-200 EfficientNet trainer.

You can edit values here OR override them on the command line, e.g.:
    python train.py --epochs 20 --batch-size 64 --no-freeze
"""

from pathlib import Path

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CUB_ROOT = DATA_DIR / "CUB_200_2011"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# --- dataset ---
NUM_CLASSES = 200
IMAGE_SIZE = 224              # EfficientNet-B0 default input resolution

# --- model ---
# EfficientNet variants available in torchvision: b0 (smallest) ... b7 (largest)
# B0 is the right choice for CPU. Bump to b2/b3 if you have a GPU.
MODEL_NAME = "efficientnet_b0"

# If True, freeze the backbone and only train the new 200-class head.
# This trains in MINUTES on CPU and gets ~60-70% accuracy.
# Set to False for full fine-tuning (best accuracy, much slower).
FREEZE_BACKBONE = True

# --- training ---
EPOCHS = 8
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2               # DataLoader workers; set to 0 on Windows if you hit issues

# Reproducibility
SEED = 42
