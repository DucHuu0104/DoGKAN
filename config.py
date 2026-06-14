import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths (SEED-V)
DATA_ROOT = "./data/SEED_V/EEG_DE_features"
LOCS_PATH = "./data/SEED_V/channel_62_pos.locs"

# Labels per session (Disgust:0, Fear:1, Sad:2, Neutral:3, Happy:4)
SESSION_LABELS = {
    1: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    3: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}

# Model
NUM_NODES = 62
IN_CHANNELS = 5
NUM_CLASSES = 5

# Training
LEARNING_RATE = 5e-4
NUM_EPOCHS = 300
BATCH_SRC = 16
BATCH_TGT = 16

# KAN
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3
KAN_REG_COEFF = 0.01

# Transformer
TRANS_HEADS = 4
TRANS_D_K = 16
TRANS_D_V = 16
TRANS_D_FF = 128
TRANS_DROPOUT = 0.1

# Reproducibility
RANDOM_SEED = 2
