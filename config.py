import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASET = "SEED_V"
# DATASET = "SEED_IV"

if DATASET == "SEED_V":

    DATA_ROOT = "./data/SEED_V/EEG_DE_features"
    LOCS_PATH = "./data/SEED_V/channel_62_pos.locs"

    NUM_CLASSES = 5

    SESSION_LABELS = {
        1: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
        2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
        3: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    }
    
    LABEL_NAMES = [
        "Disgust",
        "Fear",
        "Sad",
        "Neutral",
        "Happy"
    ]

elif DATASET == "SEED_IV":

    DATA_ROOT = "./data/SEED_IV/eeg_feature_smooth"
    LOCS_PATH = "./data/SEED_IV/channel_62_pos.locs"

    NUM_CLASSES = 4

    SESSION_LABELS = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,3,3,0,3,0,3,0],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
    }
    
    LABEL_NAMES = [
        "Happy",
        "Sad",
        "Fear",
        "Neutral"
    ]

else:
    raise ValueError(
        f"Unsupported DATASET = {DATASET}. "
        f"Choose 'SEED_V' or 'SEED_IV'."
    )

NUM_NODES = 62
IN_CHANNELS = 5

LEARNING_RATE = 5e-4
NUM_EPOCHS = 300

BATCH_SRC = 16
BATCH_TGT = 16


KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3
KAN_REG_COEFF = 0.01

TRANS_HEADS = 4
TRANS_D_K = 16
TRANS_D_V = 16
TRANS_D_FF = 128
TRANS_DROPOUT = 0.1

RANDOM_SEED = 2