import numpy as np
import pandas as pd
import torch
import random
import os


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> Global seed set to: {seed}")


def extract_feature_from_deLDS(data):
    """Convert feature from (62, T, 5) -> (T, 62, 5)."""
    if data.ndim != 3:
        raise ValueError(f"Expected (62, T, 5), got {data.shape}")
    if data.shape[0] == 62:
        data = np.transpose(data, (1, 0, 2))
    return data


def pad_T_global(list_data):
    """Zero-pad time dimension T across all samples."""
    T_max = max(d.shape[0] for d in list_data)
    print(">>> Global T_max =", T_max)

    N = len(list_data)
    X_pad = np.zeros((N, T_max, 62, 5), dtype=np.float32)

    for i, d in enumerate(list_data):
        T = d.shape[0]
        X_pad[i, :T] = d

    return X_pad


def read_locs_to_3d(path):
    """Read .locs file and convert to 3D coordinates."""
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["id", "theta_deg", "radius", "label"]
    )

    theta = np.deg2rad(df["theta_deg"].values)
    r = df["radius"].values

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(np.clip(1 - r**2, 0, 1))

    coords_3d = np.stack([x, y, z], axis=1)
    labels = df["label"].tolist()

    return labels, coords_3d


def build_adjacency_matrix(coords_3d):
    """Build combined local + global adjacency matrix (62 EEG channels)."""
    num_nodes = 62
    coords_3d = coords_3d * 20

    # Local graph (distance-based)
    d_const = 5.0
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                A[i, j] = 1.0
            else:
                dist = np.linalg.norm(coords_3d[i] - coords_3d[j])
                A[i, j] = min(1.0, d_const / (dist**2 + 1e-8))

    # Global graph (symmetric hemisphere pairs)
    global_pairs = [
        (1, 2), (4, 5), (6, 14), (7, 13), (8, 12), (9, 11),
        (15, 23), (16, 22), (17, 21), (18, 20), (24, 32),
        (25, 31), (26, 30), (27, 29), (33, 41), (42, 50),
        (43, 49), (44, 48), (45, 47), (51, 57), (52, 56),
        (53, 55), (59, 61), (58, 62),
    ]
    for i, j in global_pairs:
        A[i - 1, j - 1] = 1.0
        A[j - 1, i - 1] = 1.0

    return A
