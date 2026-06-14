# dataset.py
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from config import SESSION_LABELS
from utils import pad_T_global


class SEEDVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_seed_v(root_dir, session_ids=[1, 2, 3]):
    all_X, all_y, all_person, all_trial = [], [], [], []

    files = sorted(
        [f for f in os.listdir(root_dir) if f.endswith(".npz")],
        key=lambda x: int(os.path.splitext(x)[0].split("_")[0])
    )

    for sess in session_ids:
        print(f"\n=== Loading session {sess} ===")
        labels_15 = np.array(SESSION_LABELS[sess])
        start_idx = (sess - 1) * 15
        end_idx = sess * 15

        for file in files:
            path = os.path.join(root_dir, file)
            subject_id = int(os.path.splitext(file)[0].split("_")[0])

            data = np.load(path, allow_pickle=True)
            d = data['data']
            if isinstance(d.item(), bytes):
                d_dict = pickle.loads(d)
            else:
                d_dict = d.item()

            for i, k in enumerate(range(start_idx, end_idx)):
                trial_data = d_dict[k].reshape(-1, 62, 5)  # (T, 62, 5)
                all_X.append(trial_data)
                all_y.append(labels_15[i])
                all_person.append(f"S{sess}_P{subject_id}")
                all_trial.append(i + 1)

    X_pad = pad_T_global(all_X)
    return X_pad, np.array(all_y), all_person, all_trial

# SEED-IV
# class SEEDIVDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.from_numpy(X).float()
#         self.y = torch.from_numpy(y).long()
        
#     def __len__(self):
#         return len(self.X)
        
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# def load_seediv_session_raw(session_dir, labels_24, session_id):
#     X_list, y_list = [], []
#     person_ids, trial_ids = [], []

#     files = sorted(
#         [f for f in os.listdir(session_dir) if f.endswith(".mat")],
#         key=lambda x: int(os.path.splitext(x)[0].split("_")[0])
#     )

#     for file in files:
#         subject_id = int(os.path.splitext(file)[0].split("_")[0])
#         path = os.path.join(session_dir, file)
#         mat = sio.loadmat(path)

#         trial_keys = sorted(
#             [k for k in mat.keys() if k.startswith("de_LDS")],
#             key=lambda x: int(x.replace("de_LDS", ""))
#         )

#         for t, key in enumerate(trial_keys):
#             raw = mat[key]
#             data = extract_feature_from_deLDS(raw)
#             X_list.append(data)
#             y_list.append(labels_24[t])
#             person_ids.append(f"S{session_id}_P{subject_id}")
#             trial_ids.append(t + 1)

#     return X_list, y_list, person_ids, trial_ids

# def load_seediv_all(root_dir, session_ids=[1, 2, 3]):
#     all_X, all_y = [], []
#     all_person, all_trial = [], []

#     for sess in session_ids:
#         print(f"\n=== Loading session {sess} ===")
#         session_dir = os.path.join(root_dir, str(sess))
#         if not os.path.exists(session_dir):
#              print(f"Warning: Directory {session_dir} does not exist. Skipping.")
#              continue
             
#         labels_24 = np.array(SESSION_LABELS[sess])
#         X_list, y_list, p_list, t_list = load_seediv_session_raw(
#             session_dir, labels_24, sess
#         )
#         all_X += X_list
#         all_y += y_list
#         all_person += p_list
#         all_trial += t_list

#     X_pad = pad_T_global(all_X)
#     return X_pad, np.array(all_y), all_person, all_trial

# def make_source_target(X, y, persons, target_person):
#     src_idx = [i for i, p in enumerate(persons) if p != target_person]
#     tgt_idx = [i for i, p in enumerate(persons) if p == target_person]
#     return (X[src_idx], y[src_idx]), (X[tgt_idx], y[tgt_idx])

# def get_dann_loaders(X, y, persons, target_person, batch_src=32, batch_tgt=32, val_size=0.2, random_state=12):
#     (X_src, y_src), (X_tgt, y_tgt) = make_source_target(X, y, persons, target_person)

#     # Chia tập Source thành Train và Validation
#     X_train_src, X_val_src, y_train_src, y_val_src = train_test_split(
#         X_src, y_src, test_size=val_size, stratify=y_src, random_state=random_state
#     )

#     src_train_ds = SEEDIVDataset(X_train_src, y_train_src)
#     src_val_ds = SEEDIVDataset(X_val_src, y_val_src)
#     tgt_ds = SEEDIVDataset(X_tgt, y_tgt)

#     src_train_loader = DataLoader(src_train_ds, batch_size=batch_src, shuffle=True, drop_last=True)
#     src_val_loader = DataLoader(src_val_ds, batch_size=batch_src, shuffle=False, drop_last=False)
#     tgt_loader = DataLoader(tgt_ds, batch_size=batch_tgt, shuffle=True, drop_last=True) # Domain discriminator learning needs shuffle target

#     return src_train_loader, src_val_loader, tgt_loader, (X_tgt, y_tgt)


def make_source_target(X, y, persons, target_person):
    src_idx = [i for i, p in enumerate(persons) if p != target_person]
    tgt_idx = [i for i, p in enumerate(persons) if p == target_person]
    return (X[src_idx], y[src_idx]), (X[tgt_idx], y[tgt_idx])


def get_dann_loaders(X, y, persons, target_person, batch_src=32, batch_tgt=32, val_size=0.2, random_state=12):
    (X_src, y_src), (X_tgt, y_tgt) = make_source_target(X, y, persons, target_person)

    X_train, X_val, y_train, y_val = train_test_split(
        X_src, y_src, test_size=val_size, stratify=y_src, random_state=random_state
    )

    src_train_loader = DataLoader(SEEDVDataset(X_train, y_train), batch_size=batch_src, shuffle=True, drop_last=True)
    src_val_loader = DataLoader(SEEDVDataset(X_val, y_val), batch_size=batch_src, shuffle=False)
    tgt_loader = DataLoader(SEEDVDataset(X_tgt, y_tgt), batch_size=batch_tgt, shuffle=True)

    return src_train_loader, src_val_loader, tgt_loader, (X_tgt, y_tgt)
