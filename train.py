# train.py
import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from dataset import get_dann_loaders
from model import EEG_Transformer_KAN, grad_reverse
from config import *


def get_lambda(epoch, total_epochs, step, steps_per_epoch):
    p = (epoch - 1 + step / steps_per_epoch) / total_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def train_loso_dann(target_person, X, y, persons, adj=None, save_prefix=""):
    print(f"\n===== LOSO TRAINING (Target = {target_person}) =====")

    src_train_loader, src_val_loader, tgt_loader, (X_tgt, y_tgt) = get_dann_loaders(
        X, y, persons, target_person, BATCH_SRC, BATCH_TGT
    )

    model = EEG_Transformer_KAN(
        num_nodes=NUM_NODES, in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES, adj=adj
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_em_fn = nn.CrossEntropyLoss()
    loss_dom_fn = nn.CrossEntropyLoss()

    history = {
        "epoch": [], "lambda": [], "loss_em": [], "loss_dom": [],
        "loss_total": [], "val_acc": [], "best_val_acc": []
    }

    best_val_acc = 0.0
    best_state_val, best_epoch_val = None, 0
    best_state_dom, best_epoch_dom = None, 0
    min_domain_diff = float("inf")
    last_state, last_epoch = None, 0
    steps_per_epoch = len(src_train_loader)

    for ep in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
        model.train()
        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_loader)
        L_em_ep, L_dom_ep, L_total_ep = 0, 0, 0
        last_lambda, steps = 0.0, 0

        for i in range(steps_per_epoch):
            lam = get_lambda(ep, NUM_EPOCHS, i, steps_per_epoch)
            last_lambda = lam

            try: Xs, ys = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train_loader); Xs, ys = next(src_iter)
            try: Xt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader); Xt, _ = next(tgt_iter)

            Xs, ys, Xt = Xs.to(DEVICE), ys.to(DEVICE), Xt.to(DEVICE)
            opt.zero_grad()

            fs = model.extract_features(Xs)
            ft = model.extract_features(Xt)

            L_em = loss_em_fn(model.fc_emotion(fs), ys)

            f_all = torch.cat([fs, ft], dim=0)
            dom_labels = torch.cat([
                torch.zeros(fs.size(0), dtype=torch.long),
                torch.ones(ft.size(0), dtype=torch.long)
            ]).to(DEVICE)
            L_dom = loss_dom_fn(model.fc_domain(grad_reverse(f_all, lam)), dom_labels)

            L_total = L_em + L_dom + KAN_REG_COEFF * model.kan_reg()
            L_total.backward()
            opt.step()

            L_em_ep += L_em.item()
            L_dom_ep += L_dom.item()
            L_total_ep += L_total.item()
            steps += 1

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_val, y_val in src_val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                preds = model.predict_emotion(X_val).argmax(dim=1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_dom = L_dom_ep / steps

        # Strategy 1: Best validation accuracy
        if val_acc > best_val_acc or best_state_val is None:
            best_val_acc = val_acc
            best_epoch_val = ep
            best_state_val = copy.deepcopy(model.state_dict())

        # Strategy 2: Domain loss closest to log(2) (confused discriminator)
        dom_diff = abs(avg_dom - np.log(2))
        if best_state_dom is None:
            best_state_dom = copy.deepcopy(model.state_dict())
            best_epoch_dom = ep
        if ep > NUM_EPOCHS // 2 and dom_diff < min_domain_diff:
            min_domain_diff = dom_diff
            best_epoch_dom = ep
            best_state_dom = copy.deepcopy(model.state_dict())

        # Strategy 3: Last epoch
        last_state = copy.deepcopy(model.state_dict())
        last_epoch = ep

        history["epoch"].append(ep)
        history["lambda"].append(float(last_lambda))
        history["loss_em"].append(L_em_ep / steps)
        history["loss_dom"].append(avg_dom)
        history["loss_total"].append(L_total_ep / steps)
        history["val_acc"].append(val_acc)
        history["best_val_acc"].append(best_val_acc)

    # Target evaluation
    os.makedirs("checkpoints", exist_ok=True)
    prefix = f"{save_prefix}_" if save_prefix else ""
    del model

    def evaluate_state(state, name, epoch):
        m = EEG_Transformer_KAN(
            num_nodes=NUM_NODES, in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES, adj=adj
        ).to(DEVICE)
        m.load_state_dict(state)
        m.eval()

        torch.save(state, f"checkpoints/{DATASET}_best_model_{prefix}{target_person}_{name}.pth")

        Xt_eval = torch.from_numpy(X_tgt).float().to(DEVICE)
        yt_eval = torch.from_numpy(y_tgt).long().to(DEVICE)

        with torch.no_grad():
            preds = m.predict_emotion(Xt_eval).argmax(dim=1)
            acc = (preds == yt_eval).float().mean().item()
            y_t = yt_eval.cpu().numpy()
            y_p = preds.cpu().numpy()
            p, r, f, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)

        print(f">>> {target_person} [{name.upper()}] Ep:{epoch} Acc:{acc:.4f} F1:{f:.4f}")
        return y_t, y_p, acc, epoch, {"precision": p, "recall": r, "f1": f}

    print(f"\nEvaluating 3 strategies for {target_person}...")
    return None, history, {
        "val": evaluate_state(best_state_val, "val_acc", best_epoch_val),
        "dom": evaluate_state(best_state_dom, "dom_loss", best_epoch_dom),
        "last": evaluate_state(last_state, "last_ep", last_epoch),
        "best_val_acc": best_val_acc
    }


def train_loso_dann_all_subjects(X, y, persons, adj=None, subject_list=None, save_prefix=""):
    subjects = subject_list or sorted(set(persons))
    results = {}
    all_y_true = {"val": [], "dom": [], "last": []}
    all_y_pred = {"val": [], "dom": [], "last": []}

    for target in subjects:
        _, history, eval_results = train_loso_dann(target, X, y, persons, adj=adj, save_prefix=save_prefix)
        results[target] = {"history": history, "eval_results": eval_results}

        for strat in ["val", "dom", "last"]:
            if strat in eval_results:
                all_y_true[strat].extend(eval_results[strat][0])
                all_y_pred[strat].extend(eval_results[strat][1])

    session_summary = {
        strat: {"y_true": np.array(all_y_true[strat]), "y_pred": np.array(all_y_pred[strat])}
        for strat in ["val", "dom", "last"]
    }
    return results, session_summary
