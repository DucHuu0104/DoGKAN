# main.py
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from config import *
from dataset import load_seed_v
from train import train_loso_dann_all_subjects
from utils import read_locs_to_3d, build_adjacency_matrix, set_seed
from model import EEG_Transformer_KAN

LABEL_NAMES = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_cm_plot(y_true, y_pred, title, log_dir, filename):
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f"Confusion Matrix: {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(plot_dir, f"{filename}.png"))
    plt.close()


def save_training_log(results, session_summary, sessions, seed, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    sess_str = "sess" + "".join(map(str, sessions))
    log_path = os.path.join(log_dir, f"train_{sess_str}_seed{seed}_{timestamp}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("SEED-V TRAINING LOG (DoGKAN-DANN)\n")
        f.write(f"Timestamp: {timestamp}\nSessions: {sessions}\nSeed: {seed}\n")
        f.write(f"LR: {LEARNING_RATE} | Epochs: {NUM_EPOCHS}\n\n")

        # Session summary
        f.write("SESSION TOTAL SUMMARY\n")
        strat_names = {"val": "Best Val_Acc", "dom": "Best Dom_Loss", "last": "Last Epoch"}
        for strat, sname in strat_names.items():
            if strat not in session_summary:
                continue
            y_t = session_summary[strat]["y_true"]
            y_p = session_summary[strat]["y_pred"]
            if len(y_t) == 0:
                continue
            p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)
            acc = (y_t == y_p).mean()
            save_cm_plot(y_t, y_p, f"Total ({sname})", log_dir,
                         f"total_{strat}_{sess_str}_seed{seed}_{timestamp}")
            f.write(f"{sname} Acc:{acc:.4f} P:{p:.4f} R:{r:.4f} F1:{f1:.4f}\n")

        f.write("\n" + "=" * 60 + "\n\n")

        # Per-subject results
        for person, res in results.items():
            f.write(f"SUBJECT: {person}\n")
            er = res.get("eval_results", res)
            f.write(f"Best Val Acc: {er.get('best_val_acc', 0):.4f}\n")

            for k, dname in [("val", "Val_Acc"), ("dom", "Dom_Loss"), ("last", "Last")]:
                if k in er:
                    _, _, acc, ep, m = er[k]
                    f.write(f"  [{dname}] Ep:{ep} Acc:{acc:.4f} P:{m['precision']:.4f} R:{m['recall']:.4f} F1:{m['f1']:.4f}\n")

            if "dom" in er:
                save_cm_plot(er["dom"][0], er["dom"][1], person, log_dir,
                             f"cm_{person}_dom_{timestamp}")

            # Training history
            h = res["history"]
            f.write(f"{'Ep':<6}|{'Lambda':<9}|{'ValAcc':<9}|{'BestVal':<9}|{'L_Emo':<9}|{'L_Dom':<9}\n")
            for i in range(len(h["epoch"])):
                f.write(f"{h['epoch'][i]:<6}|{h['lambda'][i]:<9.4f}|{h['val_acc'][i]:<9.4f}|"
                        f"{h['best_val_acc'][i]:<9.4f}|{h['loss_em'][i]:<9.4f}|{h['loss_dom'][i]:<9.4f}\n")
            f.write("\n" + "=" * 60 + "\n\n")

    print(f"\n>>> Log saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="SEED-V DoGKAN-DANN Training")
    parser.add_argument("--sessions", nargs="+", type=int, default=[2])
    parser.add_argument("--subjects", nargs="+", type=str, default=None)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    print(f"Loading SEED-V (Sessions: {args.sessions})")
    X, y, persons, trials = load_seed_v(DATA_ROOT, session_ids=args.sessions)
    print(f"Data: X={X.shape}, y={y.shape}")

    A = None
    if os.path.exists(LOCS_PATH):
        _, coords = read_locs_to_3d(LOCS_PATH)
        A = build_adjacency_matrix(coords)
        print("Adjacency matrix built.")

    tmp = EEG_Transformer_KAN(num_nodes=62, in_channels=5, num_classes=NUM_CLASSES, adj=A)
    print(f"MODEL: DoGKAN | Params: {count_parameters(tmp):,}\n")
    del tmp

    print("Starting LOSO Training")
    results, summary = train_loso_dann_all_subjects(X, y, persons, adj=A, subject_list=args.subjects)
    save_training_log(results, summary, args.sessions, RANDOM_SEED)
    print("\nTraining Completed")


if __name__ == "__main__":
    main()
