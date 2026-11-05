"""Evaluate the trained model on the held-out TEST set and make report figures.

Usage:
    python evaluate.py --data_dir COVID-19_Radiography_Dataset
Creates in outputs/: classification_report.txt, metrics.json,
confusion_matrix.png, training_curves.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)

from lung_common import CLASSES, FocalLoss, get_splits, make_dataset  # noqa: F401 (registers FocalLoss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="COVID-19_Radiography_Dataset")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model", default=None, help="default: <out_dir>/lung_disease_densenet_se.keras")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    model_path = args.model or os.path.join(args.out_dir, "lung_disease_densenet_se.keras")
    model = tf.keras.models.load_model(model_path, compile=False)

    _, _, (te_p, te_y) = get_splits(args.data_dir)
    test_ds = make_dataset(te_p, te_y, args.batch_size)
    print(f"Test images: {len(te_p)}")

    probs = model.predict(test_ds).astype("float64")
    probs /= probs.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(te_y, y_pred)
    macro_f1 = f1_score(te_y, y_pred, average="macro")
    auc = roc_auc_score(te_y, probs, multi_class="ovr", average="macro")
    report = classification_report(te_y, y_pred, target_names=CLASSES, digits=4)
    print(report)
    print(f"Test accuracy : {acc:.4f}\nMacro F1      : {macro_f1:.4f}\nROC-AUC (OvR) : {auc:.4f}")

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report + f"\nAccuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\nROC-AUC: {auc:.4f}\n")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"test_images": int(len(te_p)), "accuracy": acc,
                   "macro_f1": macro_f1, "roc_auc_macro_ovr": auc}, f, indent=2)

    # confusion matrix (counts + row-normalised)
    cm = confusion_matrix(te_y, y_pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, ax=ax[0])
    ax[0].set_title("Confusion Matrix (counts)")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, ax=ax[1], vmin=0, vmax=1)
    ax[1].set_title("Confusion Matrix (row-normalised)")
    for a in ax:
        a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # training curves (stage 2 = fine-tuning)
    hpath = os.path.join(args.out_dir, "history.json")
    if os.path.exists(hpath):
        h = json.load(open(hpath))["stage2"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, k, t in zip(ax, ["accuracy", "loss"], ["Accuracy", "Loss"]):
            a.plot(h[k], label="train")
            a.plot(h["val_" + k], label="validation")
            a.set_title(t + " (fine-tuning stage)"); a.set_xlabel("Epoch"); a.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=200)
        plt.close()
    print(f"\nSaved figures + metrics in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
