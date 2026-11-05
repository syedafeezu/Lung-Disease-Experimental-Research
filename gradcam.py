"""Grad-CAM heatmaps: one test-set example per class (for the report).

Usage:
    python gradcam.py --data_dir COVID-19_Radiography_Dataset
Creates outputs/gradcam_<class>.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from lung_common import CLASSES, FocalLoss, get_splits, gradcam_overlay  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="COVID-19_Radiography_Dataset")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model", default=None)
    ap.add_argument("--per_class", type=int, default=1)
    args = ap.parse_args()

    model_path = args.model or os.path.join(args.out_dir, "lung_disease_densenet_se.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    _, _, (te_p, te_y) = get_splits(args.data_dir)

    for c, name in enumerate(CLASSES):
        for k, path in enumerate(te_p[te_y == c][: args.per_class]):
            gray, overlay, idx, conf, _ = gradcam_overlay(model, path)
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(gray, cmap="gray"); ax[0].set_title(f"Actual: {name}"); ax[0].axis("off")
            ax[1].imshow(overlay)
            ax[1].set_title(f"Predicted: {CLASSES[idx]} ({conf * 100:.1f}%)"); ax[1].axis("off")
            out = os.path.join(args.out_dir, f"gradcam_{name.replace(' ', '_')}_{k + 1}.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print("saved", out)


if __name__ == "__main__":
    main()
