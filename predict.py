"""Predict the class of a single chest X-ray (the 'For Validation' step).

Usage:
    python predict.py covid.png
    python predict.py covid.png --gradcam      # also saves a heatmap overlay
"""
import argparse
import os

import numpy as np
import tensorflow as tf

from lung_common import CLASSES, FocalLoss, gradcam_overlay, load_and_enhance  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=os.path.join("outputs", "lung_disease_densenet_se.keras"))
    ap.add_argument("--gradcam", action="store_true")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    x = load_and_enhance(args.image)[None, ...]
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))

    print("\nPrediction:", CLASSES[i])
    print("Confidence:", round(float(probs[i]) * 100, 2), "%")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:16s}: {p * 100:6.2f} %")

    if args.gradcam:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gray, overlay, idx, conf, _ = gradcam_overlay(model, args.image)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
        ax[1].imshow(overlay); ax[1].set_title(f"{CLASSES[idx]} ({conf * 100:.1f}%)"); ax[1].axis("off")
        out = "prediction_gradcam.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print("Saved heatmap ->", out)


if __name__ == "__main__":
    main()
