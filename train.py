"""Two-stage training of DenseNet121 + SE attention on chest X-rays.

Usage:
    python train.py --data_dir COVID-19_Radiography_Dataset
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, metrics as kmetrics, optimizers

from lung_common import (CLASSES, NUM_CLASSES, SEED, FocalLoss, build_model,
                         get_splits, make_dataset, setup_precision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="COVID-19_Radiography_Dataset")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--batch_size", type=int, default=32, help="use 16 if you run out of memory")
    ap.add_argument("--head_epochs", type=int, default=8)
    ap.add_argument("--ft_epochs", type=int, default=20)
    ap.add_argument("--unfreeze", type=int, default=120, help="number of top DenseNet layers to fine-tune")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    setup_precision()

    # ---------------- data ----------------
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = get_splits(args.data_dir)
    print(f"train / val / test : {len(tr_p)} / {len(va_p)} / {len(te_p)}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:16s} train={np.sum(tr_y == i):5d}  val={np.sum(va_y == i):5d}  test={np.sum(te_y == i):5d}")

    train_ds = make_dataset(tr_p, tr_y, args.batch_size, training=True)
    val_ds = make_dataset(va_p, va_y, args.batch_size)

    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=tr_y)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("class weights:", {CLASSES[i]: round(w, 3) for i, w in class_weight.items()})

    # ---------------- model ----------------
    model, base, head = build_model()
    model.summary()
    metrics = ["accuracy", kmetrics.AUC(name="auc")]

    # ---------------- stage 1: head only ----------------
    print("\n===== STAGE 1: training the attention head (backbone frozen) =====")
    model.compile(optimizer=optimizers.AdamW(1e-3, weight_decay=1e-4),
                  loss=FocalLoss(gamma=2.0), metrics=metrics)
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.head_epochs,
        class_weight=class_weight,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ])

    # ---------------- stage 2: fine-tuning ----------------
    print("\n===== STAGE 2: fine-tuning top DenseNet layers =====")
    base.trainable = True
    for layer in base.layers[:-args.unfreeze]:
        layer.trainable = False
    for layer in base.layers:                       # keep BatchNorm statistics frozen
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    steps = int(np.ceil(len(tr_p) / args.batch_size)) * args.ft_epochs
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-4, decay_steps=steps)
    model.compile(optimizer=optimizers.AdamW(lr, weight_decay=1e-4),
                  loss=FocalLoss(gamma=2.0), metrics=metrics)
    best_path = os.path.join(args.out_dir, "best_lung_model.keras")
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.ft_epochs,
        class_weight=class_weight,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
        ])

    # ---------------- save ----------------
    final_path = os.path.join(args.out_dir, "lung_disease_densenet_se.keras")
    model.save(final_path)
    hist = {"stage1": {k: [float(v) for v in vals] for k, vals in h1.history.items()},
            "stage2": {k: [float(v) for v in vals] for k, vals in h2.history.items()}}
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    best = {k: v[-1] for k, v in h2.history.items()}
    print("\nFinal-epoch metrics (stage 2):", {k: round(v, 4) for k, v in best.items()})
    print(f"Saved model   -> {final_path}")
    print(f"Saved history -> {os.path.join(args.out_dir, 'history.json')}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
