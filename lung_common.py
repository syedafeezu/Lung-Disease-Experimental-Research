"""Shared configuration, data pipeline, model, loss and Grad-CAM helpers
for the Automated Lung Disease Classification project.

Model: DenseNet121 (ImageNet) + Squeeze-and-Excitation attention head.
"""
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
SEED = 42


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------
def setup_precision():
    """Use mixed precision on GPU (faster); plain float32 on CPU."""
    from tensorflow.keras import mixed_precision

    if tf.config.list_physical_devices("GPU"):
        mixed_precision.set_global_policy("mixed_float16")
        print("[info] GPU found -> mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")
        print("[info] No GPU found -> float32 (training will be slow; use Colab GPU)")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def collect_images(data_dir):
    """Return (paths, labels) for all four classes."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(data_dir, cls, "images")
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected folder not found: {folder}\n"
                f"Check --data_dir. It must contain: {CLASSES}, each with an 'images' sub-folder."
            )
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, f))
                labels.append(idx)
    return np.array(paths), np.array(labels)


def get_splits(data_dir):
    """Stratified 70/15/15 split. Deterministic (fixed SEED), so evaluate.py
    and gradcam.py always see exactly the same test set used by train.py."""
    paths, labels = collect_images(data_dir)
    tr_p, tmp_p, tr_y, tmp_y = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    va_p, te_p, va_y, te_y = train_test_split(
        tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=SEED)
    return (tr_p, tr_y), (va_p, va_y), (te_p, te_y)


def load_and_enhance(path):
    """Read X-ray -> resize -> CLAHE -> 3 channels -> DenseNet normalisation."""
    if isinstance(path, bytes):
        path = path.decode()
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = np.stack([img] * 3, axis=-1).astype("float32")
    return preprocess_input(img)


def make_dataset(paths, labels, batch_size=32, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.numpy_function(load_and_enhance, [path], tf.float32)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img, tf.one_hot(label, NUM_CLASSES)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="lung")
class FocalLoss(tf.keras.losses.Loss):
    """Categorical focal loss: FL = -(1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma=2.0, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(tf.pow(1.0 - y_pred, self.gamma) * ce, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma})
        return cfg


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def se_block(x, ratio=16):
    """Squeeze-and-Excitation channel attention."""
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(ch // ratio, activation="relu")(s)
    s = layers.Dense(ch, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])


def build_model():
    """Return (model, base, head). Layer names are fixed so Grad-CAM can
    retrieve 'densenet121' and 'se_head' from a saved model."""
    base = DenseNet121(include_top=False, weights="imagenet",
                       input_shape=(IMG_SIZE, IMG_SIZE, 3), name="densenet121")
    base.trainable = False

    feat_in = layers.Input(tuple(base.output.shape[1:]))      # 7 x 7 x 1024
    x = se_block(feat_in)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    head = models.Model(feat_in, out, name="se_head")

    # active only during training; no horizontal flip (unrealistic for X-rays)
    augment = models.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    out = head(base(augment(inp), training=False))
    model = models.Model(inp, out, name="DenseNet121_SE")
    return model, base, head


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
def grad_cam(model, img_batch):
    """Grad-CAM on the last DenseNet feature map (7x7x1024).
    Returns (cam in [0,1], predicted class index, confidence, all probs)."""
    base = model.get_layer("densenet121")
    head = model.get_layer("se_head")
    img_batch = tf.convert_to_tensor(img_batch)
    with tf.GradientTape() as tape:
        conv_out = base(img_batch, training=False)
        tape.watch(conv_out)
        preds = head(conv_out)
        class_idx = tf.argmax(preds[0])
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0]
    cam = tf.cast(cam, tf.float32).numpy()
    cam = cam / (cam.max() + 1e-8)
    probs = preds[0].numpy().astype("float64")
    return cam, int(class_idx), float(probs[int(class_idx)]), probs


def gradcam_overlay(model, image_path):
    """Return (gray image, RGB overlay, class_idx, confidence, probs)."""
    x = load_and_enhance(image_path)[None, ...]
    cam, idx, conf, probs = grad_cam(model, x)
    gray = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap(
        np.uint8(255 * cv2.resize(cam, (IMG_SIZE, IMG_SIZE))), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.6, heat, 0.4, 0)
    return gray, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), idx, conf, probs
