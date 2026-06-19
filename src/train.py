"""
train.py — Training pipeline for AutoClaim AI.

Usage:
    python -m src.train                        # train custom CNN only
    python -m src.train --model transfer       # train transfer model only
    python -m src.train --model both           # train both sequentially

Steps:
  1. Prepare processed dataset folder structure (if not already done).
  2. Build tf.data.Dataset pipelines with augmentation on train only.
  3. Build and compile the selected model(s).
  4. Train with: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
  5. Save best model weights and training history.
  6. Plot and save loss/accuracy curves.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_utils import compute_class_weights, get_datasets, prepare_dataset
from src.model import build_custom_cnn, build_transfer_cnn, unfreeze_top_layers


# ─── Callbacks ────────────────────────────────────────────────────────────────

def make_callbacks(model_path: str, monitor: str = "val_loss") -> list:
    """
    1. EarlyStopping (class-aligned) — stops when val_loss stops improving.
    2. ModelCheckpoint (class-aligned) — saves model only on improvement.
    3. ReduceLROnPlateau (BEYOND CLASS MATERIAL) — halves LR on plateau.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        # Beyond class material
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def save_history(history, name: str):
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    path = os.path.join(config.METRICS_DIR, f"history_{name}.json")
    with open(path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"History saved → {path}")


def plot_history(history, name: str):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"],     label="train loss")
    ax1.plot(history.history["val_loss"], label="val loss")
    ax1.set_title(f"{name} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"],     label="train accuracy")
    ax2.plot(history.history["val_accuracy"], label="val accuracy")
    ax2.set_title(f"{name} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.FIG_DIR, f"training_curves_{name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


# ─── Training routines ────────────────────────────────────────────────────────

def train_custom(train_ds, val_ds, num_classes: int):
    print("\n" + "=" * 60)
    print("PHASE 1 — Custom CNN (class-aligned)")
    print("=" * 60)

    model = build_custom_cnn(num_classes)
    model.summary()

    # sparse_categorical_crossentropy matches label_mode='int' (integer labels).
    # categorical_crossentropy is for one-hot labels — using the wrong one
    # caused the previous training run to be stuck at random performance.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(train_ds)
    callbacks = make_callbacks(config.CUSTOM_MODEL_PATH)

    history = model.fit(
        train_ds,
        epochs=config.EPOCHS_CUSTOM,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    save_history(history, "custom")
    plot_history(history, "custom")

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nCustom CNN — val loss: {loss:.4f} | val accuracy: {acc:.4f}")
    return model, history


def train_transfer(train_ds, val_ds, num_classes: int):
    """
    BEYOND CLASS MATERIAL: Transfer learning — two-phase training.

    Phase A: frozen base, train head.
    Phase B: unfreeze top-30 layers, fine-tune with LR/10.
    """
    print("\n" + "=" * 60)
    print("PHASE 2 — Transfer Learning: MobileNetV2 (Beyond class material)")
    print("=" * 60)

    model = build_transfer_cnn(num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(train_ds)
    callbacks_a = make_callbacks(config.TRANSFER_MODEL_PATH)

    print("\n[Phase A] Training classification head (base frozen) …")
    history_a = model.fit(
        train_ds,
        epochs=config.EPOCHS_TRANSFER,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_a,
        verbose=1,
    )

    print("\n[Phase B] Fine-tuning top 30 base layers …")
    model = unfreeze_top_layers(model, n_layers=30)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE / 10
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_b = make_callbacks(config.TRANSFER_MODEL_PATH)
    history_b = model.fit(
        train_ds,
        epochs=20,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_b,
        verbose=1,
    )

    # Merge histories for plotting
    merged = {
        k: history_a.history[k] + history_b.history[k]
        for k in history_a.history
    }

    class _MergedHistory:
        def __init__(self, h):
            self.history = h

    merged_history = _MergedHistory(merged)
    save_history(merged_history, "transfer")
    plot_history(merged_history, "transfer")

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nTransfer CNN — val loss: {loss:.4f} | val accuracy: {acc:.4f}")
    return model, merged_history


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoClaim AI Training")
    parser.add_argument(
        "--model",
        choices=["custom", "transfer", "both"],
        default="custom",
    )
    parser.add_argument("--force-prep", action="store_true")
    args = parser.parse_args()

    import tensorflow as tf
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("=" * 60)
    print("Preparing dataset …")
    summary = prepare_dataset(force=args.force_prep)

    with open(config.CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"\n{num_classes} classes: {class_names}")

    train_ds, val_ds, _ = get_datasets()

    if args.model in ("custom", "both"):
        train_custom(train_ds, val_ds, num_classes)

    if args.model in ("transfer", "both"):
        train_ds, val_ds, _ = get_datasets()
        train_transfer(train_ds, val_ds, num_classes)

    print("\nTraining complete. Run `python -m src.evaluate` for metrics.")


if __name__ == "__main__":
    main()
