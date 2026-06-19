"""
data_utils.py — Dataset preparation and data-pipeline helpers.

The CarDD dataset ships in COCO instance-segmentation format.
We convert it to a folder-per-class layout compatible with
keras.utils.image_dataset_from_directory.

KEY KERAS 3 DESIGN DECISION
  Rescaling (1/255) and data augmentation live INSIDE the model, not in
  the dataset pipeline.  This is the recommended Keras 3 pattern because:
    • Augmentation layers in tf.data.map() with AUTOTUNE parallelism are
      unreliable in Keras 3 graph mode (traced as static ops, lose randomness).
    • class_weight only works correctly with INTEGER (sparse) labels in Keras 3;
      one-hot labels cause incorrect weight application and collapsed training.
  The dataset therefore returns raw uint8→float32 [0, 255] images with
  integer class labels.  The model applies Rescaling then augmentation as
  its first layers (skipped automatically at inference via training=False).
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

import keras
import config


# ─── COCO → folder structure ──────────────────────────────────────────────────

def _load_coco_labels(ann_path: str) -> dict:
    """Return {image_filename: dominant_class_name} from a COCO annotation file."""
    with open(ann_path) as f:
        data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_area = defaultdict(lambda: defaultdict(float))
    for ann in data["annotations"]:
        img_area[ann["image_id"]][ann["category_id"]] += ann["area"]
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    labels = {}
    for img_id, areas in img_area.items():
        dominant_cat_id = max(areas, key=areas.get)
        labels[id_to_filename[img_id]] = cat_map[dominant_cat_id]
    return labels


def _safe_class_folder(class_name: str) -> str:
    return class_name.replace(" ", "_")


def prepare_dataset(force: bool = False) -> dict:
    """Copy CarDD images into data/processed/{split}/{class}/ folders."""
    proc = Path(config.PROC_DIR)

    if proc.exists() and not force:
        splits_ok = all(
            (proc / s).exists() and len(list((proc / s).iterdir())) > 0
            for s in ("train", "val", "test")
        )
        if splits_ok:
            print("Processed dataset already exists. Pass force=True to rebuild.")
            return _count_processed()

    if proc.exists():
        shutil.rmtree(proc)
    proc.mkdir(parents=True)

    split_cfg = [
        ("train", config.COCO_TRAIN_DIR,
         os.path.join(config.COCO_ANN_DIR, "instances_train2017.json")),
        ("val",   config.COCO_VAL_DIR,
         os.path.join(config.COCO_ANN_DIR, "instances_val2017.json")),
        ("test",  config.COCO_TEST_DIR,
         os.path.join(config.COCO_ANN_DIR, "instances_test2017.json")),
    ]

    summary = {}
    for split_name, img_dir, ann_path in split_cfg:
        print(f"\nPreparing {split_name} split …")
        labels = _load_coco_labels(ann_path)
        split_counts = defaultdict(int)
        skipped = 0
        for filename, class_name in labels.items():
            src = os.path.join(img_dir, filename)
            if not os.path.exists(src):
                skipped += 1
                continue
            dst_dir = proc / split_name / _safe_class_folder(class_name)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / filename
            if not dst.exists():
                shutil.copy2(src, dst)
            split_counts[class_name] += 1
        if skipped:
            print(f"  Skipped {skipped} images (file not found).")
        for cls, cnt in sorted(split_counts.items()):
            print(f"  {cls:>14}: {cnt:>4} images")
        summary[split_name] = dict(split_counts)

    print("\nDataset preparation complete.")
    _save_class_names()
    return summary


def _count_processed() -> dict:
    summary = {}
    for split in ("train", "val", "test"):
        split_dir = Path(config.PROC_DIR) / split
        counts = {}
        if split_dir.exists():
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    counts[cls_dir.name.replace("_", " ")] = len(list(cls_dir.glob("*.jpg")))
        summary[split] = counts
    return summary


def _save_class_names():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    train_dir = Path(config.PROC_DIR) / "train"
    class_names = sorted(
        d.name.replace("_", " ") for d in train_dir.iterdir() if d.is_dir()
    )
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved → {config.CLASS_NAMES_PATH}")
    print(f"Classes: {class_names}")
    return class_names


# ─── Image quality check ──────────────────────────────────────────────────────

def check_corrupted_images(root_dir=None) -> list:
    if root_dir is None:
        root_dir = config.PROC_DIR
    bad = []
    for path in Path(root_dir).rglob("*.jpg"):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            bad.append(str(path))
    print(f"Corrupted / unreadable images found: {len(bad)}")
    return bad


# ─── Dataset loaders ──────────────────────────────────────────────────────────

def get_datasets(
    img_size: tuple = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
):
    """
    Return (train_ds, val_ds, test_ds) as tf.data.Dataset objects.

    Labels are INTEGER class indices (label_mode='int' = sparse labels).
    Images are float32 in [0, 255] — rescaling is the model's responsibility.

    Why label_mode='int' instead of 'categorical' (one-hot):
        Keras 3's class_weight parameter requires integer labels to correctly
        map per-sample weights.  With one-hot labels, class_weight fails silently,
        applying incorrect (or all-zero) gradients and causing the loss to stay
        stuck at ln(num_classes) ≈ 1.79 throughout training.

    Why no augmentation here:
        Keras 3 augmentation layers inside tf.data.map() with AUTOTUNE are
        compiled as static TF graphs and lose proper random seeding.  The
        recommended pattern is to place augmentation as the first layers of
        the model itself (see model.py), where training=True/False controls
        whether augmentation is applied.
    """
    import tensorflow as tf

    train_ds = (
        keras.utils.image_dataset_from_directory(
            os.path.join(config.PROC_DIR, "train"),
            image_size=img_size,
            batch_size=batch_size,
            label_mode="int",    # sparse integer labels
            shuffle=True,
            seed=config.RANDOM_SEED,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        keras.utils.image_dataset_from_directory(
            os.path.join(config.PROC_DIR, "val"),
            image_size=img_size,
            batch_size=batch_size,
            label_mode="int",
            shuffle=False,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        keras.utils.image_dataset_from_directory(
            os.path.join(config.PROC_DIR, "test"),
            image_size=img_size,
            batch_size=batch_size,
            label_mode="int",
            shuffle=False,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


get_generators = get_datasets  # backward-compat alias


def get_true_labels(dataset) -> np.ndarray:
    """Collect all integer class labels from a tf.data.Dataset."""
    import tensorflow as tf
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels, axis=0).astype(int)


# ─── Class-weight helper (Beyond class material) ──────────────────────────────

def compute_class_weights(train_ds) -> dict:
    """
    BEYOND CLASS MATERIAL: class weights for imbalanced data.

    Requires integer labels (label_mode='int') — class_weight in Keras 3
    does not work with one-hot (categorical) labels.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = get_true_labels(train_ds)
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=labels
    )
    # Cap weights at 2.5 to prevent oscillation.
    # Full balanced weights give crack a weight of 7.7× vs scratch's 0.5×.
    # Such extreme ratios cause gradient oscillation: each crack batch fires a
    # massive update that undoes what scratch batches just learned, keeping
    # val_loss stuck near ln(6) = 1.79 (random).  Capping at 2.5 still gives
    # rare classes extra attention without destabilising the optimiser.
    MAX_WEIGHT = 2.5
    cw = {int(c): min(float(w), MAX_WEIGHT) for c, w in zip(unique_classes, weights)}

    try:
        with open(config.CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        print("Class weights (capped at 2.5):")
        for idx, w in sorted(cw.items()):
            name = class_names[idx] if idx < len(class_names) else str(idx)
            print(f"  {name:>15}: {w:.3f}")
    except FileNotFoundError:
        print("Class weights:", cw)

    return cw


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_class_distribution(summary: dict, save_path=None):
    splits = list(summary.keys())
    all_classes = sorted({cls for s in summary.values() for cls in s})
    x = np.arange(len(all_classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, split in enumerate(splits):
        counts = [summary[split].get(cls, 0) for cls in all_classes]
        ax.bar(x + i * width, counts, width, label=split)
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_classes, rotation=25, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("CarDD — Image count per class and split")
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


def plot_sample_images(n_per_class: int = 3, save_path=None):
    train_dir = Path(config.PROC_DIR) / "train"
    class_dirs = sorted(train_dir.iterdir())
    n_classes = len(class_dirs)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 3, n_classes * 2.5))
    rng = np.random.default_rng(config.RANDOM_SEED)
    for row, cls_dir in enumerate(class_dirs):
        images = sorted(cls_dir.glob("*.jpg"))
        chosen = rng.choice(images, size=min(n_per_class, len(images)), replace=False)
        class_label = cls_dir.name.replace("_", " ")
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(chosen):
                ax.imshow(Image.open(chosen[col]).convert("RGB"))
            ax.axis("off")
            if col == 0:
                ax.set_title(class_label, fontsize=9, fontweight="bold")
    plt.suptitle("Sample images per damage class (training split)", fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
