"""
model.py — CNN architecture definitions.

KERAS 3 AUGMENTATION PATTERN
  Data augmentation and pixel rescaling are the FIRST LAYERS of each model.
  This means:
    • model(x, training=True)  → augmentation + rescaling applied  (training)
    • model(x, training=False) → only rescaling applied             (inference)
  Images enter the model as float32 in [0, 255] from the dataset loader.

Two models are provided:
  1. Custom CNN (class-aligned) — built from scratch
  2. Transfer Learning CNN (Beyond class material) — MobileNetV2
"""

import keras
from keras import layers
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Rescaling,
)
from keras.models import Model, Sequential

import config


# ─── Shared augmentation block ────────────────────────────────────────────────

def _augmentation_layers() -> list:
    """
    Return the list of augmentation layers added to every model.

    Class-covered augmentations (same parameters as ImageDataGenerator):
        RandomFlip      — horizontal_flip=True
        RandomRotation  — rotation_range=15 → factor = 15/360
        RandomZoom      — zoom_range=0.20

    All three are no-ops when the model is called with training=False,
    so they never affect validation or inference.
    """
    return [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(config.ROTATION_RANGE / 360.0),
        layers.RandomZoom(config.ZOOM_RANGE),
    ]


# ─── Model 1: Custom CNN (class-aligned) ─────────────────────────────────────

def build_custom_cnn(num_classes: int) -> Sequential:
    """
    Three-block convolutional network, aligned with class material.

    First layers (preprocessing + augmentation):
        Rescaling(1/255)  — normalises [0,255] → [0,1], same as
                            ImageDataGenerator(rescale=1./255) in class.
        RandomFlip/Rotation/Zoom — same augmentations as class, but placed
                            inside the model (Keras 3 best practice).

    Conv blocks:
        Conv2D(32)+ReLU → MaxPool(2×2)   → 128×128 → 64×64
        Conv2D(64)+ReLU → MaxPool(2×2)   → 64×64   → 32×32
        Conv2D(128)+ReLU → MaxPool(2×2)  → 32×32   → 16×16

    Classification head:
        GlobalAveragePooling2D  — averages each of 128 filter channels
                                  over the 16×16 spatial grid → 128 values.
                                  Replaces Flatten to avoid parameter explosion
                                  (see note below).
        Dense(128)+ReLU
        Dropout(0.5)            — prevents overfitting (class-aligned)
        Dense(num_classes)+Softmax

    WHY GlobalAveragePooling INSTEAD OF Flatten:
        The class notebook used 64×64 images → 8×8×32 = 2048 values after
        Flatten, giving ~6M Dense parameters.  At 128×128 with 128 filters,
        Flatten gives 16×16×128 = 32,768 → Dense(256) = 8.4M parameters.
        With only ~2,800 training images, this ratio is ~3,000 params/sample:
        guaranteed overfitting.  This caused the first run to achieve 17%
        val accuracy (random chance).

        GlobalAveragePooling compresses 16×16×128 → 128 values.
        Dense(128) then has 128×128 = 16,384 parameters total (~46 params/sample).
        The same layer appears in the class Transfer Learning notebook head.
    """
    model = Sequential(name="custom_cnn")
    model.add(Input(shape=config.IMG_SHAPE))

    # Preprocessing (always active)
    model.add(Rescaling(1.0 / 255))

    # Augmentation (training=True only — skipped at inference)
    for aug_layer in _augmentation_layers():
        model.add(aug_layer)

    # Block 1 — 128×128 → 64×64
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2 — 64×64 → 32×32
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3 — 32×32 → 16×16
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Classification head
    model.add(GlobalAveragePooling2D())   # 16×16×128 → 128
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


# ─── Model 2: Transfer Learning CNN (Beyond class material) ──────────────────

def build_transfer_cnn(num_classes: int) -> Model:
    """
    BEYOND CLASS MATERIAL: Transfer Learning with MobileNetV2.

    The class notebook showed ResNet50 on a binary task.
    Here: MobileNetV2 on a 6-class problem — concept is class-covered,
    specific backbone and fine-tuning strategy are extensions.

    MobileNetV2 expects pixels in [-1, 1].
    keras.applications.mobilenet_v2.preprocess_input handles this mapping
    from [0, 255] → [-1, 1].

    Architecture:
        Augmentation (training only, same as custom CNN)
        preprocess_input ([0,255] → [-1,1])
        MobileNetV2 (frozen in phase A, partially unfrozen in phase B)
        GlobalAveragePooling2D
        BatchNormalization  ← Beyond class material
        Dense(256)+ReLU
        Dropout(0.4)
        Dense(128)+ReLU
        Dropout(0.3)
        Dense(num_classes)+Softmax
    """
    inputs = Input(shape=config.IMG_SHAPE)

    # Augmentation (training only)
    x = inputs
    for aug_layer in _augmentation_layers():
        x = aug_layer(x)

    # MobileNetV2 preprocessing: [0,255] → [-1,1]
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    base = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=config.IMG_SHAPE,
    )
    base.trainable = False
    x = base(x, training=False)

    x = GlobalAveragePooling2D()(x)
    # BEYOND CLASS MATERIAL: BatchNormalization stabilises activations
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="transfer_mobilenetv2")


def unfreeze_top_layers(model: Model, n_layers: int = 30) -> Model:
    """BEYOND CLASS MATERIAL: Fine-tuning — unfreeze top N base layers."""
    base = model.get_layer("mobilenetv2_1.00_128")
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    total = sum(1 for l in base.layers if l.trainable)
    print(f"Fine-tuning: {total} of {len(base.layers)} base layers unfrozen.")
    return model
