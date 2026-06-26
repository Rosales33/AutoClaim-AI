"""
AutoClaim AI — Central configuration.

All paths, hyperparameters, and thresholds live here so every script
reads from one place and results are reproducible.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
PROC_DIR   = os.path.join(DATA_DIR, "processed")  # folder-per-class copies

COCO_DIR        = os.path.join(BASE_DIR, "CarDD_release", "CarDD_COCO")
COCO_TRAIN_DIR  = os.path.join(COCO_DIR, "train2017")
COCO_VAL_DIR    = os.path.join(COCO_DIR, "val2017")
COCO_TEST_DIR   = os.path.join(COCO_DIR, "test2017")
COCO_ANN_DIR    = os.path.join(COCO_DIR, "annotations")

MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR     = os.path.join(OUTPUTS_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
DEMO_DIR    = os.path.join(BASE_DIR, "demo_images")

CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_custom.keras")
TRANSFER_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_transfer.keras")
CLASS_NAMES_PATH  = os.path.join(MODELS_DIR, "class_names.json")

# ─── Image parameters ─────────────────────────────────────────────────────────
# 128×128: better detail than class's 64×64, fast enough on CPU (~3× faster
# than 224×224), and avoids the parameter explosion that causes ~17% val
# accuracy when Flatten is used after 3 MaxPool blocks on a 224×224 input.
# The custom CNN uses GlobalAveragePooling so image size has minimal impact
# on parameter count; 128 is a practical training speed choice.
IMG_SIZE    = (128, 128)     # height × width fed into the network
IMG_SHAPE   = (128, 128, 3)  # includes colour channels
BATCH_SIZE  = 32
RANDOM_SEED = 42

# ─── Training hyperparameters ─────────────────────────────────────────────────
EPOCHS_CUSTOM   = 50   # max epochs; early stopping will usually stop sooner
EPOCHS_TRANSFER = 30   # transfer learning converges faster

LEARNING_RATE        = 0.001   # Adam default (same as class)
EARLY_STOP_PATIENCE  = 8       # stop if val_loss does not improve for N epochs
LR_REDUCE_PATIENCE   = 4       # halve LR if val_loss stalls for N epochs
LR_REDUCE_FACTOR     = 0.5

# ─── Augmentation (applied ONLY to training split) ────────────────────────────
ROTATION_RANGE   = 15
ZOOM_RANGE       = 0.20
BRIGHTNESS_RANGE = (0.8, 1.2)  # Beyond class material — see data_utils.py

# ─── Class-weight cap (Beyond class material) ─────────────────────────────────
# Full balanced weighting gives 'crack' a weight of 7.7× vs 'scratch' 0.5×.
# Such extremes destabilise training (gradient oscillation).  Cap at 2.5.
CLASS_WEIGHT_MAX = 2.5

# ─── Triage thresholds (configurable) ─────────────────────────────────────────
TRIAGE_HIGH_CONFIDENCE   = 0.80  # minor class (dent/scratch) fast-track threshold
TRIAGE_SEVERE_CONFIDENCE = 0.70  # severe class priority threshold

# Damage categories considered severe (high-repair / safety-critical)
SEVERE_CLASSES = {"glass shatter", "tire flat", "lamp broken", "crack"}

# ─── Class map (auto-populated from COCO, kept here for reference) ─────────────
# 1: dent | 2: scratch | 3: crack | 4: glass shatter | 5: lamp broken | 6: tire flat
