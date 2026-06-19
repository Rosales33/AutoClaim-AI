# AutoClaim AI — CNN-Based Car Damage Classification

**IE University | Deep Learning Final Project**

> An MVP that classifies car damage from images and generates a first-level insurance triage recommendation using a Convolutional Neural Network.

---

## Business Problem

Insurance companies receive thousands of car damage claim images every day.
Manual first-review is slow, repetitive, and expensive.
AutoClaim AI helps a claims handler upload a car photo and instantly receive:

- Predicted damage type (one of 6 categories)
- Confidence score (0–100%)
- Top-3 probable classes
- Triage decision: **Fast-track / Priority Assessment / Human Review**

---

## Project Structure

```
.
├── config.py                   ← All paths, hyperparameters, thresholds
├── requirements.txt
├── data/
│   └── processed/              ← Auto-generated from CarDD (run train.py)
│       ├── train/{class}/
│       ├── val/{class}/
│       └── test/{class}/
├── CarDD_release/              ← Original CarDD COCO dataset
│   └── CarDD_COCO/
│       ├── annotations/
│       ├── train2017/
│       ├── val2017/
│       └── test2017/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_inference_demo.ipynb
├── src/
│   ├── data_utils.py           ← COCO→folder converter + generators
│   ├── model.py                ← CNN architectures
│   ├── train.py                ← Training script
│   ├── evaluate.py             ← Metrics + figures
│   ├── predict.py              ← Inference backend (used by frontend)
│   └── triage.py               ← Business triage rules
├── app/
│   └── streamlit_app.py        ← Streamlit web frontend
├── models/
│   ├── best_model_custom.keras
│   ├── best_model_transfer.keras
│   └── class_names.json
├── outputs/
│   ├── figures/                ← Confusion matrix, training curves, samples
│   ├── metrics/                ← model_metrics.csv, history JSON
│   └── reports/                ← data_inventory.md, model_evaluation.md
├── demo_images/                ← Sample images for live demo
└── presentation/
    └── presentation_outline.md
```

---

## Dataset

**CarDD** — Car Damage Detection Dataset (COCO format)

| Split | Images | Annotations |
|-------|--------|-------------|
| train | 2 816  | 6 211       |
| val   |   810  | —           |
| test  |   374  | —           |

**Classes** (auto-detected from annotations):

| Class | Severity | Triage default |
|-------|----------|----------------|
| dent | Low | Fast-track (high conf.) |
| scratch | Low | Fast-track (high conf.) |
| crack | **High** | Priority assessment |
| glass shatter | **High** | Priority assessment |
| lamp broken | **High** | Priority assessment |
| tire flat | **High** | Priority assessment |

**Note**: Each image may contain multiple damage types.
We assign the **dominant class** (category with the largest total annotation area)
as the single label — this converts instance segmentation data into a
classification problem suitable for the CNN.

---

## How to Install

```bash
# 1. Clone / navigate to project folder
cd FinalProyect

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Train

```bash
# Train the custom CNN (class-aligned, required)
python -m src.train

# Train the transfer learning model (MobileNetV2, Beyond class material)
python -m src.train --model transfer

# Train both sequentially
python -m src.train --model both

# Force re-preparation of the processed dataset
python -m src.train --force-prep
```

Training outputs:
- `models/best_model_custom.keras`
- `models/best_model_transfer.keras`
- `models/class_names.json`
- `outputs/metrics/history_custom.json`
- `outputs/figures/training_curves_custom.png`

---

## How to Evaluate

```bash
# Evaluate custom CNN on test set
python -m src.evaluate

# Evaluate transfer model
python -m src.evaluate --model transfer

# Evaluate both
python -m src.evaluate --model both
```

Evaluation outputs:
- `outputs/metrics/model_metrics.csv`
- `outputs/figures/confusion_matrix_{name}.png`
- `outputs/figures/per_class_metrics_{name}.png`
- `outputs/figures/sample_predictions_{name}.png`
- `outputs/reports/model_evaluation.md`

---

## How to Run the Frontend

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## How to Run Notebooks

```bash
jupyter notebook notebooks/
```

Recommended order:
1. `01_data_exploration.ipynb` — dataset prep and visualisation
2. `02_cnn_training.ipynb` — train both models
3. `03_model_evaluation.ipynb` — metrics and curves
4. `04_inference_demo.ipynb` — end-to-end inference demo

---

## Example Prediction Output

```json
{
  "predicted_class": "scratch",
  "confidence": 0.87,
  "top_3": [
    {"class": "scratch",      "confidence": 0.87},
    {"class": "dent",         "confidence": 0.09},
    {"class": "crack",        "confidence": 0.03}
  ]
}
```

Triage decision: **🟢 Fast-Track Claim**
> 'scratch' is low-severity damage detected with 87% confidence.
> This claim can proceed through the standard automated pipeline.

---

## Model Architectures

### Custom CNN (Class-Aligned)

```
Input: 128×128×3
├── Rescaling(1/255)                   ← normalisation (= ImageDataGenerator rescale)
├── RandomFlip / RandomRotation / RandomZoom  ← augmentation (training only)
├── Conv2D(32, 3×3) + ReLU
├── MaxPooling(2×2)
├── Conv2D(64, 3×3) + ReLU
├── MaxPooling(2×2)
├── Conv2D(128, 3×3) + ReLU
├── MaxPooling(2×2)
├── GlobalAveragePooling2D             ← replaces Flatten to avoid 8.4M-param explosion
├── Dense(128) + ReLU
├── Dropout(0.5)
└── Dense(6) + Softmax
```

Total parameters: ~110 K. Architecture mirrors the class Simpsons notebook,
extended to 6 output classes. GlobalAveragePooling2D is used instead of Flatten
because Flatten on 128×128 input produces 32 768 values → 8.4 M Dense parameters,
which causes severe overfitting with only 2 816 training images.

### Transfer Learning CNN — MobileNetV2 (Beyond Class Material)

```
Input: 128×128×3
├── Augmentation (training only)
├── mobilenet_v2.preprocess_input      ← [0,255] → [-1,1]
├── MobileNetV2 (ImageNet pre-trained, frozen in phase A)
├── GlobalAveragePooling2D
├── BatchNormalization
├── Dense(256) + ReLU
├── Dropout(0.4)
├── Dense(128) + ReLU
├── Dropout(0.3)
└── Dense(6) + Softmax
```

Phase A: frozen base, train head.
Phase B: unfreeze top 30 base layers, fine-tune with LR/10.

---

## Class-Aligned vs Beyond Class Material

| Feature | Class-Aligned | Beyond Class Material |
|---------|-------------|----------------------|
| Framework | TensorFlow/Keras | — |
| `image_dataset_from_directory` | ✅ | — |
| Conv2D + MaxPooling | ✅ | — |
| Dense + Dropout | ✅ | — |
| Data Augmentation (flip, rotation, zoom) | ✅ | — |
| Batch Normalization | ✅ | — |
| Adam + sparse_categorical_crossentropy | ✅ | — |
| EarlyStopping | ✅ | — |
| ModelCheckpoint | ✅ | — |
| Transfer Learning (concept) | ✅ (ResNet50 in class) | — |
| MobileNetV2 backbone | — | ✅ |
| Two-phase fine-tuning | — | ✅ |
| ReduceLROnPlateau | — | ✅ |
| Class weights for imbalance | — | ✅ |
| COCO annotation parsing | — | ✅ |
| GlobalAveragePooling2D | — | ✅ |

---

## Triage Thresholds (configurable in config.py)

| Rule | Threshold | Decision |
|------|-----------|---------|
| confidence < 60% | any class | Human Review |
| severe class + conf ≥ 70% | crack, glass shatter, lamp broken, tire flat | Priority Assessment |
| minor class + conf ≥ 80% | dent, scratch | Fast-Track |
| all other cases | — | Human Review |

---

## Known Limitations

1. **Domain shift**: Model trained on CarDD; performance may differ on images with unusual camera angles, extreme weather, or non-Western car models.
2. **Multi-damage images**: We assign one dominant label; images with equally severe combined damage may be mislabelled.
3. **Class imbalance**: `crack` has ~15× fewer training images than `scratch`; crack recall may be lower.
4. **Resolution**: Some CarDD images are low-resolution; resize to 128×128 may lose fine detail such as small cracks.
5. **Not a final decision**: This tool is first-level triage only. All recommendations require human review before claim approval or denial.

---

## Future Improvements

- Object detection / segmentation (e.g. YOLOv8) for bounding box localisation
- Severity estimation and repair cost prediction
- Multi-label classification (one image → multiple damage types)
- Active learning pipeline for continuous model improvement
- REST API (FastAPI) for production deployment
