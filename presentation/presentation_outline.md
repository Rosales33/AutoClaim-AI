# AutoClaim AI — Presentation Outline
## IE University | Deep Learning Final Project
### Estimated duration: 15 minutes + 5 min Q&A

---

## Slide 1 — Title

**AutoClaim AI**
*CNN-Based Car Damage Classification for Insurance Claims*

- Course: Deep Learning
- University: IE University
- [Team member names]
- Date: [Presentation date]

**Speaker note**: Open with a brief hook — "Every day, an insurance company receives thousands of claim photos. What if a computer could read them in under a second?"

---

## Slide 2 — Business Problem

**The problem: Manual first-review is slow and expensive**

- Insurance adjusters spend 15–30 minutes on first-review of each car damage photo
- High-volume claims create backlogs → slower payouts → unhappy customers
- Human reviewers are inconsistent; fatigue increases error rates
- **Cost**: A mid-sized insurer handling 500 claims/day → ~200 hours/day in manual triage

**Pain points** (use a 2×2 grid):
| | Low Severity | High Severity |
|-|-------------|---------------|
| **Correctly identified** | Fast-track → quick settlement | Priority → protect liability |
| **Misidentified** | Overpayment | Underpayment + legal risk |

**Speaker note**: This is where you establish business relevance. Keep it concrete and relatable.

---

## Slide 3 — Proposed Solution

**AutoClaim AI: A CNN-based first-level triage tool**

Architecture diagram:
```
[Claims Handler]
     ↓ uploads photo
[AutoClaim AI Web App]
     ↓ preprocesses (128×128, RGB, normalised)
[Trained CNN Model]
     ↓ outputs probabilities
[Triage Engine]
     ↓ applies business rules
[Decision] → Fast-Track | Priority | Human Review
```

**Value proposition**:
- Processes an image in < 1 second
- Consistent — same rules every time
- Scalable — handles peak claim volumes
- Augments (not replaces) claims handlers

---

## Slide 4 — Dataset

**CarDD — Car Damage Detection Dataset**

| Split | Images |
|-------|--------|
| Train | 2 816  |
| Val   |   810  |
| Test  |   374  |
| **Total** | **4 000** |

**6 damage categories**:
- dent, scratch (minor — fast-track eligible)
- crack, glass shatter, lamp broken, tire flat (severe — priority eligible)

**Show sample images grid** (from `outputs/figures/sample_images.png`)

**Key challenge**: Dataset is imbalanced — scratch has 15× more images than crack.

**Speaker note**: Show the class distribution chart (`outputs/figures/class_distribution.png`). Explain why imbalance matters.

---

## Slide 5 — Why CNNs?

**Why a Convolutional Neural Network?**

Traditional ML (Random Forest, SVM) on raw pixels:
- ❌ Loses spatial relationships
- ❌ Cannot handle variable-size inputs
- ❌ Doesn't generalise across zoom/rotation

CNN advantages:
- ✅ **Local connectivity**: filters detect edges/textures at any position
- ✅ **Hierarchical features**: edges → textures → damage patterns
- ✅ **Translation invariance**: via MaxPooling
- ✅ **Proven on image tasks**: standard approach in the field

**Visual**: Show a conv filter map — Layer 1 detects edges; Layer 3 detects crack patterns.

---

## Slide 6 — Data Preprocessing & Augmentation

**Preprocessing** (applied to all splits):
1. **Convert to RGB** — ensures consistent 3-channel input
2. **Resize to 128×128** — fixed input shape required by the CNN (chosen over 224×224 to avoid parameter explosion — see Slide 7)
3. **Normalise [0, 1]** — `pixel / 255.0` — stabilises gradient updates

**Augmentation** (training only — class material):
- Random horizontal flip
- Rotation ±15°
- Zoom ±20%
- *(Beyond class material)* Brightness adjustment — simulates sun/shade in parking lots

**Why augmentation on training only?**
Validation and test sets measure real-world performance. Augmenting them would hide true accuracy.

**Show preprocessing pipeline image** (from `outputs/figures/` — notebook 04)

---

## Slide 7 — Custom CNN Architecture

**Class-aligned architecture** — built from scratch

```
Input (128×128×3)
├── Rescaling(1/255)                      →  [0,1] normalised
├── Augmentation (training only)          →  flip, rotation, zoom
├── Block 1: Conv2D(32) + ReLU + MaxPool  →  64×64×32
├── Block 2: Conv2D(64) + ReLU + MaxPool  →  32×32×64
├── Block 3: Conv2D(128) + ReLU + MaxPool →  16×16×128
├── GlobalAveragePooling2D                →  128 values
├── Dense(128) + ReLU
├── Dropout(0.5)
└── Dense(6) + Softmax                    →  6 probabilities
```

**Total: ~110 K parameters** — deliberately small for 2 816 training images.

**Key design decision — GlobalAveragePooling vs Flatten**:
- Flatten after 3 MaxPool on 128×128 → 32 768 values → Dense(256) = **8.4 M parameters**
- With 2 816 training images: ~3 000 params/sample → **guaranteed overfitting** (first run: 17% val acc)
- GlobalAveragePooling: 16×16×128 → **128 values** → 110 K params total (~46/sample)
- Same layer used in the class Transfer Learning notebook

**Each choice explained**:
| Component | What it does |
|-----------|-------------|
| Conv2D + ReLU | Detects visual patterns at each location |
| MaxPooling | Shrinks feature maps, adds translation robustness |
| GlobalAveragePooling2D | Averages each filter channel → avoids parameter explosion |
| Dense + ReLU | Non-linear combination of features |
| Dropout(0.5) | Zeroes 50% of neurons → prevents overfitting |
| Softmax | Converts outputs to probabilities (sum = 1) |

---

## Slide 8 — Transfer Learning (Beyond Class Material)

> **EXTENDS CLASS MATERIAL** — Transfer learning concept was introduced in class with ResNet50. This extends it with MobileNetV2 and two-phase fine-tuning. Dropout and BatchNormalization are covered in class; what goes beyond is the specific backbone and the two-phase training protocol.

**MobileNetV2 pre-trained on ImageNet**:
- 1.28 million training images, 1 000 classes
- Already learned: edges, textures, shapes, objects
- We add a classification head for 6 damage categories

**Two-phase training**:
1. **Phase A** — Freeze base, train head only (fast convergence)
2. **Phase B** — Unfreeze top 30 layers, fine-tune with LR/10

**Why MobileNetV2 over ResNet50**:
- 3.4 M vs 25 M parameters → 7× faster on CPU
- Efficient depthwise convolutions
- Input 128×128 matches our custom CNN (fair comparison)

**Show**: side-by-side summary table (custom CNN vs. transfer CNN parameters)

---

## Slide 9 — Evaluation Results

**Show results table** (from `outputs/metrics/model_metrics.csv`):

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Custom CNN (from scratch) | 41.4% | 0.374 | 0.387 |
| **Transfer MobileNetV2** | **76.5%** | **0.728** | **0.761** |

**+85% relative improvement** — from 41% to 76% — is the transfer learning story.

**Key metrics explanation**:
- **Accuracy**: % of correct predictions overall
- **Macro F1**: average F1 across all 6 classes — penalises rare class failures equally
- **Weighted F1**: F1 weighted by class frequency — reflects real-world distribution

**Show confusion matrix** (`outputs/figures/confusion_matrix_*.png`)

**Speaker note**: Highlight which classes are easiest/hardest and explain why (data volume, visual similarity).

---

## Slide 10 — Overfitting Prevention

**How we prevent overfitting**:

| Technique | Where | Effect |
|-----------|-------|--------|
| Dropout(0.5) | Dense layer | Randomly disables 50% of neurons per batch |
| Early Stopping | Training | Stops when val_loss stops improving |
| Data Augmentation | Training only | Creates synthetic image variety |
| Batch Normalization | Transfer model head | Stabilises layer inputs, speeds training |
| Class Weights* | Loss function | Balances imbalanced classes |
| ReduceLROnPlateau* | Optimiser | Reduces LR when val_loss plateaus |

*Beyond class material (used only in transfer learning model)

**Show**: training curves (`outputs/figures/training_curves_custom.png`)

**Interpretation**: If val_loss curve stays close to train_loss → low overfitting.

---

## Slide 11 — MVP Demo Flow

**Live demo walkthrough**:

```
streamlit run app/streamlit_app.py
```

1. Open browser → `http://localhost:8501`
2. Upload a car damage photo (or pick from demo_images/)
3. Click **Predict**
4. Review:
   - Damage type predicted
   - Confidence % + colour bar
   - Top-3 classes
   - Triage decision with explanation
5. Expand "Raw JSON" for technical output

**Show screenshots or live demo**

---

## Slide 12 — Business Value

**Impact of AutoClaim AI**:

| Metric | Before | After |
|--------|--------|-------|
| First-review time | 15–30 min/claim | < 1 second |
| Consistency | Variable (human fatigue) | Deterministic |
| Scalability | Linear with headcount | Serverless, unlimited |
| Severe claim detection | ~72% caught on first review* | ~[model recall]% |

*industry estimate

**Cost example** (illustrative):
- 500 claims/day × 20 min saved = 167 hours/day
- At €30/hour → **€5 000/day saved** in triage labour

**Customer experience**: faster settlement → higher satisfaction scores

---

## Slide 13 — Limitations

**What this tool cannot do** (transparency builds trust):

1. **Not a final decision** — all outputs require human validation before payment
2. **Category granularity** — `dent` covers everything from a parking-lot scuff to a major collision. The dataset does not distinguish between cosmetic dents and structurally significant ones. The triage system classifies all dents as "minor", which underestimates severity in serious cases. A production system would need finer categories (e.g. `dent-minor` / `dent-severe`) or a separate damage-extent estimator.
3. **Image quality dependency** — blurry, poorly lit, or extreme-angle photos reduce accuracy
4. **Multi-damage complexity** — a single label is assigned; combined damage (e.g., dent + crack) may be misclassified
5. **Class imbalance** — crack F1 = 0.36 due to only 61 training images (15× fewer than scratch)
6. **Domain shift** — trained on CarDD; performance on a different insurer's photo library is unknown until validated
7. **No severity score** — the model predicts damage type, not repair cost or extent

**Speaker note**: Point 2 is a real example from testing — a heavily crushed front end was labelled `dent` by the dataset and routed as "minor" by the triage. Use this as a concrete illustration of dataset limitations.

---

## Slide 14 — Future Improvements

**What comes next**:

1. **Object detection** (YOLOv8) — locate damage regions with bounding boxes
2. **Multi-label classification** — predict all damage types in one image
3. **Severity estimation** — regress repair cost from visual features
4. **Active learning** — continuously improve with new labelled claims
5. **REST API** (FastAPI) — integrate directly into claims management system
6. **Explainability** (Grad-CAM) — highlight which image regions drove the prediction → builds adjuster trust

---

## Slide 15 — Final Takeaway

**AutoClaim AI in one sentence**:
> A CNN-based system that reads car damage photos in under a second and routes each claim to the right processing track — saving hundreds of hours per week and improving customer experience.

**Key technical achievements**:
- ✅ Full CNN pipeline from data → training → evaluation → inference
- ✅ Custom CNN aligned with course material
- ✅ Transfer learning extension (MobileNetV2)
- ✅ Working Streamlit demo
- ✅ Business-ready triage logic with configurable thresholds

**Business case**:
- Solves a real, high-volume problem
- Immediate ROI through labour reduction
- Foundation for a full intelligent claims platform

**Thank you — questions?**

---

*Total slides: 15 | Estimated time: 13 min presentation + 2 min Q&A buffer*
*Demo: 2–3 minutes within slide 11*
