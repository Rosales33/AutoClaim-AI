# AutoClaim AI — Model Evaluation Report
**Dataset**: CarDD  |  **Classes**: crack, dent, glass shatter, lamp broken, scratch, tire flat

---
## custom_cnn

| Metric | Value |
|--------|-------|
| Accuracy       | 0.4144 |
| Macro F1       | 0.3743 |
| Weighted F1    | 0.3874 |
| Macro Precision| 0.3786 |
| Macro Recall   | 0.4565 |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| crack | 0.000 | 0.000 | 0.000 | 8 |
| dent | 0.358 | 0.220 | 0.273 | 109 |
| glass shatter | 0.696 | 0.846 | 0.764 | 65 |
| lamp broken | 0.230 | 0.762 | 0.354 | 42 |
| scratch | 0.511 | 0.197 | 0.284 | 122 |
| tire flat | 0.476 | 0.714 | 0.571 | 28 |

## transfer_mobilenetv2

| Metric | Value |
|--------|-------|
| Accuracy       | 0.7647 |
| Macro F1       | 0.7284 |
| Weighted F1    | 0.7613 |
| Macro Precision| 0.7676 |
| Macro Recall   | 0.7292 |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| crack | 0.667 | 0.250 | 0.364 | 8 |
| dent | 0.704 | 0.743 | 0.723 | 109 |
| glass shatter | 0.901 | 0.985 | 0.941 | 65 |
| lamp broken | 0.579 | 0.786 | 0.667 | 42 |
| scratch | 0.790 | 0.648 | 0.712 | 122 |
| tire flat | 0.964 | 0.964 | 0.964 | 28 |

---

## Business Interpretation

- **Scratch / dent**: easiest (most common, fast-track eligible).
- **Crack / tire flat**: hardest (few samples, visually subtle).
- Confusing *dent* ↔ *scratch* is low-risk (same triage path).
- Confusing *crack* → *scratch* is high-risk: cracks are structural.
- Confidence thresholds in `config.py` route borderline cases to human review.
