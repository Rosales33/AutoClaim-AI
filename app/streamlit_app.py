"""
streamlit_app.py — AutoClaim AI frontend.

Run:
    streamlit run app/streamlit_app.py

The frontend is intentionally thin: it calls src/predict.py and src/triage.py
for all logic. No prediction code lives here.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.predict import load_model, predict_image  # noqa: E402 — after sys.path
from src.triage import get_triage_decision, get_triage_explanation  # noqa

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoClaim AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 4px solid #2563eb;
}
.triage-priority   { border-left-color: #dc2626; background: #fef2f2; color: #7f1d1d !important; }
.triage-fast_track { border-left-color: #16a34a; background: #f0fdf4; color: #14532d !important; }
.triage-review     { border-left-color: #d97706; background: #fffbeb; color: #78350f !important; }
.triage-priority h3, .triage-priority p   { color: #7f1d1d !important; }
.triage-fast_track h3, .triage-fast_track p { color: #14532d !important; }
.triage-review h3, .triage-review p       { color: #78350f !important; }
.warn-box {
    background: #fffbeb;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    color: #78350f;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AutoClaim AI")
    st.markdown("**CNN-Based Car Damage Triage**")
    st.markdown("---")
    st.markdown(
        "This tool uses a Convolutional Neural Network trained on the "
        "[CarDD dataset](https://github.com/CarDD-USTC/CarDD-USTC.github.io) "
        "to classify car damage and generate a first-level triage recommendation."
    )
    st.markdown("---")
    st.markdown("**Damage categories:**")
    try:
        with open(config.CLASS_NAMES_PATH) as f:
            classes = json.load(f)
        for c in classes:
            st.markdown(f"• {c.title()}")
    except FileNotFoundError:
        st.warning("class_names.json not found. Train the model first.")
        classes = []

    st.markdown("---")
    st.markdown("**Model settings**")
    model_choice = st.radio(
        "Select model",
        options=["Auto (best available)", "Custom CNN", "Transfer (MobileNetV2)"],
        index=0,
    )
    model_path_map = {
        "Auto (best available)": None,
        "Custom CNN":            config.CUSTOM_MODEL_PATH,
        "Transfer (MobileNetV2)": config.TRANSFER_MODEL_PATH,
    }
    chosen_model_path = model_path_map[model_choice]

    st.markdown("---")
    st.markdown("**Triage thresholds**")
    st.markdown(f"🟢 **Fast-Track** — dent/scratch + conf ≥ {config.TRIAGE_HIGH_CONFIDENCE:.0%}")
    st.markdown(f"🔴 **Priority** — severe damage + conf ≥ {config.TRIAGE_SEVERE_CONFIDENCE:.0%}")
    st.markdown(f"🟡 **Human Review** — dent/scratch + conf < {config.TRIAGE_HIGH_CONFIDENCE:.0%}")
    st.markdown(f"🟡 **Human Review** — severe damage + conf < {config.TRIAGE_SEVERE_CONFIDENCE:.0%}")
    st.caption(
        "Minor = dent, scratch  |  "
        f"Severe = {', '.join(sorted(config.SEVERE_CLASSES))}"
    )

# ─── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_model(path):
    try:
        return load_model(path)
    except FileNotFoundError as e:
        return None

model = get_model(chosen_model_path)

# ─── Main area ────────────────────────────────────────────────────────────────
st.title("AutoClaim AI — Car Damage Triage")
st.markdown(
    "Upload a car damage photo to receive an automated damage classification "
    "and triage recommendation. This tool assists insurance agents with "
    "first-level claim screening — it does **not** replace a full assessment."
)
st.markdown("---")

if model is None:
    st.error(
        "No trained model found. Please run `python -m src.train` first, "
        "then restart this app."
    )
    st.stop()

# ─── Image input ──────────────────────────────────────────────────────────────
col_upload, col_sample = st.columns([3, 2])

with col_upload:
    st.subheader("Upload an image")
    uploaded = st.file_uploader(
        "Choose a car damage image (JPG / PNG / BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
    )

with col_sample:
    st.subheader("Or try a sample image")
    demo_dir = Path(config.DEMO_DIR)
    sample_images = sorted(demo_dir.rglob("*.jpg")) + sorted(demo_dir.rglob("*.png"))
    if sample_images:
        sample_labels = [p.parent.name + "/" + p.name for p in sample_images]
        selected_label = st.selectbox("Pick a sample", ["(none)"] + sample_labels)
        sample_path = None
        if selected_label != "(none)":
            idx = sample_labels.index(selected_label)
            sample_path = sample_images[idx]
    else:
        st.info("No sample images found in demo_images/.")
        sample_path = None

# Determine which image to use
pil_image = None
if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")
elif sample_path is not None:
    pil_image = Image.open(sample_path).convert("RGB")

# ─── Prediction ───────────────────────────────────────────────────────────────
if pil_image is not None:
    st.markdown("---")
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Input Image")
        st.image(pil_image, use_container_width=True)
        st.caption(f"Size: {pil_image.width}×{pil_image.height} px  →  resized to {config.IMG_SIZE[0]}×{config.IMG_SIZE[1]} for inference")

    with right:
        with st.spinner("Running inference…"):
            result = predict_image(model, pil_image)
            triage = get_triage_decision(result["predicted_class"], result["confidence"])
            explanation = get_triage_explanation(triage["level"])

        # ── Primary prediction ──
        st.subheader("Prediction Result")
        conf_pct = result["confidence"] * 100

        pred_col, conf_col = st.columns(2)
        pred_col.metric("Damage Type", result["predicted_class"].title())
        conf_col.metric("Confidence", f"{conf_pct:.1f}%")

        # Confidence bar
        bar_color = (
            "#16a34a" if conf_pct >= 80 else
            "#d97706" if conf_pct >= 60 else
            "#dc2626"
        )
        st.markdown(
            f"<div style='background:#e5e7eb;border-radius:4px;height:12px;'>"
            f"<div style='width:{conf_pct:.0f}%;background:{bar_color};"
            f"height:12px;border-radius:4px;'></div></div>",
            unsafe_allow_html=True
        )
        st.markdown("")

        # ── Top-3 predictions ──
        st.subheader("Top-3 Predictions")
        for i, t in enumerate(result["top_3"]):
            pct = t["confidence"] * 100
            medal = ["🥇", "🥈", "🥉"][i]
            st.markdown(
                f"{medal} **{t['class'].title()}** — {pct:.1f}%  "
                f"<progress value='{pct:.0f}' max='100' "
                f"style='width:180px;accent-color:{bar_color};'></progress>",
                unsafe_allow_html=True
            )

        # ── Triage decision ──
        st.subheader("Triage Recommendation")
        level_class = f"triage-{triage['level']}"
        st.markdown(
            f"<div class='metric-card {level_class}'>"
            f"<h3>{triage['icon']} {triage['decision']}</h3>"
            f"<p>{triage['reason']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.expander("What does this mean?"):
            st.markdown(explanation)

    # ── Disclaimer ──
    st.markdown("---")
    st.markdown(
        "<div class='warn-box'>"
        "<strong>Important:</strong> This tool provides an automated first-level "
        "triage suggestion only. It is <strong>not</strong> a final claim decision. "
        "All recommendations must be reviewed by a qualified claims handler before "
        "any payment or denial is issued. Model accuracy is not 100%. "
        "Image quality, unusual angles, and rare damage combinations may affect results."
        "</div>",
        unsafe_allow_html=True
    )

    # ── Full JSON output (collapsible, for technical review) ──
    with st.expander("Raw prediction output (JSON)"):
        st.json({
            "predicted_class": result["predicted_class"],
            "confidence":      result["confidence"],
            "top_3":           result["top_3"],
            "triage": {
                "decision": triage["decision"],
                "level":    triage["level"],
                "reason":   triage["reason"],
            }
        })

else:
    # Placeholder
    st.info("Upload a car damage image above or select a sample to begin.")
    st.markdown("---")
    st.markdown("### How it works")
    how_cols = st.columns(4)
    steps = [
        ("1. Upload", "Choose a photo of a damaged car"),
        ("2. Classify", "CNN predicts the damage type"),
        ("3. Score", "Confidence score (0–100%)"),
        ("4. Triage", "Fast-track / Priority / Review"),
    ]
    for col, (title, desc) in zip(how_cols, steps):
        col.markdown(f"**{title}**\n\n{desc}")
