"""
predict.py — Inference backend for AutoClaim AI.

This module is the SINGLE source of prediction logic.
Both the CLI and the Streamlit frontend import from here.

Public API:
    load_model(model_path)           → keras.Model
    load_class_names()               → list[str]
    preprocess_image(source)         → np.ndarray  (1, 224, 224, 3)
    predict_image(model, source)     → dict
    predict_top_k(model, img_arr, k) → dict

Example output:
    {
        "predicted_class": "scratch",
        "confidence": 0.87,
        "top_3": [
            {"class": "scratch",  "confidence": 0.87},
            {"class": "dent",     "confidence": 0.09},
            {"class": "crack",    "confidence": 0.03}
        ]
    }
"""

import json
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


_model_cache: dict = {}


def load_model(model_path: str = None) -> keras.Model:
    """Load and cache a Keras model.  Auto-selects best available if no path given."""
    if model_path is None:
        if os.path.exists(config.TRANSFER_MODEL_PATH):
            model_path = config.TRANSFER_MODEL_PATH
        else:
            model_path = config.CUSTOM_MODEL_PATH

    if model_path not in _model_cache:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `python -m src.train` first."
            )
        print(f"Loading model from {model_path} …")
        _model_cache[model_path] = keras.models.load_model(model_path)
        print("Model loaded.")
    return _model_cache[model_path]


def load_class_names() -> list:
    if not os.path.exists(config.CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"class_names.json not found at {config.CLASS_NAMES_PATH}. "
            "Run `python -m src.train` first."
        )
    with open(config.CLASS_NAMES_PATH) as f:
        return json.load(f)


def preprocess_image(source: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Convert any image source into (1, H, W, 3) float32 in [0, 255].

    Steps:
      1. Open as PIL.Image in RGB — ensures 3 channels regardless of source.
      2. Resize to config.IMG_SIZE (128×128) — CNNs require fixed input shape.
      3. Keep raw [0, 255] — the model's Rescaling(1/255) layer handles this.
      4. Expand batch dimension — Keras expects (batch, H, W, C).
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    elif isinstance(source, Image.Image):
        img = source
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    img = img.convert("RGB")
    img = img.resize(config.IMG_SIZE)
    # Do NOT divide by 255 — the model contains a Rescaling(1/255) layer
    # as its first layer and expects raw [0, 255] float32 input.
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_top_k(
    model: keras.Model,
    img_array: np.ndarray,
    k: int = 3,
) -> dict:
    """Run inference and return top-k predictions."""
    class_names = load_class_names()
    probs = model.predict(img_array, verbose=0)[0]
    top_k_idx = np.argsort(probs)[::-1][:k]

    top_k = [
        {"class": class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in top_k_idx
    ]

    return {
        "predicted_class": top_k[0]["class"],
        "confidence":      top_k[0]["confidence"],
        "top_3":           top_k,
    }


def predict_image(
    model: keras.Model,
    source: Union[str, Path, Image.Image, np.ndarray],
) -> dict:
    """End-to-end prediction from any image source."""
    img_array = preprocess_image(source)
    return predict_top_k(model, img_array, k=3)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AutoClaim AI — predict a single image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model", default=None, help="Path to saved model (optional)")
    args = parser.parse_args()

    from src.triage import get_triage_decision

    model = load_model(args.model)
    result = predict_image(model, args.image_path)
    triage = get_triage_decision(result["predicted_class"], result["confidence"])

    print(f"\n{'='*50}")
    print(f"Image         : {args.image_path}")
    print(f"Prediction    : {result['predicted_class']}")
    print(f"Confidence    : {result['confidence']:.1%}")
    print(f"\nTop-3 predictions:")
    for i, t in enumerate(result["top_3"], 1):
        print(f"  {i}. {t['class']:>14} — {t['confidence']:.1%}")
    print(f"\nTriage decision : {triage['decision']}")
    print(f"Reason          : {triage['reason']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
