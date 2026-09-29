from pathlib import Path

import numpy as np
import pickle
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Digit Recognizer")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "digit_model.pkl", "rb"))


model = load_model()

st.title("Digit Recognizer")
st.write(
    "A gradient boosting model (trained on 70,000 MNIST handwritten digits) recognizes a handwritten digit "
    "(0-9). Write a single digit clearly on paper, or in a paint app, then upload a photo or take one with "
    "your camera."
)

source = st.radio("Image source", ["Upload a photo", "Use my camera"], horizontal=True)
file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"]) if source == "Upload a photo" else st.camera_input("Take a photo")

invert = st.checkbox(
    "Dark digit on a light background",
    value=True,
    help="Leave this on for pen/pencil on paper. Turn it off if you drew a white digit on a black background.",
)

if file is not None:
    img = Image.open(file).convert("L")
    st.image(img, caption="Original", width=150)

    small = img.resize((28, 28))
    if invert:
        small = ImageOps.invert(small)
    pixels = np.array(small, dtype=float).reshape(1, -1) / 255.0

    if st.button("Predict"):
        if pixels.max() < 0.05:
            st.warning("The image looks empty or the digit is too faint. Try a clearer photo.")
        else:
            proba = model.predict_proba(pixels)[0]
            pred = int(np.argmax(proba))
            st.success(f"Predicted digit: **{pred}**  ({proba[pred]:.1%} confidence)")
            st.image(small.resize((140, 140)), caption="What the model sees (28x28)")
            st.bar_chart(proba)

st.caption(
    "Model: HistGradientBoosting on 28x28 pixel values (validation accuracy ≈ 97.8% on the MNIST dataset). "
    "The model expects a single centered digit on a plain background, similar to MNIST. Cluttered backgrounds, "
    "multiple digits, or poor lighting will confuse it."
)
