from pathlib import Path

import numpy as np
import pickle
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "digit_model.pkl", "rb"))


model = load_model()

st.title("Digit Recognizer")
st.write(
    "A gradient boosting model (trained on 70,000 MNIST handwritten digits) recognizes a digit you draw. "
    "Draw a single digit (0-9), centered and filling most of the box, then click Predict."
)

canvas = st_canvas(
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict") and canvas.image_data is not None:
    img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L").resize((28, 28))
    pixels = np.array(img, dtype=float).reshape(1, -1) / 255.0

    if pixels.max() < 0.05:
        st.warning("The canvas looks empty. Draw a digit first.")
    else:
        proba = model.predict_proba(pixels)[0]
        pred = int(np.argmax(proba))
        st.success(f"Predicted digit: **{pred}**  ({proba[pred]:.1%} confidence)")
        st.bar_chart(proba)

if st.button("Clear"):
    st.rerun()

st.caption(
    "Model: HistGradientBoosting on 28x28 pixel values (validation accuracy ≈ 97.8% on the MNIST dataset). "
    "Digits drawn off-center, too small, or too thin are harder to recognize, since the model was trained on "
    "centered, normalized digit images."
)
