import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

# ---------------- CONFIG ----------------
MODEL_PATH = "newwwwwaste_model.keras"
CLASS_JSON = "newwwwclass_indices.json"
IMG_SIZE = 50  # model input size

# ---------------- LOAD CLASS MAPPING ----------------
with open(CLASS_JSON, "r") as f:
    class_map = json.load(f)
idx_to_class = {v: k for k, v in class_map.items()}


# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=True)
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model


model = load_trained_model()

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(
    page_title="ASL Alphabet Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Settings")
show_probs = st.sidebar.checkbox("Show all class probabilities", value=False)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: gray;'>Made by Joel & Deepali</p>", unsafe_allow_html=True)

# ---------------- MAIN HEADER ----------------
st.markdown(
    "<h1 style='text-align: center; color: #4B0082;'>American Sign Language Alphabet Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Upload an image of a hand sign and get instant predictions!</p>",
    unsafe_allow_html=True
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------------- PREDICTION ----------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        img_array = preprocess_image(img)
        preds = model.predict(img_array, verbose=0)[0]
        class_index = int(np.argmax(preds))
        confidence = float(preds[class_index])

    st.success(f"Predicted Letter: **{idx_to_class[class_index]}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    if show_probs:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("All Class Probabilities")
        prob_dict = {idx_to_class[i]: f"{preds[i] * 100:.2f}%" for i in range(len(preds))}
        st.table(prob_dict)

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color: gray;'>Made with ❤️ by Joel & Deepali</p>",
    unsafe_allow_html=True
)
