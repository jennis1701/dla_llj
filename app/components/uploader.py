from io import BytesIO
from PIL import Image
import numpy as np
import streamlit as st


def upload_image(label: str = "Upload an activity image"):
    uploaded_file = st.file_uploader(label, type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        return None
    try:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
        return image
    except Exception:
        st.error("Unable to read the uploaded image. Please upload a valid PNG/JPG file.")
        return None


def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    image = image.resize(target_size)
    image_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image_array, axis=0)
