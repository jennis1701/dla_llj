import streamlit as st
import os
from pathlib import Path
import sys
import tensorflow as tf

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.components.uploader import upload_image, preprocess_image
from training.utils import load_config, create_image_generators

st.set_page_config(page_title="Prediction", page_icon="🤖")
st.title("Activity Prediction")

config = load_config("config/config.yaml")
save_dir = Path(config["training"]["save_dir"])
model_files = sorted(save_dir.glob("*_model.h5")) if save_dir.exists() else []

if not model_files:
    st.warning("No saved models found. Train at least one model with `python training/train.py --all`.")
else:
    model_name = st.selectbox("Select a saved model", [m.name for m in model_files])
    model_path = save_dir / model_name
    uploaded_image = upload_image()

    if uploaded_image is not None and st.button("Predict Activity"):
        try:
            model = tf.keras.models.load_model(model_path)
            preprocessed = preprocess_image(uploaded_image, target_size=(config["dataset"]["image_size"], config["dataset"]["image_size"]))
            class_names = create_image_generators(config)[3]
            prediction = model.predict(preprocessed)
            predicted_index = int(tf.argmax(prediction[0]).numpy())
            st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
            st.success(f"Predicted activity: **{class_names[predicted_index]}**")
            st.write(f"Confidence: {float(prediction[0][predicted_index]):.2f}")
        except Exception as exc:
            st.error(f"Failed to run prediction: {exc}")
