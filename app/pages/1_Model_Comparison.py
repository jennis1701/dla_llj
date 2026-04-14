import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.utils import load_config, create_image_generators


st.set_page_config(page_title="Model Comparison", page_icon="📊")
st.title("Model Comparison")

config = load_config("config/config.yaml")
train_dir = Path(config["dataset"]["train_dir"])

if not train_dir.exists():
    st.warning("The training data folder does not exist yet. Place your images under the data/train directory, organized by class.")
else:
    st.markdown("### Dataset overview")
    st.write(f"Training data directory: `{train_dir}`")
    try:
        train_gen, _, _, class_names = create_image_generators(config)
        st.write(f"Detected classes: **{', '.join(class_names)}**")
        st.write(f"Images per class: {train_gen.class_indices}")
    except Exception as exc:
        st.error(f"Unable to build dataset preview: {exc}")

st.markdown("---")
st.markdown(
    "This app includes four model architectures: **CNN (MobileNetV2 transfer learning)**, **RNN**, **LSTM**, and **GRU**. "
    "Use the training script to fit models on your dataset and then compare saved model performance in the Confusion Matrix page."
)

st.markdown("### Training command")
st.code("python training/train.py --all", language="bash")
st.markdown("or train a single model:")
st.code("python training/train.py --model cnn --epochs 35", language="bash")
