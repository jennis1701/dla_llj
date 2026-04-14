import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.utils import load_config

st.set_page_config(page_title="Human Activity Prediction", page_icon="🏃", layout="wide")

config = load_config("config/config.yaml")
save_dir = Path(config["training"]["save_dir"])

st.title("Human Activity Prediction")
st.markdown(
    "Welcome to the human activity prediction app. Train one of the available models, then use the Prediction page to classify a new image or the Confusion Matrix page to evaluate model performance."
)

st.sidebar.header("Quick links")
st.sidebar.write("- [Model Comparison](app/pages/1_Model_Comparison.py)")
st.sidebar.write("- [Prediction](app/pages/2_Prediction.py)")
st.sidebar.write("- [Confusion Matrix](app/pages/3_Confusion_Matrix.py)")

if save_dir.exists():
    model_files = sorted(save_dir.glob("*_model.h5"))
    if model_files:
        st.success(f"Saved models detected: {', '.join(m.name for m in model_files)}")
    else:
        st.info("No saved models found yet. Train a model using `python training/train.py --all`.")
else:
    st.info("No saved model directory found. A `saved_models` folder will be created when you train your first model.")

st.markdown("---")
st.subheader("How to use this project")
cols = st.columns(3)
cols[0].markdown("**1. Prepare data**\nPlace labeled images in `data/train/<class_name>` and `data/test/<class_name>`.")
cols[1].markdown("**2. Train models**\nRun `python training/train.py --all` or select a specific model.")
cols[2].markdown("**3. Predict and evaluate**\nUse the Streamlit pages for prediction and confusion matrix analysis.")
