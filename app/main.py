import streamlit as st
from pathlib import Path
import sys

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Human Activity Prediction",
    page_icon="🏃",
    layout="wide"
)

# ✅ Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# ✅ Load config safely
config = {"training": {"save_dir": "models"}}

try:
    from training.utils import load_config
    config = load_config("config/config.yaml")
except Exception as e:
    st.warning(f"Using default config. Error: {e}")

# ✅ UI
st.title("🏃 Human Activity Prediction")

st.sidebar.success("Use the sidebar to navigate between pages")

# ✅ Always show something (prevents blank screen)
st.write("Welcome to the Human Activity Prediction App!")

# ✅ Model check
save_dir = Path(config["training"]["save_dir"])

if save_dir.exists():
    model_files = list(save_dir.glob("*_model.h5"))
    if model_files:
        st.success(f"Saved models: {', '.join(m.name for m in model_files)}")
    else:
        st.info("No trained models found yet.")
else:
    st.info("Model directory not found. Train a model first.")

st.markdown("---")

st.subheader("📌 Instructions")
st.write("""
1. Train models using your training script  
2. Go to **Prediction page** to test images  
3. Use **Confusion Matrix page** for evaluation  
""")
