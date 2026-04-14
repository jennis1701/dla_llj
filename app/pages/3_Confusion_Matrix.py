import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.evaluate import evaluate_model, plot_confusion_matrix
from training.utils import load_config

st.set_page_config(page_title="Confusion Matrix", page_icon="📈")
st.title("Confusion Matrix")

config = load_config("config/config.yaml")
save_dir = Path(config["training"]["save_dir"])
model_files = sorted(save_dir.glob("*_model.h5")) if save_dir.exists() else []

if not model_files:
    st.warning("No saved model files were detected. Train a model first to generate a confusion matrix.")
else:
    model_name = st.selectbox("Choose a model for evaluation", [m.name for m in model_files])
    model_path = save_dir / model_name
    if st.button("Evaluate model"):
        try:
            results = evaluate_model(str(model_path), config)
            st.write("### Accuracy")
            st.metric(label="Test accuracy", value=f"{results['accuracy']:.3f}")
            st.write("### Classification report")
            st.text(results["report"])
            fig = plot_confusion_matrix(results["confusion_matrix"], results["class_names"])
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
