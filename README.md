# Human Activity Prediction

A Streamlit app for human activity image classification built from a Jupyter notebook. The project includes CNN, RNN, LSTM, and GRU model definitions, training utilities, evaluation workflows, and a simple prediction interface.

## Structure

- `app/` — Streamlit frontend and pages
- `models/` — model architecture definitions
- `training/` — data loaders, training CLI, and evaluation
- `config/` — YAML configuration for datasets and training
- `saved_models/` — trained model artifacts
- `data/` — place your training and test images here
- `requirements.txt` — Python dependencies

## Quick start

1. Create the dataset structure:
   - `data/train/<class_name>/...`
   - `data/test/<class_name>/...`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train models:
   ```bash
   python training/train.py --all
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app/main.py
   ```

## Notes

- The app supports image-based activity prediction using saved `.h5` models.
- The `app/pages/` folder provides separate pages for model comparison, prediction, and confusion matrix evaluation.
- Adjust `config/config.yaml` to change dataset folders, image size, and training parameters.
