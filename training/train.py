import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.utils import load_config, create_image_generators
from models import build_cnn, build_rnn, build_lstm, build_gru

MODEL_BUILDERS = {
    "cnn": build_cnn,
    "rnn": build_rnn,
    "lstm": build_lstm,
    "gru": build_gru,
}


def get_models(num_classes: int):
    return {name: builder(num_classes) for name, builder in MODEL_BUILDERS.items()}


def train_model(name: str, model, train_gen, val_gen, epochs: int, save_dir: str) -> None:
    print(f"Training {name}...")
    try:
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=2
        )
    except PermissionError as e:
        print(f"Permission error during training: {e}")
        print("This might be due to OneDrive file locking. Try moving the dataset outside OneDrive.")
        raise

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = save_dir_path / f"{name}_model.h5"
    model.save(model_path)
    print(f"Saved {name} model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train human activity prediction models.")
    parser.add_argument("--model", choices=list(MODEL_BUILDERS), help="Which model to train.")
    parser.add_argument("--all", action="store_true", help="Train all supported models.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_gen, val_gen, _, class_names = create_image_generators(config)
    num_classes = len(class_names)
    if num_classes <= 0:
        raise ValueError(
            "No classes found in the training dataset. "
            "Add labeled subfolders under data/train before training."
        )
    epochs = args.epochs or config["training"]["epochs"]
    save_dir = config["training"]["save_dir"]

    if args.all or args.model is None:
        for name, builder in MODEL_BUILDERS.items():
            model = builder(num_classes)
            train_model(name, model, train_gen, val_gen, epochs, save_dir)
    else:
        model = MODEL_BUILDERS[args.model](num_classes)
        train_model(args.model, model, train_gen, val_gen, epochs, save_dir)


if __name__ == "__main__":
    main()
