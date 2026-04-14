import matplotlib.pyplot as plt


def plot_accuracy_history(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["accuracy"], label="Train Accuracy")
    ax.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
    ax.set_title("Accuracy over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    return fig


def plot_loss_history(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history.get("val_loss", []), label="Validation Loss")
    ax.set_title("Loss over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    return fig
