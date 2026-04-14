import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(num_classes: int, image_size: int = 224) -> tf.keras.Model:
    model = models.Sequential([
        layers.Reshape((image_size, image_size * 3), input_shape=(image_size, image_size, 3)),
        layers.LSTM(128),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
