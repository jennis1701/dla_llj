import os
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def create_image_generators(config: dict):
    # Import TensorFlow only when needed (lazy import for cloud compatibility)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    image_size = config["dataset"]["image_size"]
    batch_size = config["dataset"]["batch_size"]
    validation_split = config["dataset"]["validation_split"]

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_dir = os.path.expanduser(config["dataset"]["train_dir"])
    test_dir = os.path.expanduser(config["dataset"]["test_dir"])

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    if train_generator.num_classes <= 0:
        raise ValueError(
            f"No class directories detected in training data path: {train_dir}. "
            "Make sure subfolders exist for each activity label and contain images."
        )

    class_names = list(train_generator.class_indices.keys())
    return train_generator, validation_generator, test_generator, class_names
