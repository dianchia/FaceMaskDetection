import gc
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.applications import mobilenet_v2, resnet50, inception_v3
from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Dropout,
    Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)

try:
    from rich.logging import RichHandler
except ModuleNotFoundError:
    os.system("pip install rich")
    from rich.logging import RichHandler

handler = RichHandler()
formatter = logging.Formatter("%(message)s")
logger = logging.getLogger("main")

handler.setFormatter(formatter)
if len(logger.handlers) == 0:
    logger.addHandler(handler)


"""Script for training Face Mask Detector

This is the script used to train the Face Mask Detector.
"""


def build_model_mnet():
    """
    Building the transfer learning model with MobileNet_V2
    Returns
    -------
    tensorflow.keras.Model
        The model for training.
    """
    baseModel = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def build_model_resnet():
    """
    Building the transfer learning model with ResNet50
    Returns
    -------
    tensorflow.keras.Model
        The model for training.
    """
    baseModel = ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def build_model_inception():
    """
    Building the transfer learning model with Inception_V3
    Returns
    -------
    tensorflow.keras.Model
        The model for training.
    """
    baseModel = InceptionV3(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def main():
    """
    Main function.
    """
    EPOCHS = 8
    BATCH_SIZE = 32
    TRAIN_TEST_SPLIT = 0.8  # Training: 0.8, Validation: 0.2

    DATASET_PATH = "../input/facemaskdetection/dataset"
    TOTAL_DATA_COUNT = len(list(Path(DATASET_PATH).rglob("**/*.jpg")))
    TRAIN_DATA_COUNT = np.ceil(TOTAL_DATA_COUNT * TRAIN_TEST_SPLIT).astype("int32")
    VAL_DATA_COUNT = TOTAL_DATA_COUNT - TRAIN_DATA_COUNT
    TRAIN_STEPS_PER_EPOCH = np.ceil(TRAIN_DATA_COUNT / BATCH_SIZE).astype("int32")
    VAL_STEPS_PER_EPOCH = np.ceil(VAL_DATA_COUNT / BATCH_SIZE).astype("int32")

    logger.info(f"Epochs: {EPOCHS}\nBatch size: {BATCH_SIZE}")
    logger.info(
        f"Total data size: {TOTAL_DATA_COUNT}. Splitted {TRAIN_TEST_SPLIT * 100}% as training."
    )
    logger.info(
        f"Train data size: {TRAIN_DATA_COUNT}\tValidation data size: {VAL_DATA_COUNT}"
    )
    logger.info(
        f"Steps per epoch:\nTraining: {TRAIN_STEPS_PER_EPOCH}\tValidation: {VAL_STEPS_PER_EPOCH}"
    )

    logger.info("Building models...")
    models = [
        {
            "name": "MobileNetV2",
            "model": build_model_mnet(),
            "preprocessor": mobilenet_v2.preprocess_input,
        },
        {
            "name": "ResNet50",
            "model": build_model_resnet(),
            "preprocessor": resnet50.preprocess_input,
        },
        {
            "name": "InceptionV3",
            "model": build_model_inception(),
            "preprocessor": inception_v3.preprocess_input,
        },
    ]

    for model_dict in models:
        model_name = model_dict["name"]
        model = model_dict["model"]
        preprocessor = model_dict["preprocessor"]

        logger.info(f"Training {model_name}...")
        optimizer = optimizers.Adam(learning_rate=1e-4, decay=1e-4 / 20)
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        aug = ImageDataGenerator(
            preprocessing_function=preprocessor,
            rotation_range=20,
            zoom_range=0.15,
            shear_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.2,
        )

        train_generator = aug.flow_from_directory(
            DATASET_PATH,
            target_size=(224, 224),
            color_mode="rgb",
            classes=["WMFD", "CMFD", "IMFD"],
            class_mode="categorical",
            batch_size=BATCH_SIZE,
            subset="training",
        )

        val_generator = aug.flow_from_directory(
            DATASET_PATH,
            target_size=(224, 224),
            color_mode="rgb",
            classes=["WMFD", "CMFD", "IMFD"],
            class_mode="categorical",
            batch_size=BATCH_SIZE,
            subset="validation",
        )

        logger.info(f"Class indices: {train_generator.class_indices}")

        H = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
            verbose=1,
        )

        plt.style.use("seaborn-darkgrid")
        plt.figure()
        plt.plot(H.history["loss"], label="Train Loss")
        plt.plot(H.history["val_loss"], label="Validation Loss")
        plt.plot(H.history["accuracy"], label="Train Accuracy")
        plt.plot(H.history["val_accuracy"], label="Validation Accuracy")
        plt.title(f"Loss and Accuracy Graph for {model_name}")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss / Accuracy")
        plt.legend(loc="lower left")
        plt.show()
        model.save(f"{model_name}_MaskDetectorFull.h5")
        print(f"Collected: {gc.collect()}")
