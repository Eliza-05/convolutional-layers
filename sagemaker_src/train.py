#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def build_cnn(conv_depth: int, learning_rate: float, dropout: float) -> tf.keras.Model:
    if conv_depth not in (1, 2, 3):
        raise ValueError("conv_depth must be 1, 2, or 3")

    filters = [32, 64, 128][:conv_depth]

    model = Sequential(name=f"fashion_cnn_{conv_depth}layers")
    model.add(tf.keras.layers.Input(shape=(28, 28, 1), name="input"))

    for i, f in enumerate(filters, start=1):
        model.add(Conv2D(f, (3, 3), activation="relu", padding="same", name=f"conv{i}"))
        model.add(BatchNormalization(name=f"bn{i}"))

        # Pooling logic (matches your notebook-style behavior)
        # conv_depth 1: pool in conv1
        # conv_depth 2: pool in conv1 and conv2
        # conv_depth 3: pool in conv1 and conv2 (not in conv3)
        if conv_depth in (1, 2):
            use_pool = True
        else:
            use_pool = i in (1, 2)

        if use_pool:
            model.add(MaxPooling2D((2, 2), name=f"pool{i}"))

    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation="relu", name="dense1"))
    model.add(Dropout(dropout, name="dropout"))
    model.add(Dense(10, activation="softmax", name="output"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def parse_args():
    p = argparse.ArgumentParser()
    # SageMaker sets SM_MODEL_DIR automatically for Training Jobs
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--conv_depth", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load Fashion-MNIST (downloads if not cached)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = build_cnn(
        conv_depth=args.conv_depth,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"test_loss={test_loss:.4f}")
    print(f"test_accuracy={test_acc:.4f}")

    # IMPORTANT: Everything saved inside SM_MODEL_DIR will be packaged into model.tar.gz by SageMaker
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model as SavedModel under /opt/ml/model/1 (common serving format)
    export_dir = model_dir / "1"
    export_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(export_dir))

    # Save metrics and metadata alongside the model
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "epochs_trained": int(len(history.history.get("loss", []))),
        "conv_depth": int(args.conv_depth),
        "learning_rate": float(args.learning_rate),
        "dropout": float(args.dropout),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
    }

    with open(model_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(model_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print(f"\n✅ Saved artifacts to: {model_dir}")
    print("✅ SageMaker will automatically create model.tar.gz from this folder.")

if __name__ == "__main__":
    main()
