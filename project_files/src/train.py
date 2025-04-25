#!/usr/bin/env python
"""
Training module for TF binding prediction.

This module handles the training of CNN models for TF binding prediction.
"""

import os

# Disable GPU/MPS to avoid Metal plugin errors on Mac
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Hide all GPU devices
try:
    tf.config.set_visible_devices([], "GPU")
    print("GPU disabled, using CPU for training.")
except Exception:
    pass

import sys
import argparse
import logging
import numpy as np
import json

# Ensure non-interactive backend for matplotlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import model-building utilities
from .model import (
    create_binding_cnn,
    create_baseline_model,
    create_advanced_binding_cnn,
    compile_model,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tf_binding_train")


def load_data(data_dir):
    """
    Load preprocessed training, validation, and test data.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    try:
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        sys.exit(1)
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_callbacks(patience=10, model_dir=None):
    """
    Create a list of callbacks for model training.
    """
    cb_list = []

    # Early stopping
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )
    cb_list.append(es)

    # Model checkpoint
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "model_checkpoint.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
        cb_list.append(mc)

        # TensorBoard
        tb_logdir = os.path.join(
            model_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(tb_logdir, exist_ok=True)
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=tb_logdir, histogram_freq=1, write_graph=True
        )
        cb_list.append(tb)

    return cb_list


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=32,
    epochs=100,
    callbacks_list=None,
):
    """
    Train the model on the data.
    """
    logger.info(f"Training: epochs={epochs}, batch_size={batch_size}")

    # Class weights for imbalance
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    frac_pos = n_pos / len(y_train)
    if frac_pos < 0.25 or frac_pos > 0.75:
        class_weights = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
        logger.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None
        logger.info("Dataset is balanced; not using class weights")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1,
    )
    logger.info(f"Finished training ({len(history.history['loss'])} epochs)")
    return model, history


def save_training_results(model, history, output_dir, tf_name):
    """
    Save model, history, plots, and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "model.h5")
    model.save(model_path)
    logger.info(f"Saved model to {model_path}")

    # Save summary
    with open(os.path.join(output_dir, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Save history JSON
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(hist_dict, f, indent=4)

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    plt.plot(history.history[acc_key], label="train_acc")
    plt.plot(history.history[val_acc_key], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved training curves to {plot_path}")

    # Save metadata
    metadata = {
        "tf_name": tf_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": type(model).__name__,
        "num_parameters": int(
            np.sum([np.prod(w.shape) for w in model.trainable_weights])
        ),
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved metadata")


def main():
    parser = argparse.ArgumentParser(description="Train TF binding model")
    parser.add_argument("--tf", required=True, help="Transcription factor name")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "advanced", "baseline"],
        default="cnn",
        help="Type of model architecture",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    args = parser.parse_args()

    logger.info(f"Loading data from {args.data_dir}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
    logger.info(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    # Build model
    input_shape = X_train.shape[1:]
    if args.model_type == "cnn":
        model = create_binding_cnn(input_shape)
    elif args.model_type == "baseline":
        model = create_baseline_model(input_shape)
    else:
        model = create_advanced_binding_cnn(input_shape)

    # Compile with defaults
    model = compile_model(model)
    model.summary()

    # Callbacks
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    callbacks_list = get_callbacks(patience=args.patience, model_dir=checkpoint_dir)

    # Train
    model, history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks_list=callbacks_list,
    )

    # Save everything
    save_training_results(model, history, args.output_dir, args.tf)
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()

def run_training_pipeline(tf_name):
    import os
    import numpy as np
    import json
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # Paths
    data_dir = f"data/processed/{tf_name}"
    output_dir = f"models/{tf_name}"
    os.makedirs(output_dir, exist_ok=True)

    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    X_val = np.load(f"{data_dir}/X_val.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")

    # Define model
    model = Sequential([
        Conv1D(64, 15, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])

    # Callbacks
    checkpoint_path = os.path.join(output_dir, "model.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=40,
                        batch_size=32,
                        callbacks=[checkpoint, early_stop])

    # Save training history
    hist_path = os.path.join(output_dir, f"{tf_name}_training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history.history, f)

    # Plot training curve
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title(f"{tf_name} Training Loss Curve")
    plt.savefig(os.path.join(output_dir, f"{tf_name}_training_curves.png"))
    plt.close()

    print(f"✅ Saved model and training history for {tf_name} in {output_dir}")
