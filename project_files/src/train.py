#!/usr/bin/env python
"""
Training module for TF binding prediction.

This module handles the training of CNN models for TF binding prediction.
"""

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf

# Ensure non-interactive backend for matplotlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import model-building utilities
from model import (
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
    if model_dir:
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
    if n_pos / len(y_train) < 0.25 or n_pos / len(y_train) > 0.75:
        cw = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
        logger.info(f"Using class weights: {cw}")
    else:
        cw = None
        logger.info("No class weights")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_list,
        class_weight=cw,
        verbose=1,
    )
    logger.info(f"Finished training ({len(history.history['loss'])} epochs)")
    return model, history


def save_training_results(model, history, output_dir, tf_name):
    """
    Save model, history, plots, and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Model
    m_path = os.path.join(output_dir, "model.h5")
    model.save(m_path)
    logger.info(f"Saved model to {m_path}")
    # Summary
    with open(os.path.join(output_dir, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    # History
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(hist, f, indent=4)
    # Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    ak = "accuracy" if "accuracy" in history.history else "acc"
    vak = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    plt.plot(history.history[ak], label="train_acc")
    plt.plot(history.history[vak], label="val_acc")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plot_p = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_p)
    plt.close()
    logger.info(f"Saved plots to {plot_p}")
    # Metadata
    meta = {
        "tf_name": tf_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": type(model).__name__,
        "params": int(np.sum([np.prod(w.shape) for w in model.trainable_weights])),
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
    logger.info("Saved metadata")


def main():
    p = argparse.ArgumentParser(description="Train TF binding model")
    p.add_argument("--tf", required=True, help="TF name")
    p.add_argument("--data-dir", required=True, help="Processed data dir")
    p.add_argument("--output-dir", required=True, help="Output dir")
    p.add_argument(
        "--model-type", default="cnn", choices=["cnn", "advanced", "baseline"]
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading data from {args.data_dir}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
    logger.info(f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}")

    # Build & compile model
    inp_shape = X_train.shape[1:]
    if args.model_type == "cnn":
        model = create_binding_cnn(inp_shape)
    elif args.model_type == "baseline":
        model = create_baseline_model(inp_shape)
    else:
        model = create_advanced_binding_cnn(inp_shape)
    model = compile_model(model)
    model.summary()

    # Callbacks
    cbs = get_callbacks(
        patience=args.patience, model_dir=os.path.join(args.output_dir, "checkpoints")
    )

    # Train
    model, hist = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks_list=cbs,
    )

    # Save results
    save_training_results(model, hist, args.output_dir, args.tf)
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main()
