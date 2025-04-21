#!/usr/bin/env python
"""
Evaluation module for TF binding prediction.

This module handles the evaluation of trained models for TF binding prediction.
"""

import os

# Disable GPU to avoid Metal plugin errors on Mac
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
    print("GPU disabled, using CPU for evaluation.")
except Exception:
    pass

import argparse
import logging
import numpy as np

# Use non-interactive backend for plots
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tf_binding_eval")


def load_model_and_data(model_path, data_dir):
    """
    Load a trained model and test data.
    """
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Loading test data from {data_dir}")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate a trained model on test data and save metrics & plots.

    Parameters:
    -----------
    model : tf.keras.Model
    X_test : np.ndarray
    y_test : np.ndarray
    output_dir : str
        Directory where metrics.json, roc_curve.png, pr_curve.png will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Predict probabilities
    logger.info("Running model predictions")
    y_pred = model.predict(X_test)

    # Compute ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    logger.info(f"ROC AUC = {roc_auc:.3f}, PR AUC = {pr_auc:.3f}")

    # Save metrics
    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "num_test_samples": int(len(y_test)),
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")

    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"Saved ROC curve to {roc_path}")

    # Plot and save Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(pr_path)
    plt.close()
    logger.info(f"Saved PR curve to {pr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained TF binding model.")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to saved .h5 model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing X_test.npy and y_test.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save evaluation metrics and plots",
    )
    args = parser.parse_args()

    model, X_test, y_test = load_model_and_data(args.model_path, args.data_dir)
    evaluate_model(model, X_test, y_test, args.output_dir)

