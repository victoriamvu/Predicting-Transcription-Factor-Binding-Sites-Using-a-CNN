#!/usr/bin/env python
"""
Evaluation module for TF binding prediction.

This module handles the evaluation of trained models for TF binding prediction.
"""

import os
import argparse
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tf_binding_evaluate')


def load_model_and_data(model_path, data_dir):
    """
    Load a trained model and test data.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    data_dir : str
        Directory with test data
    
    Returns:
    --------
    tuple: (model, X_test, y_test)
    """
    # TODO: Implement model and data loading
    # 1. Load the trained model
    # 2. Load test data from data_dir
    # 3. Return model and test data
    model = tf.keras.models.load_model(model_path)
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    X_test, y_test : numpy arrays
        Test data
    
    Returns:
    --------
    tuple: (metrics_dict, fpr, tpr, precision, recall)
    """
    # TODO: Implement model evaluation
    # 1. Get model predictions on test data
    # 2. Calculate metrics: accuracy, loss, AUC
    # 3. Calculate ROC and PR curve data
    # 4. Return metrics and curve data
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predicted probabilities
    y_scores = model.predict(X_test).flatten()
    
    # Binarize predictions at threshold = 0.5
    y_pred = (y_scores >= 0.5).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_scores)
    
    # Print and save metrics
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc_score:.4f}")
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nAUC: {auc_score:.4f}\n")

    # Plot and save ROC and PR curves
    plot_roc_curve(y_test, y_scores, save_path=os.path.join(output_dir, "roc_curve.png"))
    plot_pr_curve(y_test, y_scores, save_path=os.path.join(output_dir, "pr_curve.png"))


def plot_roc_curve(fpr, tpr, output_file=None):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    fpr : numpy array
        False positive rates
    tpr : numpy array
        True positive rates
    output_file : str or None
        Path to save the plot
    
    Returns:
    --------
    float: Area under ROC curve
    """
    # TODO: Implement ROC curve plotting
    # 1. Calculate AUC
    # 2. Create ROC curve plot
    # 3. Save plot if output_file is provided
    # 4. Return AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(precision, recall, output_file=None):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    precision : numpy array
        Precision values
    recall : numpy array
        Recall values
    output_file : str or None
        Path to save the plot
    
    Returns:
    --------
    float: Area under PR curve
    """
    # TODO: Implement PR curve plotting
    # 1. Calculate AUC
    # 2. Create precision-recall curve plot
    # 3. Save plot if output_file is provided
    # 4. Return AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Avg Precision = {avg_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_motifs(model, sequence_length=200, output_dir=None):
    """
    Analyze model to extract motif information.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    sequence_length : int
        Length of input sequences
    output_dir : str or None
        Directory to save results
    
    Returns:
    --------
    dict: Discovered motif information
    """
    # TODO: Implement motif analysis
    # 1. Extract weights from first convolutional layer
    # 2. Visualize filters as sequence logos
    # 3. Save visualizations if output_dir is provided
    # 4. Return motif information
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract weights from first convolutional layer
    conv_layer = None
    for layer in model.layers:
        if "Conv1D" in layer.__class__.__name__:
            conv_layer = layer
            break

    if conv_layer is None:
        print("No Conv1D layer found.")
        return []

    weights = conv_layer.get_weights()[0]  # shape: (kernel_size, 4, num_filters)
    kernel_size, _, num_filters = weights.shape
    motif_list = []

    # 2. Visualize filters as sequence logos
    for i in range(num_filters):
        pwm = weights[:, :, i].T  # shape: 4 x kernel_size
        motif_list.append(pwm)

        plt.figure(figsize=(kernel_size / 2, 2))
        plt.imshow(pwm, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Weight')
        plt.yticks(range(4), labels=['A', 'C', 'G', 'T'])
        plt.title(f"Filter {i + 1}")
        plt.xlabel("Position")
        plt.tight_layout()

        # 3. Save visualizations
        plt.savefig(os.path.join(output_dir, f"filter_{i + 1}.png"))
        plt.close()
        
        return motif_list

def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate TF binding prediction model')
    parser.add_argument('--tf', type=str, required=True, help='Transcription factor name')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with test data')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--analyze-motifs', action='store_true', help='Analyze model for motifs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data(args.model_path, args.data_dir)
    
    # Evaluate model
    metrics, fpr, tpr, precision, recall = evaluate_model(model, X_test, y_test)
    
    # TODO: Generate and save ROC and PR curves
    
    # TODO: Save evaluation metrics
    
    # TODO: Analyze motifs if requested

    if args.analyze_motifs:
        logger.info("Analyzing motifs...")
        analyze_motifs(model, output_dir=args.output_dir)
        logger.info(f"Evaluation complete. Results saved to {args.output_dir}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained TF binding model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved .h5 model")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with test data")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save results")
    args = parser.parse_args()

    # Load model and test data
    model, X_test, y_test = load_model_and_data(args.model_path, args.data_dir)

    # Evaluate and generate plots
    evaluate_model(model, X_test, y_test, args.output_dir)
