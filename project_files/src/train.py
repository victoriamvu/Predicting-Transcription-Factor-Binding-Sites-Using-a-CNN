#!/usr/bin/env python
"""
Training module for TF binding prediction.

This module handles the training of CNN models for TF binding prediction.
"""

import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import from local modules
from model import create_binding_cnn, create_baseline_model, create_advanced_binding_cnn, compile_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tf_binding_train')


def load_data(data_dir):
    """
    Load preprocessed training, validation, and test data.
    
    Parameters:
    -----------
    data_dir : str
        Directory with preprocessed data
    
    Returns:
    --------
    tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # TODO: Implement data loading
    # 1. Load X_train.npy, y_train.npy, etc.
    # 2. Return loaded data
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_callbacks(patience=10, model_dir=None):
    """
    Create a list of callbacks for model training.
    """
    callback_list = []
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callback_list.append(early_stopping)
    
    # Add model checkpoint if directory is provided
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'model_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callback_list.append(model_checkpoint)
    
    # Add TensorBoard for visualization
    log_dir = os.path.join(model_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")) if model_dir else None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callback_list.append(tensorboard)
    
    return callback_list


def train_model(model, X_train, y_train, X_val, y_val, 
               batch_size=32, epochs=100, callbacks_list=None):
    """
    Train a model on the given dataset.
    """
    logger.info(f"Starting model training with {epochs} max epochs and batch size {batch_size}")
    
    # Calculate class weights to handle imbalanced data
    # (TF binding sites are typically sparse compared to non-binding regions)
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    
    # Only use class weights if there's significant imbalance
    if n_pos / len(y_train) < 0.25 or n_pos / len(y_train) > 0.75:
        logger.info(f"Dataset is imbalanced: {n_pos} positive samples, {n_neg} negative samples")
        class_weight = {
            0: 1.0 * len(y_train) / (2 * n_neg),
            1: 1.0 * len(y_train) / (2 * n_pos)
        }
        logger.info(f"Using class weights: {class_weight}")
    else:
        logger.info("Dataset is relatively balanced, not using class weights")
        class_weight = None
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=1
    )
    
    logger.info(f"Model training completed after {len(history.history['loss'])} epochs")
    
    return model, history


def save_training_results(model, history, output_dir, tf_name):
    """
    Save trained model and training history.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model architecture and weights
    model_path = os.path.join(output_dir, f"model.h5")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save model summary to a text file
    with open(os.path.join(output_dir, f"model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Save training history as JSON
    if history is not None and hasattr(history, 'history'):
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(value) for value in values]
        
        history_path = os.path.join(output_dir, f"training_history.json")
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=4)
        logger.info(f"Training history saved to {history_path}")
    
    # Generate and save training curves
    if history is not None:
        plot_training_history(history, output_dir)
    
    # Save metadata
    metadata = {
        "tf_name": tf_name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model.__class__.__name__,
        "num_parameters": int(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]))
    }
    
    metadata_path = os.path.join(output_dir, f"model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to {metadata_path}")


def plot_training_history(history, output_dir):
    """
    Plot training curves.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        Training history
    output_dir : str
        Directory to save plots
    """
    # Create figure with two subplots for loss and accuracy
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        acc_key = 'accuracy'
        val_acc_key = 'val_accuracy'
    else:
        acc_key = 'acc'  # For older TensorFlow versions
        val_acc_key = 'val_acc'
        
    plt.plot(history.history[acc_key], label='Training Accuracy')
    plt.plot(history.history[val_acc_key], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Training curves saved to {plot_path}")


def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description='Train TF binding prediction model')
    parser.add_argument('--tf', type=str, required=True, help='Transcription factor name')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with processed data')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--model-type', type=str, default='cnn', choices=['cnn', 'advanced', 'baseline'],
                      help='Type of model to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
    
    if X_train is None or y_train is None:
        logger.error("Failed to load training data")
        return
    
    logger.info(f"Data loaded: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
    # Create and compile model based on the model type
    logger.info(f"Creating {args.model_type} model")
    input_shape = X_train.shape[1:]  # Get input shape from training data
    
    if args.model_type == 'cnn':
        model = create_binding_cnn(input_shape)
    elif args.model_type == 'baseline':
        model = create_baseline_model(input_shape)
    elif args.model_type == 'advanced':
        model = create_advanced_binding_cnn(input_shape) 
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return
    
    # Compile the model
    model = compile_model(model)
    model.summary()
    
    # Setup callbacks
    logger.info("Setting up training callbacks")
    model_dir = os.path.join(args.output_dir, 'checkpoints')
    callbacks_list = get_callbacks(patience=args.patience, model_dir=model_dir)
    
    # Train model
    logger.info(f"Starting training with batch size {args.batch_size} and max {args.epochs} epochs")
    model, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks_list=callbacks_list
    )
    
    # Save results
    logger.info(f"Training completed, saving results to {args.output_dir}")
    save_training_results(
        model=model,
        history=history,
        output_dir=args.output_dir,
        tf_name=args.tf
    )
    
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
