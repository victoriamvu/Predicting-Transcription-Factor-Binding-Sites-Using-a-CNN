#!/usr/bin/env python
"""
Model architecture module for TF binding prediction.

This module defines CNN architectures for predicting transcription factor binding sites.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def create_binding_cnn(input_shape=(200, 4), num_filters=None, 
                      kernel_sizes=None, pool_sizes=None,
                      dense_units=64, dropout_rate=0.3):
    """
    Create a 1D CNN model for transcription factor binding prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, 4)
    num_filters : tuple or None
        Number of filters in each convolutional layer (e.g., (32, 64))
    kernel_sizes : tuple or None
        Size of kernels in each convolutional layer (e.g., (8, 4))
    pool_sizes : tuple or None
        Size of pooling windows in each pooling layer (e.g., (2, 2))
    dense_units : int
        Number of units in the dense layer
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    tensorflow.keras.Model: CNN model for binding prediction
    """
    # Set default values for parameters if None
    if num_filters is None:
        num_filters = (16, 32)  # Default: 16 filters then 32 filters
    if kernel_sizes is None:
        kernel_sizes = (8, 4)   # Default: kernel size 8 then 4
    if pool_sizes is None:
        pool_sizes = (2, 2)     # Default: pool size 2 for both layers
    
    # Create model
    model = models.Sequential()
    
    # Input layer 
    model.add(layers.InputLayer(input_shape=input_shape))
    
    # First convolutional block
    model.add(layers.Conv1D(filters=num_filters[0], 
                          kernel_size=kernel_sizes[0],
                          activation='relu',
                          padding='same'))
    model.add(layers.MaxPooling1D(pool_size=pool_sizes[0]))
    model.add(layers.Dropout(dropout_rate))
    
    # Second convolutional block
    model.add(layers.Conv1D(filters=num_filters[1], 
                          kernel_size=kernel_sizes[1],
                          activation='relu',
                          padding='same'))
    model.add(layers.MaxPooling1D(pool_size=pool_sizes[1]))
    model.add(layers.Dropout(dropout_rate))
    
    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer with sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model


def create_advanced_binding_cnn(input_shape=(200, 4), 
                              conv_layers=None,
                              dense_layers=None,
                              attention=False,
                              batch_norm=True,
                              dropout_rate=0.3):
    """
    Create an advanced CNN model with options like attention and batch normalization.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, 4)
    conv_layers : list or None
        List of (filters, kernel_size) tuples for each conv layer
    dense_layers : list or None
        List of units for each dense layer
    attention : bool
        Whether to use attention mechanism
    batch_norm : bool
        Whether to use batch normalization
    dropout_rate : float
        Dropout rate for regularization
    
    Returns:
    --------
    tensorflow.keras.Model: Advanced CNN model for binding prediction
    """
    # Set default values for parameters if None
    if conv_layers is None:
        conv_layers = [(32, 8), (64, 4), (128, 4)]  # Three conv layers
    if dense_layers is None:
        dense_layers = [128, 64]  # Two dense layers
    
    # Create model using the Functional API
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Add convolutional layers
    for i, (filters, kernel_size) in enumerate(conv_layers):
        x = layers.Conv1D(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        padding='same')(x)
        
        if batch_norm:
            x = layers.BatchNormalization()(x)
            
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Add attention mechanism if specified
    if attention:
        # Simple self-attention mechanism
        attention_scores = layers.Dense(1)(x)
        attention_weights = layers.Softmax(axis=1)(attention_scores)
        context_vector = tf.keras.layers.Multiply()([x, attention_weights])
        x = layers.GlobalAveragePooling1D()(context_vector)
    else:
        x = layers.GlobalMaxPooling1D()(x)
    
    # Add dense layers
    for units in dense_layers:
        x = layers.Dense(units, activation='relu')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def create_baseline_model(input_shape=(200, 4), hidden_units=64):
    """
    Create a simple baseline model for comparison.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, 4)
    hidden_units : int
        Number of hidden units
    
    Returns:
    --------
    tensorflow.keras.Model: Baseline model for binding prediction
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=input_shape))
    
    # Flatten the input (convert 2D to 1D)
    model.add(layers.Flatten())
    
    # Add a simple dense hidden layer
    model.add(layers.Dense(hidden_units, activation='relu'))
    
    # Output layer with sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model


def compile_model(model, learning_rate=0.001, metrics=None):
    """
    Compile a model with appropriate loss function and metrics.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Model to compile
    learning_rate : float
        Learning rate for optimizer
    metrics : list or None
        List of metrics (default: accuracy and AUC)
    
    Returns:
    --------
    tensorflow.keras.Model: Compiled model
    """
    # Set default metrics if None
    if metrics is None:
        metrics = [
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    
    # Compile model with binary crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    model = create_binding_cnn()
    model = compile_model(model)
    print("Model structure will be implemented by team members")
