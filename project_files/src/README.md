# Source Code Directory

This directory contains the core implementation of the transcription factor binding prediction model.

## Files

```
src/
├── README.md        # This file
├── data.py          # Data loading and preprocessing
├── model.py         # CNN model implementations
├── train.py         # Training functions
└── evaluate.py      # Evaluation metrics and functions
```

## Module Descriptions

### data.py

Handles all data processing for the project, including:

- Parsing JASPAR motif files (PFM, PWM)
- Extracting sequences from reference genome based on ChIP-seq peaks
- Converting DNA sequences to one-hot encoding
- Generating negative samples for training
- Preparing training, validation, and test datasets

**Key Functions:**
- `parse_jaspar_pfm(pfm_file)`: Parse JASPAR Position Frequency Matrix
- `pfm_to_pwm(pfm_df, pseudocount)`: Convert PFM to Position Weight Matrix
- `one_hot_encode(sequence)`: Convert DNA sequence to one-hot encoding
- `reverse_complement(sequence)`: Get reverse complement of DNA sequence
- `extract_sequences_from_chip(bed_file, genome_file)`: Extract sequences from ChIP-seq peaks
- `generate_negative_samples(positive_sequences, method)`: Generate negative samples
- `prepare_dataset(positive_sequences, negative_sequences)`: Prepare data for model training

### model.py

Implements various CNN architectures for transcription factor binding prediction:

- Basic CNN model with convolutional and dense layers
- Advanced model with options for attention and batch normalization
- Baseline models for comparison

**Key Functions:**
- `create_binding_cnn(sequence_length, num_filters, ...)`: Create basic CNN model
- `create_advanced_binding_cnn(sequence_length, ...)`: Create advanced CNN with additional options
- `create_baseline_model(sequence_length)`: Create simple baseline model
- `compile_model(model, learning_rate, metrics)`: Compile a model with appropriate settings

### train.py

Handles model training, including:

- Loading and preparing data
- Setting up callbacks (early stopping, checkpointing)
- Training models
- Handling class imbalance
- Saving trained models and training history

**Key Functions:**
- `load_data(data_dir)`: Load preprocessed data
- `get_callbacks(patience, model_dir)`: Create callbacks for training
- `train_model(model, X_train, y_train, ...)`: Train model on given dataset
- `save_training_results(model, history, output_dir)`: Save model and training history
- `plot_training_history(history, output_dir)`: Plot training curves

### evaluate.py

Provides evaluation metrics and visualization for model assessment:

- Model loading and evaluation
- Performance metrics calculation (accuracy, AUC-ROC, PR curve)
- Visualization of results
- Motif analysis from model weights

**Key Functions:**
- `load_model_and_data(model_path, data_dir)`: Load model and test data
- `evaluate_model(model, X_test, y_test)`: Evaluate model performance
- `plot_roc_curve(fpr, tpr, output_file)`: Plot ROC curve
- `plot_pr_curve(precision, recall, output_file)`: Plot precision-recall curve
- `analyze_motifs(model, sequence_length, output_dir)`: Extract motifs from model weights

## Usage

The modules in this directory are typically called by scripts in the `scripts/` directory, but they can also be imported and used directly in Python code or notebooks:

```python
from src.data import one_hot_encode, extract_sequences_from_chip
from src.model import create_binding_cnn, compile_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_roc_curve
```
