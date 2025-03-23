# Source Code Directory

This directory contains the core implementation of the transcription factor binding prediction model.

## Table of Contents
1. [Files](#files)
2. [Tasks](#implementation-tasks)
   - [data.py](#datapy)
   - [model.py](#modelpy)
   - [train.py](#trainpy)
   - [evaluate.py](#evaluatepy)
3. [Dependencies](#dependencies)
4. [Development Workflow](#development-workflow)
5. [Example Usage](#example-usage)

## Files

```
src/
├── README.md        # This file
├── data.py          # Data loading and preprocessing
├── model.py         # CNN model implementations
├── train.py         # Training functions
└── evaluate.py      # Evaluation metrics and functions
```

## Implementation Tasks

Each source file contains boilerplate code with function signatures and documentation. The implementation tasks are in order of importance and should be completed in that order.

### data.py


The following functions need to be implemented:

- `parse_jaspar_pfm(pfm_file)`: Parse JASPAR Position Frequency Matrix
- `pfm_to_pwm(pfm_df, pseudocount)`: Convert PFM to Position Weight Matrix
- `one_hot_encode(sequence)`: Convert DNA sequence to one-hot encoding
- `reverse_complement(sequence)`: Get reverse complement of DNA sequence
- `extract_sequences_from_chip(bed_file, genome_file)`: Extract sequences from ChIP-seq peaks
- `generate_negative_samples(positive_sequences, method)`: Generate negative samples
- `prepare_dataset(positive_sequences, negative_sequences)`: Prepare data for model training

Look for `# TODO:` comments in the file for implementation guidance. The data processing pipeline needs to:
1. Read JASPAR motifs
2. Extract sequences from the genome using ChIP-seq coordinates
3. Generate negative examples
4. Convert sequences to one-hot encoding
5. Split data into train/val/test sets

Unit tests for these functions are provided in `tests/test_data.py`.

### model.py


The following functions need to be implemented:

- `create_binding_cnn(sequence_length, num_filters, ...)`: Create basic CNN model
- `create_advanced_binding_cnn(sequence_length, ...)`: Create advanced CNN with additional options
- `create_baseline_model(sequence_length)`: Create simple baseline model
- `compile_model(model, learning_rate, metrics)`: Compile a model with appropriate settings

Goal:
1. Takes one-hot encoded sequences (shape: N×200×4) as input
2. Uses convolutional layers to detect sequence motifs
3. Includes pooling and regularization
4. Outputs a single probability value via sigmoid activation

Reference the documentation for details

### train.py



The following functions need to be implemented:

- `load_data(data_dir)`: Load preprocessed data from NumPy files
- `get_callbacks(patience, model_dir)`: Create callbacks for training
- `train_model(model, X_train, y_train, ...)`: Train model on given dataset
- `save_training_results(model, history, output_dir)`: Save model and training history
- `plot_training_history(history, output_dir)`: Plot training curves

The training pipeline should:
1. Load data from processed NumPy files
2. Set up appropriate callbacks (early stopping, model checkpointing)
3. Train the model with validation
4. Save the model and visualize training curves

### evaluate.py


The following functions need to be implemented:

- `load_model_and_data(model_path, data_dir)`: Load model and test data
- `evaluate_model(model, X_test, y_test)`: Evaluate model performance
- `plot_roc_curve(fpr, tpr, output_file)`: Plot ROC curve
- `plot_pr_curve(precision, recall, output_file)`: Plot precision-recall curve
- `analyze_motifs(model, sequence_length, output_dir)`: Extract motifs from model weights

The evaluation pipeline should:
1. Load a trained model
2. Evaluate on test data
3. Calculate performance metrics
4. Visualize results with appropriate plots
5. Optionally analyze the learned motifs

## Dependencies

The code depends on the following libraries:
- NumPy and Pandas for data manipulation
- TensorFlow/Keras for deep learning models
- Biopython and Pyfaidx for biological sequence handling
- Matplotlib and Seaborn for visualization

## Development Workflow

1. Start by implementing `data.py` and testing with `tests/test_data.py`
2. Once data processing works, implement the model architecture in `model.py`
3. Implement training functions in `train.py`
4. Finally, implement evaluation metrics in `evaluate.py`
5. Use the master script `scripts/run_workflow.sh` to run the complete pipeline

## Example Usage

The modules in this directory will be called by scripts in the `scripts/` directory, but they can also be imported and used directly:

```python
from src.data import one_hot_encode, extract_sequences_from_chip
from src.model import create_binding_cnn, compile_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_roc_curve
```
