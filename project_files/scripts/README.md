# Scripts Directory

This directory contains utility scripts for running the transcription factor binding prediction project.

## Files

```
scripts/
├── README.md                  # This file
├── download_data.sh           # Script to download all data
├── download_jaspar_script.py  # Script to download JASPAR data
├── train_model.py             # Script to train the model
├── evaluate_model.py          # Script to evaluate the model
├── verify_setup.py            # Script to verify project structure
└── run_workflow.sh            # Master script to run entire workflow
```

## Script Descriptions

### download_data.sh

Bash script to download all necessary data for the project.

**Functionality:**
- Creates directory structure
- Downloads JASPAR motif data
- Downloads reference genome (hg38)
- Downloads ChIP-seq data from ENCODE
- Indexes the genome with pyfaidx

**Usage:**
```bash
bash scripts/download_data.sh
```

**Notes:**
- Checks for existing files before downloading to avoid overwriting
- Uses the `-k` option with gunzip to preserve original compressed files
- Provides informative messages about which steps are being skipped or executed

### download_jaspar_script.py

Python script for downloading and processing transcription factor binding motifs from the JASPAR database.

**Functionality:**
- Downloads Position Frequency Matrices (PFMs) for specific TFs
- Converts PFMs to Position Weight Matrices (PWMs)
- Retrieves TF metadata
- Saves files in appropriate formats

**Usage:**
```bash
# Download a single TF
python scripts/download_jaspar_script.py --tf CTCF --output-dir ./data/raw/jaspar

# Download multiple TFs
python scripts/download_jaspar_script.py --tf-list "CTCF,GATA1,CEBPA,TP53" --output-dir ./data/raw/jaspar

# List available TFs
python scripts/download_jaspar_script.py --list --species human
```

**Notes:**
- Supports downloading by TF name or JASPAR ID
- Downloads multiple file formats (PFM, JASPAR, MEME)
- Includes functions to convert between matrix formats

### train_model.py

Script for training transcription factor binding prediction models.

**Functionality:**
- Loads configuration from YAML file
- Loads processed data
- Creates and compiles model
- Trains model with early stopping
- Saves trained model and training history

**Usage:**
```bash
python scripts/train_model.py --tf CTCF --data-dir data/processed/CTCF --output-dir models/CTCF --model-type cnn
```

**Options:**
- `--tf`: Transcription factor name
- `--data-dir`: Directory with processed data
- `--output-dir`: Directory to save model and results
- `--config`: Path to config file (default: config.yaml)
- `--model-type`: Type of model to train (cnn, advanced, baseline)
- `--batch-size`: Batch size for training
- `--epochs`: Maximum number of epochs

### evaluate_model.py

Script for evaluating trained transcription factor binding prediction models.

**Functionality:**
- Loads trained model
- Loads test data
- Evaluates model performance
- Generates ROC and PR curves
- Optionally analyzes learned motifs

**Usage:**
```bash
python scripts/evaluate_model.py --tf CTCF --model-path models/CTCF/model.h5 --data-dir data/processed/CTCF --output-dir results/CTCF
```

**Options:**
- `--tf`: Transcription factor name
- `--model-path`: Path to the trained model
- `--data-dir`: Directory with test data
- `--output-dir`: Directory to save evaluation results
- `--config`: Path to config file (default: config.yaml)
- `--analyze-motifs`: Flag to analyze model for motifs

### verify_setup.py

Script to check if all necessary files and directories exist in the project structure.

**Functionality:**
- Checks directory structure
- Verifies source files exist
- Checks for data files (JASPAR, ENCODE, genome)
- Provides next steps based on current state

**Usage:**
```bash
python scripts/verify_setup.py
```

**Notes:**
- Uses symbols (✓, ⚠, ✗) to indicate status
- Checks file sizes to ensure files aren't empty
- Suggests specific next steps based on what's missing

### run_workflow.sh

Master script that runs the entire workflow from data download to model evaluation.

**Functionality:**
- Verifies project structure
- Downloads data if not already present
- Processes data for each TF
- Trains models for each TF
- Evaluates models and analyzes motifs
- Generates a summary report

**Usage:**
```bash
bash scripts/run_workflow.sh
```

**Notes:**
- Checks for existing files at each step to avoid unnecessary reprocessing
- Provides colorful output to track progress
- Asks for confirmation before retraining existing models
- Creates all necessary directories
- Generates a final HTML report if Jupyter is installed

## Execution Order

For a complete workflow, you can either:

1. **Run the master script:**
   ```bash
   bash scripts/run_workflow.sh
   ```

2. **Run individual scripts in this order:**
   - `download_data.sh` - Download all data
   - `verify_setup.py` - Verify data was downloaded correctly
   - Use `src/data.py` to process data for each TF
   - `train_model.py` - Train models for each TF
   - `evaluate_model.py` - Evaluate models
