#!/usr/bin/env bash
# Master script to run the entire TF binding prediction workflow
# This script runs all steps in a logical order from data download to model evaluation

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print a section header
print_header() {
    echo -e "\n${YELLOW}========== $1 ==========${NC}\n"
}

# Function to print a step
print_step() {
    echo -e "${BLUE}---> $1${NC}"
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Create models and results directories
mkdir -p models/{CTCF,GATA1,CEBPA,TP53}
mkdir -p results/{CTCF,GATA1,CEBPA,TP53}

# List of TFs to process
TFS=("CTCF" "GATA1" "CEBPA" "TP53")

# Verify project structure before starting
print_header "VERIFYING PROJECT STRUCTURE"
python scripts/verify_setup.py
print_success "Project structure verified"

# Step 1: Download data if it doesn't exist
print_header "DOWNLOADING DATA"
if [ ! -f data/raw/genome/hg38.fa ] || [ ! -d data/raw/jaspar ] || [ ! -d data/raw/encode ]; then
    print_step "Running download script..."
    bash scripts/download_data.sh
    print_success "Data download complete"
else
    print_success "Data already downloaded, skipping download step"
fi

# Step 2: Process data for each TF
print_header "PROCESSING DATA"
for TF in "${TFS[@]}"; do
    print_step "Processing data for $TF..."
    
    # Determine the correct ChIP-seq file
    CHIP_FILE="data/raw/encode/$TF/ENCFF*.bed"
    
    # Check if processed data already exists
    if [ -f "data/processed/$TF/X_train.npy" ]; then
        print_success "Processed data for $TF already exists, skipping"
    else
        # Process data
        python src/data.py --tf $TF \
                          --jaspar-dir data/raw/jaspar \
                          --chip-seq-file $CHIP_FILE \
                          --genome data/raw/genome/hg38.fa \
                          --output-dir data/processed/$TF
        print_success "Data processing for $TF complete"
    fi
done

# Step 3: Train models for each TF
print_header "TRAINING MODELS"
for TF in "${TFS[@]}"; do
    print_step "Training model for $TF..."
    
    # Check if model already exists
    if [ -f "models/$TF/model.h5" ]; then
        read -p "Model for $TF already exists. Retrain? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Skipping training for $TF"
            continue
        fi
    fi
    
    # Train model
    python scripts/train_model.py --tf $TF \
                                 --data-dir data/processed/$TF \
                                 --output-dir models/$TF
    print_success "Model training for $TF complete"
done

# Step 4: Evaluate models for each TF
print_header "EVALUATING MODELS"
for TF in "${TFS[@]}"; do
    print_step "Evaluating model for $TF..."
    
    # Check if model exists before evaluation
    if [ ! -f "models/$TF/model.h5" ]; then
        echo "No model found for $TF. Skipping evaluation."
        continue
    fi
    
    # Evaluate model
    python scripts/evaluate_model.py --tf $TF \
                                    --model-path models/$TF/model.h5 \
                                    --data-dir data/processed/$TF \
                                    --output-dir results/$TF \
                                    --analyze-motifs
    print_success "Model evaluation for $TF complete"
done

# Final step: Generate summary report
print_header "GENERATING SUMMARY"
print_step "Running Jupyter notebook..."

# Check if Jupyter is installed
if command -v jupyter &> /dev/null; then
    # Execute the results notebook to generate report
    jupyter nbconvert --to html --execute notebooks/results.ipynb --output summary_report.html
    print_success "Summary report generated: notebooks/summary_report.html"
else
    echo "Jupyter not found. Please run notebooks/results.ipynb manually to generate the summary report."
fi

print_header "WORKFLOW COMPLETE"
echo "Results for each TF are available in the results/ directory"
echo "You can analyze the results further using the notebooks/results.ipynb notebook"
