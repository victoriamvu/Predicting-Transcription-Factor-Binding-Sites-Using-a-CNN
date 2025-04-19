#!/usr/bin/env bash
# Master script to run the entire TF binding prediction workflow
# This script runs all steps in a logical order from data download to model evaluation
# Automatically handles project root detection and path resolution

# Don't exit on error - we want to continue even if some steps fail
set +e

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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

# Function to print warning message
print_warning() {
    echo -e "${RED}! $1${NC}"
}

# Create necessary directories
mkdir -p "$PROJECT_ROOT/models/{CTCF,GATA1,CEBPA,TP53}"
mkdir -p "$PROJECT_ROOT/results/{CTCF,GATA1,CEBPA,TP53}"
mkdir -p "$PROJECT_ROOT/data/raw/{genome,jaspar,encode/{CTCF,GATA1,CEBPA,TP53}}"
mkdir -p "$PROJECT_ROOT/data/processed/{CTCF,GATA1,CEBPA,TP53}"
mkdir -p "$PROJECT_ROOT/src"
mkdir -p "$PROJECT_ROOT/notebooks"

# List of TFs to process
TFS=("CTCF" "GATA1" "CEBPA" "TP53")

# Verify project structure before starting
print_header "VERIFYING PROJECT STRUCTURE"
if [ -f "$SCRIPT_DIR/verify_setup.py" ]; then
    python "$SCRIPT_DIR/verify_setup.py"
    if [ $? -eq 0 ]; then
        print_success "Project structure verified"
    else
        print_warning "Project structure verification failed, but continuing..."
    fi
else
    print_warning "verify_setup.py not found, skipping verification step"
    print_step "Creating minimal directory structure..."
fi

# Check for gdown for Google Drive downloads
print_step "Checking for gdown (needed for Google Drive downloads)..."
if ! command -v gdown &> /dev/null; then
    print_step "Installing gdown..."
    pip install gdown
    if [ $? -ne 0 ]; then
        print_warning "Failed to install gdown. Some downloads may fail."
    else
        print_success "gdown installed successfully"
    fi
else
    print_success "gdown is already installed"
fi

# Step 1: Download data if it doesn't exist
print_header "DOWNLOADING DATA"
if [ ! -f "$PROJECT_ROOT/data/raw/genome/hg38.fa" ] || [ ! -d "$PROJECT_ROOT/data/raw/jaspar" ] || [ ! -d "$PROJECT_ROOT/data/raw/encode" ]; then
    print_step "Running download script..."
    if [ -f "$SCRIPT_DIR/download_data.sh" ]; then
        bash "$SCRIPT_DIR/download_data.sh"
        if [ $? -eq 0 ]; then
            print_success "Data download complete"
        else
            print_warning "Data download encountered errors, but continuing..."
            # Fallback for common files
            if [ ! -f "$PROJECT_ROOT/data/raw/genome/hg38.fa" ]; then
                print_step "Attempting to download hg38 genome directly..."
                mkdir -p "$PROJECT_ROOT/data/raw/genome"
                wget -O "$PROJECT_ROOT/data/raw/genome/hg38.fa.gz" https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
                gunzip "$PROJECT_ROOT/data/raw/genome/hg38.fa.gz"
            fi
            if [ ! -d "$PROJECT_ROOT/data/raw/jaspar" ]; then
                print_step "Attempting to download JASPAR data directly..."
                mkdir -p "$PROJECT_ROOT/data/raw/jaspar"
                wget -O "$PROJECT_ROOT/data/raw/jaspar/JASPAR2022_CORE_vertebrates_non-redundant_pfms_jaspar.txt" \
                    http://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_jaspar.txt
            fi
        fi
    else
        print_warning "download_data.sh not found in scripts!"
        print_step "Please add $SCRIPT_DIR/download_data.sh or download data manually."
    fi
else
    print_success "Data already downloaded, skipping download step"
fi

# Step 2: Process data for each TF
print_header "PROCESSING DATA"
for TF in "${TFS[@]}"; do
    print_step "Processing data for $TF..."

    # Locate the ChIP-seq BED file
    shopt -s nullglob
    bed_files=( "$PROJECT_ROOT/data/raw/encode/$TF"/ENCFF*.bed )
    if [ ${#bed_files[@]} -eq 0 ]; then
        print_warning "ChIP-seq file not found for $TF in $PROJECT_ROOT/data/raw/encode/$TF"
        mkdir -p "$PROJECT_ROOT/data/processed/$TF"
        touch "$PROJECT_ROOT/data/processed/$TF"/{X_train.npy,y_train.npy,X_val.npy,y_val.npy,X_test.npy,y_test.npy}
        echo "{\"status\": \"placeholder\", \"message\": \"ChIP-seq file missing\"}" > \
            "$PROJECT_ROOT/data/processed/$TF/metadata.json"
        shopt -u nullglob
        continue
    fi
    CHIP_FILE="${bed_files[0]}"
    shopt -u nullglob

    if [ -f "$PROJECT_ROOT/src/data.py" ]; then
        python "$PROJECT_ROOT/src/data.py" --tf "$TF" \
            --jaspar-dir "$PROJECT_ROOT/data/raw/jaspar" \
            --chip-seq-file "$CHIP_FILE" \
            --genome "$PROJECT_ROOT/data/raw/genome/hg38.fa" \
            --output-dir "$PROJECT_ROOT/data/processed/$TF"
        if [ $? -ne 0 ]; then
            print_warning "Data processing for $TF failed. Creating placeholders..."
            mkdir -p "$PROJECT_ROOT/data/processed/$TF"
            touch "$PROJECT_ROOT/data/processed/$TF"/{X_train.npy,y_train.npy,X_val.npy,y_val.npy,X_test.npy,y_test.npy}
            echo "{\"status\": \"failed\", \"message\": \"Data processing failed\"}" > \
                "$PROJECT_ROOT/data/processed/$TF/metadata.json"
        else
            print_success "Data processing for $TF complete"
        fi
    else
        print_warning "src/data.py not found! Cannot process data for $TF"
        mkdir -p "$PROJECT_ROOT/data/processed/$TF"
        touch "$PROJECT_ROOT/data/processed/$TF"/{X_train.npy,y_train.npy,X_val.npy,y_val.npy,X_test.npy,y_test.npy}
        echo "{\"status\": \"placeholder\", \"message\": \"src/data.py not implemented yet\"}" > \
            "$PROJECT_ROOT/data/processed/$TF/metadata.json"
    fi

done

# Step 3: Train models for each TF
print_header "TRAINING MODELS"
for TF in "${TFS[@]}"; do
    print_step "Checking training script for $TF..."
    if [ ! -f "$SCRIPT_DIR/train_model.py" ]; then
        print_warning "train_model.py not found! Cannot train model for $TF"
        mkdir -p "$PROJECT_ROOT/models/$TF"
        echo "{\"status\": \"placeholder\", \"message\": \"train_model.py not implemented yet\"}" > \
            "$PROJECT_ROOT/models/$TF/model.json"
        touch "$PROJECT_ROOT/models/$TF/model.h5"
        continue
    fi
    print_step "Training model for $TF..."
    if [ -f "$PROJECT_ROOT/models/$TF/model.h5" ]; then
        read -p "Model for $TF already exists. Retrain? (y/n) " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Skipping training for $TF"
            continue
        fi
    fi
    python "$SCRIPT_DIR/train_model.py" --tf "$TF" \
        --data-dir "$PROJECT_ROOT/data/processed/$TF" \
        --output-dir "$PROJECT_ROOT/models/$TF"
    if [ $? -ne 0 ]; then
        print_warning "Model training for $TF failed. Creating placeholders..."
        mkdir -p "$PROJECT_ROOT/models/$TF"
        echo "{\"status\": \"failed\", \"message\": \"Model training failed\"}" > \
            "$PROJECT_ROOT/models/$TF/model.json"
        touch "$PROJECT_ROOT/models/$TF/model.h5"
    else
        print_success "Model training for $TF complete"
    fi
done

# Step 4: Evaluate models for each TF
print_header "EVALUATING MODELS"
for TF in "${TFS[@]}"; do
    print_step "Checking evaluation script for $TF..."
    if [ ! -f "$SCRIPT_DIR/evaluate_model.py" ]; then
        print_warning "evaluate_model.py not found! Cannot evaluate model for $TF"
        mkdir -p "$PROJECT_ROOT/results/$TF"
        echo "{\"status\": \"placeholder\", \"message\": \"evaluate_model.py not implemented yet\"}" > \
            "$PROJECT_ROOT/results/$TF/results.json"
        continue
    fi
    print_step "Evaluating model for $TF..."
    if [ ! -f "$PROJECT_ROOT/models/$TF/model.h5" ]; then
        print_warning "No model found for $TF. Skipping evaluation."
        mkdir -p "$PROJECT_ROOT/results/$TF"
        echo "{\"status\": \"skipped\", \"message\": \"No model available\"}" > \
            "$PROJECT_ROOT/results/$TF/results.json"
        continue
    fi
    python "$SCRIPT_DIR/evaluate_model.py" --tf "$TF" \
        --model-path "$PROJECT_ROOT/models/$TF/model.h5" \
        --data-dir "$PROJECT_ROOT/data/processed/$TF" \
        --output-dir "$PROJECT_ROOT/results/$TF" \
        --analyze-motifs
    if [ $? -ne 0 ]; then
        print_warning "Model evaluation for $TF failed. Creating placeholders..."
        mkdir -p "$PROJECT_ROOT/results/$TF"
        echo "{\"status\": \"failed\", \"message\": \"Model evaluation failed\"}" > \
            "$PROJECT_ROOT/results/$TF/results.json"
    else
        print_success "Model evaluation for $TF complete"
    fi
done

# Final step: Generate summary report
print_header "GENERATING SUMMARY"
if [ -f "$PROJECT_ROOT/notebooks/results.ipynb" ] && command -v jupyter &> /dev/null; then
    print_step "Executing results notebook to HTML..."
    jupyter nbconvert --to html --execute "$PROJECT_ROOT/notebooks/results.ipynb" \
        --output "$PROJECT_ROOT/notebooks/summary_report.html"
    if [ $? -eq 0 ]; then
        print_success "Summary report generated: notebooks/summary_report.html"
    else
        print_warning "Failed to execute notebook. Placeholder created."
        echo "<html><body><h1>Summary Report</h1><p>Execution failed.</p></body></html>" > \
            "$PROJECT_ROOT/notebooks/summary_report.html"
    fi
else
    print_warning "Notebook or Jupyter not found. Skipping summary generation."
fi

print_header "WORKFLOW COMPLETE"
echo "Results for each TF are available in the results/ directory"

echo "Missing components detected during execution:"
[ ! -f "$PROJECT_ROOT/src/data.py" ] && echo "- src/data.py (data processing)"
[ ! -f "$SCRIPT_DIR/train_model.py" ] && echo "- scripts/train_model.py (model training)"
[ ! -f "$SCRIPT_DIR/evaluate_model.py" ] && echo "- scripts/evaluate_model.py (model evaluation)"
[ ! -f "$PROJECT_ROOT/notebooks/results.ipynb" ] && echo "- notebooks/results.ipynb (results analysis)"
