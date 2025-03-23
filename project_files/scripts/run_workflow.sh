#!/usr/bin/env bash
# Master script to run the entire TF binding prediction workflow
# This script runs all steps in a logical order from data download to model evaluation
# Modified to handle missing source files and ensure Google Drive downloads work

# Don't exit on error - we want to continue even if some steps fail
set +e

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
mkdir -p models/{CTCF,GATA1,CEBPA,TP53}
mkdir -p results/{CTCF,GATA1,CEBPA,TP53}
mkdir -p data/raw/{genome,jaspar,encode/{CTCF,GATA1,CEBPA,TP53}}
mkdir -p data/processed/{CTCF,GATA1,CEBPA,TP53}
mkdir -p src
mkdir -p notebooks

# List of TFs to process
TFS=("CTCF" "GATA1" "CEBPA" "TP53")

# Verify project structure before starting
print_header "VERIFYING PROJECT STRUCTURE"
if [ -f scripts/verify_setup.py ]; then
    python scripts/verify_setup.py
    if [ $? -eq 0 ]; then
        print_success "Project structure verified"
    else
        print_warning "Project structure verification failed, but continuing..."
    fi
else
    print_warning "verify_setup.py not found, skipping verification step"
    # Create minimal directories needed
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
if [ ! -f data/raw/genome/hg38.fa ] || [ ! -d data/raw/jaspar ] || [ ! -d data/raw/encode ]; then
    print_step "Running download script..."
    if [ -f download_data.sh ]; then
        bash scripts/download_data.sh
        if [ $? -eq 0 ]; then
            print_success "Data download complete"
        else
            print_warning "Data download encountered errors, but continuing..."
            
            # Fallback for common files if download script fails
            if [ ! -f data/raw/genome/hg38.fa ]; then
                print_step "Attempting to download hg38 genome directly..."
                mkdir -p data/raw/genome
                wget -O data/raw/genome/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
                gunzip data/raw/genome/hg38.fa.gz
            fi
            
            # Attempt to download JASPAR data
            if [ ! -d data/raw/jaspar/CTCF ]; then
                print_step "Attempting to download JASPAR data directly..."
                mkdir -p data/raw/jaspar
                wget -O data/raw/jaspar/JASPAR2022_CORE_vertebrates_non-redundant_pfms_jaspar.txt http://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_jaspar.txt
            fi
        fi
    else
        print_warning "download_data.sh not found!"
        print_step "Creating a minimal download_data.sh script..."
        
        # Create a minimal download script
        cat > scripts/download_data.sh << 'EOF'
#!/usr/bin/env bash
# Minimal download script for TF binding prediction project

set -e
mkdir -p data/raw/genome
mkdir -p data/raw/jaspar
mkdir -p data/raw/encode/{CTCF,GATA1,CEBPA,TP53}

# Download hg38 genome (small part for testing)
echo "Downloading hg38 genome (chr21 only for testing)..."
wget -O data/raw/genome/hg38.chr21.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz
gunzip -f data/raw/genome/hg38.chr21.fa.gz
mv data/raw/genome/hg38.chr21.fa data/raw/genome/hg38.fa

# Download JASPAR data
echo "Downloading JASPAR data..."
wget -O data/raw/jaspar/JASPAR2022_CORE_vertebrates_non-redundant_pfms_jaspar.txt http://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_jaspar.txt

# Download ENCODE ChIP-seq data
echo "Downloading sample ENCODE ChIP-seq data..."
# CTCF
wget -O data/raw/encode/CTCF/ENCFF002CEL.bed.gz https://www.encodeproject.org/files/ENCFF002CEL/@@download/ENCFF002CEL.bed.gz
gunzip -f data/raw/encode/CTCF/ENCFF002CEL.bed.gz

# CEBPA (already exists in project)
if [ ! -f data/raw/encode/CEBPA/ENCFF002CWG.bed ]; then
    cp project_files/data/raw/encode/CEBPA/ENCFF002CWG.bed data/raw/encode/CEBPA/
fi

# Use gdown for Google Drive files if needed
if command -v gdown &> /dev/null; then
    echo "Using gdown to download additional files..."
    # Add gdown commands here if you have Google Drive file IDs
    # Example: gdown --id YOUR_GOOGLE_DRIVE_FILE_ID -O output_path
fi

echo "Download complete!"
EOF
        
        chmod +x scripts/download_data.sh
        bash scripts/download_data.sh
        if [ $? -ne 0 ]; then
            print_warning "Auto-generated download script failed. You may need to download data manually."
        else
            print_success "Basic data download successful"
        fi
    fi
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
        continue
    fi
    
    # Check if data.py exists
    if [ ! -f "src/data.py" ]; then
        print_warning "src/data.py not found! Cannot process data for $TF"
        
        # Create dummy processed data
        print_step "Creating placeholder processed data files..."
        mkdir -p "data/processed/$TF"
        touch "data/processed/$TF/X_train.npy"
        touch "data/processed/$TF/y_train.npy"
        touch "data/processed/$TF/X_val.npy"
        touch "data/processed/$TF/y_val.npy"
        touch "data/processed/$TF/X_test.npy"
        touch "data/processed/$TF/y_test.npy"
        touch "data/processed/$TF/metadata.json"
        echo "{\"status\": \"placeholder\", \"message\": \"src/data.py not implemented yet\"}" > "data/processed/$TF/metadata.json"
        continue
    fi
    
    # Process data
    python src/data.py --tf $TF \
                      --jaspar-dir data/raw/jaspar \
                      --chip-seq-file $CHIP_FILE \
                      --genome data/raw/genome/hg38.fa \
                      --output-dir data/processed/$TF
    
    if [ $? -ne 0 ]; then
        print_warning "Data processing for $TF failed. Creating placeholders..."
        # Create dummy processed data
        mkdir -p "data/processed/$TF"
        touch "data/processed/$TF/X_train.npy"
        touch "data/processed/$TF/y_train.npy"
        touch "data/processed/$TF/X_val.npy"
        touch "data/processed/$TF/y_val.npy"
        touch "data/processed/$TF/X_test.npy"
        touch "data/processed/$TF/y_test.npy"
        echo "{\"status\": \"failed\", \"message\": \"Data processing failed\"}" > "data/processed/$TF/metadata.json"
    else
        print_success "Data processing for $TF complete"
    fi
done

# Step 3: Train models for each TF
print_header "TRAINING MODELS"
for TF in "${TFS[@]}"; do
    print_step "Checking training script for $TF..."
    
    # Check if training script exists
    if [ ! -f "scripts/train_model.py" ]; then
        print_warning "scripts/train_model.py not found! Cannot train model for $TF"
        # Create dummy model
        mkdir -p "models/$TF"
        echo "{\"status\": \"placeholder\", \"message\": \"train_model.py not implemented yet\"}" > "models/$TF/model.json"
        touch "models/$TF/model.h5"
        continue
    fi
    
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
    
    if [ $? -ne 0 ]; then
        print_warning "Model training for $TF failed. Creating placeholders..."
        # Create dummy model files
        mkdir -p "models/$TF"
        echo "{\"status\": \"failed\", \"message\": \"Model training failed\"}" > "models/$TF/model.json"
        touch "models/$TF/model.h5"
    else
        print_success "Model training for $TF complete"
    fi
done

# Step 4: Evaluate models for each TF
print_header "EVALUATING MODELS"
for TF in "${TFS[@]}"; do
    print_step "Checking evaluation script for $TF..."
    
    # Check if evaluation script exists
    if [ ! -f "scripts/evaluate_model.py" ]; then
        print_warning "scripts/evaluate_model.py not found! Cannot evaluate model for $TF"
        # Create dummy results
        mkdir -p "results/$TF"
        echo "{\"status\": \"placeholder\", \"message\": \"evaluate_model.py not implemented yet\"}" > "results/$TF/results.json"
        continue
    fi
    
    print_step "Evaluating model for $TF..."
    
    # Check if model exists before evaluation
    if [ ! -f "models/$TF/model.h5" ]; then
        print_warning "No model found for $TF. Skipping evaluation."
        # Create dummy results
        mkdir -p "results/$TF"
        echo "{\"status\": \"skipped\", \"message\": \"No model available\"}" > "results/$TF/results.json"
        continue
    fi
    
    # Evaluate model
    python scripts/evaluate_model.py --tf $TF \
                                    --model-path models/$TF/model.h5 \
                                    --data-dir data/processed/$TF \
                                    --output-dir results/$TF \
                                    --analyze-motifs
    
    if [ $? -ne 0 ]; then
        print_warning "Model evaluation for $TF failed. Creating placeholders..."
        # Create dummy results
        mkdir -p "results/$TF"
        echo "{\"status\": \"failed\", \"message\": \"Model evaluation failed\"}" > "results/$TF/results.json"
    else
        print_success "Model evaluation for $TF complete"
    fi
done

# Final step: Generate summary report
print_header "GENERATING SUMMARY"
print_step "Checking for results notebook..."

# Check if results notebook exists
if [ ! -f "notebooks/results.ipynb" ]; then
    print_warning "notebooks/results.ipynb not found! Cannot generate summary report."
    # Create dummy summary
    echo "<html><body><h1>Summary Report</h1><p>This is a placeholder. The actual notebook needs to be created.</p></body></html>" > "notebooks/summary_report.html"
else
    print_step "Running Jupyter notebook..."
    
    # Check if Jupyter is installed
    if command -v jupyter &> /dev/null; then
        # Execute the results notebook to generate report
        jupyter nbconvert --to html --execute notebooks/results.ipynb --output summary_report.html
        if [ $? -eq 0 ]; then
            print_success "Summary report generated: notebooks/summary_report.html"
        else
            print_warning "Failed to execute notebook. Creating placeholder..."
            echo "<html><body><h1>Summary Report</h1><p>This is a placeholder. The notebook execution failed.</p></body></html>" > "notebooks/summary_report.html"
        fi
    else
        print_warning "Jupyter not found. Please run notebooks/results.ipynb manually to generate the summary report."
        echo "<html><body><h1>Summary Report</h1><p>This is a placeholder. Please run the notebook manually.</p></body></html>" > "notebooks/summary_report.html"
    fi
fi

print_header "WORKFLOW COMPLETE"
echo "Results for each TF are available in the results/ directory"
echo "You can analyze the results further using the notebooks/results.ipynb notebook"
echo ""
echo "Missing components detected during execution:"
[ ! -f "src/data.py" ] && echo "- src/data.py (data processing)"
[ ! -f "scripts/train_model.py" ] && echo "- scripts/train_model.py (model training)"
[ ! -f "scripts/evaluate_model.py" ] && echo "- scripts/evaluate_model.py (model evaluation)"
[ ! -f "notebooks/results.ipynb" ] && echo "- notebooks/results.ipynb (results analysis)"

echo ""
echo "This script created placeholder files where needed, so your colleagues can"
echo "see the expected structure and implement the missing components."
