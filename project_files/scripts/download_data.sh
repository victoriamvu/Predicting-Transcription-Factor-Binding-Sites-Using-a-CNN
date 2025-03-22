#!/usr/bin/env bash
# Script to download all necessary data for TF binding prediction project

# Exit on any error
set -e

# Function to check if a file exists
file_exists() {
    if [ -f "$1" ]; then
        echo "File already exists: $1"
        return 0
    else
        return 1
    fi
}

# Function to check if a directory exists
dir_exists() {
    if [ -d "$1" ]; then
        echo "Directory already exists: $1"
        return 0
    else
        mkdir -p "$1"
        return 1
    fi
}

# Create necessary directories (won't fail if they already exist)
echo "Setting up directories..."
mkdir -p data/raw/jaspar
mkdir -p data/raw/encode/{CTCF,GATA1,CEBPA,TP53}
mkdir -p data/raw/genome

# Download JASPAR data if not already present
echo "Checking JASPAR data..."
if ! ls data/raw/jaspar/MA0139.1_*.* &> /dev/null; then
    echo "Downloading CTCF, GATA1, CEBPA from JASPAR..."
    python scripts/download_jaspar_script.py --tf-list "CTCF,GATA1,CEBPA" --output-dir ./data/raw/jaspar
else
    echo "JASPAR data for CTCF, GATA1, CEBPA already exists, skipping download"
fi

if ! ls data/raw/jaspar/MA0106.3_*.* &> /dev/null; then
    echo "Downloading TP53 (p53) from JASPAR..."
    python scripts/download_jaspar_script.py --tf MA0106.3 --output-dir ./data/raw/jaspar
else
    echo "JASPAR data for TP53 already exists, skipping download"
fi

# Download reference genome if not already present
GENOME_FILE="data/raw/genome/hg38.fa"
GENOME_GZ="${GENOME_FILE}.gz"

if ! file_exists "$GENOME_FILE"; then
    echo "Downloading reference genome (hg38)..."
    if ! file_exists "$GENOME_GZ"; then
        wget -c https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -P data/raw/genome
    else
        echo "Compressed genome file already exists, skipping download"
    fi
    
    echo "Extracting genome file..."
    gunzip -k "$GENOME_GZ"
else
    echo "Genome file already exists, skipping download and extraction"
fi

# Download ChIP-seq data from ENCODE if not already present
echo "Checking ChIP-seq data from ENCODE..."

# CTCF ChIP-seq peaks
CTCF_BED="data/raw/encode/CTCF/ENCFF002CEL.bed"
CTCF_BED_GZ="${CTCF_BED}.gz"
if ! file_exists "$CTCF_BED"; then
    echo "Downloading CTCF ChIP-seq peaks..."
    if ! file_exists "$CTCF_BED_GZ"; then
        wget -c https://www.encodeproject.org/files/ENCFF002CEL/@@download/ENCFF002CEL.bed.gz -P data/raw/encode/CTCF
    else
        echo "Compressed CTCF file already exists, skipping download"
    fi
    gunzip -k "$CTCF_BED_GZ"
else
    echo "CTCF ChIP-seq data already exists, skipping download"
fi

# GATA1 ChIP-seq peaks
GATA1_BED="data/raw/encode/GATA1/ENCFF002CUQ.bed"
GATA1_BED_GZ="${GATA1_BED}.gz"
if ! file_exists "$GATA1_BED"; then
    echo "Downloading GATA1 ChIP-seq peaks..."
    if ! file_exists "$GATA1_BED_GZ"; then
        wget -c https://www.encodeproject.org/files/ENCFF002CUQ/@@download/ENCFF002CUQ.bed.gz -P data/raw/encode/GATA1
    else
        echo "Compressed GATA1 file already exists, skipping download"
    fi
    gunzip -k "$GATA1_BED_GZ"
else
    echo "GATA1 ChIP-seq data already exists, skipping download"
fi

# CEBPA ChIP-seq peaks
CEBPA_BED="data/raw/encode/CEBPA/ENCFF002CWG.bed"
CEBPA_BED_GZ="${CEBPA_BED}.gz"
if ! file_exists "$CEBPA_BED"; then
    echo "Downloading CEBPA ChIP-seq peaks..."
    if ! file_exists "$CEBPA_BED_GZ"; then
        wget -c https://www.encodeproject.org/files/ENCFF002CWG/@@download/ENCFF002CWG.bed.gz -P data/raw/encode/CEBPA
    else
        echo "Compressed CEBPA file already exists, skipping download"
    fi
    gunzip -k "$CEBPA_BED_GZ"
else
    echo "CEBPA ChIP-seq data already exists, skipping download"
fi

# TP53 (p53) ChIP-seq peaks
TP53_BED="data/raw/encode/TP53/ENCFF002CXC.bed"
TP53_BED_GZ="${TP53_BED}.gz"
if ! file_exists "$TP53_BED"; then
    echo "Downloading TP53 ChIP-seq peaks..."
    if ! file_exists "$TP53_BED_GZ"; then
        wget -c https://www.encodeproject.org/files/ENCFF002CXC/@@download/ENCFF002CXC.bed.gz -P data/raw/encode/TP53
    else
        echo "Compressed TP53 file already exists, skipping download"
    fi
    gunzip -k "$TP53_BED_GZ"
else
    echo "TP53 ChIP-seq data already exists, skipping download"
fi

# Index genome with pyfaidx for faster access
echo "Checking genome index..."
if ! file_exists "${GENOME_FILE}.fai"; then
    echo "Indexing genome..."
    python -c "from pyfaidx import Fasta; Fasta('${GENOME_FILE}')"
else
    echo "Genome index already exists, skipping indexing"
fi

echo "Data setup complete!"
echo "Downloaded data structure:"
echo "  - JASPAR motifs: data/raw/jaspar/"
echo "  - ChIP-seq data: data/raw/encode/"
echo "  - Reference genome: data/raw/genome/hg38.fa"
echo ""
echo "Next steps:"
echo "1. Process the raw data using src/data.py"
echo "2. Train models using scripts/train_model.py"
echo "3. Evaluate results using scripts/evaluate_model.py"
