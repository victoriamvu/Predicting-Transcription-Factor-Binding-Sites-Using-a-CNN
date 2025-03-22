#!/usr/bin/env python
"""
Simple verification script to check if all necessary files and directories
exist in the project structure.
"""

import os
import sys
import logging
from pathlib import Path


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger('verify_setup')


def check_directory(directory):
    """Check if a directory exists."""
    if os.path.exists(directory):
        if os.path.isdir(directory):
            logger.info(f"✓ Directory exists: {directory}")
            return True
        else:
            logger.error(f"✗ Path exists but is not a directory: {directory}")
            return False
    else:
        logger.error(f"✗ Directory not found: {directory}")
        return False


def check_file(file_path, min_size_kb=0.1):
    """Check if a file exists and has content."""
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            if size_kb >= min_size_kb:
                logger.info(f"✓ File exists ({size_kb:.2f} KB): {file_path}")
                return True
            else:
                logger.warning(f"⚠ File exists but may be empty ({size_kb:.2f} KB): {file_path}")
                return False
        else:
            logger.error(f"✗ Path exists but is not a file: {file_path}")
            return False
    else:
        logger.error(f"✗ File not found: {file_path}")
        return False


def check_directory_structure():
    """Check if the basic directory structure exists."""
    directories = [
        ".",
        "data",
        "data/raw",
        "data/processed",
        "data/raw/jaspar",
        "data/raw/encode",
        "data/raw/genome",
        "notebooks",
        "src",
        "scripts",
        "tests"
    ]
    
    all_good = True
    for directory in directories:
        if not check_directory(directory):
            all_good = False
    
    return all_good


def check_source_files():
    """Check if all source files exist."""
    source_files = [
        "src/data.py",
        "src/model.py",
        "src/train.py",
        "src/evaluate.py",
        "config.yaml",
        "README.md",
        "requirements.txt",
        "scripts/download_data.sh",
        "tests/test_data.py"
    ]
    
    all_good = True
    for file_path in source_files:
        if not check_file(file_path):
            all_good = False
    
    # Check script files that might be generated
    script_files = [
        "scripts/download_jaspar_script.py",
        "scripts/train_model.py",
        "scripts/evaluate_model.py"
    ]
    
    for file_path in script_files:
        if os.path.exists(file_path):
            check_file(file_path)
    
    return all_good


def check_data_files():
    """Check if data files exist."""
    # Check JASPAR files
    jaspar_dir = "data/raw/jaspar"
    if check_directory(jaspar_dir):
        # Look for PFM files (MA*.pfm)
        pfm_files = list(Path(jaspar_dir).glob("MA*.pfm"))
        if pfm_files:
            logger.info(f"✓ Found {len(pfm_files)} JASPAR PFM files")
            for pfm_file in pfm_files:
                check_file(pfm_file)
        else:
            logger.error(f"✗ No JASPAR PFM files found in {jaspar_dir}")
            logger.error("  Run: scripts/download_jaspar_script.py to download JASPAR data")
    
    # Check ENCODE files
    encode_dir = "data/raw/encode"
    if check_directory(encode_dir):
        # Look for TF directories
        tf_dirs = [d for d in os.listdir(encode_dir) if os.path.isdir(os.path.join(encode_dir, d))]
        if tf_dirs:
            logger.info(f"✓ Found {len(tf_dirs)} TF directories in ENCODE data")
            
            bed_files_found = False
            for tf_dir in tf_dirs:
                tf_path = os.path.join(encode_dir, tf_dir)
                bed_files = list(Path(tf_path).glob("*.bed"))
                if bed_files:
                    bed_files_found = True
                    logger.info(f"✓ Found ChIP-seq BED file(s) for {tf_dir}")
                    for bed_file in bed_files:
                        check_file(bed_file)
            
            if not bed_files_found:
                logger.error("✗ No ChIP-seq BED files found")
                logger.error("  Run: scripts/download_data.sh to download ChIP-seq data")
        else:
            logger.error(f"✗ No TF directories found in {encode_dir}")
            logger.error("  Run: scripts/download_data.sh to download ChIP-seq data")
    
    # Check genome file
    genome_file = "data/raw/genome/hg38.fa"
    genome_index = "data/raw/genome/hg38.fa.fai"
    
    if check_file(genome_file, min_size_kb=100000):  # Expect genome to be large
        logger.info(f"✓ Genome file exists")
        if os.path.exists(genome_index):
            logger.info(f"✓ Genome index exists: {genome_index}")
        else:
            logger.warning(f"⚠ Genome index not found: {genome_index}")
            logger.warning("  Run: python -c \"from pyfaidx import Fasta; Fasta('data/raw/genome/hg38.fa')\"")
    else:
        logger.error(f"✗ Genome file not found or too small: {genome_file}")
        logger.error("  Run: scripts/download_data.sh to download the reference genome")
    
    # Check processed data
    processed_dir = "data/processed"
    if check_directory(processed_dir):
        # Look for TF directories with processed data
        tf_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        if tf_dirs:
            logger.info(f"✓ Found {len(tf_dirs)} TF directories with processed data")
            for tf_dir in tf_dirs:
                tf_path = os.path.join(processed_dir, tf_dir)
                npy_files = list(Path(tf_path).glob("*.npy"))
                if npy_files:
                    logger.info(f"✓ Found processed data for {tf_dir}")
                else:
                    logger.warning(f"⚠ No processed data files (*.npy) found for {tf_dir}")
        else:
            logger.warning(f"⚠ No processed data directories found in {processed_dir}")
            logger.warning("  Need to process raw data using src/data.py")


def main():
    """Main function."""
    logger.info("Verifying project structure...")
    
    # Check directories
    dir_check = check_directory_structure()
    
    # Check source files
    src_check = check_source_files()
    
    # Check data files
    check_data_files()
    
    # Summary and next steps
    logger.info("\n=== Summary ===")
    if dir_check and src_check:
        logger.info("✅ Basic project structure is complete")
        
        # Check for specific files to determine next steps
        if not os.path.exists("data/raw/jaspar/MA0139.1_CTCF.pfm"):
            logger.info("\nNext steps:")
            logger.info("1. Download JASPAR data:")
            logger.info("   python scripts/download_jaspar_script.py --tf-list \"CTCF,GATA1,CEBPA,TP53\" --output-dir ./data/raw/jaspar")
            return
        
        if not os.path.exists("data/raw/genome/hg38.fa"):
            logger.info("\nNext steps:")
            logger.info("1. Download reference genome and ChIP-seq data:")
            logger.info("   bash scripts/download_data.sh")
            return
        
        if not any(Path("data/processed").glob("*/*.npy")):
            logger.info("\nNext steps:")
            logger.info("1. Process data for TF binding prediction:")
            logger.info("   python src/data.py --tf CTCF --jaspar-dir data/raw/jaspar --chip-seq-file data/raw/encode/CTCF/*.bed --genome data/raw/genome/hg38.fa --output-dir data/processed/CTCF")
            return
        
        logger.info("\nNext steps:")
        logger.info("1. Train models:")
        logger.info("   python scripts/train_model.py --tf CTCF --data-dir data/processed/CTCF --output-dir models/CTCF")
        logger.info("2. Evaluate models:")
        logger.info("   python scripts/evaluate_model.py --tf CTCF --model-path models/CTCF/model.h5 --data-dir data/processed/CTCF --output-dir results/CTCF")
    else:
        logger.warning("⚠ Some parts of the project structure are missing")
        logger.info("\nNext steps:")
        logger.info("1. Create missing directories and files")
        logger.info("2. Run this verification script again")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
