#!/usr/bin/env python
"""
Script to verify the project structure and necessary data files
for the Transcription Factor Binding Prediction project.

This script focuses on verifying the presence of large data files from Google Drive
in the correct directories and offers to download them if they're missing.

Usage:
    python verify_setup.py  # Check setup only
    python verify_setup.py --download  # Check and download missing files
"""

import os
import sys
import argparse
import importlib
import subprocess
from pathlib import Path
import yaml

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Required directories - only checking essential ones
REQUIRED_DIRS = [
    "data/raw/jaspar",
    "data/raw/encode",
    "data/raw/encode/CTCF",
    "data/raw/encode/GATA1", 
    "data/raw/encode/CEBPA",
    "data/raw/encode/TP53",
    "data/raw/genome",
    "data/processed"
]

# Configuration file is required
CONFIG_FILE = "config.yaml"

# Required Python packages (only essential ones)
REQUIRED_PACKAGES = [
    "tensorflow",  # Base package name - version specifics checked separately
    "numpy", 
    "pandas", 
    "pyfaidx",     # For genome indexing
    "gdown"        # For Google Drive downloads
]

# Google Drive files that need to be downloaded
DRIVE_FILES = {
    # Format: "file_path": ("google_drive_id", file_type, size_in_mb)
    "data/raw/encode/CTCF/ENCFF002CEL.bed": ("1xUYGGQPZs_oVPKdnPMfFgRwQaKSPygGn", "ChIP-seq", 5),
    "data/raw/encode/GATA1/ENCFF002CUQ.bed": ("1sEpFXs2Z-GMp37gFtvs0vJ1ZuHIrTQOP", "ChIP-seq", 4),
    "data/raw/encode/CEBPA/ENCFF002CWG.bed": ("1qPYcblVm6zeBTiNCg3J8NeImQkiuH6BF", "ChIP-seq", 4),
    "data/raw/encode/TP53/ENCFF002CXC.bed": ("19rbwAr_E6QCNFi3fVFLNBUTlBJv6pHM8", "ChIP-seq", 3),
    "data/raw/genome/hg38.fa": ("1eKJ7M1B5zKEFQDGf93d3vHM5Z_5LLkKx", "Reference Genome", 3200)
}

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_status(message, status, color=Colors.OKGREEN):
    """Print a status message with color"""
    status_str = f"{color}{status}{Colors.ENDC}"
    print(f"{message.ljust(60)} [{status_str}]")

def check_directories():
    """Check if all required directories exist"""
    print(f"\n{Colors.BOLD}Checking directory structure...{Colors.ENDC}")
    missing_dirs = []

    for directory in REQUIRED_DIRS:
        dir_path = ROOT_DIR / directory
        if not dir_path.exists():
            missing_dirs.append(directory)
            print_status(f"Directory: {directory}", "MISSING", Colors.WARNING)
        else:
            print_status(f"Directory: {directory}", "OK")

    return missing_dirs

def check_config():
    """Check if the configuration file is valid"""
    print(f"\n{Colors.BOLD}Checking configuration file...{Colors.ENDC}")
    config_path = ROOT_DIR / CONFIG_FILE
    
    if not config_path.exists():
        print_status("Configuration file (config.yaml)", "MISSING", Colors.FAIL)
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Only check for the main sections that are necessary
        required_sections = ["data", "transcription_factors", "model"]
        missing_sections = []
        
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
                print_status(f"Config section: {section}", "MISSING", Colors.FAIL)
            else:
                print_status(f"Config section: {section}", "OK")
        
        if missing_sections:
            return False
            
        return True
    except Exception as e:
        print_status(f"Configuration file validation", "ERROR", Colors.FAIL)
        print(f"  Error details: {str(e)}")
        return False

def check_dependencies():
    """Check if essential Python packages are installed"""
    print(f"\n{Colors.BOLD}Checking Python dependencies...{Colors.ENDC}")
    missing_packages = []
    
    # First check for TensorFlow - this is platform specific
    try:
        # Try to import tensorflow
        import tensorflow as tf
        tf_version = tf.__version__
        print_status(f"Package: tensorflow (version {tf_version})", "INSTALLED")
    except ImportError:
        # If that fails, check if we're on Mac with Apple Silicon
        if os.uname().sysname == "Darwin" and os.uname().machine == "arm64":
            try:
                import tensorflow_macos
                print_status(f"Package: tensorflow-macos", "INSTALLED")
            except ImportError:
                missing_packages.append("tensorflow-macos tensorflow-metal")
                print_status(f"Package: tensorflow-macos", "MISSING", Colors.FAIL)
        else:
            missing_packages.append("tensorflow")
            print_status(f"Package: tensorflow", "MISSING", Colors.FAIL)
    
    # Check other required packages
    for package in REQUIRED_PACKAGES[1:]:  # Skip tensorflow, already checked
        try:
            module = importlib.import_module(package)
            print_status(f"Package: {package}", "INSTALLED")
        except ImportError:
            missing_packages.append(package)
            print_status(f"Package: {package}", "MISSING", Colors.FAIL)

    return missing_packages

def check_google_drive_files():
    """Check if required Google Drive files exist"""
    print(f"\n{Colors.BOLD}Checking required data files from Google Drive...{Colors.ENDC}")
    missing_files = {}
    
    for file_path, (drive_id, file_type, size_mb) in DRIVE_FILES.items():
        full_path = ROOT_DIR / file_path
        if not full_path.exists():
            print_status(f"{file_type}: {file_path} ({size_mb} MB)", "MISSING", Colors.WARNING)
            missing_files[file_path] = (drive_id, file_type, size_mb)
        else:
            # For large files like genome, check file size is reasonable
            if size_mb > 1000:  # If file is over 1GB
                actual_size_mb = full_path.stat().st_size / (1024 * 1024)
                if actual_size_mb < size_mb * 0.8:  # If file is less than 80% of expected size
                    print_status(f"{file_type}: {file_path} ({actual_size_mb:.0f}/{size_mb} MB)", 
                                 "INCOMPLETE", Colors.WARNING)
                    missing_files[file_path] = (drive_id, file_type, size_mb)
                else:
                    print_status(f"{file_type}: {file_path} ({actual_size_mb:.0f} MB)", "OK")
            else:
                print_status(f"{file_type}: {file_path}", "OK")
    
    # If genome exists, check for index
    genome_file = ROOT_DIR / "data/raw/genome/hg38.fa"
    genome_idx = ROOT_DIR / "data/raw/genome/hg38.fa.fai"
    
    if genome_file.exists() and not genome_idx.exists():
        print_status("Genome index (hg38.fa.fai)", "MISSING", Colors.WARNING)
        # The index will be created later
    elif genome_file.exists():
        print_status("Genome index (hg38.fa.fai)", "OK")
    
    return missing_files

def check_gdown_availability():
    """Check if gdown is installed and working"""
    try:
        import gdown
        print_status("gdown package for Google Drive downloads", "INSTALLED")
        return True
    except ImportError:
        print_status("gdown package for Google Drive downloads", "MISSING", Colors.FAIL)
        return False

def create_missing_directories(missing_dirs):
    """Create missing directories"""
    if not missing_dirs:
        return
        
    print(f"\n{Colors.BOLD}Creating missing directories...{Colors.ENDC}")
    for directory in missing_dirs:
        dir_path = ROOT_DIR / directory
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print_status(f"Created directory: {directory}", "OK")
        except Exception as e:
            print_status(f"Failed to create: {directory}", "ERROR", Colors.FAIL)
            print(f"  Error details: {str(e)}")

def install_missing_packages(missing_packages):
    """Prompt to install missing packages"""
    if not missing_packages:
        return
        
    print(f"\n{Colors.BOLD}Missing packages detected.{Colors.ENDC}")
    print("Run the following command to install them:")
    print(f"\npip install {' '.join(missing_packages)}\n")

def download_files(missing_files):
    """Download missing files from Google Drive"""
    if not missing_files:
        print(f"\n{Colors.OKGREEN}All required files are present.{Colors.ENDC}")
        return True
    
    try:
        import gdown
    except ImportError:
        print(f"\n{Colors.FAIL}Cannot download files: gdown package is not installed.{Colors.ENDC}")
        print("Please install it with: pip install gdown")
        return False
    
    print(f"\n{Colors.BOLD}Downloading missing files from Google Drive...{Colors.ENDC}")
    
    success = True
    for file_path, (drive_id, file_type, size_mb) in missing_files.items():
        full_path = ROOT_DIR / file_path
        
        # Create parent directory if it doesn't exist
        os.makedirs(full_path.parent, exist_ok=True)
        
        # Download file
        print(f"\nDownloading {file_type}: {file_path} ({size_mb} MB)...")
        
        try:
            url = f"https://drive.google.com/uc?id={drive_id}"
            
            # For the genome file, check if we're downloading the gzipped version
            if file_path.endswith("hg38.fa") and drive_id.endswith("LLkKx"):  # This is the genome ID
                gzip_path = str(full_path) + ".gz"
                gdown.download(url, gzip_path, quiet=False)
                
                # Check if download was successful
                if not os.path.exists(gzip_path):
                    print(f"{Colors.FAIL}Failed to download {file_path}.gz{Colors.ENDC}")
                    success = False
                    continue
                
                # Extract the file
                print(f"Extracting {gzip_path}...")
                try:
                    import gzip
                    with gzip.open(gzip_path, 'rb') as f_in:
                        with open(str(full_path), 'wb') as f_out:
                            f_out.write(f_in.read())
                    
                    # Remove gzip file after extraction
                    os.remove(gzip_path)
                    print(f"{Colors.OKGREEN}Successfully extracted {file_path}{Colors.ENDC}")
                    
                    # Create index for the genome file
                    if file_path.endswith("hg38.fa"):
                        print("Creating genome index with pyfaidx...")
                        try:
                            from pyfaidx import Fasta
                            Fasta(str(full_path), build_index=True)
                            print(f"{Colors.OKGREEN}Successfully created genome index{Colors.ENDC}")
                        except Exception as e:
                            print(f"{Colors.WARNING}Failed to create genome index: {str(e)}{Colors.ENDC}")
                    
                except Exception as e:
                    print(f"{Colors.FAIL}Failed to extract {gzip_path}: {str(e)}{Colors.ENDC}")
                    success = False
            else:
                # Direct download for other files
                gdown.download(url, str(full_path), quiet=False)
                
                # Check if download was successful
                if not os.path.exists(str(full_path)):
                    print(f"{Colors.FAIL}Failed to download {file_path}{Colors.ENDC}")
                    success = False
                else:
                    print(f"{Colors.OKGREEN}Successfully downloaded {file_path}{Colors.ENDC}")
        
        except Exception as e:
            print(f"{Colors.FAIL}Error downloading {file_path}: {str(e)}{Colors.ENDC}")
            success = False
    
    return success

def main():
    """Main function to verify project setup and download missing files"""
    parser = argparse.ArgumentParser(description='Verify project setup for TF binding prediction')
    parser.add_argument('--download', action='store_true', help='Download missing files automatically')
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}{Colors.BOLD}Transcription Factor Binding Prediction Project Setup Verification{Colors.ENDC}")
    print(f"Working directory: {ROOT_DIR}")
    
    # Check if directory structure is correct
    missing_dirs = check_directories()
    
    # If critical directories are missing, create them
    if missing_dirs:
        create_missing_directories(missing_dirs)
    
    # Check configuration file
    config_ok = check_config()
    
    # Check if required packages are installed
    missing_packages = check_dependencies()
    if missing_packages:
        install_missing_packages(missing_packages)
        
        # If gdown is missing, we can't download files
        if "gdown" in missing_packages:
            args.download = False  # Disable download
    
    # Check if Google Drive files exist
    missing_files = check_google_drive_files()
    
    # If download flag is set, download missing files
    if args.download and missing_files:
        if check_gdown_availability():
            download_success = download_files(missing_files)
            if download_success:
                # Re-check files to update status
                missing_files = check_google_drive_files()
    elif missing_files:
        print(f"\n{Colors.WARNING}Missing files detected. Run with --download to download them.{Colors.ENDC}")
    
    # Print summary
    print(f"\n{Colors.BOLD}Setup Verification Summary:{Colors.ENDC}")
    all_ok = (not missing_dirs) and config_ok and (not missing_packages) and (not missing_files)
    
    if all_ok:
        print(f"\n{Colors.OKGREEN}✓ All checks passed! The project is correctly set up.{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}! Some issues were detected in your project setup.{Colors.ENDC}")
        
        if missing_dirs:
            print(f"  {Colors.WARNING}✗ Missing directories: {', '.join(missing_dirs)}{Colors.ENDC}")
        if not config_ok:
            print(f"  {Colors.WARNING}✗ Configuration file issues{Colors.ENDC}")
        if missing_packages:
            print(f"  {Colors.WARNING}✗ Missing packages: {', '.join(missing_packages)}{Colors.ENDC}")
        if missing_files:
            print(f"  {Colors.WARNING}✗ Missing files from Google Drive: {len(missing_files)}{Colors.ENDC}")
            for file_path in missing_files:
                print(f"    - {file_path}")
            
        print(f"\nRun with --download to download missing files.")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
