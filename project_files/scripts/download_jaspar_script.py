#!/usr/bin/env python
"""
Script to download transcription factor binding motifs from JASPAR database
and prepare them for use in the TF binding prediction project.

This script:
1. Downloads PFM matrices for specified TFs from JASPAR
2. Converts them to PWM format
3. Saves them in a format compatible with the project
"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
import json
import sys
from io import StringIO
import logging
from pathlib import Path


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('jaspar_downloader')


def download_jaspar_pfm(tf_id=None, tf_name=None, version='2022', 
                      output_dir='./data/raw/jaspar'):
    """
    Download Position Frequency Matrix (PFM) for a transcription factor from JASPAR.
    
    Parameters:
    -----------
    tf_id : JASPAR matrix ID (e.g., 'MA0139.1' for CTCF)
    tf_name : Name of the TF (alternative to tf_id)
    version : JASPAR database version
    output_dir : Directory to save downloaded files
    
    Returns:
    --------
    dict: Dictionary containing file paths and metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = f"https://jaspar.genereg.net/api/v1/matrix/"
    
    # If tf_id is provided, directly fetch that matrix
    if tf_id:
        logger.info(f"Downloading matrix for TF ID: {tf_id}")
        matrix_url = f"{base_url}{tf_id}/"
        response = requests.get(matrix_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to download matrix for {tf_id}. Status code: {response.status_code}")
            return None
        
        matrix_data = response.json()
        tf_name = matrix_data.get('name')
        
    # If tf_name is provided but not tf_id, search for the TF
    elif tf_name:
        logger.info(f"Searching for TF: {tf_name}")
        search_url = f"{base_url}?name={tf_name}&version={version}&format=json"
        response = requests.get(search_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to search for {tf_name}. Status code: {response.status_code}")
            return None
        
        results = response.json().get('results', [])
        
        if not results:
            logger.error(f"No matches found for {tf_name}")
            return None
        
        # Take the first result (can be modified to be more selective)
        matrix_data = results[0]
        tf_id = matrix_data.get('matrix_id')
        
    else:
        logger.error("Either tf_id or tf_name must be provided")
        return None
    
    # Download PFM in various formats
    formats = {
        'pfm': 'PFM',
        'jaspar': 'JASPAR',
        'meme': 'MEME'
    }
    
    output_files = {}
    for fmt, label in formats.items():
        download_url = f"{base_url}{tf_id}.{fmt}"
        response = requests.get(download_url)
        
        if response.status_code == 200:
            file_path = os.path.join(output_dir, f"{tf_id}_{tf_name}.{fmt}")
            with open(file_path, 'w') as f:
                f.write(response.text)
            output_files[fmt] = file_path
            logger.info(f"Downloaded {label} format to {file_path}")
        else:
            logger.warning(f"Failed to download {label} format. Status code: {response.status_code}")
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f"{tf_id}_{tf_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(matrix_data, f, indent=2)
    output_files['metadata'] = metadata_file
    
    logger.info(f"Successfully downloaded data for {tf_name} ({tf_id})")
    
    return {
        'tf_id': tf_id,
        'tf_name': tf_name,
        'files': output_files,
        'metadata': matrix_data
    }


def parse_jaspar_pfm(pfm_file):
    """
    Parse a JASPAR PFM file and return the position frequency matrix.
    
    Parameters:
    -----------
    pfm_file : Path to PFM file
    
    Returns:
    --------
    tuple: (matrix_id, tf_name, pfm_df)
    """
    # Initialize variables
    matrix_id = None
    tf_name = None
    nucleotides = []
    counts = []
    
    with open(pfm_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Extract matrix ID and TF name
                parts = line[1:].split()
                matrix_id = parts[0]
                tf_name = ' '.join(parts[1:]) if len(parts) > 1 else None
            elif line and line[0] in 'ACGT':
                # Parse nucleotide counts
                parts = line.split()
                if len(parts) > 1:
                    nucleotides.append(parts[0])
                    counts.append([float(x) for x in parts[1:]])
    
    if not nucleotides or not counts:
        return None, None, None
    
    # Create DataFrame
    pfm_df = pd.DataFrame(counts, index=nucleotides)
    
    return matrix_id, tf_name, pfm_df


def pfm_to_pwm(pfm_file_or_df, pseudocount=0.1, background=None, output_file=None):
    """
    Convert a Position Frequency Matrix (PFM) to a Position Weight Matrix (PWM).
    
    Parameters:
    -----------
    pfm_file_or_df : Path to PFM file or DataFrame containing PFM
    pseudocount : Small value to avoid log(0)
    background : Dictionary of background probabilities (default: uniform)
    output_file : Path to save the PWM file
    
    Returns:
    --------
    pandas.DataFrame: PWM as a DataFrame
    """
    # If input is a file, parse it
    if isinstance(pfm_file_or_df, (str, Path)):
        matrix_id, tf_name, pfm_df = parse_jaspar_pfm(pfm_file_or_df)
    else:
        pfm_df = pfm_file_or_df
    
    if pfm_df is None:
        logger.error("Failed to parse PFM file or invalid DataFrame provided")
        return None
    
    # Set default background if not provided
    if background is None:
        background = {nuc: 0.25 for nuc in 'ACGT'}
    
    # Add pseudocount to avoid zeros
    pwm_df = pfm_df.copy()
    for nuc in pwm_df.index:
        pwm_df.loc[nuc] = pwm_df.loc[nuc] + pseudocount
    
    # Convert to probabilities
    col_sums = pwm_df.sum(axis=0)
    for nuc in pwm_df.index:
        pwm_df.loc[nuc] = pwm_df.loc[nuc] / col_sums
    
    # Calculate log-odds scores
    for nuc in pwm_df.index:
        pwm_df.loc[nuc] = np.log2(pwm_df.loc[nuc] / background[nuc])
    
    # Save to file if output_file provided
    if output_file:
        pwm_df.to_csv(output_file, sep='\t')
        logger.info(f"Saved PWM to {output_file}")
    
    return pwm_df


def get_tf_metadata(tf_id=None, tf_name=None, version='2022'):
    """
    Get metadata for a transcription factor from JASPAR.
    
    Parameters:
    -----------
    tf_id : JASPAR matrix ID
    tf_name : Name of the TF
    version : JASPAR database version
    
    Returns:
    --------
    dict: Dictionary of metadata
    """
    base_url = "https://jaspar.genereg.net/api/v1/matrix/"
    
    # If tf_id is provided, directly fetch that matrix
    if tf_id:
        matrix_url = f"{base_url}{tf_id}/"
        response = requests.get(matrix_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to get metadata for {tf_id}. Status code: {response.status_code}")
            return None
        
        return response.json()
        
    # If tf_name is provided but not tf_id, search for the TF
    elif tf_name:
        search_url = f"{base_url}?name={tf_name}&version={version}&format=json"
        response = requests.get(search_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to search for {tf_name}. Status code: {response.status_code}")
            return None
        
        results = response.json().get('results', [])
        
        if not results:
            logger.error(f"No matches found for {tf_name}")
            return None
        
        # Return the first result
        return results[0]
    
    else:
        logger.error("Either tf_id or tf_name must be provided")
        return None


def list_available_tfs(tax_id=None, tf_class=None, version='2022'):
    """
    List available transcription factors in JASPAR.
    
    Parameters:
    -----------
    tax_id : Taxonomy ID to filter by species (e.g., 9606 for human)
    tf_class : Filter by TF class (e.g., 'Zinc-coordinating')
    version : JASPAR database version
    
    Returns:
    --------
    pandas.DataFrame: DataFrame with TF information
    """
    base_url = "https://jaspar.genereg.net/api/v1/matrix/"
    params = {'version': version, 'format': 'json', 'page_size': 100}
    
    if tax_id:
        params['tax_id'] = tax_id
    
    if tf_class:
        params['class'] = tf_class
    
    all_results = []
    page = 1
    
    while True:
        params['page'] = page
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            logger.error(f"Failed to list TFs. Status code: {response.status_code}")
            break
        
        data = response.json()
        results = data.get('results', [])
        
        if not results:
            break
        
        all_results.extend(results)
        
        if not data.get('next'):
            break
        
        page += 1
    
    if not all_results:
        logger.warning("No TFs found matching the criteria")
        return pd.DataFrame()
    
    # Extract relevant information
    tf_info = []
    for result in all_results:
        tf_info.append({
            'matrix_id': result.get('matrix_id'),
            'name': result.get('name'),
            'class': result.get('class'),
            'family': result.get('family'),
            'species': ', '.join([sp.get('name') for sp in result.get('species', [])]),
            'tax_id': ', '.join([str(sp.get('tax_id')) for sp in result.get('species', [])]),
            'collection': result.get('collection'),
            'version': result.get('version')
        })
    
    return pd.DataFrame(tf_info)


def download_and_prepare_tfs(tf_list, output_dir='./data/raw/jaspar', version='2022'):
    """
    Download and prepare data for multiple transcription factors.
    
    Parameters:
    -----------
    tf_list : List of TF IDs or names
    output_dir : Directory to save downloaded files
    version : JASPAR database version
    
    Returns:
    --------
    dict: Dictionary mapping TF IDs to their data
    """
    os.makedirs(output_dir, exist_ok=True)
    pwm_dir = os.path.join(output_dir, 'pwm')
    os.makedirs(pwm_dir, exist_ok=True)
    
    results = {}
    
    for tf in tf_list:
        # Check if tf is ID or name
        if tf.startswith('MA'):
            tf_data = download_jaspar_pfm(tf_id=tf, version=version, output_dir=output_dir)
        else:
            tf_data = download_jaspar_pfm(tf_name=tf, version=version, output_dir=output_dir)
        
        if tf_data:
            # Convert PFM to PWM
            pfm_file = tf_data['files'].get('pfm')
            if pfm_file:
                tf_id = tf_data['tf_id']
                tf_name = tf_data['tf_name']
                pwm_file = os.path.join(pwm_dir, f"{tf_id}_{tf_name}.pwm")
                pwm = pfm_to_pwm(pfm_file, output_file=pwm_file)
                tf_data['files']['pwm'] = pwm_file
                tf_data['pwm'] = pwm
            
            results[tf_data['tf_id']] = tf_data
    
    # Create a summary file
    summary_data = []
    for tf_id, data in results.items():
        summary_data.append({
            'matrix_id': tf_id,
            'name': data['tf_name'],
            'pfm_file': data['files'].get('pfm'),
            'pwm_file': data['files'].get('pwm'),
            'metadata_file': data['files'].get('metadata')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'tf_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"Downloaded and prepared data for {len(results)} transcription factors")
    logger.info(f"Summary saved to {summary_file}")
    
    return results


def main():
    """
    Main function to run the data download script.
    """
    parser = argparse.ArgumentParser(description='Download TF data from JASPAR')
    parser.add_argument('--tf', type=str, help='Transcription factor ID or name')
    parser.add_argument('--tf-list', type=str, help='Comma-separated list of TF IDs or names')
    parser.add_argument('--list', action='store_true', help='List available TFs')
    parser.add_argument('--species', type=str, default='human', help='Species (human, mouse, etc.)')
    parser.add_argument('--tax-id', type=int, help='Taxonomy ID (e.g., 9606 for human)')
    parser.add_argument('--output-dir', type=str, default='./data/jaspar', help='Output directory')
    parser.add_argument('--version', type=str, default='2022', help='JASPAR version')
    
    args = parser.parse_args()
    
    # Set up human taxonomy ID if species is provided
    if args.species and not args.tax_id:
        species_to_tax = {
            'human': 9606,
            'mouse': 10090,
            'rat': 10116,
            'fruit fly': 7227,
            'nematode': 6239,
            'zebrafish': 7955,
            'yeast': 4932
        }
        args.tax_id = species_to_tax.get(args.species.lower())
    
    # List available TFs
    if args.list:
        logger.info(f"Listing available TFs for {'tax_id=' + str(args.tax_id) if args.tax_id else 'all species'}")
        tf_df = list_available_tfs(tax_id=args.tax_id, version=args.version)
        
        if not tf_df.empty:
            print(tf_df.to_string())
            # Save to CSV
            output_file = os.path.join(args.output_dir, 'available_tfs.csv')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            tf_df.to_csv(output_file, index=False)
            logger.info(f"Saved available TFs to {output_file}")
        return
    
    # Download a single TF
    if args.tf:
        logger.info(f"Downloading data for TF: {args.tf}")
        tf_data = download_jaspar_pfm(
            tf_id=args.tf if args.tf.startswith('MA') else None,
            tf_name=None if args.tf.startswith('MA') else args.tf,
            version=args.version,
            output_dir=args.output_dir
        )
        
        if tf_data:
            # Convert PFM to PWM
            pfm_file = tf_data['files'].get('pfm')
            if pfm_file:
                pwm_dir = os.path.join(args.output_dir, 'pwm')
                os.makedirs(pwm_dir, exist_ok=True)
                
                tf_id = tf_data['tf_id']
                tf_name = tf_data['tf_name']
                pwm_file = os.path.join(pwm_dir, f"{tf_id}_{tf_name}.pwm")
                pfm_to_pwm(pfm_file, output_file=pwm_file)
                logger.info(f"Converted PFM to PWM: {pwm_file}")
        return
    
    # Download multiple TFs
    if args.tf_list:
        tf_list = [tf.strip() for tf in args.tf_list.split(',')]
        logger.info(f"Downloading data for {len(tf_list)} TFs: {', '.join(tf_list)}")
        download_and_prepare_tfs(tf_list, output_dir=args.output_dir, version=args.version)
        return
    
    # If no specific action, show help
    parser.print_help()


if __name__ == "__main__":
    main()
