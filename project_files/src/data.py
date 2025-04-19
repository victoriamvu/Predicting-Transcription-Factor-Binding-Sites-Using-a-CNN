#!/usr/bin/env python
"""
Enhanced data processing module for TF binding prediction.

This module handles loading and processing of transcription factor binding data
from JASPAR and ChIP-seq experiments with improved error handling, data validation,
and additional features.
"""

import os
import argparse
import logging
import json
import random
from datetime import datetime
from Bio import motifs
from Bio.Seq import Seq
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pyfaidx import Fasta


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tf_binding_data")


def parse_jaspar_pfm(pfm_file):
    """
    Parse a JASPAR PFM file and return the position frequency matrix.

    Args:
        pfm_file: Path to the PFM file

    Returns:
        tuple: (matrix_id, tf_name, pfm_df)

    Raises:
        FileNotFoundError: If PFM file doesn't exist
        ValueError: If PFM file is malformed
    """
    try:
        with open(pfm_file, "r") as f:
            motif = motifs.read(f, "jaspar")
    except FileNotFoundError:
        logger.error(f"File {pfm_file} not found.")
        raise
    except Exception as e:
        logger.error(f"Error parsing {pfm_file}: {str(e)}")
        raise ValueError(f"Malformed PFM file: {pfm_file}")

    pfm_data = {base: motif.counts[base] for base in "ACGT"}
    pfm_df = pd.DataFrame(data=pfm_data)

    return motif.matrix_id, motif.name, pfm_df


def pfm_to_pwm(pfm_df, pseudocount=0.1):
    """
    Convert a Position Frequency Matrix (PFM) to a Position Weight Matrix (PWM).

    Args:
        pfm_df: PFM as a DataFrame with rows for A, C, G, T
        pseudocount: Small value to avoid log(0)

    Returns:
        numpy.ndarray: PWM as a 2D array
    """
    pfm = np.array(pfm_df)
    num_seqs = np.sum(pfm, axis=0, keepdims=True)
    ppm = (pfm + pseudocount) / (num_seqs + 4 * pseudocount)
    pwm = np.log2(ppm / 0.25)  # Using log2 for bits

    return pwm


def one_hot_encode(sequence):
    """
    Convert a DNA sequence to one-hot encoding.

    Args:
        sequence: DNA sequence string

    Returns:
        numpy.ndarray: One-hot encoded matrix with shape (len(sequence), 4)
    """
    base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    one_hot = np.zeros((len(sequence), 4))

    for i, base in enumerate(sequence.upper()):
        if base in base_to_index:
            one_hot[i, base_to_index[base]] = 1
        # Else leave as zero (handles N's and other characters)

    return one_hot


def reverse_complement(sequence):
    """
    Get the reverse complement of a DNA sequence.

    Args:
        sequence: DNA sequence string

    Returns:
        str: Reverse complement of the sequence
    """
    complement = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
    }
    return "".join(complement.get(base, "N") for base in reversed(sequence.upper()))


def extract_sequences_from_chip(bed_file, genome_file, sequence_length=200):
    """
    Extract DNA sequences centered on ChIP-seq peaks.

    Args:
        bed_file: Path to BED file with ChIP-seq peaks
        genome_file: Path to reference genome FASTA file
        sequence_length: Length of sequences to extract

    Returns:
        list: Extracted DNA sequences

    Raises:
        ValueError: If inputs are invalid or processing fails
    """
    if not os.path.exists(bed_file):
        raise ValueError(f"BED file not found: {bed_file}")
    if not os.path.exists(genome_file):
        raise ValueError(f"Genome file not found: {genome_file}")

    try:
        genome = Fasta(genome_file)
    except Exception as e:
        raise ValueError(f"Failed to load genome: {str(e)}")

    sequences = []
    half_len = sequence_length // 2

    try:
        with open(bed_file, "r") as bed:
            for line_num, line in enumerate(bed, 1):
                if line.startswith("#") or line.strip() == "":
                    continue

                try:
                    parts = line.strip().split("\t")
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])

                    # Validate coordinates
                    if start >= end:
                        logger.warning(
                            f"Invalid coordinates in BED line {line_num}: start >= end"
                        )
                        continue

                    peak_center = (start + end) // 2
                    seq_start = peak_center - half_len
                    seq_end = peak_center + half_len

                    # Check chromosome exists in genome
                    if chrom not in genome:
                        logger.warning(f"Chromosome {chrom} not found in genome")
                        continue

                    # Check bounds
                    if seq_start < 0 or seq_end > len(genome[chrom]):
                        logger.warning(
                            f"Sequence bounds out of range in line {line_num}"
                        )
                        continue

                    sequence = genome[chrom][seq_start:seq_end].seq.upper()

                    # Validate sequence
                    if len(sequence) == sequence_length and "N" not in sequence:
                        sequences.append(sequence)
                    else:
                        logger.debug(
                            f"Skipping sequence with N's or wrong length in line {line_num}"
                        )

                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping malformed BED line {line_num}: {str(e)}")
                    continue

    except Exception as e:
        raise ValueError(f"Error processing BED file: {str(e)}")

    if not sequences:
        raise ValueError("No valid sequences extracted from BED file")

    return sequences


def generate_negative_samples(
    positive_sequences, method="dinucleotide_shuffle", genome_file=None, n_samples=None
):
    """
    Generate negative samples for training.

    Args:
        positive_sequences: List of positive DNA sequences
        method: Generation method ('shuffle', 'dinucleotide_shuffle', 'random_genomic')
        genome_file: Path to genome FASTA (required for 'random_genomic')
        n_samples: Number of negatives to generate (default: match positives)

    Returns:
        list: Negative DNA sequences

    Raises:
        ValueError: For invalid inputs or methods
    """
    if not positive_sequences:
        raise ValueError("No positive sequences provided")

    seq_len = len(positive_sequences[0])
    n_samples = n_samples or len(positive_sequences)
    negatives = []

    if method == "dinucleotide_shuffle":
        for seq in positive_sequences[:n_samples]:
            try:
                shuffled = str(Seq(seq).dinucleotide_shuffle())
                negatives.append(shuffled)
            except Exception as e:
                logger.warning(f"Shuffling failed: {str(e)}")
                continue

    elif method == "random_genomic":
        if not genome_file:
            raise ValueError("Genome file required for random genomic samples")
        negatives = extract_random_genomic_sequences(genome_file, seq_len, n_samples)

    else:  # simple shuffle
        for seq in positive_sequences[:n_samples]:
            seq_list = list(seq)
            random.shuffle(seq_list)
            negatives.append("".join(seq_list))

    return negatives


def extract_random_genomic_sequences(genome_file, seq_length, n_samples):
    """
    Extract random genomic sequences for negative samples.

    Args:
        genome_file: Path to genome FASTA
        seq_length: Length of sequences to extract
        n_samples: Number of sequences to extract

    Returns:
        list: Random genomic sequences
    """
    try:
        genome = Fasta(genome_file)
        chromosomes = [
            chrom for chrom in genome.keys() if len(genome[chrom]) > seq_length
        ]
        sequences = []

        while len(sequences) < n_samples and chromosomes:
            chrom = random.choice(chromosomes)
            max_start = len(genome[chrom]) - seq_length
            start = random.randint(0, max_start)
            seq = genome[chrom][start : start + seq_length].seq.upper()

            if "N" not in seq:
                sequences.append(seq)

        return sequences

    except Exception as e:
        logger.error(f"Error extracting random sequences: {str(e)}")
        raise ValueError("Failed to generate random genomic sequences")


def prepare_dataset(
    positive_sequences,
    negative_sequences=None,
    test_size=0.2,
    val_size=0.1,
    augment=False,
):
    """
    Prepare training, validation, and test datasets.

    Args:
        positive_sequences: List of positive DNA sequences
        negative_sequences: List of negative sequences (if None, generates them)
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        augment: Whether to augment data with reverse complements

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)

    Raises:
        ValueError: If inputs are invalid or processing fails
    """
    if not positive_sequences:
        raise ValueError("No positive sequences provided")

    # Generate negatives if not provided
    if negative_sequences is None:
        negative_sequences = generate_negative_samples(positive_sequences)

    # Validate sequence lengths
    seq_len = len(positive_sequences[0])
    if any(len(seq) != seq_len for seq in positive_sequences + negative_sequences):
        raise ValueError("All sequences must be of equal length")

    # Combine and label data
    sequences = positive_sequences + negative_sequences
    labels = [1] * len(positive_sequences) + [0] * len(negative_sequences)

    # Data augmentation
    if augment:
        logger.info("Performing data augmentation")
        aug_pos = [reverse_complement(seq) for seq in positive_sequences]
        aug_neg = [reverse_complement(seq) for seq in negative_sequences]
        sequences.extend(aug_pos + aug_neg)
        labels.extend([1] * len(aug_pos) + [0] * len(aug_neg))

        # Add shifted versions
        shift_size = min(5, seq_len // 10)  # Shift by 5bp or 10% of sequence length
        for seq in positive_sequences + negative_sequences:
            shifted = seq[shift_size:] + seq[:shift_size]
            sequences.append(shifted)
            labels.append(1 if seq in positive_sequences else 0)

    # Convert to numpy arrays for sklearn
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=y_train,
    )

    # One-hot encode sequences
    X_train_encoded = np.array([one_hot_encode(seq) for seq in X_train])
    X_val_encoded = np.array([one_hot_encode(seq) for seq in X_val])
    X_test_encoded = np.array([one_hot_encode(seq) for seq in X_test])

    return (
        X_train_encoded,
        np.array(y_train),
        X_val_encoded,
        np.array(y_val),
        X_test_encoded,
        np.array(y_test),
    )


def process_tf_data(
    tf_name,
    jaspar_dir,
    chip_seq_file,
    genome_file,
    output_dir,
    sequence_length=200,
    test_size=0.2,
    val_size=0.1,
    augment=False,
):
    """
    Process data for a transcription factor.

    Args:
        tf_name: Name of the transcription factor
        jaspar_dir: Directory with JASPAR motif files
        chip_seq_file: Path to ChIP-seq peaks BED file
        genome_file: Path to reference genome FASTA
        output_dir: Directory to save processed data
        sequence_length: Length of sequences to extract
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        augment: Whether to augment training data

    Raises:
        ValueError: If any processing step fails
    """
    # Validate inputs
    if not os.path.isdir(jaspar_dir):
        raise ValueError(f"JASPAR directory not found: {jaspar_dir}")
    if not os.path.isfile(chip_seq_file):
        raise ValueError(f"ChIP-seq file not found: {chip_seq_file}")
    if not os.path.isfile(genome_file):
        raise ValueError(f"Genome file not found: {genome_file}")

    logger.info(f"Processing data for TF: {tf_name}")

    try:
        # Find JASPAR files
        jaspar_files = [
            f
            for f in os.listdir(jaspar_dir)
            if tf_name.upper() in f.upper()
            and (f.endswith(".pfm") or f.endswith(".jaspar"))
        ]
        if not jaspar_files:
            raise ValueError(f"No JASPAR files found for TF: {tf_name}")

        # Extract sequences
        logger.info(f"Extracting sequences from {chip_seq_file}")
        sequences = extract_sequences_from_chip(
            chip_seq_file, genome_file, sequence_length
        )
        if not sequences:
            raise ValueError(f"No valid sequences extracted from {chip_seq_file}")
        logger.info(f"Extracted {len(sequences)} sequences")

        # Generate negatives
        logger.info("Generating negative samples")
        negative_sequences = generate_negative_samples(
            sequences,
            method="dinucleotide_shuffle",
            genome_file=genome_file,
            n_samples=len(sequences),
        )

        # Prepare datasets
        logger.info("Preparing train/val/test splits")
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(
            sequences, negative_sequences, test_size, val_size, augment
        )

        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)

        # Save data with TF-specific filenames
        base_path = os.path.join(output_dir, tf_name)
        np.save(f"{base_path}_X_train.npy", X_train)
        np.save(f"{base_path}_y_train.npy", y_train)
        np.save(f"{base_path}_X_val.npy", X_val)
        np.save(f"{base_path}_y_val.npy", y_val)
        np.save(f"{base_path}_X_test.npy", X_test)
        np.save(f"{base_path}_y_test.npy", y_test)

        # Save comprehensive metadata
        metadata = {
            "tf_name": tf_name,
            "creation_date": datetime.now().isoformat(),
            "processing_parameters": {
                "sequence_length": sequence_length,
                "test_size": test_size,
                "val_size": val_size,
                "augment": augment,
            },
            "statistics": {
                "num_positive": len(sequences),
                "num_negative": len(negative_sequences),
                "train_samples": len(y_train),
                "val_samples": len(y_val),
                "test_samples": len(y_test),
            },
            "source_files": {
                "jaspar_files": jaspar_files,
                "chip_seq_file": os.path.basename(chip_seq_file),
                "genome_file": os.path.basename(genome_file),
            },
        }

        metadata_path = os.path.join(output_dir, f"{tf_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully processed data for TF: {tf_name}")
        logger.info(f"Saved processed data to: {output_dir}")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(
            f"Failed to process data for TF {tf_name}: {str(e)}", exc_info=True
        )
        raise ValueError(f"Processing failed for {tf_name}: {str(e)}") from e


def main():
    """Main function to run the data processing script."""
    parser = argparse.ArgumentParser(
        description="Process data for TF binding prediction"
    )
    parser.add_argument("--tf", required=True, help="Transcription factor name")
    parser.add_argument(
        "--jaspar-dir", required=True, help="Directory with JASPAR files"
    )
    parser.add_argument(
        "--chip-seq-file", required=True, help="ChIP-seq peaks BED file"
    )
    parser.add_argument("--genome", required=True, help="Reference genome FASTA")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--sequence-length", type=int, default=200, help="Sequence length"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data for testing"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of training data for validation",
    )
    parser.add_argument("--augment", action="store_true", help="Augment training data")

    args = parser.parse_args()

    try:
        process_tf_data(
            tf_name=args.tf,
            jaspar_dir=args.jaspar_dir,
            chip_seq_file=args.chip_seq_file,
            genome_file=args.genome,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            test_size=args.test_size,
            val_size=args.val_size,
            augment=args.augment,
        )
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
