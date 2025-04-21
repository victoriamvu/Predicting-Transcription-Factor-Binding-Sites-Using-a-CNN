#!/usr/bin/env python
"""
Data processing module for TF binding prediction.

This script parses JASPAR motifs, extracts ChIP-seq sequences,
 generates negative samples, encodes sequences, and splits into
 train/validation/test sets for a given transcription factor.
"""
import os
import sys
import argparse
import logging
import json
import random
from datetime import datetime
from Bio import motifs
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from pyfaidx import Fasta

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tf_binding_data")


def parse_jaspar_pfm(pfm_path):
    """
    Parse a JASPAR PFM file and return a DataFrame of counts.
    """
    # Placeholder: implement actual parsing
    return motifs.read(pfm_path, "jaspar").counts


def pfm_to_pwm(pfm_df, pseudocount=1.0):
    """Convert a PFM to a PWM with pseudocount."""
    pfm = np.array(pfm_df)
    total = np.sum(pfm, axis=0, keepdims=True)
    ppm = (pfm + pseudocount) / (total + 4 * pseudocount)
    return np.log2(ppm / 0.25)


def one_hot_encode(sequence):
    """One-hot encode a DNA sequence."""
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    return np.array([mapping.get(nuc.upper(), [0, 0, 0, 0]) for nuc in sequence])


def extract_sequences_from_chip(bed_file, genome_file, sequence_length=200):
    """Extract sequences centered on ChIP-seq peaks, clipping at chromosome edges."""
    genome = Fasta(genome_file)
    half = sequence_length // 2
    seqs = []
    with open(bed_file) as f:
        for i, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start_s, end_s = line.split()[:3]
            start, end = int(start_s), int(end_s)
            if start >= end:
                logger.warning(f"Line {i}: start >= end")
                continue
            center = (start + end) // 2
            s, e = center - half, center + half
            if chrom not in genome:
                logger.warning(f"Chromosome {chrom} missing in genome")
                continue
            chrom_len = len(genome[chrom])
            if s < 0:
                s, e = 0, sequence_length
            if e > chrom_len:
                e, s = chrom_len, chrom_len - sequence_length
            seq = genome[chrom][s:e].seq.upper()
            if len(seq) == sequence_length and "N" not in seq:
                seqs.append(seq)
    return seqs


def generate_negative_samples(
    pos_seqs, method="dinucleotide_shuffle", genome_file=None, n_samples=None
):
    """Generate negative samples by shuffling or sampling genomic regions."""
    seq_len = len(pos_seqs[0]) if pos_seqs else 0
    n_samples = n_samples or len(pos_seqs)
    negs = []
    if method == "random_genomic":
        if not genome_file:
            raise ValueError("Genome file required for genomic sampling")
        negs = extract_random_genomic_sequences(genome_file, seq_len, n_samples)
    else:
        for seq in pos_seqs[:n_samples]:
            lst = list(seq)
            random.shuffle(lst)
            negs.append("".join(lst))
    return negs


def extract_random_genomic_sequences(genome_file, seq_length, n_samples):
    """Placeholder: extract random genomic windows."""
    # Implement as needed
    return []


def prepare_dataset(pos_seqs, neg_seqs, test_size, val_size, augment=False):
    # Combine positive and negative sequences
    X = np.array(pos_seqs + neg_seqs)
    y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

    # Shuffle the data
    X, y = sklearn.utils.shuffle(X, y, random_state=42)

    # One-hot encode all sequences (turn (N,) into (N, 200, 4))
    X_encoded = np.array([one_hot_encode(seq) for seq in X])

    # Split into train + temp (for further val/test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y, test_size=test_size + val_size, stratify=y, random_state=42
    )

    # Calculate proportion for validation split
    val_ratio = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0

    # Split temp into validation and test sets
    if val_ratio > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - val_ratio, stratify=y_temp, random_state=42
        )
    else:
        X_val, y_val = np.array([]), np.array([])
        X_test, y_test = X_temp, y_temp

    return X_train, y_train, X_val, y_val, X_test, y_test


def process_tf_data(
    tf_name,
    jaspar_dir,
    chip_seq_file,
    genome_file,
    output_dir,
    sequence_length,
    test_size,
    val_size,
    augment,
):
    """Full pipeline: motifs -> sequences -> negatives -> dataset -> save."""
    if not os.path.isdir(jaspar_dir):
        raise ValueError(f"JASPAR dir not found: {jaspar_dir}")
    if not os.path.isfile(chip_seq_file):
        raise ValueError(f"ChIP-seq file not found: {chip_seq_file}")
    if not os.path.isfile(genome_file):
        raise ValueError(f"Genome file not found: {genome_file}")
    logger.info(f"Processing {tf_name}")
    pos_seqs = extract_sequences_from_chip(chip_seq_file, genome_file, sequence_length)
    neg_seqs = generate_negative_samples(
        pos_seqs,
        method="dinucleotide_shuffle",
        genome_file=genome_file,
        n_samples=len(pos_seqs),
    )
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(
        pos_seqs, neg_seqs, test_size, val_size, augment
    )
    # Save processed arrays and metadata
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    md = {
        "tf_name": tf_name,
        "date": datetime.now().isoformat(),
        "sequence_length": sequence_length,
        "test_size": test_size,
        "val_size": val_size,
        "augment": augment,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(md, f, indent=4)
    logger.info(f"Processed data for {tf_name}, saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process TF binding data for a given TF"
    )
    parser.add_argument("--tf", required=True, help="Transcription factor name")
    parser.add_argument(
        "--jaspar-dir", required=True, help="Directory with JASPAR PFM files"
    )
    parser.add_argument("--chip-seq-file", required=True, help="ChIP-seq BED file path")
    parser.add_argument("--genome", required=True, help="Reference genome FASTA path")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write processed data"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=200,
        help="Length of sequences to extract",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data for testing"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="Fraction of data for validation"
    )
    parser.add_argument(
        "--augment", action="store_true", help="Augment data with shifted sequences"
    )
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
        logger.error(f"Data processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
