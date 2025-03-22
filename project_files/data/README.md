# Data Directory

This directory contains the data for the transcription factor binding prediction project.

## Directory Structure

```
data/
├── README.md             # This file
├── raw/                  # Raw, unprocessed data
│   ├── jaspar/           # JASPAR motif data
│   ├── encode/           # ChIP-seq data from ENCODE
│   └── genome/           # Reference genome
└── processed/            # Processed data for model training
    ├── CTCF/             # Processed data for CTCF
    ├── GATA1/            # Processed data for GATA1
    ├── CEBPA/            # Processed data for CEBPA
    └── TP53/             # Processed data for TP53
```

## Raw Data Sources

### JASPAR Motifs

The raw motif data comes from the JASPAR database (http://jaspar.genereg.net/). We use the following transcription factors:

- CTCF (MA0139.1): Chromatin organizer and insulator
- GATA1 (MA0035.4): Hematopoietic development regulator
- CEBPA (MA0102.2): Cell differentiation and liver function regulator
- TP53 (MA0106.3): Tumor suppressor (p53)

Each TF has the following files:
- `.pfm`: Position Frequency Matrix
- `.jaspar`: JASPAR format file
- `.meme`: MEME format file
- `_metadata.json`: Metadata about the TF

### ChIP-seq Data

ChIP-seq peak data comes from the ENCODE project (https://www.encodeproject.org/):

- CTCF: ENCFF002CEL.bed
- GATA1: ENCFF002CUQ.bed 
- CEBPA: ENCFF002CWG.bed
- TP53: ENCFF002CXC.bed

These BED files contain the genomic coordinates of the TF binding sites determined experimentally.

### Reference Genome

We use the human reference genome (hg38) from UCSC:
- `hg38.fa`: Human genome assembly GRCh38/hg38

## Processed Data Format

The processed data directory contains the following for each TF:

- `X_train.npy`: One-hot encoded training sequences (shape: n_samples × 200 × 4)
- `y_train.npy`: Binary labels for training (1 = binding site, 0 = no binding)
- `X_val.npy`: One-hot encoded validation sequences
- `y_val.npy`: Binary labels for validation
- `X_test.npy`: One-hot encoded test sequences
- `y_test.npy`: Binary labels for testing
- `metadata.json`: Information about the dataset (number of samples, class distribution, etc.)

## Data Processing

Data is processed from raw to processed using the `src/data.py` script:

```bash
python src/data.py --tf CTCF --jaspar-dir data/raw/jaspar --chip-seq-file data/raw/encode/CTCF/ENCFF002CEL.bed --genome data/raw/genome/hg38.fa --output-dir data/processed/CTCF
```

The processing pipeline:
1. Extracts 200bp sequences centered on ChIP-seq peaks (positive samples)
2. Generates negative samples using dinucleotide shuffling or random genomic regions
3. Creates one-hot encoded representations
4. Splits data into training, validation, and test sets
5. Saves the processed data in numpy format

## Notes

- Sequences with ambiguous nucleotides (N) are filtered out
- The class distribution is balanced (equal numbers of positive and negative examples)
- Data augmentation may be applied during training
