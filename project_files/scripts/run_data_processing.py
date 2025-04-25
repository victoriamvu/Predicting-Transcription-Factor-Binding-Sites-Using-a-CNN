import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data import process_tf_data

# Mapping TF names to their respective BED filenames
BED_FILES = {
    "TP53": "ENCFF403HEQ.bed",
    "GATA1": "ENCFF226NEO.bed",
    "CEBPA": "ENCFF002CWG.bed",
    "CTCF": "ENCFF002CEL.bed"
}

if __name__ == "__main__":
    import sys

    tf_name = sys.argv[sys.argv.index("--tf") + 1]
    chip_seq_filename = BED_FILES.get(tf_name)

    if not chip_seq_filename:
        raise ValueError(f"No BED file defined for TF: {tf_name}")

    process_tf_data(
        tf_name=tf_name,
        jaspar_dir="data/raw/jaspar",
        chip_seq_file=f"data/raw/encode/{tf_name}/{chip_seq_filename}",
        genome_file="data/raw/genome/hg38.fa",
        output_dir=f"data/processed/{tf_name}",
        sequence_length=200,
        test_size=0.2,
        val_size=0.1,
        augment=False,
    )
