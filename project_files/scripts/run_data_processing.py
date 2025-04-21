from project_files.src.data import process_tf_data

process_tf_data(
    tf_name="TP53",
    jaspar_dir="data/jaspar",
    chip_seq_file="data/raw/encode/TP53.bed",
    genome_file="data/raw/genome/hg38.fa",
    sequence_length=200,
    test_size=0.2,
    val_size=0.2,
    augment=False,
    output_dir="data/processed/TP53"
)



