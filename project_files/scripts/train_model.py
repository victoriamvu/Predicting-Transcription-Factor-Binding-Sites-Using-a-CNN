import argparse
from project_files.src.train import run_training_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", required=True, help="Transcription Factor name (e.g., TP53, CEBPA)")
    args = parser.parse_args()

    run_training_pipeline(tf_name=args.tf)
