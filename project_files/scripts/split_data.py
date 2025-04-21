import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse

def split_and_save(tf_name, input_dir="data/raw", output_dir="data/processed"):
    # Paths
    X_path = os.path.join(input_dir, f"{tf_name}_X.npy")
    y_path = os.path.join(input_dir, f"{tf_name}_y.npy")
    
    # Load data
    X = np.load(X_path)
    y = np.load(y_path)

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

    # Create output directory
    tf_dir = os.path.join(output_dir, tf_name)
    os.makedirs(tf_dir, exist_ok=True)

    # Save
    np.save(os.path.join(tf_dir, "X_train.npy"), X_train)
    np.save(os.path.join(tf_dir, "y_train.npy"), y_train)
    np.save(os.path.join(tf_dir, "X_val.npy"), X_val)
    np.save(os.path.join(tf_dir, "y_val.npy"), y_val)
    np.save(os.path.join(tf_dir, "X_test.npy"), X_test)
    np.save(os.path.join(tf_dir, "y_test.npy"), y_test)
    
    print(f"{tf_name} data split and saved to {tf_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", required=True, help="Name of transcription factor (e.g., TP53)")
    args = parser.parse_args()

    split_and_save(args.tf)