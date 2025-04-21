import numpy as np
import os

tf_names = ["CEBPA", "TP53", "GATA1", "CTCF"]

for tf in tf_names:
    try:
        X = np.load(f"data/processed/{tf}/X_test.npy")
        y = np.load(f"data/processed/{tf}/y_test.npy")
        print(f"{tf}: X shape = {X.shape}, y shape = {y.shape}")
    except Exception as e:
        print(f"{tf}: Error - {e}")