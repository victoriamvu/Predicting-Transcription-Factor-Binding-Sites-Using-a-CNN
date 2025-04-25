import os
import numpy as np
from tensorflow.keras.models import load_model

tfs = ["CEBPA", "CTCF", "GATA1", "TP53"]

for tf in tfs:
    try:
        print(f"Processing {tf}...")

        # Paths
        model_path = f"models/{tf}/model.h5"
        data_path = f"data/processed/{tf}/X_test.npy"
        labels_path = f"data/processed/{tf}/y_test.npy"
        save_pred_path = f"models/{tf}/{tf}_y_pred.npy"
        save_true_path = f"models/{tf}/{tf}_y_true.npy"

        # Load model and data
        model = load_model(model_path)
        X_test = np.load(data_path)
        y_test = np.load(labels_path)

        # Predict
        y_pred = model.predict(X_test).flatten()

        # Save
        np.save(save_pred_path, y_pred)
        np.save(save_true_path, y_test)

        print(f"[✓] Saved predictions for {tf}")
    except Exception as e:
        print(f"[{tf}] ❌ Error: {e}")
