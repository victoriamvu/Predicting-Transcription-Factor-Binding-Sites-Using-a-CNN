import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model

TF_NAMES = ["CEBPA", "CTCF", "GATA1", "TP53"]

for tf in TF_NAMES:
    print(f"[{tf}] Generating ROC curve...")

    try:
        model_path = f"models/{tf}/model.h5"
        data_dir = f"data/processed/{tf}"
        model = load_model(model_path)

        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))

        y_pred = model.predict(X_test).ravel()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"{tf} ROC (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{tf} ROC Curve")
        plt.legend(loc="lower right")

        plot_path = f"models/{tf}/{tf}_roc_curve.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"[{tf}] ✅ Saved ROC plot: {plot_path}")
    except Exception as e:
        print(f"[{tf}] ❌ Failed to plot ROC: {e}")
