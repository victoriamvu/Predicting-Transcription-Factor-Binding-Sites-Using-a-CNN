import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os
from tensorflow.keras.models import load_model

tf_names = ["CEBPA", "TP53", "GATA1", "CTCF"]
colors = ['blue', 'red', 'green', 'orange']

plt.figure(figsize=(8, 6))

for tf_name, color in zip(tf_names, colors):
    X_test = np.load(f"data/processed/{tf_name}/X_test.npy")
    y_test = np.load(f"data/processed/{tf_name}/y_test.npy")

    model = load_model("models/model.h5")
    y_score = model.predict(X_test).ravel()

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color=color, lw=2, label=f'{tf_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("models/all_roc_curves.png")
plt.show()
