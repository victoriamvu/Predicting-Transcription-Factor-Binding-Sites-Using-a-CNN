
import joblib  
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

import numpy as np
import sys
tf_name = sys.argv[1] 

# Load CEBPA data
X_test = np.load("data/processed/CEBPA/X_test.npy")
y_test = np.load("data/processed/CEBPA/y_test.npy")

# Load model
from tensorflow.keras.models import load_model
model = load_model("models/model.h5")

# Predict scores
y_scores = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

plt.savefig(f"models/{tf_name}_roc_curve.png")

