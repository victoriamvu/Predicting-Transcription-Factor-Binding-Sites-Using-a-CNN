import os
import json
import matplotlib.pyplot as plt

TF_LIST = ["CEBPA", "CTCF", "GATA1", "TP53"]
HISTORY_DIR = "models"

for tf_name in TF_LIST:
    history_path = os.path.join(HISTORY_DIR, f"{tf_name}_training_history.json")
    if not os.path.exists(history_path):
        print(f"❌ Missing training history: {history_path}")
        continue

    with open(history_path, "r") as f:
        history = json.load(f)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{tf_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    plt.subplot(1, 2, 2)
    plt.plot(history[acc_key], label="Train Accuracy")
    plt.plot(history[val_acc_key], label="Val Accuracy")
    plt.title(f"{tf_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_path = os.path.join(HISTORY_DIR, f"{tf_name}_training_curves.png")
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Saved: {output_path}")
