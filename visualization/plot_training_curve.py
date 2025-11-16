# -*- coding: utf-8 -*-
"""
Plot training and validation loss curves.
"""

import matplotlib.pyplot as plt
import pickle
import os

HISTORY_PATH = "../modeling/training_history.pkl"  # 由 trainer.py 保存
OUT_FIG = "fig_training_curve.png"

if __name__ == "__main__":
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError("请确保 trainer.py 保存了 training_history.pkl")

    with open(HISTORY_PATH, "rb") as f:
        history = pickle.load(f)

    loss = history["loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(8,5))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Curve (Loss vs. Val Loss)")
    plt.legend()
    plt.grid(True)

    plt.savefig(OUT_FIG)
    plt.close()
    print(f"训练曲线已保存到 {OUT_FIG}")