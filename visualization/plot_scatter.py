# -*- coding: utf-8 -*-
"""
Scatter plot of true vs predicted values.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os

DATA_DIR = "../wordle_preprocessing/preprocessed_output"
MODEL_PATH = "../modeling/saved_models/lstm.h5"
OUT_FIG = "fig_scatter_true_pred.png"

if __name__ == "__main__":
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    y_scaler = joblib.load(os.path.join(DATA_DIR, "y_scaler.pkl"))

    model = load_model(MODEL_PATH, compile=False)
    y_pred_scaled = model.predict(X_test).reshape(-1)

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test.reshape(-1,1)).reshape(-1)

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)],
             [min(y_true), max(y_true)],
             color="red", linestyle="--", label="y=x")

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted (Scatter Plot)")
    plt.legend()
    plt.grid(True)

    plt.savefig(OUT_FIG)
    plt.close()

    print(f"散点图已保存为 {OUT_FIG}")