# -*- coding: utf-8 -*-
"""
Plot residual distribution on test set.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os

DATA_DIR = "../wordle_preprocessing/preprocessed_output"
MODEL_PATH = "../modeling/saved_models/lstm.h5"
OUT_FIG = "fig_residuals.png"

if __name__ == "__main__":
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    y_scaler = joblib.load(os.path.join(DATA_DIR, "y_scaler.pkl"))

    model = load_model(MODEL_PATH, compile=False)
    y_pred_scaled = model.predict(X_test).reshape(-1)

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test.reshape(-1,1)).reshape(-1)

    residuals = y_true - y_pred

    plt.figure(figsize=(8,5))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.title("Residual Distribution (y_true - y_pred)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(OUT_FIG)
    plt.close()

    print(f"残差分布图保存为 {OUT_FIG}")