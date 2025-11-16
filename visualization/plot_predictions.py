# -*- coding: utf-8 -*-
"""
Plot y_true vs y_pred curves on the test set.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os

DATA_DIR = "../wordle_preprocessing/preprocessed_output"
MODEL_PATH = "../modeling/saved_models/lstm.h5"
OUT_FIG = "fig_prediction_curve.png"

if __name__ == "__main__":
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    y_scaler = joblib.load(os.path.join(DATA_DIR, "y_scaler.pkl"))

    model = load_model(MODEL_PATH, compile=False)
    y_pred_scaled = model.predict(X_test).reshape(-1)

    # 反标准化
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test.reshape(-1,1)).reshape(-1)

    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="True Values")
    plt.plot(y_pred, label="Predicted Values")
    plt.title("Prediction vs True Values (Test Set)")
    plt.xlabel("Sample Index")
    plt.ylabel("Average Tries")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_FIG)
    plt.close()

    print(f"预测曲线图已保存为 {OUT_FIG}")