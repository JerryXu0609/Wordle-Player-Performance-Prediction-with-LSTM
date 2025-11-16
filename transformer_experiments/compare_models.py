# -*- coding: utf-8 -*-
"""
Compare saved LSTM predictions with Transformer predictions.
Requires:
- saved_models/lstm.h5 (or saved_models/your_lstm_preds.npy)
- saved_models/transformer_preds.npy
- preprocessed_output/y_scaler.pkl
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from config_transformer import MODEL_DIR, Y_SCALER

LSTM_PRED_PATH = os.path.join(MODEL_DIR, "lstm_preds.npy")   # optional
TRANS_PRED_PATH = os.path.join(MODEL_DIR, "transformer_preds.npy")
TRUE_PATH = os.path.join(MODEL_DIR, "transformer_true.npy")

def load_preds():
    if os.path.exists(LSTM_PRED_PATH):
        lstm_pred = np.load(LSTM_PRED_PATH)
    else:
        lstm_pred = None
    trans_pred = np.load(TRANS_PRED_PATH)
    y_true = np.load(TRUE_PATH)
    return lstm_pred, trans_pred, y_true

def compare_and_plot():
    lstm_pred, trans_pred, y_true = load_preds()
    # if lstm_pred is None, try to load via trainer outputs (not required)
    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        return mae, rmse, mse

    mae_t, rmse_t, mse_t = metrics(y_true, trans_pred)
    print("Transformer on test -> MAE: {:.4f}, RMSE: {:.4f}, MSE: {:.4f}".format(mae_t, rmse_t, mse_t))
    if lstm_pred is not None:
        mae_l, rmse_l, mse_l = metrics(y_true, lstm_pred)
        print("LSTM on test -> MAE: {:.4f}, RMSE: {:.4f}, MSE: {:.4f}".format(mae_l, rmse_l, mse_l))

    # plot true vs preds
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="True")
    if lstm_pred is not None:
        plt.plot(lstm_pred, label="LSTM Pred")
    plt.plot(trans_pred, label="Transformer Pred")
    plt.legend()
    plt.title("Model Comparison: True vs Predictions")
    plt.xlabel("Sample Index")
    plt.ylabel("Average Tries")
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "comparison_true_preds.png"))
    plt.close()
    print("saved comparison plot to", os.path.join(MODEL_DIR, "comparison_true_preds.png"))

if __name__ == "__main__":
    compare_and_plot()