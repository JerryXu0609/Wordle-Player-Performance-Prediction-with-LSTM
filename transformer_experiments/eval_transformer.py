# -*- coding: utf-8 -*-
"""
Evaluate transformer model on test set and save predictions.
"""
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from config_transformer import *

def evaluate(model_path=TRANSFORMER_MODEL_PATH):
    X_test = np.load(X_TEST)
    y_test = np.load(Y_TEST)
    y_scaler = joblib.load(Y_SCALER)

    model = load_model(model_path, compile=False)
    y_pred_scaled = model.predict(X_test).reshape(-1)
    # inverse transform
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test.reshape(-1,1)).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    print("Transformer MAE:", mae)
    print("Transformer RMSE:", rmse)
    print("Transformer MSE:", mse)
    # save preds
    np.save(os.path.join(MODEL_DIR, "transformer_preds.npy"), y_pred)
    np.save(os.path.join(MODEL_DIR, "transformer_true.npy"), y_true)
    return mae, rmse, mse

if __name__ == "__main__":
    evaluate()