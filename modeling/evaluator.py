# -*- coding: utf-8 -*-
"""
Model evaluation script
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from config_model import *

def evaluate(model_path):

    X_test = np.load(X_TEST)
    y_test = np.load(Y_TEST)

    model = load_model(model_path, compile=False)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MSE:", mse)

if __name__ == "__main__":
    evaluate("saved_models/lstm.h5")