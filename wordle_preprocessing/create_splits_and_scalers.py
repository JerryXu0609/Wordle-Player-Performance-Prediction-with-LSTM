# -*- coding: utf-8 -*-
"""
Create time-ordered train/val/test splits and fit/save scalers.
"""
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = "preprocessed_output"
X_PATH = os.path.join(ROOT, "X_time_series.npy")
Y_PATH = os.path.join(ROOT, "y_time_series.npy")

OUT_X_TRAIN = os.path.join(ROOT, "X_train.npy")
OUT_Y_TRAIN = os.path.join(ROOT, "y_train.npy")
OUT_X_VAL   = os.path.join(ROOT, "X_val.npy")
OUT_Y_VAL   = os.path.join(ROOT, "y_val.npy")
OUT_X_TEST  = os.path.join(ROOT, "X_test.npy")
OUT_Y_TEST  = os.path.join(ROOT, "y_test.npy")

SCALER_FEATURE_PATH = os.path.join(ROOT, "feature_scaler.pkl")
SCALER_Y_PATH = os.path.join(ROOT, "y_scaler.pkl")


def load_inputs():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("未找到 X_time_series.npy 或 y_time_series.npy")
        sys.exit(1)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"Loaded X {X.shape}, y {y.shape}")
    return X, y


def time_order_split(X, y, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    n = X.shape[0]
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    X_train, y_train = X[:i_train], y[:i_train]
    X_val,   y_val   = X[i_train:i_val], y[i_train:i_val]
    X_test,  y_test  = X[i_val:],        y[i_val:]

    print(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def fit_and_transform_scalers(X_train, y_train, X_val, y_val, X_test, y_test):
    n_samples, seq_len, n_features = X_train.shape

    # ---- Feature scaler ----
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)

    def transform_X(X):
        n, s, f = X.shape
        flat = X.reshape(-1, f)
        return scaler.transform(flat).reshape(n, s, f)

    X_train_s = transform_X(X_train)
    X_val_s   = transform_X(X_val)
    X_test_s  = transform_X(X_test)

    # ---- Target scaler ----
    y_scaler = StandardScaler()
    y_train_r = y_train.reshape(-1, 1)
    y_scaler.fit(y_train_r)

    y_train_s = y_scaler.transform(y_train_r).reshape(-1)
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
    y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    # ---- Save scalers ----
    joblib.dump(scaler, SCALER_FEATURE_PATH)
    joblib.dump(y_scaler, SCALER_Y_PATH)

    print("Saved feature scaler ->", SCALER_FEATURE_PATH)
    print("Saved y scaler ->", SCALER_Y_PATH)

    return X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s


def save_splits(X_train, y_train, X_val, y_val, X_test, y_test):
    np.save(OUT_X_TRAIN, X_train)
    np.save(OUT_Y_TRAIN, y_train)
    np.save(OUT_X_VAL, X_val)
    np.save(OUT_Y_VAL, y_val)
    np.save(OUT_X_TEST, X_test)
    np.save(OUT_Y_TEST, y_test)

    print("Saved standardized splits:")
    for f in [OUT_X_TRAIN, OUT_Y_TRAIN, OUT_X_VAL, OUT_Y_VAL, OUT_X_TEST, OUT_Y_TEST]:
        print(" -", f)


def main():
    X, y = load_inputs()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_order_split(X, y)

    X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s = fit_and_transform_scalers(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    save_splits(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s)

    print("Shapes:")
    print("  Train:", X_train_s.shape, y_train_s.shape)
    print("  Val  :", X_val_s.shape, y_val_s.shape)
    print("  Test :", X_test_s.shape, y_test_s.shape)


if __name__ == "__main__":
    main()