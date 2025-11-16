# -*- coding: utf-8 -*-
"""
Create sliding-window time series samples for LSTM input
"""
import numpy as np
from config import DATE_COL

def create_time_series_samples(df, feature_cols, target_col, window_size=7, step=1):
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    data = df_sorted[feature_cols].values
    targets = df_sorted[target_col].values
    Xs, ys = [], []
    for start in range(0, len(df_sorted) - window_size, step):
        end = start + window_size
        Xs.append(data[start:end])
        ys.append(targets[end])
    return np.stack(Xs, axis=0), np.array(ys)