# -*- coding: utf-8 -*-
"""
LSTM model for time series regression
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config_model import SEQ_LEN, FEATURE_DIM, LSTM_UNITS, DROPOUT

def build_lstm_model():
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(SEQ_LEN, FEATURE_DIM), return_sequences=False),
        Dropout(DROPOUT),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model