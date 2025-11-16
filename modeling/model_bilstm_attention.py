# -*- coding: utf-8 -*-
"""
BiLSTM + Attention model
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from layers_attention import AttentionLayer
from config_model import SEQ_LEN, FEATURE_DIM, LSTM_UNITS, DROPOUT

def build_bilstm_attention_model():
    inputs = Input(shape=(SEQ_LEN, FEATURE_DIM))
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(inputs)
    x = AttentionLayer()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model