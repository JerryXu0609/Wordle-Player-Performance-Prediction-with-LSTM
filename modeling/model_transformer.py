# -*- coding: utf-8 -*-
"""
Transformer Encoder model for regression
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from config_model import SEQ_LEN, FEATURE_DIM

def transformer_encoder(inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    x = layers.Dropout(dropout)(x)
    x = x + inputs

    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(ff_dim, activation="relu")(y)
    y = layers.Dense(inputs.shape[-1])(y)
    return x + y

def build_transformer_model():
    inputs = layers.Input(shape=(SEQ_LEN, FEATURE_DIM))
    x = transformer_encoder(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model