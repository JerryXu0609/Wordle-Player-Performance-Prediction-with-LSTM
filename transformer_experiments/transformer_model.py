# -*- coding: utf-8 -*-
"""
Transformer model builder.
Provides:
- build_transformer_model(): standard Keras model -> outputs prediction
- build_transformer_with_attention(): model that also outputs attention scores
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
from config_transformer import SEQ_LEN, FEATURE_DIM, HEAD_SIZE, NUM_HEADS, FF_DIM, DROPOUT, NUM_ENCODER_BLOCKS

def single_transformer_block(inputs, head_size=HEAD_SIZE, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT, return_att=False):
    """
    Build one encoder block. If return_att True, returns (out, att_scores)
    """
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    # Use MultiHeadAttention with return_attention_scores
    mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
    if return_att:
        att_out, att_scores = mha(query=x, value=x, key=x, return_attention_scores=True)
    else:
        att_out = mha(query=x, value=x, key=x)
        att_scores = None
    att_out = layers.Dropout(dropout)(att_out)
    out1 = layers.Add()([x, att_out])

    y = layers.LayerNormalization(epsilon=1e-6)(out1)
    y = layers.Dense(ff_dim, activation="relu")(y)
    y = layers.Dense(inputs.shape[-1])(y)
    y = layers.Dropout(dropout)(y)
    out2 = layers.Add()([out1, y])
    return out2, att_scores

def build_transformer_model(seq_len=SEQ_LEN, feature_dim=FEATURE_DIM,
                            head_size=HEAD_SIZE, num_heads=NUM_HEADS,
                            ff_dim=FF_DIM, num_blocks=NUM_ENCODER_BLOCKS,
                            dropout=DROPOUT):
    """
    Build standard transformer model that outputs a single regression value.
    """
    inputs = Input(shape=(seq_len, feature_dim), name="inputs")
    x = inputs
    for _ in range(num_blocks):
        x, _ = single_transformer_block(x, head_size, num_heads, ff_dim, dropout, return_att=False)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="pred")(x)
    model = Model(inputs=inputs, outputs=outputs, name="transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

def build_transformer_with_attention(seq_len=SEQ_LEN, feature_dim=FEATURE_DIM,
                                     head_size=HEAD_SIZE, num_heads=NUM_HEADS,
                                     ff_dim=FF_DIM, num_blocks=NUM_ENCODER_BLOCKS,
                                     dropout=DROPOUT):
    """
    Build transformer model that returns both prediction and attention scores.
    Returns a Keras Model with outputs [pred, attention_scores].
    attention_scores shape: (batch, heads, seq_len, seq_len)
    We'll return the attention from the last block (if multiple).
    """
    inputs = Input(shape=(seq_len, feature_dim), name="inputs")
    x = inputs
    att_scores = None
    for i in range(num_blocks):
        x, att = single_transformer_block(x, head_size, num_heads, ff_dim, dropout, return_att=True)
        att_scores = att  # save last block's attention scores
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    pred = layers.Dense(1, name="pred")(x)
    # att_scores is Tensor with shape (batch, num_heads, seq_len, seq_len)
    model = Model(inputs=inputs, outputs=[pred, att_scores], name="transformer_with_att")
    # compile with a dummy loss for attention (we only train on pred)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss={"pred": "mse"})
    return model