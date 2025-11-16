# -*- coding: utf-8 -*-
"""
Train Transformer model and save history & weights.
"""
import os
import pickle
import numpy as np
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config_transformer import *
from transformer_model import build_transformer_model, build_transformer_with_attention

def load_data():
    X_train = np.load(X_TRAIN)
    y_train = np.load(Y_TRAIN)
    X_val = np.load(X_VAL)
    y_val = np.load(Y_VAL)
    return X_train, y_train, X_val, y_val

def train(save_with_attention=False):
    X_train, y_train, X_val, y_val = load_data()

    model = build_transformer_model()
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        ModelCheckpoint(TRANSFORMER_MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    # save history
    with open(TRANSFORMER_HISTORY, "wb") as f:
        pickle.dump(history.history, f)
    model.save(TRANSFORMER_MODEL_PATH)
    print("Saved transformer model to", TRANSFORMER_MODEL_PATH)

    # optionally save a model that outputs attention (weights transfer)
    if save_with_attention:
        att_model = build_transformer_with_attention()
        # transfer weights for layers that match by name where possible
        # simple approach: load weights from saved model file into att_model by layer-wise copying
        # load base model to get weights
        base = model
        # map weights by layer order (careful)
        base_weights = base.get_weights()
        try:
            att_model.set_weights(base_weights)
            att_model.save(TRANSFORMER_ATT_MODEL_PATH)
            print("Saved transformer_with_att to", TRANSFORMER_ATT_MODEL_PATH)
        except Exception as e:
            print("Warning: unable to set weights to attention model automatically:", e)
            print("You may need to retrain attention model or manually transfer weights.")

if __name__ == "__main__":
    # set save_with_attention=True if you want the additional model saved
    train(save_with_attention=True)