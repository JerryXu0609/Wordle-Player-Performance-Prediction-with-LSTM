# -*- coding: utf-8 -*-
"""
Model trainer
"""

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config_model import *
from model_lstm import build_lstm_model
from model_bilstm_attention import build_bilstm_attention_model
from model_transformer import build_transformer_model

def load_data():
    X_train = np.load(X_TRAIN)
    y_train = np.load(Y_TRAIN)
    X_val = np.load(X_VAL)
    y_val = np.load(Y_VAL)
    return X_train, y_train, X_val, y_val

def train(model_name="lstm"):

    X_train, y_train, X_val, y_val = load_data()

    if model_name == "lstm":
        model = build_lstm_model()
    elif model_name == "bilstm":
        model = build_bilstm_attention_model()
    elif model_name == "transformer":
        model = build_transformer_model()
    else:
        raise ValueError("Unknown model name.")

    ckpt_path = f"{MODEL_DIR}/{model_name}.h5"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    import pickle
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print(f"模型已保存到 {ckpt_path}")
    print("训练曲线已保存为 training_history.pkl")
    return model, history



if __name__ == "__main__":
    model, history = train("lstm")

    # 调用 utils_model.py 生成训练曲线图
    from utils_model import plot_history
    plot_history(history, name="lstm_training_curve")