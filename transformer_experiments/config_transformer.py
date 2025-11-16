# -*- coding: utf-8 -*-
"""
Configuration for Transformer experiments
"""
import os

BASE_DATA_DIR = "../wordle_preprocessing/preprocessed_output"
MODEL_DIR = "../modeling/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Data files (created by preprocessing & split script)
X_TRAIN = os.path.join(BASE_DATA_DIR, "X_train.npy")
Y_TRAIN = os.path.join(BASE_DATA_DIR, "y_train.npy")
X_VAL   = os.path.join(BASE_DATA_DIR, "X_val.npy")
Y_VAL   = os.path.join(BASE_DATA_DIR, "y_val.npy")
X_TEST  = os.path.join(BASE_DATA_DIR, "X_test.npy")
Y_TEST  = os.path.join(BASE_DATA_DIR, "y_test.npy")
Y_SCALER = os.path.join(BASE_DATA_DIR, "y_scaler.pkl")

# Model save names
TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "transformer.h5")
TRANSFORMER_ATT_MODEL_PATH = os.path.join(MODEL_DIR, "transformer_with_att.h5")
TRANSFORMER_HISTORY = os.path.join(MODEL_DIR, "transformer_history.pkl")

# Hyperparameters (you can tune these)
SEQ_LEN = 7
FEATURE_DIM = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

# Transformer hyperparams
HEAD_SIZE = 32
NUM_HEADS = 2
FF_DIM = 64
NUM_ENCODER_BLOCKS = 1
DROPOUT = 0.1