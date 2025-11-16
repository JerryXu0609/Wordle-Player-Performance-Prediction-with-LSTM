# -*- coding: utf-8 -*-
"""
Model configuration file
"""

import os

BASE_DIR = "../wordle_preprocessing/preprocessed_output"

# 数据路径
X_TRAIN = os.path.join(BASE_DIR, "X_train.npy")
Y_TRAIN = os.path.join(BASE_DIR, "y_train.npy")
X_VAL   = os.path.join(BASE_DIR, "X_val.npy")
Y_VAL   = os.path.join(BASE_DIR, "y_val.npy")
X_TEST  = os.path.join(BASE_DIR, "X_test.npy")
Y_TEST  = os.path.join(BASE_DIR, "y_test.npy")

# 模型保存目录
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 模型超参数
SEQ_LEN = 7            # 时间序列窗口
FEATURE_DIM = 3        # mean_tries, success_rate, hard_mode_pct

LSTM_UNITS = 64
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001