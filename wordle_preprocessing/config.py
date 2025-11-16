# -*- coding: utf-8 -*-
"""
Global configuration for Wordle preprocessing
"""
import os

# 数据路径与输出目录
DATA_PATH = "../datasets/2023_MCM_Problem_C_Data.csv"
OUTPUT_DIR = "preprocessed_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 列名定义
DATE_COL = "Date"
WORD_COL = "Word"
CONTEST_COL = "Contest number"

# 尝试次数列
TRY_COLS = [
    "1 try", "2 tries", "3 tries", "4 tries",
    "5 tries", "6 tries", "7 or more tries (X)"
]