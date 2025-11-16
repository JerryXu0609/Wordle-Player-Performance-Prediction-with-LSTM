# -*- coding: utf-8 -*-
"""
Data loading and basic inspection
"""
import pandas as pd
from config import DATA_PATH, DATE_COL

def load_and_inspect():
    df = pd.read_csv(DATA_PATH)
    print("原始数据大小:", df.shape)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    print(df.head())
    print(df.info())
    return df