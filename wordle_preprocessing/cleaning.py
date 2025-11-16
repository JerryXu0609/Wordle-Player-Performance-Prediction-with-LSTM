# -*- coding: utf-8 -*-
"""
Data cleaning and normalization
"""
import pandas as pd
import numpy as np
from config import DATE_COL, TRY_COLS

def basic_cleaning(df):
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.dropna(subset=[DATE_COL])
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    for col in TRY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print("缺失统计：")
    print(df.isna().sum())
    return df

def normalize_try_columns(df):
    df = df.copy()
    for col in TRY_COLS:
        if col in df.columns:
            df[col] = df[col] / 100.0
    return df