# -*- coding: utf-8 -*-
"""
Feature engineering: numeric, success rate, hard mode ratio, date features
"""
import numpy as np
from config import DATE_COL, WORD_COL, TRY_COLS

def feature_engineering(df):
    weights = np.array([1,2,3,4,5,6,7], dtype=float)
    props = df[TRY_COLS].values
    df["mean_tries"] = np.nansum(props * weights, axis=1) / np.nansum(props, axis=1)
    df["success_rate"] = df[TRY_COLS[:-1]].sum(axis=1)
    if "Number in hard mode" in df.columns and "Number of  reported results" in df.columns:
        df["hard_mode_pct"] = df["Number in hard mode"] / df["Number of  reported results"]
    else:
        df["hard_mode_pct"] = np.nan
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["day"] = df[DATE_COL].dt.day
    df["weekday"] = df[DATE_COL].dt.weekday
    df["word_len"] = df[WORD_COL].astype(str).str.len()
    df[WORD_COL] = df[WORD_COL].astype(str).str.lower()
    return df