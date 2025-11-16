# -*- coding: utf-8 -*-
"""
Main script: orchestrates the preprocessing pipeline
"""
import os
from config import OUTPUT_DIR, WORD_COL
from data_loader import load_and_inspect
from cleaning import basic_cleaning, normalize_try_columns
from eda import eda_plots
from feature_engineering import feature_engineering
from encoding import build_char_vocab, words_to_char_indices
from timeseries_builder import create_time_series_samples
from utils import save_json, save_numpy

def main():
    df = load_and_inspect()
    df = basic_cleaning(df)
    df = normalize_try_columns(df)
    eda_plots(df)
    df = feature_engineering(df)
    df.to_csv(os.path.join(OUTPUT_DIR, "df_features.csv"), index=False)

    char2idx, idx2char = build_char_vocab(df[WORD_COL].values)
    char_seqs = words_to_char_indices(df[WORD_COL].values, char2idx)
    save_numpy(char_seqs, os.path.join(OUTPUT_DIR, "char_seqs.npy"))

    feature_cols = ["mean_tries", "success_rate", "hard_mode_pct"]
    if df["hard_mode_pct"].isna().all():
        df["hard_mode_pct"] = 0.0
    X, y = create_time_series_samples(df, feature_cols, "mean_tries", window_size=7)
    save_numpy(X, os.path.join(OUTPUT_DIR, "X_time_series.npy"))
    save_numpy(y, os.path.join(OUTPUT_DIR, "y_time_series.npy"))

    summary = {
        "original_shape": df.shape,
        "date_min": str(df["Date"].min()),
        "date_max": str(df["Date"].max()),
        "n_samples_time_series": X.shape[0],
        "feature_cols": feature_cols
    }
    save_json(summary, os.path.join(OUTPUT_DIR, "preprocessing_summary.json"))
    print("数据预处理完成，结果已保存到:", OUTPUT_DIR)

if __name__ == "__main__":
    main()