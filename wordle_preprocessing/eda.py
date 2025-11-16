# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA)
"""
import os
import matplotlib.pyplot as plt
from config import DATE_COL, TRY_COLS, OUTPUT_DIR

def eda_plots(df):
    df_sorted = df.sort_values(DATE_COL)
    for col in TRY_COLS:
        if col in df_sorted.columns:
            plt.figure(figsize=(12,3))
            plt.plot(df_sorted[DATE_COL], df_sorted[col], marker='.', linewidth=0.8)
            plt.title(f"{col} over time")
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, f"time_series_{col.replace(' ','_')}.png"))
            plt.close()
    print("EDA 图表已保存到:", OUTPUT_DIR)