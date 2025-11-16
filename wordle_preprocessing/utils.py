# -*- coding: utf-8 -*-
"""
Utility functions for saving outputs and summaries
"""
import os
import json
import numpy as np

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_numpy(arr, path):
    np.save(path, arr)
    print(f"Saved: {path}")