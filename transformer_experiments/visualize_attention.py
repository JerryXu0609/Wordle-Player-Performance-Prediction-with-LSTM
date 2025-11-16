# -*- coding: utf-8 -*-
"""
Visualize attention weights from transformer_with_att model.
Note: model must be the one that outputs [pred, att_scores],
and att_scores shape is (batch, num_heads, seq_len, seq_len).
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import joblib
from config_transformer import TRANSFORMER_ATT_MODEL_PATH, X_TEST, MODEL_DIR

OUT_DIR = os.path.join(MODEL_DIR, "attention_maps")
os.makedirs(OUT_DIR, exist_ok=True)

def visualize(model_path=TRANSFORMER_ATT_MODEL_PATH, max_plots=8):
    X_test = np.load(X_TEST)
    # load model (must be saved as model with two outputs)
    model = load_model(model_path, compile=False)
    # model.predict returns [preds, att_scores]
    preds, att = model.predict(X_test, batch_size=32)
    # att shape: (n_samples, num_heads, seq_len, seq_len)
    n_samples = att.shape[0]
    num_heads = att.shape[1]
    seq_len = att.shape[2]
    sample_indices = list(range(min(max_plots, n_samples)))
    for idx in sample_indices:
        att_sample = att[idx]  # (num_heads, seq_len, seq_len)
        # average across heads
        att_avg = np.mean(att_sample, axis=0)
        plt.figure(figsize=(6,5))
        plt.imshow(att_avg, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title(f"Avg Attention - Sample {idx}")
        plt.xlabel("Key position")
        plt.ylabel("Query position")
        plt.savefig(os.path.join(OUT_DIR, f"att_avg_sample_{idx}.png"))
        plt.close()
        # also save each head
        for h in range(num_heads):
            plt.figure(figsize=(4,3))
            plt.imshow(att_sample[h], cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"Head {h} - Sample {idx}")
            plt.savefig(os.path.join(OUT_DIR, f"att_head_{h}_sample_{idx}.png"))
            plt.close()
    print("Saved attention maps to", OUT_DIR)

if __name__ == "__main__":
    visualize()