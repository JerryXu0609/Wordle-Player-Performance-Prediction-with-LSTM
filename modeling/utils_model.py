# -*- coding: utf-8 -*-
"""
Utility functions for plotting and saving training curves
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, name="training_curve"):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig(f"{name}.png")
    plt.close()