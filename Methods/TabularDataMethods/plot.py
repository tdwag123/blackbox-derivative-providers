from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_model(monotone_gp, x_scaler, y_mean: float, y_scale: float, 
               X_train_raw: np.ndarray, y_train_raw: np.ndarray, 
               learn_neg_flux: bool, n_plot: int = 60):
    
    """
    Plots reconstructed flux law. Call after fitting and predictions.
    RMK: this is directly from the monotone GP (Matern), but can 
    be adapted to make plots for other methods.
    
    """
    
    s_plot = np.linspace(X_train_raw[:, 0].min(), X_train_raw[:, 0].max(), n_plot)
    T_plot = np.linspace(X_train_raw[:, 1].min(), X_train_raw[:, 1].max(), n_plot)
    
    
    S, TT = np.meshgrid(s_plot, T_plot, indexing="xy")
    X_plot_raw = np.column_stack([S.ravel(), TT.ravel()])

    # GP was trained on standardized inputs, so standardize the plot grid too
    X_plot = x_scaler.transform(X_plot_raw)

    # predict posterior mean of f on the grid
    f_plot_std, _ = monotone_gp.predict_mean_and_ds(X_plot)

    # convert from standardized f back to raw f units
    f_plot_raw = y_mean + y_scale * f_plot_std

    # convert from f back to q if the model learned f = -q
    if learn_neg_flux:
        q_plot_raw = -f_plot_raw
        q_train_obs = -y_train_raw
    else:
        q_plot_raw = f_plot_raw
        q_train_obs = y_train_raw

    Q = q_plot_raw.reshape(S.shape)

    # 3D surface plot of reconstructed q
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(S, TT, Q, alpha=0.75, linewidth=0, antialiased=True)
    ax.scatter(
        X_train_raw[:, 0],
        X_train_raw[:, 1],
        q_train_obs,
        s=12,
        alpha=0.5,
        label="noisy training flux",
    )

    ax.set_xlabel("s = dT/dx")
    ax.set_ylabel("T")
    ax.set_zlabel("reconstructed q(s, T)")
    ax.set_title("Reconstructed constitutive flux law")
    ax.legend()
    plt.tight_layout()
