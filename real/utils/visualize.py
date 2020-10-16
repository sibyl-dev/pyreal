"""
Includes basic visualization methods, mostly used to testing purposes.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_top_contributors(contributions, select_by="absolute", n=5, values=None):
    """
    Plot the most contributing features

    Args:
        contributions (Series or DataFrame of shape (1, n_features):
            Contributions, with feature names as the column names
        select_by (one of "absolute", "max", "min"):
            Which contributions to plot.
        n (int):
            Number of features to plot
        values (Series or DataFrame of shape (1, n_features):
            If given, show the corresponding values alongside the feature names

    Returns:
        pyplot figure
    """
    features = contributions.columns.to_numpy()
    if values is not None:
        features = np.array(["%s (%s)" % (feature, values[feature]) for feature in features])

    if contributions.ndim == 2:
        contributions = contributions.iloc[0]
    contributions = contributions.to_numpy()
    order = None
    if select_by == "min":
        order = np.argsort(contributions)
    if select_by == "max":
        order = np.argsort(contributions)[::-1]
    if select_by == "absolute":
        order = np.argsort(abs(contributions))[::-1]

    if order is None:
        raise ValueError("Invalid select_by option %s, should be one of 'min', 'max', 'absolute'"
                         % select_by)

    to_plot = order[0:n]
    plt.barh(features[to_plot][::-1], contributions[to_plot][::-1])
    plt.show()
