"""
Includes basic visualization methods, mostly used to testing purposes.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_top_contributors(contributions, select_by="absolute", n=5):
    """
    Plot the most contributing features

    Args:
        contributions (DataFrame of shape (1, n_features):
            Contributions, with feature names as the column names
        select_by (one of "absolute", "max", "min"):
            Which contributions to plot.
        n (int):
            Number of features to plot

    Returns:
        pyplot figure
    """
    features = contributions.columns.to_numpy()
    values = contributions.iloc[0].to_numpy()
    order = None
    if select_by == "min":
        order = np.argsort(values)
    if select_by == "max":
        order = np.argsort(values)[::-1]
    if select_by == "absolute":
        order = np.argsort(abs(values))[::-1]

    if order is None:
        raise ValueError("Invalid select_by option %s, should be one of 'min', 'max', 'absolute'"
                         % select_by)

    to_plot = order[0:n]
    plt.barh(features[to_plot][::-1], values[to_plot][::-1])
    plt.show()
