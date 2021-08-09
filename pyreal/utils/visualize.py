"""
Includes basic visualization methods, mostly used to testing purposes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree


def plot_top_contributors(contributions, select_by="absolute", n=5, values=None,
                          flip_colors=False):
    """
    Plot the most contributing features

    Args:
        contributions (Series or DataFrame of shape (1, n_features):
            Contributions, with feature names as the column names
        select_by (one of "absolute", "max", "min"):
            Which explanation to plot.
        n (int):
            Number of features to plot
        values (Series or DataFrame of shape (1, n_features):
            If given, show the corresponding values alongside the feature names
        flip_colors (Boolean):
            If True, make the positive explanation red and negative explanation blue.
            Useful if the target variable has a negative connotation

    Returns:
        pyplot figure
            Bar plot of top contributors
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

    negative_color = "#ef8a62"
    positive_color = "#67a9cf"
    if not flip_colors:
        colors = \
            [negative_color if (c < 0) else positive_color for c in contributions[to_plot][::-1]]
    else:
        colors = \
            [positive_color if (c < 0) else negative_color for c in contributions[to_plot][::-1]]
    plt.barh(features[to_plot][::-1], contributions[to_plot][::-1], color=colors)
    plt.title("Contribution by feature")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(x=0, color="black")
    plt.show()


def plot_decision_tree(dte, size=(30, 30), class_names=None, fontsize=10, color=True):
    """
    Plot the tree structure of a decision tree explainer

    Args:
        dte:
            The decision tree explainer
        size (a tuple of two integers):
            The size of the plot, see matplotlib
        class_names (a list of n_label strings or None):
            Set the names of the labels
        fontsize (int):
            The size of the font in the plot
        color (bool):
            If true, the tree nodes will be colored
    """
    x_explain = dte.transform_to_x_explain(dte.x_train_orig)
    features = dte.convert_columns_to_interpretable(x_explain).columns

    decision_tree = dte.produce()

    fig, ax = plt.subplots(figsize=size)
    plot_tree(decision_tree, feature_names=features, class_names=class_names,
              impurity=False, fontsize=fontsize, ax=ax, filled=color)

    plt.show()


def plot_tree_importances(dte, select_by="absolute", n=5, values=None,
                          flip_colors=False):
    """
    Plot the most contributing features (for a decision tree explainer)

    Args:
        dte:
            The decision tree explainer.
        select_by (one of "absolute", "max", "min"):
            Which explanation to plot.
        n (int):
            Number of features to plot
        values (Series or DataFrame of shape (1, n_features):
            If given, show the corresponding values alongside the feature names
        flip_colors (Boolean):
            If True, make the positive explanation red and negative explanation blue.
            Useful if the target variable has a negative connotation

    Returns:
        pyplot figure
            Bar plot of top contributors
    """
    decision_tree = dte.produce()
    x_explain = dte.transform_to_x_explain(dte.x_train_orig)
    features = dte.convert_columns_to_interpretable(x_explain).columns

    importances = decision_tree.feature_importances_
    df = pd.DataFrame(importances[None, :], columns=features)

    plot_top_contributors(df, select_by=select_by, n=n, values=values, flip_colors=flip_colors)
