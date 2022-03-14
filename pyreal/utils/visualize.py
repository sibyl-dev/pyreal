"""
Includes basic visualization methods, mostly used to testing purposes.
"""
import matplotlib.pyplot as plt
import numpy as np
from pyreal.utils._plot_tree import TreeExporter


def plot_top_contributors(contributions, select_by="absolute", n=5, values=None,
                          flip_colors=False, precision=2, show=False, filename=None):
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
        precision (int):
            Number of decimal places to print for numeric float values
        show (Boolean):
            Show the figure
        filename (string or None):
            If not None, save the figure as filename

    Returns:
        pyplot figure
            Bar plot of top contributors
    """
    features = contributions.columns.to_numpy()
    if values is not None:
        features = np.array(["%s (%.*f)" % (feature, precision, values[feature])
                             if isinstance(values[feature], float)
                             else "%s (%s)" % (feature, values[feature])
                             for feature in features])

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

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()


def plot_tree_explanation(dte, transparent=False,
                          class_names=None, label='all',
                          filled=True, rounded=True, impurity=False,
                          proportion=False, precision=3, fontsize=10):
    """
    Plot the decision tree given the decision tree explainer

    Args:
        dte:
            Decision tree explainer
        transparent (bool):
            Determines if the output figure is transparent or not
    """

    negative_color = "#ef8a62"
    positive_color = "#67a9cf"

    decision_tree = dte.produce()
    feature_names = dte.return_features()
    figsize = (dte.max_depth*4+10, dte.max_depth*2)
    if transparent:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor='w')

    exporter = TreeExporter(
        max_depth=dte.max_depth,
        feature_names=feature_names,
        class_names=class_names,
        positive_color=positive_color,
        negative_color=negative_color,
        label=label,
        filled=filled,
        impurity=impurity,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
    )
    exporter.export(decision_tree, ax=ax)
    # plot_tree(decision_tree, feature_names=feature_names,
    #           impurity=impurity, filled=filled, rounded=rounded,
    #           proportion=proportion, fontsize=fontsize, ax=ax)
    plt.show()
