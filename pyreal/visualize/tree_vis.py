import matplotlib.pyplot as plt

from pyreal.utils._plot_tree import TreeExporter
from pyreal.visualize.visualize_config import NEGATIVE_COLOR, POSITIVE_COLOR


def plot_tree_explanation(
    dte,
    transparent=False,
    class_names=None,
    label="all",
    filled=True,
    rounded=True,
    impurity=False,
    proportion=False,
    precision=3,
    fontsize=10,
    filename=None,
):
    """
    Plot the decision tree given the decision tree explainer

    Args:
        dte:
            Decision tree explainer.
        transparent (Boolean):
            Determines if the output figure is transparent or not.
        class_names (list of str):
            Names of each of the target classes in ascending numerical order.
        label ('all', 'root', or 'none'):
            Options include 'all' to show at every node, 'root' to show only at
            the top root node, or 'none' to not show at any node.
        filled (Boolean):
            If set to True, paint the nodes based on the majority class of the node.
        rounded (Boolean):
            If set to True, the box representing each node will have rounded corners.
        impurity (Boolean):
            If set to True, show the impurity at each node.
        proportion (Boolean):
            If set to True, change the display of 'values' and/or 'samples' to be
            proportions and percentages respectively.
        precision (int):
            Number of digits of precision for floating point numbers.
        filename (string or None):
            If not None, save the figure as filename.
    """

    decision_tree = dte.produce().get()
    feature_names = dte.return_features()
    if dte.max_depth is None:
        max_depth = 6
    else:
        max_depth = dte.max_depth
    figsize = (max_depth * 4 + 10, max_depth * 2)
    if transparent:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor="w")

    exporter = TreeExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        positive_color=POSITIVE_COLOR,
        negative_color=NEGATIVE_COLOR,
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
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")

    plt.show()
