"""
Includes basic visualization methods, mostly used to testing purposes.
"""
import matplotlib.colors as color
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pyreal.utils._plot_tree import TreeExporter

negative_color = "#ef8a62"
positive_color = "#67a9cf"
WIDTH = 5


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
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")

    plt.show()


def plot_shapelet(timeSeriesData, shapeletIndices, shapeletLength):
    # only support plotting one instance at a time
    index = timeSeriesData.index
    assert index.size == 1
    columns = timeSeriesData.columns.get_level_values(0).unique()
    timestamps = timeSeriesData.columns.get_level_values(1).unique()
    fig, axs = plt.subplots(columns.size)

    for var in columns:
        axs.plot(timestamps, timeSeriesData.iloc[0].loc[(var, slice(None))], color=negative_color)
        for idx in shapeletIndices:
            axs.plot(
                np.arange(idx, idx + shapeletLength),
                timeSeriesData.iloc[0].loc[(var, slice(idx, idx + shapeletLength - 1))],
                color=positive_color,
            )
    plt.show()


def plot_time_series_explanation(timeSeriesData, contribution):
    index = timeSeriesData.index
    assert index.size == 1
    columns = timeSeriesData.columns.get_level_values(0).unique()
    timestamps = timeSeriesData.columns.get_level_values(1).unique()
    fig, axs = plt.subplots(columns.size)
    cmap = color.LinearSegmentedColormap.from_list("posnegcmap", [negative_color, positive_color])

    for var in columns:
        axs.scatter(
            timestamps,
            timeSeriesData.iloc[0].loc[(var, slice(None))],
            c=contribution,
            cmap=cmap,
            vmin=-1,
            vmax=1,
        )

    plt.show()


"""def plot_timeseries_saliency(X, saliency, timesteps=None, show=True):
    if timesteps is None:
        timesteps = np.arange(X.shape[0])
    plt.plot(timesteps, X, c=saliency)
    if show:
        plt.show()"""


def plot_timeseries_saliency(
    data, colors, title=None, fig=None, scale=True, mincol=None, maxcol=None
):
    y = data
    x = np.arange(len(y))
    # y = preprocessing.scale(y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if mincol is None:
        mincol = colors.min()
    if maxcol is None:
        maxcol = colors.max()
    norm = plt.Normalize(mincol, maxcol)
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)
    lc.set_array(colors)
    lc.set_linewidth(WIDTH)
    # if(fig is None or ax is None):
    #    fig, ax = plt.subplots()
    if fig is None:
        fig = plt.figure()
    ax = plt.gca()
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

    if scale:
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

    ax.set_title(title)

    # plt.show()
