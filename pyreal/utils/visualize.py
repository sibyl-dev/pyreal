"""
Includes basic visualization methods, mostly used to testing purposes.
"""
import matplotlib.colors as color
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection

from pyreal.utils._plot_tree import TreeExporter

negative_color = "#ef8a62"
positive_color = "#67a9cf"
WIDTH = 5


def plot_top_contributors(
    contributions,
    select_by="absolute",
    n=5,
    values=None,
    transparent=False,
    flip_colors=False,
    precision=2,
    show=False,
    filename=None,
):
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
        transparent (Boolean):
            If True, the background of the figure is set to transparent.
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
        features = np.array(
            [
                "%s (%.*f)" % (feature, precision, values[feature])
                if isinstance(values[feature], float)
                else "%s (%s)" % (feature, values[feature])
                for feature in features
            ]
        )

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
        raise ValueError(
            "Invalid select_by option %s, should be one of 'min', 'max', 'absolute'" % select_by
        )

    to_plot = order[0:n]

    negative_color = "#ef8a62"
    positive_color = "#67a9cf"
    if not flip_colors:
        colors = [
            negative_color if (c < 0) else positive_color for c in contributions[to_plot][::-1]
        ]
    else:
        colors = [
            positive_color if (c < 0) else negative_color for c in contributions[to_plot][::-1]
        ]

    if transparent:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(facecolor="w")
    plt.barh(features[to_plot][::-1], contributions[to_plot][::-1], color=colors)
    plt.title("Contribution by feature")
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(x=0, color="black")

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()


def swarm_plot(contributions, values, type="swarm", n=5, show=False, filename=None, **kwargs):
    """
    Generates a strip plot (type="strip") or a swarm plot (type="swarm") from a set of feature
    contributions.

    Args:
        contributions (Series or DataFrame of shape (n_instances, n_features):
            Contributions, with feature names as the column names
        values (Series or DataFrame of shape (n_instances, n_features):
            If given, show the corresponding values alongside the feature names
        type (String, one of ["strip", "swarm"]:
            The type of plot to generate
        n (int):
            Number of features to show
        show (Boolean):
            Whether or not to show the figure
        filename (string or None):
            If not None, save the figure as filename
        **kwargs:
            Additional arguments to pass to seaborn.swarmplot or seaborn.stripplot
    """
    average_importance = np.mean(abs(contributions), axis=0)
    order = np.argsort(average_importance)[::-1]
    for i in range(n):
        hues = values.iloc[:, order[i : i + 1]]
        hues = hues.melt()["value"]
        if type == "strip":
            ax = sns.stripplot(
                x="value",
                y="variable",
                hue=hues,
                data=contributions.iloc[:, order[i : i + 1]].melt(),
                palette="coolwarm_r",
                legend=False,
                size=3,
                **kwargs
            )
        elif type == "swarm":
            ax = sns.swarmplot(
                x="value",
                y="variable",
                hue=hues,
                data=contributions.iloc[:, order[i : i + 1]].melt(),
                palette="coolwarm_r",
                legend=False,
                size=3,
                **kwargs
            )
        else:
            raise ValueError("Invalid type %s. Type must be one of [strip, swarm]." % type)
        plt.axvline(x=0, color="black", linewidth=1)
        ax.grid(axis="y")
        ax.set_ylabel("")
        ax.set_xlabel("Contributions")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()


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

    decision_tree = dte.produce()
    feature_names = dte.return_features()
    figsize = (dte.max_depth * 4 + 10, dte.max_depth * 2)
    if transparent:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor="w")

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
