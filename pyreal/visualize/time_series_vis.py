import matplotlib.colors as color
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pyreal.visualize.visualize_config import NEGATIVE_COLOR, POSITIVE_COLOR

WIDTH = 5


def plot_time_series_explanation(timeSeriesData, contribution):
    index = timeSeriesData.index
    assert index.size == 1
    columns = timeSeriesData.columns.get_level_values(0).unique()
    timestamps = timeSeriesData.columns.get_level_values(1).unique()
    fig, axs = plt.subplots(columns.size)
    cmap = color.LinearSegmentedColormap.from_list("posnegcmap", [NEGATIVE_COLOR, POSITIVE_COLOR])

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


def plot_timeseries_saliency(
    data, colors, title=None, fig=None, scale=True, mincol=None, maxcol=None, show=False
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

    if show:
        plt.show()


def plot_shapelet(timeSeriesData, shapeletIndices, shapeletLength, show=False):
    # only support plotting one instance at a time
    index = timeSeriesData.index
    assert index.size == 1
    columns = timeSeriesData.columns.get_level_values(0).unique()
    timestamps = timeSeriesData.columns.get_level_values(1).unique()
    fig, axs = plt.subplots(columns.size)

    for var in columns:
        axs.plot(timestamps, timeSeriesData.iloc[0].loc[(var, slice(None))], color=NEGATIVE_COLOR)
        for idx in shapeletIndices:
            axs.plot(
                np.arange(idx, idx + shapeletLength),
                timeSeriesData.iloc[0].loc[(var, slice(idx, idx + shapeletLength - 1))],
                color=POSITIVE_COLOR,
            )

    if show:
        plt.show()
