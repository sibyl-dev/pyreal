import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyreal.realapp import realapp
from pyreal.types.explanations.feature_based import (
    FeatureContributionExplanation,
    FeatureImportanceExplanation,
)
from pyreal.visualize.visualize_config import (
    NEGATIVE_COLOR,
    NEGATIVE_COLOR_LIGHT,
    NEUTRAL_COLOR,
    PALETTE_CMAP,
    POSITIVE_COLOR,
    POSITIVE_COLOR_LIGHT,
)


def _parse_multi_contribution(explanation):
    if isinstance(explanation, FeatureContributionExplanation):
        contributions = explanation.get()
        values = explanation.get()
    else:
        contribution_list = [explanation[i]["Contribution"] for i in explanation]
        value_list = [explanation[i]["Feature Value"] for i in explanation]
        feature_list = explanation[next(iter(explanation))]["Feature Name"].values
        contributions = pd.DataFrame(contribution_list)
        contributions.columns = feature_list
        values = pd.DataFrame(value_list)
        values.columns = feature_list
    return contributions, values


def plot_top_contributors(
    explanation,
    select_by="absolute",
    n=5,
    transparent=False,
    flip_colors=False,
    precision=2,
    prediction=None,
    include_averages=False,
    include_axis=True,
    show=False,
    filename=None,
):
    """
    Plot the most contributing features

    Args:
        explanation (DataFrame or FeatureBased):
            One output DataFrame from RealApp.produce_feature_contributions or
            RealApp.prepare_feature_importance OR FeatureBased explanation object
        select_by (one of "absolute", "max", "min"):
            Which explanation to plot.
        n (int):
            Number of features to plot
        transparent (Boolean):
            If True, the background of the figure is set to transparent.
        flip_colors (Boolean):
            If True, make the positive explanation red and negative explanation blue.
            Useful if the target variable has a negative connotation
        precision (int):
            Number of decimal places to print for numeric float values
        prediction (numeric or string):
            Prediction to display in the title
        include_averages (Boolean):
            If True, include the mean values in the visualization (if provided in explanation)
        include_axis (Boolean):
            If True, include the contribution axis
        show (Boolean):
            Show the figure
        filename (string or None):
            If not None, save the figure as filename

    Returns:
        pyplot figure
            Bar plot of top contributors
    """
    if isinstance(explanation, FeatureContributionExplanation):
        explanation = realapp.format_feature_contribution_output(explanation)
        explanation = explanation[next(iter(explanation))]
    elif isinstance(explanation, FeatureImportanceExplanation):
        explanation = realapp.format_feature_importance_output(explanation)

    features = explanation["Feature Name"].to_numpy()
    if "Feature Value" in explanation:
        values = explanation["Feature Value"].to_numpy()
        if include_averages and "Average/Mode" in explanation:
            averages = explanation["Average/Mode"].to_numpy()
            features = np.array(
                [
                    (
                        "%s - %.*g (mean: %.*g)"
                        % (features[i], precision, values[i], precision, averages[i])
                        if isinstance(values[i], (float, np.float, int, np.integer))
                        else "%s - %s (mode: %s)" % (features[i], values[i], averages[i])
                    )
                    for i in range(len(features))
                ]
            )
        else:
            features = np.array(
                [
                    (
                        "%s (%.*f)" % (features[i], precision, values[i])
                        if isinstance(values[i], float)
                        else "%s (%s)" % (features[i], values[i])
                    )
                    for i in range(len(features))
                ]
            )
    if "Contribution" in explanation:
        contributions = explanation["Contribution"]
    elif "Importance" in explanation:
        contributions = explanation["Importance"]
    else:
        raise ValueError("Provided DataFrame has neither Contribution nor Importance column")

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

    if not flip_colors:
        colors = [
            NEGATIVE_COLOR if (c < 0) else POSITIVE_COLOR for c in contributions[to_plot][::-1]
        ]
    else:
        colors = [
            POSITIVE_COLOR if (c < 0) else NEGATIVE_COLOR for c in contributions[to_plot][::-1]
        ]

    if transparent:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(facecolor="w")
    plt.barh(features[to_plot][::-1], contributions[to_plot][::-1], color=colors)
    plt.title("Contributions by feature", fontsize=18)
    if prediction is not None:
        plt.title("Overall prediction: %s" % prediction, fontsize=12)
        plt.suptitle("Contributions by feature", fontsize=18, y=1)
    if include_axis:
        plt.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
        plt.xlabel("Contribution")
    else:
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


def swarm_plot(
    explanation, type="swarm", n=5, discrete=False, show=False, filename=None, **kwargs
):
    """
    Generates a strip plot (type="strip") or a swarm plot (type="swarm") from a set of feature
    contributions.

    Args:
        explanation (DataFrame or FeatureBased):
            One output DataFrame from RealApp.produce_feature_contributions OR
            FeatureContributions explanation object
        type (String, one of ["strip", "swarm"]:
            The type of plot to generate
        n (int):
            Number of features to show
        discrete (Boolean):
            If true, give discrete legends for each row. Otherwise, give a colorbar legend
        show (Boolean):
            If True, show the figure
        filename (string or None):
            If not None, save the figure as filename
        legend (Boolean):
            If True, show a colorbar legend
        **kwargs:
            Additional arguments to pass to seaborn.swarmplot or seaborn.stripplot
    """
    contributions, values = _parse_multi_contribution(explanation)

    average_importance = np.mean(abs(contributions), axis=0)
    order = np.argsort(average_importance)[::-1]
    num_cats = []

    if discrete:
        legend = "brief"
    else:
        legend = False

    for i in range(n):
        hues = values.iloc[:, order[i : i + 1]]
        hues = hues.melt()["value"]
        num_colors = len(np.unique(hues.astype("str")))
        palette = sns.blend_palette(
            [NEGATIVE_COLOR_LIGHT, NEUTRAL_COLOR, POSITIVE_COLOR_LIGHT], n_colors=num_colors
        )
        if type == "strip":
            ax = sns.stripplot(
                x="value",
                y="variable",
                hue=hues,
                data=contributions.iloc[:, order[i : i + 1]].melt(),
                palette=palette,
                legend=legend,
                size=3,
                **kwargs
            )
        elif type == "swarm":
            ax = sns.swarmplot(
                x="value",
                y="variable",
                hue=hues,
                data=contributions.iloc[:, order[i : i + 1]].melt(),
                palette=palette,
                legend=legend,
                size=3,
                **kwargs
            )
        else:
            raise ValueError("Invalid type %s. Type must be one of [strip, swarm]." % type)

        handles, labels = ax.get_legend_handles_labels()
        num_cats.append(len(labels) - sum(num_cats))
        plt.axvline(x=0, color="black", linewidth=1)
        ax.grid(axis="y")
        ax.set_ylabel("")
        ax.set_xlabel("Contributions")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    legends = []
    if discrete:
        handles, labels = ax.get_legend_handles_labels()
        shift = 1 / len(num_cats)
        r = 0
        for i in range(0, len(num_cats)):
            if num_cats[i] <= 5:
                l1 = ax.legend(
                    handles[r : r + num_cats[i]],
                    labels[r : r + num_cats[i]],
                    bbox_to_anchor=(1, 1 - (i * shift)),
                    loc="upper left",
                    ncol=num_cats[i],
                    labelspacing=0.2,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    frameon=False,
                )
                legends.append(l1)
            else:
                step = math.ceil(num_cats[i] / 5)
                l1 = ax.legend(
                    handles[r : r + num_cats[i] : step],
                    labels[r : r + num_cats[i] : step],
                    bbox_to_anchor=(1, 1 - (i * shift)),
                    loc="upper left",
                    ncol=num_cats[i],
                    labelspacing=0.2,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    frameon=False,
                )
                legends.append(l1)
            r += num_cats[i]
        for labels in legends[:-1]:
            ax.add_artist(labels)

    else:
        ax = plt.gca()
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=PALETTE_CMAP, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm)
        cbar.ax.get_yaxis().set_ticks([])
        cbar.ax.text(1.5, 0.05, "low", ha="left", va="center")
        cbar.ax.text(1.5, 0.95, "high", ha="left", va="center")
        cbar.ax.set_ylabel("Feature Value", rotation=270)
        cbar.ax.get_yaxis().labelpad = 15

    if filename is not None:
        plt.gcf().savefig(filename, bbox_extra_artists=legends, bbox_inches="tight")
    if show:
        plt.show()


def feature_scatter_plot(
    explanation, feature, predictions=None, discrete=None, show=False, filename=None
):
    """
    Plot a contribution scatter plot for one feature

    Args:
        explanation (DataFrame or FeatureBased):
            One output DataFrame from RealApp.produce_feature_contributions OR
            FeatureContributions explanation object
        feature (column label):
            Label of column to visualize
        predictions (array-like of length n_instances):
            Predictions corresponding to explained instances
        discrete (Boolean):
            If true, plot x as discrete data. Defaults to True if x is not numeric.
        show (Boolean):
            If True, show the figure
        filename (string or None):
            If not None, save the figure as filename

    Returns:

    """
    contributions, values = _parse_multi_contribution(explanation)

    contributions = contributions[feature]
    values = values[feature]

    if isinstance(predictions, dict):
        predictions = np.array([predictions[i] for i in predictions]).reshape(-1)

    legend_type = "discrete"
    if predictions is None:
        legend_type = "none"
        predictions = np.zeros_like(contributions)

    data = pd.DataFrame(
        {"Contribution": contributions.values, "Value": values.values, "Prediction": predictions}
    )

    num_colors = len(np.unique(predictions.astype("str")))
    palette = sns.blend_palette(
        [NEGATIVE_COLOR_LIGHT, NEUTRAL_COLOR, POSITIVE_COLOR_LIGHT], n_colors=num_colors
    )

    if (
        legend_type != "none"
        and isinstance(predictions[0], float)
        or (isinstance(predictions[0], int) and num_colors > 6)
    ):
        legend_type = "continuous"

    plot_legend = False
    if legend_type == "discrete":
        plot_legend = True

    if discrete is None:
        discrete = not pd.api.types.is_numeric_dtype(values)
    if discrete:
        ax = sns.stripplot(
            x="Value",
            y="Contribution",
            data=data,
            hue="Prediction",
            palette=palette,
            legend=plot_legend,
            alpha=0.5,
            zorder=0,
        )
    else:
        ax = sns.scatterplot(
            x="Value",
            y="Contribution",
            data=data,
            hue="Prediction",
            palette=palette,
            legend=plot_legend,
            alpha=0.5,
        )

    plt.axhline(0, color="black", zorder=0)
    plt.xlabel("Values for %s" % feature)
    if legend_type == "continuous":
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=PALETTE_CMAP, norm=norm)
        min_val = predictions.min()
        max_val = predictions.max()
        sm.set_array([])
        cbar = ax.figure.colorbar(sm)
        cbar.ax.get_yaxis().set_ticks([])
        cbar.ax.text(1.5, 0.05, ("%.2f" % min_val).rstrip("0").rstrip("."), ha="left", va="center")
        cbar.ax.text(1.5, 0.95, ("%.2f" % max_val).rstrip("0").rstrip("."), ha="left", va="center")
        cbar.ax.set_ylabel("Prediction", rotation=270)
        cbar.ax.get_yaxis().labelpad = 15

    if filename is not None:
        plt.gcf().savefig(filename, bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()
