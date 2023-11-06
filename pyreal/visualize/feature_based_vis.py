import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyreal.explanation_types import FeatureContributionExplanation, FeatureImportanceExplanation
from pyreal.realapp import realapp
from pyreal.utils import get_top_contributors
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
        values = explanation.get_values()
    else:
        contribution_list = [explanation[i]["Contribution"] for i in explanation]
        value_list = [explanation[i]["Feature Value"] for i in explanation]
        feature_list = explanation[next(iter(explanation))]["Feature Name"].values
        contributions = pd.DataFrame(contribution_list)
        contributions.columns = feature_list
        values = pd.DataFrame(value_list)
        values.columns = feature_list
    return contributions, values


def feature_bar_plot(
    explanation,
    select_by="absolute",
    num_features=5,
    transparent=False,
    flip_colors=False,
    precision=2,
    prediction=None,
    include_averages=False,
    include_axis=True,
    show=False,
    filename=None,
    **kwargs
):
    """
    Plot the most contributing features

    Args:
        explanation (DataFrame or FeatureBased):
            One output DataFrame from RealApp.produce_feature_contributions or
            RealApp.prepare_feature_importance OR FeatureBased explanation object
        select_by (one of "absolute", "max", "min"):
            Method to use when selecting features.
        num_features (int):
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
        **kwargs:
            Additional parameters to pass into plt.barh

    Returns:
        pyplot figure
            Bar plot of top contributors
    """
    if isinstance(explanation, FeatureContributionExplanation):
        explanation = realapp.format_feature_contribution_output(explanation)
        explanation = explanation[next(iter(explanation))]
    elif isinstance(explanation, FeatureImportanceExplanation):
        explanation = realapp.format_feature_importance_output(explanation)

    if isinstance(explanation, dict):
        raise ValueError(
            "Invalid explanation. Expected feature contribution explanation on a single instance"
            " or feature importance explanation. If you are passing in an explanation from"
            " RealApp.produce_feature_contributions(), please index to get a single instance, ie"
            " explanation[0]."
        )
    if not isinstance(explanation, pd.DataFrame):
        raise ValueError(
            "Invalid explanation type, expected DataFrame or"
            " FeatureContributionExplanation/FeatureImportanceExplanation object"
        )

    explanation = get_top_contributors(explanation, num_features=num_features, select_by=select_by)

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
    are_importances = False
    if "Contribution" in explanation:
        contributions = explanation["Contribution"]
    elif "Importance" in explanation:
        contributions = explanation["Importance"]
        are_importances = True
    else:
        raise ValueError("Provided DataFrame has neither Contribution nor Importance column")

    if contributions.ndim == 2:
        contributions = contributions.iloc[0]
    contributions = contributions.to_numpy()

    if not flip_colors:
        colors = [NEGATIVE_COLOR if (c < 0) else POSITIVE_COLOR for c in contributions[::-1]]
    else:
        colors = [POSITIVE_COLOR if (c < 0) else NEGATIVE_COLOR for c in contributions[::-1]]

    if transparent:
        _, ax = plt.subplots()
    else:
        _, ax = plt.subplots(facecolor="w")
    plt.barh(features[::-1], contributions[::-1], color=colors, **kwargs)
    if are_importances:
        title = "Feature Importance Scores"
    else:
        title = "Feature Contributions"
    plt.title(title, fontsize=18)
    if prediction is not None:
        plt.title("Overall prediction: %s" % prediction, fontsize=12)
        plt.suptitle(title, fontsize=18, y=1)
    if include_axis:
        plt.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
        if are_importances:
            plt.xlabel("Importance")
        else:
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


def strip_plot(
    explanation,
    type="strip",
    num_features=5,
    discrete=False,
    show=False,
    filename=None,
    marker_size=3,
    palette=None,
    show_legend=True,
    **kwargs
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
        num_features (int):
            Number of features to show
        discrete (Boolean):
            If true, give discrete legends for each row. Otherwise, give a colorbar legend
        show (Boolean):
            If True, show the figure
        filename (string or None):
            If not None, save the figure as filename
        marker_size (int):
            Size of markers to use in plot
        palette (seaborn palette name, list, or dict):
            Colors to use in the plot. See seaborn.color_palette for more info
        show_legend (Boolean):
            If False, hide the legend
        **kwargs:
            Additional arguments to pass to seaborn.swarmplot or seaborn.stripplot
    """
    contributions, values = _parse_multi_contribution(explanation)

    average_importance = np.mean(abs(contributions), axis=0)
    order = np.argsort(average_importance)[::-1]
    num_cats = []

    if discrete and show_legend:
        legend = "brief"
    else:
        legend = False
    generate_palette = palette is None
    for i in range(num_features):
        hues = values.iloc[:, order[i : i + 1]]
        hues = hues.melt()["value"]
        num_colors = len(np.unique(hues.astype("str")))
        if generate_palette:
            palette = sns.blend_palette(
                [NEGATIVE_COLOR_LIGHT, NEUTRAL_COLOR, POSITIVE_COLOR_LIGHT], n_colors=num_colors
            )
        if "size" in kwargs:
            marker_size = kwargs["size"]
            kwargs.pop("size", None)
        if type == "strip":
            ax = sns.stripplot(
                x="value",
                y="variable",
                hue=hues,
                data=contributions.iloc[:, order[i : i + 1]].melt(),
                palette=palette,
                legend=legend,
                size=marker_size,
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
                size=marker_size,
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

    if show_legend:
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
    explanation,
    feature,
    predictions=None,
    discrete=None,
    show=False,
    filename=None,
    palette=None,
    marker_alpha=0.5,
    marker_size=3,
    **kwargs
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
        palette (seaborn palette name, list, or dict):
            Colors to use in the plot. See seaborn.color_palette for more info
        marker_alpha (float between (0,1]):
            Alpha value to use for markers
        marker_size (int):
            Size to use for markers
        **kwargs:
            Additional arguments to pass into seaborn.stripplot or seaborn.scatterplot
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
    if palette is None:
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
            alpha=marker_alpha,
            size=marker_size,
            zorder=0,
            **kwargs
        )
    else:
        ax = sns.scatterplot(
            x="Value",
            y="Contribution",
            data=data,
            hue="Prediction",
            palette=palette,
            legend=plot_legend,
            alpha=marker_alpha,
            sizes=marker_size,
            **kwargs
        )

    plt.axhline(0, color="black", zorder=0)
    plt.xlabel("Values for %s" % feature)
    if legend_type == "continuous":
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=PALETTE_CMAP, norm=norm)
        if discrete:
            plt.xticks(rotation=45, ha="right")
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
