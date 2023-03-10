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


def plot_top_contributors(
    explanation,
    select_by="absolute",
    n=5,
    transparent=False,
    flip_colors=False,
    precision=2,
    show=False,
    filename=None,
):
    """
    Plot the most contributing features

    Args:
        explanation (DataFrame or FeatureBased):
            One output DataFrame from RealApp.produce_local_feature_contributions or
            RealApp.prepare_global_feature_importance OR FeatureBased explanation object
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


def swarm_plot(explanation, type="swarm", n=5, show=False, filename=None, legend=True, **kwargs):
    """
    Generates a strip plot (type="strip") or a swarm plot (type="swarm") from a set of feature
    contributions.

    Args:
        explanation (DataFrame or FeatureBased):
            One output DataFrame from RealApp.produce_local_feature_contributions OR
            FeatureContributions explanation object
        type (String, one of ["strip", "swarm"]:
            The type of plot to generate
        n (int):
            Number of features to show
        show (Boolean):
            Whether or not to show the figure
        filename (string or None):
            If not None, save the figure as filename
        legend (Boolean):
            If True, show a colorbar legend
        **kwargs:
            Additional arguments to pass to seaborn.swarmplot or seaborn.stripplot
    """
    if isinstance(explanation, FeatureContributionExplanation):
        contributions = explanation.get()
        values = explanation.get()
    else:
        contribution_list = [explanation[i]["Contribution"] for i in explanation]
        value_list = [explanation[i]["Feature Value"] for i in explanation]
        contributions = pd.DataFrame(contribution_list)
        values = pd.DataFrame(value_list)

    average_importance = np.mean(abs(contributions), axis=0)
    order = np.argsort(average_importance)[::-1]
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
                palette=palette,
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

    if legend:
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
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
