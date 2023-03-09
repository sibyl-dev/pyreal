import matplotlib.pyplot as plt
import numpy as np
from pyreal.types.explanations.feature_based import FeatureContributionExplanation, FeatureImportanceExplanation
from pyreal.realapp import realapp


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
