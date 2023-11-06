import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from pyreal.explanation_types import FeatureValueBased
from pyreal.visualize.visualize_config import NEGATIVE_COLOR, NEUTRAL_COLOR, POSITIVE_COLOR


def partial_dependence_plot(explanation, transparent=False, show=False, filename=None):
    """
    Create Partial Depedence Plot from PartialDependenceExplanation object

    Args:
        explanation (FeatureValueBased):
            A FeatureValueBased explanation object
        transparent (Boolean):
            If True, the background of the figure is set to transparent.
        show (Boolean):
            Show the figure
        filename (string or None):
            If not None, save the figure as filename

    Returns:
        pyplot figure
            Partial dependence plot (see tutorial for example)
    """
    if not isinstance(explanation, FeatureValueBased):
        raise TypeError("explanation must be an instance of FeatureValueBased")

    if transparent:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(facecolor="w")

    cmap = LinearSegmentedColormap.from_list(
        "pyreal_cm", colors=[NEGATIVE_COLOR, NEUTRAL_COLOR, POSITIVE_COLOR]
    )
    dim = len(explanation.explanation.grid)
    # one dimensional pdp
    if dim == 1:
        ax = sns.lineplot(
            x=explanation.explanation.grid[0], y=explanation.explanation.predictions[0]
        )
        ax.set(
            xlabel=f"Feature: {explanation.explanation.feature_names[0]}",
            ylabel="Partial Dependence",
        )
    else:
        contour_plot = ax.contourf(
            explanation.explanation.grid[0],
            explanation.explanation.grid[1],
            np.transpose(explanation.explanation.predictions[0]),
            # transpose is due to the first feature lying on x-axis
            cmap=cmap,
        )
        fig.colorbar(contour_plot, label="partial dependence")

        ax = plt.gca()
        ax.set(
            xlabel=explanation.explanation.feature_names[0],
            ylabel=explanation.explanation.feature_names[1],
        )
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
