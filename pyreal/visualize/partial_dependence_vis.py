import matplotlib.pyplot as plt
import seaborn as sns

from pyreal.types.explanations.feature_value_based import FeatureValueBased


def partial_dependence_plot(explanation, transparent=False, show=False, filename=None):
    """
    Create Partial Depedence Plot from PartialDependenceExplanation object

    Args:
        explanation (FeatureValueBased):
            A FeatureValueBased explanation object
        transparent (Boolean):
            If True, the background of the figure is set to transparent.
        flip_colors (Boolean):
            If True, make the positive explanation red and negative explanation blue.
            Useful if the target variable has a negative connotation
        include_averages (Boolean):
            If True, include the mean values in the visualization (if provided in explanation)
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

    dim = len(explanation.explanation.grid)
    # one dimensional pdp
    if dim == 1:
        ax = sns.lineplot(
            x=explanation.explanation.grid[0], y=explanation.explanation.predictions[0]
        )
        ax.set(xlabel=explanation.explanation.feature_names[0], ylabel="Partial Dependence")
    else:
        plt.contourf(
            explanation.explanation.grid[0],
            explanation.explanation.grid[1],
            explanation.explanation.predictions[0],
        )
        # Add labels to the contour lines with custom names
        # plt.clabel(
        #     cs,
        #     fontsize=10,
        #     colors="k",
        #     use_clabeltext=True,
        # )

        ax = plt.gca()
        ax.set(
            xlabel=explanation.explanation.feature_names[0],
            ylabel=explanation.explanation.feature_names[1],
        )
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
