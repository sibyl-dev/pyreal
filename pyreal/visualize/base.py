import pandas as pd

from pyreal.explanation_types.explanations.base import Explanation
from pyreal.explanation_types.explanations.decision_tree import DecisionTreeExplanation
from pyreal.explanation_types.explanations.feature_based import (
    FeatureContributionExplanation,
    FeatureImportanceExplanation,
)
from pyreal.explanation_types.explanations.feature_value_based import PartialDependenceExplanation
from pyreal.visualize import (
    feature_bar_plot,
    feature_scatter_plot,
    partial_dependence_plot,
    strip_plot,
)


def plot_explanation(
    explanation, feature=None, num_features=5, show=False, filename=None, **kwargs
):
    """
    Plots a RealApp output or Pyreal explanation using the most appropriate visualization method

    Args:
        explanation (Explanation object or valid RealApp explanation format):
            The explanation to plot
        feature (string):
            If given, generate a plot for this explanation. Currently ignored for all explanation
            types except single entity feature contribution explanations.
        num_features (int):
            If the plot shows multiple features, the number of features to plot
        show (Boolean)
            If True, show the plot after generating
        filename (string)
            If given, save the figure as filename
        **kwargs:
            Additional arguments to pass to the selected plot

    """
    if isinstance(explanation, FeatureContributionExplanation):
        if feature is None:
            return strip_plot(explanation, n=num_features, show=show, filename=filename, **kwargs)
        else:
            return feature_scatter_plot(
                explanation, feature=feature, show=show, filename=filename, **kwargs
            )
    if isinstance(explanation, FeatureImportanceExplanation):
        return feature_bar_plot(
            explanation, n=num_features, show=show, filename=filename, **kwargs
        )
    if isinstance(explanation, PartialDependenceExplanation):
        return partial_dependence_plot(explanation, show=show, filename=filename, **kwargs)
    if isinstance(explanation, DecisionTreeExplanation):
        raise ValueError(
            "Sorry - decision tree explanations are not currently supported for visualization"
        )
    if isinstance(explanation, Explanation):
        raise ValueError("Sorry - the explanation given is not of a type currently supported.")

    if isinstance(explanation, dict):  # Assume multiple feature contributions
        if feature is None:
            return strip_plot(explanation, n=num_features, show=show, filename=filename, **kwargs)
        else:
            return feature_scatter_plot(
                explanation, feature=feature, show=show, filename=filename, **kwargs
            )
    if isinstance(explanation, pd.DataFrame):  # Assume importance or one-entity contributions
        return feature_bar_plot(
            explanation, n=num_features, show=show, filename=filename, **kwargs
        )

    raise ValueError("Given an explanation of an unrecognized format.")
