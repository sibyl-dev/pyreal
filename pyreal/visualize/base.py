import pandas as pd

from pyreal.explanation_types import (
    DecisionTreeExplanation,
    Explanation,
    FeatureContributionExplanation,
    FeatureImportanceExplanation,
    PartialDependenceExplanation,
)
from pyreal.visualize import (
    example_table,
    feature_bar_plot,
    feature_scatter_plot,
    partial_dependence_plot,
    strip_plot,
)


def plot_explanation(
    explanation,
    feature=None,
    num_features=5,
    predictions=None,
    discrete=False,
    show=False,
    filename=None,
    **kwargs
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
        predictions (numeric, string, or array-like):
            Prediction(s) of inputs rows being explained, optionally shown in some visualization
            types
        discrete (Boolean):
            True if features should be plotted as having categorical or discrete values, False if
            continuous. Only used for some visualization types.
        show (Boolean)
            If True, show the plot after generating
        filename (string)
            If given, save the figure as filename
        **kwargs:
            Additional arguments to pass to the selected plot

    """
    if isinstance(explanation, FeatureContributionExplanation):
        if feature is None:
            return strip_plot(
                explanation,
                num_features=num_features,
                show=show,
                filename=filename,
                discrete=discrete,
                *kwargs,
            )
        else:
            return feature_scatter_plot(
                explanation,
                feature=feature,
                show=show,
                filename=filename,
                predictions=predictions,
                **kwargs,
            )
    if isinstance(explanation, FeatureImportanceExplanation):
        return feature_bar_plot(
            explanation, num_features=num_features, show=show, filename=filename, **kwargs
        )
    if isinstance(explanation, PartialDependenceExplanation):
        return partial_dependence_plot(explanation, show=show, filename=filename, **kwargs)
    if isinstance(explanation, DecisionTreeExplanation):
        raise ValueError(
            "Sorry - decision tree explanations are not currently supported for visualization"
        )
    if isinstance(explanation, Explanation):
        raise ValueError("Sorry - the explanation given is not of a type currently supported.")

    if isinstance(explanation, dict) and "X" in explanation:
        return example_table(explanation, **kwargs)

    if isinstance(explanation, dict):  # Assume multiple feature contributions
        if "Contribution" in explanation[next(iter(explanation))]:
            if feature is None:
                return strip_plot(
                    explanation,
                    num_features=num_features,
                    show=show,
                    filename=filename,
                    discrete=discrete,
                    **kwargs,
                )
            else:
                return feature_scatter_plot(
                    explanation,
                    feature=feature,
                    show=show,
                    filename=filename,
                    predictions=predictions,
                    **kwargs,
                )
        else:
            raise ValueError(
                "Unsupported explanation type. If this is an output from an example-based"
                " explanation, this function only supports taking one row result at a time (ie,"
                " plot_explanation(example_explanation[0])"
            )
    if isinstance(explanation, pd.DataFrame):  # Assume importance or one-entity contributions
        return feature_bar_plot(
            explanation,
            num_features=num_features,
            show=show,
            filename=filename,
            prediction=predictions,
            **kwargs,
        )

    raise ValueError("Given an explanation of an unrecognized format.")
