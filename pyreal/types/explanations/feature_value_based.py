import numpy as np
import pandas as pd

from pyreal.types.explanations.base import Explanation


class FeatureValueExplanation:
    """
    A wrapper for feature value-based explanations.

    Args:
        feature_names:
            the list of features in this explanation
        predictions:
            the predictions calculated on the grid values, this has shape
            (n_output, feature_values[0], feature_values[1],...)
        grid:
            lists of feature values
    """

    def __init__(self, feature_names, predictions, grid):
        self.feature_names = feature_names
        self.predictions = predictions
        self.grid = grid

    def rename_features(self, feature_descriptions):
        if feature_descriptions is not None:
            for i, feature in enumerate(self.feature_names):
                if feature in feature_descriptions.keys():
                    self.feature_names[i] = feature_descriptions[feature]


class FeatureValueBased(Explanation):
    """
    A type wrapper for feature value-based DataFrame type outputs from explanation algorithms.
    """

    def __init__(self, feature_names, predictions, grid):
        explanation = FeatureValueExplanation(feature_names, predictions, grid)
        super().__init__(explanation)

    def validate(self):
        """
        Validate that `self.explanation` is a valid `FeatureValueExplanation`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()
        if not isinstance(self.explanation.feature_names, (list, pd.Index)):
            raise AssertionError(
                "You must provide the explainer feature names as a list of strings, get type"
                f" {type(self.explanation.feature_names)}"
            )

        if not isinstance(self.explanation.predictions, np.ndarray):
            raise AssertionError(
                "The explanation values must be provided in the form of an Numpy array, get type"
                f" {type(self.explanation.predictions)}"
            )

        if not isinstance(self.explanation.grid, list):
            raise AssertionError(
                "The feature grids must be provided in the form of an Numpy array."
            )

        if len(self.explanation.feature_names) != len(self.explanation.grid):
            raise AssertionError("The values of feature grids must match the number of features.")

    def update_feature_names(self, feature_descriptions):
        """
        Wrapper method for renaming features.
        Validates explanation after renaming.
        """
        self.explanation.rename_features(feature_descriptions)
        self.validate()


class PartialDependenceExplanation(FeatureValueBased):
    def validate(self):
        """
        Validate that `self.explanation` is a valid `FeatureValueExplanation`
        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` is invalid
        """
        super().validate()
