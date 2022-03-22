import collections

import numpy as np
import pandas as pd

from pyreal.transformers import Transformer
from pyreal.types.explanations.dataframe import (
    AdditiveFeatureContributionExplanation, FeatureImportanceExplanation,)

from collections.abc import Sequence

class FeatureSelectTransformer(Transformer):
    """
    A transformer that selects and re-orders features to match the model's inputs
    """

    def __init__(self, columns, **kwargs):
        """
        Initializes the transformer

        Args:
            columns (dataframe column label type or list of dataframe column label type):
                Label of column to select, or an ordered list of column labels to select
        """
        if columns is not None and not isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
            columns = [columns]
        self.columns = columns
        self.dropped_columns = []
        super().__init__(**kwargs)

    def fit(self, x, **params):
        """
        Saves the columns being dropped

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit on
        Returns:
            None

        """
        self.dropped_columns = list(set(x.columns) - set(self.columns))
        super().fit(x)

    def data_transform(self, x):
        """
        Reorders and selects the features in x

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The data to transform
        Returns:
            DataFrame of shape (n_instances, len(columns)):
                The data with features selected and reordered
        """
        return x[self.columns]

    def inverse_transform_explanation_additive_contributions(self, explanation):
        """
        Sets the contribution of dropped features to 0
        Args:
            explanation (AdditiveFeatureContributionExplanationType):
                The explanation to be transformed

        Returns:
            Returns:
                AdditiveFeatureContributionExplanationType:
                    The transformed explanation

        """
        explanation_df = explanation.get()
        for col in self.dropped_columns:
            explanation_df[col] = 0
        return AdditiveFeatureContributionExplanation(explanation_df)

    def inverse_transform_explanation_feature_importance(self, explanation):
        """
        Sets the importance of dropped features to 0

        Args:
            explanation (FeatureImportanceExplanation):
                The explanation to be transformed
        Returns:
            FeatureImportanceExplanation:
                The transformed explanation

        """
        explanation_df = explanation.get()
        for col in self.dropped_columns:
            explanation_df[col] = 0
        return FeatureImportanceExplanation(explanation_df)

    def transform_explanation_additive_contributions(self, explanation):
        """
        Selects the desired columns
        Args:
            explanation (AdditiveFeatureContributionExplanationType):
                The explanation to be transformed

        Returns:
            Returns:
                AdditiveFeatureContributionExplanationType:
                    The transformed explanation

        """
        return AdditiveFeatureContributionExplanation(explanation.get()[self.columns])

    def transform_explanation_feature_importance(self, explanation):
        """
        Selects the desired columns

        Args:
            explanation (FeatureImportanceExplanation):
                The explanation to be transformed
        Returns:
            FeatureImportanceExplanation:
                The transformed explanation

        """
        return FeatureImportanceExplanation(explanation.get()[self.columns])


class ColumnDropTransformer(Transformer):
    """
    A transformer that drops a set of columns from the data
    """

    def __init__(self, columns, **kwargs):
        """
        Initializes the transformer

        Args:
            columns (dataframe column label type or list of dataframe column label type):
                Label of column to select, or an ordered list of column labels to select
        """
        if columns is not None and not isinstance(columns, Sequence):
            columns = [columns]
        self.dropped_columns = columns
        super().__init__(**kwargs)

    def data_transform(self, x):
        """
        Reorders and selects the features in x

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The data to transform
        Returns:
            DataFrame of shape (n_instances, len(columns)):
                The data with features selected and reordered
        """
        return x.drop(self.dropped_columns, axis=1)

    def inverse_transform_explanation_additive_contributions(self, explanation):
        """
        Sets the contribution of dropped features to 0
        Args:
            explanation (AdditiveFeatureContributionExplanationType):
                The explanation to be transformed

        Returns:
            Returns:
                AdditiveFeatureContributionExplanationType:
                    The transformed explanation

        """
        explanation_df = explanation.get()
        for col in self.dropped_columns:
            explanation_df[col] = 0
        return AdditiveFeatureContributionExplanation(explanation_df)

    def inverse_transform_explanation_feature_importance(self, explanation):
        """
        Sets the importance of dropped features to 0

        Args:
            explanation (FeatureImportanceExplanation):
                The explanation to be transformed
        Returns:
            FeatureImportanceExplanation:
                The transformed explanation

        """
        explanation_df = explanation.get()
        for col in self.dropped_columns:
            explanation_df[col] = 0
        return FeatureImportanceExplanation(explanation_df)
