from pyreal.transformers import Transformer
from pyreal.types.explanations.dataframe import (
    AdditiveFeatureContributionExplanation, FeatureImportanceExplanation,)


class FeatureSelectTransformer(Transformer):
    """
    A transformer that selects and re-orders features to match the model's inputs
    """

    def __init__(self, columns):
        """
        Initializes the transformer

        Args:
            columns (array-like or Index):
                An ordered list of columns to select
        """
        self.columns = columns
        self.dropped_columns = []

    def fit(self, x):
        """
        Saves the columns being dropped

        Args:
            x (DataFrame of shape (n_instances, n_features)):
                The dataset to fit on
        Returns:
            None

        """
        self.dropped_columns = list(set(x.columns) - set(self.columns))

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

    def transform_explanation_additive_contributions(self, explanation):
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

    def transform_explanation_feature_importance(self, explanation):
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


class ColumnDropTransformer(Transformer):
    """
    A transformer that drops a set of columns from the data
    """

    def __init__(self, columns):
        """
        Initializes the transformer

        Args:
            columns (array-like or Index):
                An ordered list of columns to drop
        """
        self.dropped_columns = columns

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

    def transform_explanation_additive_contributions(self, explanation):
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

    def transform_explanation_feature_importance(self, explanation):
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
