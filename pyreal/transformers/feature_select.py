from pyreal.transformers import Transformer


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

    def transform(self, data):
        """
        Reorders and selects the features in data
        Args:
            data:

        Returns:

        """
        return data[self.feature_names]


class ColumnDropTransformer(Transformer):
    """
    Removes columns that should not be predictive
    """

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def transform(self, x):
        return x.drop(self.columns_to_drop, axis="columns")

    def transform_explanation_additive_contributions(self, explanation):
        explanation_df = explanation.get()
        for col in self.columns_to_drop:
            explanation_df[col] = 0
        return AdditiveFeatureContributionExplanationType(explanation_df)

    def transform_explanation_feature_importance(self, explanation):
        for col in self.columns_to_drop:
            explanation[col] = 0
        return explanation
