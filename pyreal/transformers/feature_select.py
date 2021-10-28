from pyreal.transformers import Transformer


class FeatureSelectTransformer(Transformer):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, data):
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
