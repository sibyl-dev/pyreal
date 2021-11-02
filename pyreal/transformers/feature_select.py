from pyreal.transformers import Transformer
from pyreal.types.explanations.dataframe import AdditiveFeatureContributionExplanation


class FeatureSelectTransformer(Transformer):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, data):
        return data[self.columns]


class ColumnDropTransformer(Transformer):
    """
    Removes columns that should not be predictive
    """

    def __init__(self, columns):
        self.columns = columns

    def transform(self, x):
        return x.drop(self.columns, axis="columns")

    def transform_explanation_additive_contributions(self, explanation):
        explanation_df = explanation.get()
        for col in self.columns:
            explanation_df[col] = 0
        return AdditiveFeatureContributionExplanation(explanation_df)

    def transform_explanation_feature_importance(self, explanation):
        for col in self.columns:
            explanation[col] = 0
        return explanation
