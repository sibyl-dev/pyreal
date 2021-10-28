from pyreal.transformers import BaseTransformer


class FeatureSelectTransformer(BaseTransformer):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, data):
        return data[self.columns]


class ColumnDropTransformer(BaseTransformer):
    """
    Removes columns that should not be predictive
    """

    def __init__(self, columns):
        self.columns = columns

    def transform(self, x):
        return x.drop(self.columns, axis="columns")

    def transform_explanation_shap(self, explanation):
        for col in self.columns:
            explanation[col] = 0
        return explanation

    def transform_explanation_permutation_importance(self, explanation):
        for col in self.columns:
            explanation[col] = 0
        return explanation
