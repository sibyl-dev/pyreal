from pyreal.transformers import BaseTransformer


class FeatureSelectTransformer(BaseTransformer):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def transform(self, data):
        return data[self.feature_names]


class ColumnDropTransformer(BaseTransformer):
    """
    Removes columns that should not be predictive
    """

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def transform(self, x):
        return x.drop(self.columns_to_drop, axis="columns")

    def transform_explanation_shap(self, explanation):
        for col in self.columns_to_drop:
            explanation[col] = 0
        return explanation

    def transform_explanation_permutation_importance(self, explanation):
        for col in self.columns_to_drop:
            explanation[col] = 0
        return explanation
