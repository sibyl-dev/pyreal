import pandas as pd

from pyreal.types.explanations.base_explanation import ExplanationType


class DataFrameExplanationType(ExplanationType):
    def validate(self):
        super().validate()
        if not isinstance(self.explanation, pd.DataFrame):
            raise AssertionError("DataFrame explanations must be of type DataFrame")


class FeatureImportanceExplanationType(DataFrameExplanationType):
    def validate(self):
        super().validate()
        if self.explanation.shape[0] > 1:
            raise AssertionError("Global Feature Importance Explanations can have only one row")


class AdditiveFeatureImportanceExplanationType(FeatureImportanceExplanationType):
    def validate(self):
        super().validate()


class FeatureContributionExplanationType(DataFrameExplanationType):
    def validate(self):
        super().validate()


class AdditiveFeatureContributionExplanationType(FeatureContributionExplanationType):
    def validate(self):
        super().validate()
