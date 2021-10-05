import pandas as pd

from pyreal.types.explanations.base import ExplanationType


class DataFrameExplanationType(ExplanationType):
    @staticmethod
    def validate(explanation):
        if not isinstance(explanation, pd.DataFrame):
            raise AssertionError("DataFrame explanations must be of type DataFrame")


class FeatureImportanceExplanationType(DataFrameExplanationType):
    @staticmethod
    def validate(explanation):
        super(FeatureImportanceExplanationType, FeatureImportanceExplanationType).validate(
            explanation)


class AdditiveFeatureImportanceExplanationType(FeatureImportanceExplanationType):
    @staticmethod
    def validate(explanation):
        super(AdditiveFeatureImportanceExplanationType,
              AdditiveFeatureImportanceExplanationType).validate(explanation)


class FeatureContributionExplanationType(DataFrameExplanationType):
    @staticmethod
    def validate(explanation):
        super(FeatureContributionExplanationType, FeatureContributionExplanationType).validate(
            explanation)


class AdditiveFeatureContributionExplanationType(FeatureContributionExplanationType):
    @staticmethod
    def validate(explanation):
        super(AdditiveFeatureContributionExplanationType,
              AdditiveFeatureContributionExplanationType).validate(explanation)
