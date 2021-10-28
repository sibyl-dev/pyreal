import pandas as pd
import pytest

from pyreal.types.explanations.dataframe import (
    AdditiveFeatureContributionExplanationType, AdditiveFeatureImportanceExplanationType,
    DataFrameExplanationType, FeatureContributionExplanationType,
    FeatureImportanceExplanationType,)


def helper_test_dataframe_explanations(cls, valid_explanation, invalid_explanation):
    explanation = cls(valid_explanation)
    explanation.validate()  # assert does not raise
    assert explanation.get() is valid_explanation

    with pytest.raises(AssertionError):
        cls(invalid_explanation)
    # skip over the init validation to check validate separately
    explanation.explanation = invalid_explanation
    with pytest.raises(AssertionError):
        explanation.validate()


def test_dataframe_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    invalid_explanation = [[1, 1, 1]]

    helper_test_dataframe_explanations(DataFrameExplanationType, valid_explanation,
                                       invalid_explanation)


def test_feature_importance_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    invalid_explanation = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=["A", "B", "C"])

    helper_test_dataframe_explanations(FeatureImportanceExplanationType,
                                       valid_explanation, invalid_explanation)


def test_additive_feature_importance_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    invalid_explanation = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=["A", "B", "C"])

    helper_test_dataframe_explanations(AdditiveFeatureImportanceExplanationType,
                                       valid_explanation, invalid_explanation)


def test_feature_contributions_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=["A", "B", "C"])
    invalid_explanation = [[1, 1, 1], [2, 2, 2]]

    helper_test_dataframe_explanations(FeatureContributionExplanationType,
                                       valid_explanation, invalid_explanation)


def test_additive_feature_contributions_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=["A", "B", "C"])
    invalid_explanation = [[1, 1, 1], [2, 2, 2]]

    helper_test_dataframe_explanations(AdditiveFeatureContributionExplanationType,
                                       valid_explanation, invalid_explanation)