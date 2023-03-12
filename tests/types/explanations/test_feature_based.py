import pandas as pd
import pytest

from pyreal.types.explanations.feature_based import (
    AdditiveFeatureContributionExplanation,
    AdditiveFeatureImportanceExplanation,
    FeatureBased,
    FeatureContributionExplanation,
    FeatureImportanceExplanation,
)


def helper_test_dataframe_explanations(
    cls, valid_explanation, invalid_explanation, values=None, invalid_values=None
):
    explanation = cls(valid_explanation, values)
    explanation.validate()  # assert does not raise
    assert explanation.get() is valid_explanation

    with pytest.raises(AssertionError):
        cls(invalid_explanation, values)
    # skip over the init validation to check validate separately
    explanation.explanation = invalid_explanation
    with pytest.raises(AssertionError):
        explanation.validate()

    if invalid_values is not None:
        with pytest.raises(AssertionError):
            cls(invalid_explanation, invalid_values)


def test_dataframe_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    invalid_explanation = [[1, 1, 1]]

    helper_test_dataframe_explanations(FeatureBased, valid_explanation, invalid_explanation)


def test_feature_importance_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    invalid_explanation = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=["A", "B", "C"])

    helper_test_dataframe_explanations(
        FeatureImportanceExplanation, valid_explanation, invalid_explanation
    )

    helper_test_dataframe_explanations(
        AdditiveFeatureImportanceExplanation, valid_explanation, invalid_explanation
    )


def test_feature_contributions_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=["A", "B", "C"])
    invalid_explanation = [[1, 1, 1], [2, 2, 2]]
    values = valid_explanation.copy()
    invalid_values = pd.DataFrame([1, 1])

    helper_test_dataframe_explanations(
        FeatureContributionExplanation,
        valid_explanation,
        invalid_explanation,
        values,
        invalid_values,
    )

    helper_test_dataframe_explanations(
        AdditiveFeatureContributionExplanation,
        valid_explanation,
        invalid_explanation,
        values,
        invalid_values,
    )
