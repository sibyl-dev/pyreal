import pandas as pd
import pytest

from pyreal.types.explanations.dataframe_explanation import DataFrameExplanationType


def test_dataframe_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    explanation = DataFrameExplanationType(valid_explanation)
    explanation.validate()  # assert does not raise
    assert explanation.get() is valid_explanation

    invalid_explanation = [[1, 1, 1]]
    with pytest.raises(AssertionError):
        DataFrameExplanationType(invalid_explanation)
    # skip over the init validation to check validate separately
    explanation.explanation = invalid_explanation
    with pytest.raises(AssertionError):
        explanation.validate()


def test_feature_importance_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    explanation = DataFrameExplanationType(valid_explanation)
    explanation.validate()  # assert does not raise
    assert explanation.get() is valid_explanation

    invalid_explanation = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=["A", "B", "C"])
    with pytest.raises(AssertionError):
        DataFrameExplanationType(invalid_explanation)
    # skip over the init validation to check validate separately
    explanation.explanation = invalid_explanation
    with pytest.raises(AssertionError):
        explanation.validate()




