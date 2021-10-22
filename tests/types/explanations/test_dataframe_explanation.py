import pandas as pd

from pyreal.types.explanations.dataframe_explanation import DataFrameExplanationType


def test_dataframe_explanation_type():
    valid_explanation = pd.DataFrame([[1, 1, 1]], columns=["A", "B", "C"])
    invalid_explanation = [[1, 1, 1]]
    explanation = DataFrameExplanationType(valid_explanation)
    assert explanation.validate(valid_explanation)

    assert explanation.get() == base_explanation
