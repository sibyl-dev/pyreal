import pandas as pd
from pandas.testing import assert_frame_equal

from pyreal.transformers import (
    FeatureSelectTransformer,
    OneHotEncoder,
    Transformer,
    base,
    fit_transformers,
    run_transformers,
)
from pyreal.types.explanations.feature_based import FeatureImportanceExplanation, FeatureContributionExplanation


class TestTransformer(Transformer):
    def data_transform(self, x):
        return x

    # noinspection PyMethodMayBeStatic
    def transform_explanation_feature_based(self, explanation):
        explanation.get().iloc[0, 0] = "A"
        return explanation

    # noinspection PyMethodMayBeStatic
    def inverse_transform_explanation_feature_based(self, explanation):
        explanation.get().iloc[0, 1] = "B"
        return explanation


def test_fit_run_transformers():
    x_orig = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["A", "B", "C"])

    ohe_transformer = OneHotEncoder(columns=["A"])
    fs_transformer = FeatureSelectTransformer(columns=["A_1", "A_2", "C"])

    transformers = [ohe_transformer, fs_transformer]

    expected_result = pd.DataFrame([[1, 0, 1], [0, 1, 2], [0, 0, 3]], columns=["A_1", "A_2", "C"])

    result_fit = fit_transformers(transformers, x_orig)
    assert_frame_equal(result_fit, expected_result, check_dtype=False)

    result_run = run_transformers(transformers, x_orig)
    assert_frame_equal(result_run, expected_result, check_dtype=False)


def test_display_missing_transform_info():
    # Assert no errors
    base._display_missing_transform_info("A", "B")
    base._display_missing_transform_info_inverse("A", "B")


def test_explanation_transform_transfer():
    test_transformer = TestTransformer()
    explanation = FeatureImportanceExplanation(pd.DataFrame([["1", "2"]]))
    result1 = test_transformer.transform_explanation(explanation)
    assert result1.__class__ == FeatureImportanceExplanation
    assert_frame_equal(result1.get(), pd.DataFrame([["A", "2"]]))

    result2 = test_transformer.inverse_transform_explanation(explanation)
    assert result2.__class__ == FeatureImportanceExplanation
    assert_frame_equal(result2.get(), pd.DataFrame([["A", "B"]]))


def test_inverse_transform_explanation_keeps_values():
    values = pd.DataFrame([["0", "0"]])
    test_transformer = TestTransformer()
    explanation = FeatureContributionExplanation(pd.DataFrame([["1", "2"]]), values)

    result1 = test_transformer.transform_explanation(explanation)
    assert result1.get_values() is values


def test_transform_explanation_keeps_values():
    values = pd.DataFrame([["0", "0"]])
    test_transformer = TestTransformer()
    explanation = FeatureContributionExplanation(pd.DataFrame([["1", "2"]]), values)

    result1 = test_transformer.inverse_transform_explanation(explanation)
    assert result1.get_values() is values
