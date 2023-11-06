import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pyreal.explainers import Explainer, LocalFeatureContribution
from pyreal.explanation_types import AdditiveFeatureContributionExplanation
from pyreal.transformers import BreakingTransformError, FeatureSelectTransformer, Transformer


def breaking_transform(explanation):
    raise BreakingTransformError


def test_init_invalid_transforms(regression_no_transforms):
    invalid_transform = "invalid"
    with pytest.raises(TypeError):
        Explainer(
            regression_no_transforms["model"],
            regression_no_transforms["x"],
            transformers=invalid_transform,
            scope="testing",
        )


def test_init_invalid_model():
    invalid_model = []
    with pytest.raises(TypeError):
        Explainer(invalid_model, pd.DataFrame([0]), scope="testing")


def test_transform_to_functions(regression_one_hot):
    x = pd.DataFrame([[2, 1, 3], [4, 3, 4], [6, 7, 2]], columns=["A", "B", "C"])
    expected = pd.DataFrame(
        [[1, 3, 1, 0, 0], [3, 4, 0, 1, 0], [7, 2, 0, 0, 1]],
        columns=["B", "C", "A_2", "A_4", "A_6"],
    )

    regression_one_hot["transformers"].set_flags(model=True, interpret=True)
    feature_select = FeatureSelectTransformer(columns=["B", "A_2"], algorithm=False, model=True)
    explainer = Explainer(
        regression_one_hot["model"],
        x,
        transformers=[regression_one_hot["transformers"], feature_select],
        scope="testing",
    )
    result = explainer.transform_to_x_interpret(x)
    assert_frame_equal(result, expected, check_like=True, check_dtype=False)
    result = explainer.transform_to_x_model(x)
    assert_frame_equal(result, expected[["B", "A_2"]], check_like=True, check_dtype=False)
    result = explainer.transform_to_x_algorithm(x)
    assert_frame_equal(result, expected, check_like=True, check_dtype=False)


def test_transform_to_functions_series(regression_one_hot):
    x = pd.Series([2, 1, 3], index=["A", "B", "C"])
    expected = pd.Series([1, 3, 1, 0, 0], index=["B", "C", "A_2", "A_4", "A_6"])

    regression_one_hot["transformers"].set_flags(model=True, interpret=True)
    feature_select = FeatureSelectTransformer(columns=["B", "A_2"], algorithm=False, model=True)
    explainer = Explainer(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=[regression_one_hot["transformers"], feature_select],
        scope="testing",
    )
    result = explainer.transform_to_x_interpret(x)
    assert_series_equal(result, expected, check_dtype=False)
    result = explainer.transform_to_x_model(x)
    assert_series_equal(result, expected[["B", "A_2"]], check_dtype=False)
    result = explainer.transform_to_x_algorithm(x)
    assert_series_equal(result, expected, check_dtype=False)


def test_transform_x_from_algorithm_to_model(regression_one_hot):
    x = pd.DataFrame([[2, 1, 3], [4, 3, 4]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[2, 1], [4, 3]], columns=["A", "B"])

    x_series = pd.Series([2, 1, 3], index=["A", "B", "C"])
    expected_series = pd.Series([2, 1], index=["A", "B"])

    regression_one_hot["transformers"].set_flags(model=True, interpret=True)
    feature_select = FeatureSelectTransformer(columns=["A", "B"], algorithm=False, model=True)
    explainer = Explainer(
        regression_one_hot["model"],
        regression_one_hot["x"],
        transformers=[regression_one_hot["transformers"], feature_select],
        scope="testing",
    )
    result = explainer.transform_x_from_algorithm_to_model(x)
    assert_frame_equal(result, expected, check_like=True, check_dtype=False)
    result_series = explainer.transform_x_from_algorithm_to_model(x_series)
    assert_series_equal(result_series, expected_series, check_dtype=False)


def test_predict_regression(regression_no_transforms, regression_one_hot):
    model = regression_no_transforms
    explainer = Explainer(
        model["model"], model["x"], transformers=model["transformers"], scope="testing"
    )
    expected = np.array(model["y"]).reshape(-1)
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)

    model = regression_one_hot
    explainer = Explainer(
        model["model"], model["x"], transformers=model["transformers"], scope="testing"
    )
    expected = np.array(model["y"]).reshape(-1)
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)


def test_predict_classification(classification_no_transforms):
    model = classification_no_transforms
    explainer = Explainer(
        model["model"], model["x"], transformers=model["transformers"], scope="testing"
    )
    expected = np.array(model["y"])
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)


def test_evaluate_model(regression_no_transforms):
    explainer = LocalFeatureContribution(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        y_train=regression_no_transforms["y"],
    )
    score = explainer.evaluate_model("accuracy")
    assert score == 1

    score = explainer.evaluate_model("neg_mean_squared_error")
    assert score == 0

    new_y = regression_no_transforms["x"].iloc[:, 0:1].copy()
    new_y.iloc[0, 0] = 0
    explainer = LocalFeatureContribution(
        regression_no_transforms["model"], regression_no_transforms["x"], y_train=new_y
    )
    score = explainer.evaluate_model("accuracy")
    assert abs(score - 0.6667) <= 0.0001


def test_evaluate_model_no_dataset_on_init(regression_no_transforms):
    x = regression_no_transforms["x"]
    y = regression_no_transforms["y"]
    explainer = LocalFeatureContribution(
        regression_no_transforms["model"],
    )
    score = explainer.evaluate_model("accuracy", x, y)
    assert score == 1

    score = explainer.evaluate_model("neg_mean_squared_error", x, y)
    assert score == 0


def test_transform_explanation(regression_no_transforms):
    feature_select1 = FeatureSelectTransformer(["A", "B"])
    feature_select2 = FeatureSelectTransformer(["C"], algorithm=False, interpret=True)
    x = pd.DataFrame([[1, 1, 1, 1], [1, 1, 1, 1]], columns=["A", "B", "C", "D"])
    feature_select1.fit(x)
    feature_select2.fit(x)

    explainer = LocalFeatureContribution(
        regression_no_transforms["model"],
        x,
        y_train=regression_no_transforms["y"],
        transformers=[feature_select1, feature_select2],
    )

    explanation_base = pd.DataFrame([[1, 2], [1, 2]], columns=["A", "B"])
    explanation = AdditiveFeatureContributionExplanation(explanation_base)

    transform_explanation = explainer.transform_explanation(explanation, x).get()
    expected_explanation = pd.DataFrame([[0], [0]], columns=["C"])
    assert_frame_equal(transform_explanation, expected_explanation)

    explanation_base = pd.DataFrame([[1, 2], [1, 2]], columns=["A", "B"])
    explanation = AdditiveFeatureContributionExplanation(explanation_base)

    feature_select1.inverse_transform_explanation_additive_feature_contribution = (
        breaking_transform
    )

    transform_explanation = explainer.transform_explanation(explanation, x).get()
    expected_explanation = pd.DataFrame([[1, 2], [1, 2]], columns=["A", "B"])
    assert_frame_equal(transform_explanation, expected_explanation)


def test_transform_x_with_produce(regression_no_transforms):
    class SubtractTransformer(Transformer):
        def __init__(self, n, **kwargs):
            self.n = n
            super().__init__(**kwargs)

        def data_transform(self, x):
            return x - self.n

        def inverse_transform_explanation_additive_feature_contribution(self, explanation):
            return AdditiveFeatureContributionExplanation(explanation.get() + self.n)

    x = pd.DataFrame([[0, 1, 2, 3], [0, 1, 2, 3]], columns=["A", "B", "C", "D"])
    explanation_base = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["A", "B", "C"])
    explanation = AdditiveFeatureContributionExplanation(explanation_base)

    feature_select1 = FeatureSelectTransformer(columns=["A", "B", "C"]).fit(x)
    subtract_1 = SubtractTransformer(1)
    subtract_2 = SubtractTransformer(2, model=False, interpret=True)
    subtract_3 = SubtractTransformer(3, model=False, interpret=True)

    explainer = LocalFeatureContribution(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        y_train=regression_no_transforms["y"],
        transformers=[feature_select1, subtract_1, subtract_2, subtract_3],
    )
    transform_explanation = explainer.transform_explanation(explanation, x)
    expected_x = pd.DataFrame([[-5, -4, -3, -2], [-5, -4, -3, -2]], columns=["A", "B", "C", "D"])
    assert_frame_equal(transform_explanation.get_values(), expected_x)

    explanation_base = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["A", "B", "C"])
    explanation = AdditiveFeatureContributionExplanation(explanation_base)

    subtract_3.transform_explanation = breaking_transform
    transform_explanation = explainer.transform_explanation(explanation, x)
    expected_x = pd.DataFrame([[-2, -1, 0, 1], [-2, -1, 0, 1]], columns=["A", "B", "C", "D"])
    assert_frame_equal(transform_explanation.get_values(), expected_x)

    explanation_base = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["A", "B", "C"])
    explanation = AdditiveFeatureContributionExplanation(explanation_base)

    feature_select1.inverse_transform_explanation = breaking_transform
    transform_explanation = explainer.transform_explanation(explanation, x)
    expected_x = pd.DataFrame([[0, 1, 2], [0, 1, 2]], columns=["A", "B", "C"])
    assert_frame_equal(transform_explanation.get_values(), expected_x)


def test_break(regression_no_transforms):
    feature_select = FeatureSelectTransformer(["A"])
    feature_select.fit(pd.DataFrame([[1, 2, 3]], columns=["A", "B", "C"]))

    explanation = 10  # invalid explanation type

    explainer = LocalFeatureContribution(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=feature_select,
    )
    with pytest.raises(ValueError):
        explainer.transform_explanation(explanation)


def test_fit_transformer_param(regression_no_transforms):
    feature_select1 = FeatureSelectTransformer(["A", "B"], model=True)
    feature_select2 = FeatureSelectTransformer(["C"], model=False, interpret=True)
    x = pd.DataFrame([[1, 1, 1, 1]], columns=["A", "B", "C", "D"])

    explainer = LocalFeatureContribution(
        regression_no_transforms["model"],
        x,
        y_train=regression_no_transforms["y"],
        transformers=[feature_select1, feature_select2],
        fit_transformers=True,
    )

    explanation = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], columns=["A", "B", "C", "D"])
    explanation = AdditiveFeatureContributionExplanation(explanation, explanation.copy())

    transform_explanation = explainer.transform_explanation(explanation).get()
    expected_explanation = pd.DataFrame([[0], [0]], columns=["C"])
    assert_frame_equal(transform_explanation, expected_explanation)


def test_no_dataset_on_init(regression_no_transforms):
    model = regression_no_transforms["model"]
    explainer = Explainer(model, scope="testing")
    assert explainer.x_train_orig is None
