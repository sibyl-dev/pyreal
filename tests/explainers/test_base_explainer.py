import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyreal.explainers import LocalFeatureContribution
from pyreal.transformers import FeatureSelectTransformer


def test_init_invalid_transforms(regression_no_transforms):
    invalid_transform = "invalid"
    with pytest.raises(TypeError):
        LocalFeatureContribution(regression_no_transforms["model"], regression_no_transforms["x"],
                                 m_transformers=invalid_transform)
    with pytest.raises(TypeError):
        LocalFeatureContribution(regression_no_transforms["model"], regression_no_transforms["x"],
                                 e_transformers=invalid_transform)
    with pytest.raises(TypeError):
        LocalFeatureContribution(regression_no_transforms["model"], regression_no_transforms["x"],
                                 i_transformers=invalid_transform)


def test_init_invalid_model():
    invalid_model = []
    with pytest.raises(TypeError):
        LocalFeatureContribution(invalid_model, pd.DataFrame([0]))


def test_run_transformers(regression_one_hot):
    x = pd.DataFrame([[2, 1, 3],
                      [4, 3, 4],
                      [6, 7, 2]], columns=["A", "B", "C"])
    expected = pd.DataFrame([[1, 3, 1, 0, 0],
                             [3, 4, 0, 1, 0],
                             [7, 2, 0, 0, 1]], columns=["B", "C", "A_2", "A_4", "A_6"])

    regression_one_hot["transformers"].set_flags(model=True, interpret=True)
    feature_select = FeatureSelectTransformer(columns=["B", "A_2"], algorithm=False, model=True)
    explainer = LocalFeatureContribution(regression_one_hot["model"], x,
                                         transformers=[regression_one_hot["transformers"],
                                                       feature_select])
    result = explainer.transform_to_x_interpret(x)
    assert_frame_equal(result, expected, check_like=True, check_dtype=False)
    result = explainer.transform_to_x_model(x)
    assert_frame_equal(result, expected[["B", "A_2"]], check_like=True, check_dtype=False)
    result = explainer.transform_to_x_algorithm(x)
    assert_frame_equal(result, expected, check_like=True, check_dtype=False)


def test_predict_regression(regression_no_transforms, regression_one_hot):
    model = regression_no_transforms
    explainer = LocalFeatureContribution(model["model"], model["x"],
                                         transformers=model["transformers"])
    expected = np.array(model["y"]).reshape(-1)
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)

    model = regression_one_hot
    explainer = LocalFeatureContribution(model["model"], model["x"],
                                         transformers=model["transformers"])
    expected = np.array(model["y"]).reshape(-1)
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)


def test_predict_classification(classification_no_transforms):
    model = classification_no_transforms
    explainer = LocalFeatureContribution(model["model"], model["x"],
                                         transformers=model["transformers"])
    expected = np.array(model["y"])
    result = explainer.model_predict(model["x"])
    assert np.array_equal(result, expected)


def test_evaluate_model(regression_no_transforms):
    explainer = LocalFeatureContribution(regression_no_transforms["model"],
                                         regression_no_transforms["x"],
                                         y_orig=regression_no_transforms["y"])
    score = explainer.evaluate_model("accuracy")
    assert score == 1

    score = explainer.evaluate_model("neg_mean_squared_error")
    assert score == 0

    new_y = regression_no_transforms["x"].iloc[:, 0:1].copy()
    new_y.iloc[0, 0] = 0
    explainer = LocalFeatureContribution(regression_no_transforms["model"],
                                         regression_no_transforms["x"],
                                         y_orig=new_y)
    score = explainer.evaluate_model("accuracy")
    assert abs(score - .6667) <= 0.0001


def test_break(regression_no_transforms):
    feature_select = FeatureSelectTransformer(["A"])
    feature_select.fit(pd.DataFrame([[1, 2, 3]], columns=["A", "B", "C"]))

    explanation = 10  # invalid explanation type

    explainer = LocalFeatureContribution(regression_no_transforms["model"],
                                         regression_no_transforms["x"],
                                         transformers=feature_select)
    with pytest.raises(ValueError):
        explainer.transform_explanation(explanation)
